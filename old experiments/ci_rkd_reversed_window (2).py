"""
CI-RKD Distillation — Reversed Weight Schedule
================================================
Teacher : PGA_Unfold_J20  (120 outer × 20 inner, pretrained, frozen)
Student : PGA_Unfold_J10  (120 outer × 10 inner, trained with CI-RKD)

Change from previous version
------------------------------
Weight schedule REVERSED:

  Previous:  w_l = (K_layers - l) / weight_sum
             l=0 (T_106 → S_0) gets strongest weight  ← T_106 not well converged
             l=14 (T_120 → S_14) gets weakest weight  ← T_120 most converged

  This file:  w_l = (l + 1) / weight_sum
             l=0 (T_106 → S_0) gets weakest weight
             l=14 (T_120 → S_0) gets strongest weight  ← most converged teacher
                                                           supervises student's
                                                           first outer iteration

Rationale: the most converged teacher state (T_120) should give the strongest
signal to the student's earliest, most unstable iterations. The previous
schedule did the opposite — weakest supervision where it matters most.

Variable naming
---------------
K        : number of users  (from system_config)
K_layers : number of outer iteration pairs distilled
I        : outer iterations for both models (= n_iter_outer = 120)
"""

import time
import torch
import torch.nn.functional as F_nn
import numpy as np

from system_config import *
from utility import *
from PGA_models import *

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Data ──────────────────────────────────────────────────────────────────────
H_train, H_test0 = get_data_tensor(data_source)
H_test           = H_test0[:, :test_size, :, :].to(device)
torch.manual_seed(3407)

# ── Radar covariance cache — one R per SNR in snr_dB_list ────────────────────
# Matches original training code: random SNR chosen per batch
radar_cache = {}
for _snr_db in snr_dB_list:
    _R, _, _, _ = get_radar_data(_snr_db, H_train[:, :1, :, :])
    radar_cache[_snr_db] = _R.to(device)   # [K_u, 1, Nrf, Nrf]

# Test radar data at evaluation SNR
Rtest, at, theta, ideal_beam = get_radar_data(snr_dB, H_test.cpu())
Rtest = Rtest.to(device)

def get_R(snr_db, B):
    """Return radar covariance for given SNR, expanded to batch size B."""
    return radar_cache[snr_db].expand(-1, B, -1, -1)

I = n_iter_outer   # 120


# ══════════════════════════════════════════════════════════════════════════════
# 1.  FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def flatten_F(F_mat, B):
    """
    F_mat : [K, B, Nt, Nrf]  complex  (K = number of users)
    Returns [B, K*Nt*Nrf]    complex
    """
    return F_mat.permute(1, 0, 2, 3).reshape(B, -1)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  RKD LOSSES  (exact implementations, unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def rkd_distance_loss(teacher, student):
    """RKD distance loss: penalises discrepancy in pairwise L2 distances.
    Works for both real and complex-valued representations."""
    def _to_real(x):
        if torch.is_complex(x):
            return torch.view_as_real(x).flatten(-2)  # (N, 2D)
        return x
    t = _to_real(teacher.detach())
    s = _to_real(student)
    with torch.no_grad():
        t_dist = torch.cdist(t, t, p=2)
        t_dist = t_dist / t_dist.mean().clamp(min=1e-12)
    s_dist = torch.cdist(s, s, p=2)
    s_dist = s_dist / s_dist.mean().clamp(min=1e-12)
    return torch.nn.functional.smooth_l1_loss(s_dist, t_dist)


def rkd_angle_loss(teacher, student):
    """RKD angle loss: penalises discrepancy in triplet cosine angles.
    Works for both real and complex-valued representations."""
    def _to_real(x):
        if torch.is_complex(x):
            return torch.view_as_real(x).flatten(-2)  # (N, 2D)
        return x
    t = _to_real(teacher.detach())  # (N, D)
    s = _to_real(student)           # (N, D)
    with torch.no_grad():
        t_e = t.unsqueeze(0) - t.unsqueeze(1)          # (N, N, D)
        t_e = torch.nn.functional.normalize(t_e, p=2, dim=-1)
        t_cos = torch.bmm(t_e, t_e.permute(0, 2, 1))  # (N, N, N)
    s_e = s.unsqueeze(0) - s.unsqueeze(1)              # (N, N, D)
    s_e = torch.nn.functional.normalize(s_e, p=2, dim=-1)
    s_cos = torch.bmm(s_e, s_e.permute(0, 2, 1))      # (N, N, N)
    return torch.nn.functional.smooth_l1_loss(s_cos, t_cos)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CROSS-ITERATION RKD LOSS — REVERSED WEIGHT SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════

def ci_rkd_loss(F_T_last_Kl, F_S_first_Kl, K_layers, B,
                lambda_dist=25.0, lambda_angle=50.0):
    """
    Cross-Iteration RKD with REVERSED weight schedule.

    Matching (l = 0 ... K_layers-1):
        teacher F_T_last_Kl[l]   outer iter  I - K_layers + l
        ↕
        student F_S_first_Kl[l]  outer iter  l

    In list order:
        F_T_last_Kl[0]  = F_T(I - K_layers)   least  converged of the K_layers
        F_T_last_Kl[K-1]= F_T(I)              most converged

    REVERSED weight: w_l = (l + 1) / sum(1..K_layers)
        l=0       → w≈0   weakest:   least converged teacher → student early iter
        l=K_layers-1 → w=1  strongest: most converged teacher → student early iter

    This means F_T(I) (the best teacher state) gives the strongest gradient
    signal to F_S(K_layers-1) — but note the student index also increases with l.

    To give the STRONGEST signal specifically to student iter 0 from the MOST
    converged teacher, we use a cross-weighting approach:
    the weight at position l reflects how converged the teacher is at that index
    (higher l = more converged teacher = higher weight).

    Comparison with previous schedule:
        Previous: w_l = (K_layers - l) / weight_sum  → T(I-K) gets most weight
        This:     w_l = (l + 1)        / weight_sum  → T(I)   gets most weight
    """
    total_dist  = 0.0
    total_angle = 0.0
    weight_sum  = K_layers * (K_layers + 1) / 2   # sum of 1..K_layers

    for l in range(K_layers):
        # REVERSED: l=0 gets weight 1/weight_sum, l=K_layers-1 gets K_layers/weight_sum
        w_l    = (l + 1) / weight_sum

        t_feat = flatten_F(F_T_last_Kl[l], B)   # [B, K*Nt*Nrf] complex, detached
        s_feat = flatten_F(F_S_first_Kl[l], B)   # [B, K*Nt*Nrf] complex, in-graph

        total_dist  += w_l * rkd_distance_loss(t_feat, s_feat)
        total_angle += w_l * rkd_angle_loss(t_feat, s_feat)

    return lambda_dist * total_dist + lambda_angle * total_angle


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MODEL SUBCLASSES  (identical to previous version)
# ══════════════════════════════════════════════════════════════════════════════

class PGA_Unfold_J20_CI(PGA_Unfold_J20):
    """Teacher — stores only last K_layers F tensors (not all 120)."""

    def execute_PGA_last_Kl(self, H, R, Pt, n_iter_outer, n_iter_inner, K_layers):
        _, _, F, W = initialize(H, R, Pt, initial_normalization)
        F_traj_last = []

        for ii in range(n_iter_outer):
            for jj in range(n_iter_inner):
                grad_F_com  = get_grad_F_com(H, F, W)
                grad_F_rad  = get_grad_F_rad(F, W, R)
                delta_F_com = self.step_size[jj][ii][0] * grad_F_com
                delta_F_rad = self.step_size[jj][ii][0] * grad_F_rad
                F = F + delta_F_com * WEIGHT_F_COM - delta_F_rad * WEIGHT_F_RAD
                if sum(torch.abs(F[0, :, 0, 0])) > 1e3:
                    F = normalize_power(F, W, H, Pt)
            F = F / torch.abs(F)

            W_new      = W.clone().detach()
            grad_W_com = get_grad_W_com(H, F, W)
            grad_W_rad = get_grad_W_rad(F, W, R)
            for k in range(K):
                delta_W_com = self.step_size[0][ii][k+1] * grad_W_com[k]
                delta_W_rad = self.step_size[0][ii][k+1] * grad_W_rad[k]
                W_new[k] = (W[k].clone().detach()
                            + delta_W_com * WEIGHT_W_COM
                            - delta_W_rad * WEIGHT_W_RAD)
            F, W = normalize(F, W_new, H, Pt)

            if ii >= n_iter_outer - K_layers:
                F_traj_last.append(F.detach().clone())

        return F, W, F_traj_last


class PGA_Unfold_J10_CI(PGA_Unfold_J10):
    """Student — 120 outer × 10 inner. Stores first K_layers F for CI-RKD."""

    def execute_PGA_with_first_Kl(self, H, R, Pt,
                                    n_iter_outer, n_iter_inner, K_layers):
        rate_init, tau_init, F, W = initialize(H, R, Pt, initial_normalization)
        rate_over_iters = torch.zeros(n_iter_outer, H.shape[1], device=H.device)
        tau_over_iters  = torch.zeros(n_iter_outer, H.shape[1], device=H.device)
        F_traj = []

        for ii in range(n_iter_outer):
            for jj in range(n_iter_inner):
                grad_F_com  = get_grad_F_com(H, F, W)
                grad_F_rad  = get_grad_F_rad(F, W, R)
                delta_F_com = self.step_size[jj][ii][0] * grad_F_com
                delta_F_rad = self.step_size[jj][ii][0] * grad_F_rad
                F = F + delta_F_com * WEIGHT_F_COM - delta_F_rad * WEIGHT_F_RAD
                F = normalize_power(F, W, H, Pt)
            F = F / torch.abs(F)

            W_new      = W.clone().detach()
            grad_W_com = get_grad_W_com(H, F, W)
            grad_W_rad = get_grad_W_rad(F, W, R)
            for k in range(K):
                delta_W_com = self.step_size[0][ii][k+1] * grad_W_com[k]
                delta_W_rad = self.step_size[0][ii][k+1] * grad_W_rad[k]
                W_new[k] = (W[k].clone().detach()
                            + delta_W_com * WEIGHT_W_COM
                            - delta_W_rad * WEIGHT_W_RAD)
            F, W = normalize(F, W_new, H, Pt)

            if ii < K_layers:
                F_traj.append(F.clone())   # in-graph ✓

            rate_over_iters[ii] = get_sum_rate(H, F, W, Pt)
            tau_over_iters[ii]  = get_beam_error(H, F, W, R, Pt)

        rates = torch.cat([rate_init, rate_over_iters], dim=0)
        taus  = torch.cat([tau_init,  tau_over_iters],  dim=0)
        return (torch.transpose(rates, 0, 1),
                torch.transpose(taus,  0, 1),
                F, W, F_traj)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

if run_RKD_Distillation == 1:

    print('Running CI-RKD Distillation — REVERSED weight schedule')
    print(f'Teacher: I={I} outer × J=20 inner = {I*n_iter_inner_J20} steps/batch')
    print(f'Student: I={I} outer × J=10 inner = {I*n_iter_inner_J10} steps/batch')
    print(f'Weight: w_l = (l+1)/sum(1..K_layers)  →  T(I) gets strongest signal')
    print(f'SNR list = {snr_dB_list}  |  K (users) = {K}  |  device = {device}\n')

    # ── Hyperparameters ───────────────────────────────────────────────────────
    K_layers     = 10
    lambda_dist  = 25.0
    lambda_angle = 50.0
    E_warmup     = 40

    assert K_layers < I

    # ── Teacher ──────────────────────────────────────────────────────────────
    model_teacher = PGA_Unfold_J20_CI(step_size_UPGA_J20).to(device)
    model_teacher.load_state_dict(
        torch.load(model_file_name_UPGA_J20, map_location=device))
    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad_(False)
    print(f'Teacher loaded: {model_file_name_UPGA_J20}')

    # ── Student ───────────────────────────────────────────────────────────────
    model_student = PGA_Unfold_J10_CI(step_size_UPGA_J10).to(device)
    optimizer     = torch.optim.Adam(model_student.parameters(), lr=learning_rate)
    print(f'Student: PGA_Unfold_J10_CI  step_size shape = '
          f'{list(model_student.step_size.shape)}')
    print(f'K_layers={K_layers}  |  batch={batch_size}  |  warmup={E_warmup} epochs\n')

    for i_epoch in range(n_epoch):
        start_time         = time.time()
        epoch_loss         = 0.0
        epoch_rkd          = 0.0
        epoch_task_student = 0.0
        epoch_task_teacher = 0.0
        num_batches        = 0
        in_warmup          = 0 #(i_epoch < E_K_l)

        H_shuffled = torch.transpose(H_train, 0, 1)[
            np.random.permutation(len(H_train[0]))]  # shuffle N_train samples

        for i_batch in range(0, len(H_train), batch_size):  # iterate over K_u dim
            H = torch.transpose(
                H_shuffled[i_batch:i_batch + batch_size], 0, 1).to(device)
            B = H.shape[1]

            # Random SNR per batch — matches original training code
            snr_dB_train = np.random.choice(snr_dB_list)
            snr_train    = 10 ** (snr_dB_train / 10)
            R = get_R(snr_dB_train, B)

            # ── Teacher forward ───────────────────────────────────────────────
            with torch.no_grad():
                F_t, W_t, F_T_last_Kl = model_teacher.execute_PGA_last_Kl(
                    H, R, snr_train, I, n_iter_inner_J20, K_layers)
                teacher_task_loss = get_sum_loss(F_t, W_t, H, R, snr_train, B)

            # ── Student forward ───────────────────────────────────────────────
            rate, _, F_s, W_s, F_S_first_Kl = model_student.execute_PGA_with_first_Kl(
                H, R, snr_train, I, n_iter_inner_J10, K_layers)

            # ── Task loss ─────────────────────────────────────────────────────
            loss_task = get_sum_loss(F_s, W_s, H, R, snr_train, B)

            # ── CI-RKD loss ───────────────────────────────────────────────────
            if in_warmup:
                total_loss = loss_task
                loss_rkd   = torch.tensor(0.0, device=device)
            else:
                loss_rkd = ci_rkd_loss(
                    F_T_last_Kl, F_S_first_Kl,
                    K_layers     = K_layers,
                    B            = B,
                    lambda_dist  = lambda_dist,
                    lambda_angle = lambda_angle)
                total_loss = loss_task + loss_rkd

            # ── Backward + update ─────────────────────────────────────────────
            optimizer.zero_grad()
            total_loss.backward()

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                optimizer.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(model_student.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                model_student.step_size.data.clamp_(min=1e-8)

            epoch_loss         += total_loss.item()
            epoch_rkd          += loss_rkd.item() if not in_warmup else 0.0
            epoch_task_student += loss_task.item()
            epoch_task_teacher += teacher_task_loss.item()
            num_batches        += 1

            if device.type == 'cuda':
                torch.cuda.empty_cache()

        nb  = max(num_batches, 1)
        tag = "warmup" if in_warmup else "CI-RKD"
        print(
            f"Epoch {i_epoch:4d} [{tag}] | "
            f"Time: {time.time()-start_time:.1f}s | "
            f"Loss: {epoch_loss/nb:.4f} | "
            f"Student: {epoch_task_student/nb:.4f} | "
            f"Teacher: {epoch_task_teacher/nb:.4f} | "
            f"RKD: {epoch_rkd/nb:.4f}"
        )

    save_path = model_file_name_UPGA_J10 + f'_CI_RKD_reversed_Kl_no warmup{K_layers}.pth'
    torch.save(model_student.state_dict(), save_path)
    print(f'\nStudent saved → {save_path}')


# ══════════════════════════════════════════════════════════════════════════════
# 6.  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

if run_RKD_Distillation == 0:

    K_layers = 15   # must match training

    model_test = PGA_Unfold_J10_CI(step_size_UPGA_J10).to(device)
    model_test.load_state_dict(
        torch.load(model_file_name_UPGA_J10 + f'_CI_RKD_reversed_warmup10_Kl{K_layers}.pth',
                   map_location=device))
    model_test.eval()

    with torch.no_grad():
        (rate_iter_RKD, beam_error_iter_RKD,
         F_RKD, W_RKD, _) = model_test.execute_PGA_with_first_Kl(
            H_test, Rtest, snr, I, n_iter_inner_J10,
            K_layers=K_layers)

    rate_RKD = [r.detach().cpu().numpy()
                for r in (sum(rate_iter_RKD) / len(H_test[0]))]
    beam_error_RKD = [r.detach().cpu().numpy()
                      for r in (sum(beam_error_iter_RKD) / len(H_test[0]))]
    iter_number_RKD = np.array(list(range(I + 1)))

    print(f'\nCI-RKD Reversed  J=10 | 120 outer layers | K_layers={K_layers}')
    print(f'  Final sum rate   : {rate_RKD[-1]:.4f} bps/Hz')
    print(f'  Final beam error : {beam_error_RKD[-1]:.4f}')
