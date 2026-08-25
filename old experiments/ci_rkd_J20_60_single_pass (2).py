"""
CI-RKD: J20-inner / 60-outer Student — Single Pass + Hint + FSP
================================================================
Teacher : PGA_Unfold_J20  (120 outer × 20 inner)  — pretrained, frozen
Student : PGA_Unfold_J20_60  (60 outer × 20 inner) — trained with CI-RKD

Fix: single forward pass collects first-K and last-K F in one 60-iter run.
Previous version ran two 60-iter passes = 120 iterations total per batch.

Small batch strategy (B=10, 1 batch/epoch)
-------------------------------------------
RKD-D / RKD-A are WEAK with B=10 — only 45 pairs, signal is sparse.
Dominant losses for small batches:

  Hint (FitNet-style): direct normalised MSE on F feature vectors
    → per-sample, per-layer — B×K_layers independent gradient terms
    → STRONGEST signal regardless of batch size

  FSP (Flow of Solution): Gramian G[l] = feat[l]^T feat[l+1] / D
    → transfers convergence RATE between consecutive outer iters
    → O(B²D) not O(B³), works well at B=10

  RKD kept as secondary signal — still adds relational geometry
  constraint but weighted lower than hint for small batches.

Total loss:
  L = L_task
    + λ_D · L_CI_RKD (early + late)
    + λ_FSP · L_CI_FSP (early + late)
    + λ_hint · L_hint (early + late)
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

# ── Radar covariance cache ────────────────────────────────────────────────────
radar_cache = {}
for _snr_db in snr_dB_list:
    _R, _, _, _ = get_radar_data(_snr_db, H_train[:, :1, :, :])
    radar_cache[_snr_db] = _R.to(device)

Rtest, at, theta, ideal_beam = get_radar_data(snr_dB, H_test.cpu())
Rtest = Rtest.to(device)

def get_R(snr_db, B):
    return radar_cache[snr_db].expand(-1, B, -1, -1)

# ── Iteration counts ──────────────────────────────────────────────────────────
I_T = n_iter_outer        # teacher: 120
I_S = n_iter_outer // 2   # student:  60

# ── Student step_size — fixed init ───────────────────────────────────────────
step_size_student_J20_60 = torch.full(
    [n_iter_inner_J20, I_S, K + 1], step_size_fixed, requires_grad=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def flatten_F(F_mat, B):
    """[K_u, B, Nt, Nrf] complex → [B, K_u·Nt·Nrf] complex"""
    return F_mat.permute(1, 0, 2, 3).reshape(B, -1)

def to_real(x):
    """complex → float32 via view_as_real"""
    if torch.is_complex(x):
        return torch.view_as_real(x).flatten(-2)
    return x


# ══════════════════════════════════════════════════════════════════════════════
# 2.  LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def rkd_distance_loss(teacher, student):
    """RKD distance loss: penalises discrepancy in pairwise L2 distances.
    Works for both real and complex-valued representations."""
    def _to_real(x):
        if torch.is_complex(x):
            return torch.view_as_real(x).flatten(-2)
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
            return torch.view_as_real(x).flatten(-2)
        return x
    t = _to_real(teacher.detach())
    s = _to_real(student)
    with torch.no_grad():
        t_e   = t.unsqueeze(0) - t.unsqueeze(1)
        t_e   = torch.nn.functional.normalize(t_e, p=2, dim=-1)
        t_cos = torch.bmm(t_e, t_e.permute(0, 2, 1))
    s_e   = s.unsqueeze(0) - s.unsqueeze(1)
    s_e   = torch.nn.functional.normalize(s_e, p=2, dim=-1)
    s_cos = torch.bmm(s_e, s_e.permute(0, 2, 1))
    return torch.nn.functional.smooth_l1_loss(s_cos, t_cos)


def ci_rkd_loss(F_T_window, F_S_window, K_layers, B,
                lambda_dist=25.0, lambda_angle=50.0):
    """CI-RKD reversed weights. Secondary signal for small batches."""
    total_dist  = 0.0
    total_angle = 0.0
    weight_sum  = K_layers * (K_layers + 1) / 2
    for l in range(K_layers):
        w_l    = (l + 1) / weight_sum
        t_feat = flatten_F(F_T_window[l], B)
        s_feat = flatten_F(F_S_window[l], B)
        total_dist  += w_l * rkd_distance_loss(t_feat, s_feat)
        total_angle += w_l * rkd_angle_loss(t_feat, s_feat)
    return lambda_dist * total_dist + lambda_angle * total_angle


def ci_fsp_loss(F_T_window, F_S_window, B):
    """
    CI-FSP: Gramian between consecutive outer iteration F matrices.
    G[l] = feat[l]^T · feat[l+1] / D   shape [B, B]
    Transfers convergence RATE — how fast F moves between iterations.
    O(B²·D), works well at B=10.
    Teacher features already detached (came from no_grad block).
    """
    n_pairs = len(F_T_window) - 1
    if n_pairs <= 0:
        return torch.tensor(0.0, device=F_S_window[0].device)

    total = 0.0
    for l in range(n_pairs):
        t_f1 = to_real(flatten_F(F_T_window[l],     B))
        t_f2 = to_real(flatten_F(F_T_window[l + 1], B))
        D    = t_f1.shape[1]
        G_T  = torch.mm(t_f1, t_f2.t()) / D               # detached

        s_f1 = to_real(flatten_F(F_S_window[l],     B))   # in-graph
        s_f2 = to_real(flatten_F(F_S_window[l + 1], B))   # in-graph
        G_S  = torch.mm(s_f1, s_f2.t()) / D

        total += F_nn.mse_loss(G_S, G_T.detach())

    return total / n_pairs


def hint_loss(F_T_window, F_S_window, B):
    """
    FitNet-style hint: normalised MSE on F feature vectors per sample.

    This is the primary loss for small batches.
    With B=10 and K_layers=15 you get 150 independent gradient terms
    per forward pass — much denser than RKD's 45 pairs.

    Reversed weight: higher l = more converged teacher = higher weight.
    Teacher features already detached.
    """
    K_l        = len(F_T_window)
    weight_sum = K_l * (K_l + 1) / 2
    total      = 0.0

    for l in range(K_l):
        w_l   = (l + 1) / weight_sum
        t_f   = to_real(flatten_F(F_T_window[l], B))   # [B, D] detached
        s_f   = to_real(flatten_F(F_S_window[l], B))   # [B, D] in-graph

        # Normalise per-sample before MSE — scale invariant
        t_fn  = F_nn.normalize(t_f.detach(), dim=1)
        s_fn  = F_nn.normalize(s_f,          dim=1)

        total += w_l * F_nn.mse_loss(s_fn, t_fn)

    return total


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TEACHER MODEL
# ══════════════════════════════════════════════════════════════════════════════

class PGA_Unfold_J20_Teacher(PGA_Unfold_J20):

    def execute_PGA_window(self, H, R, Pt, n_iter_outer, n_iter_inner,
                            t_start, K_layers):
        if t_start is None:
            t_start = n_iter_outer - K_layers
        t_end = t_start + K_layers
        assert t_end <= n_iter_outer

        _, _, F, W = initialize(H, R, Pt, initial_normalization)
        F_traj = []

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

            if t_start <= ii < t_end:
                F_traj.append(F.detach().clone())

        return F, W, F_traj


# ══════════════════════════════════════════════════════════════════════════════
# 4.  STUDENT MODEL — J=20 inner, 60 outer, SINGLE PASS
# ══════════════════════════════════════════════════════════════════════════════

class PGA_Unfold_J20_60(PGA_Unfold_J20):
    """
    Student — 60 outer × 20 inner.
    step_size shape: [20, 60, K+1]

    execute_PGA_single_pass: ONE 60-iteration forward pass that collects
      F_traj_first : iters [0, K_layers)      in-graph for early losses
      F_traj_last  : iters [I_S-K_layers, I_S) in-graph for late losses

    K_layers=0: no F stored, pure task loss training.
    """

    def execute_PGA_single_pass(self, H, R, Pt,
                                 n_iter_outer, n_iter_inner, K_layers):
        rate_init, tau_init, F, W = initialize(H, R, Pt, initial_normalization)
        rate_over_iters = torch.zeros(n_iter_outer, H.shape[1], device=H.device)
        tau_over_iters  = torch.zeros(n_iter_outer, H.shape[1], device=H.device)
        I = n_iter_outer

        # Pre-compute which iterations to store
        first_set = set(range(K_layers))
        last_set  = set(range(I - K_layers, I)) if K_layers > 0 else set()

        F_traj_first = []
        F_traj_last  = []

        for ii in range(I):
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

            if ii in first_set:
                F_traj_first.append(F.clone())   # in-graph ✓
            if ii in last_set:
                F_traj_last.append(F.clone())    # in-graph ✓

            rate_over_iters[ii] = get_sum_rate(H, F, W, Pt)
            tau_over_iters[ii]  = get_beam_error(H, F, W, R, Pt)

        rates = torch.cat([rate_init, rate_over_iters], dim=0)
        taus  = torch.cat([tau_init,  tau_over_iters],  dim=0)
        return (torch.transpose(rates, 0, 1),
                torch.transpose(taus,  0, 1),
                F, W, F_traj_first, F_traj_last)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TRAINING — CI-RKD + FSP + Hint
# ══════════════════════════════════════════════════════════════════════════════

if run_RKD_Distillation == 1:

    print('CI-RKD + FSP + Hint: J20/60outer ← J20/120outer teacher')
    print(f'Teacher: {I_T}×20 = {I_T*n_iter_inner_J20} steps/batch')
    print(f'Student: {I_S}×20 = {I_S*n_iter_inner_J20} steps/batch  (single pass)')
    print(f'B={batch_size}  →  Hint+FSP dominant, RKD secondary\n')

    # ── Hyperparameters ───────────────────────────────────────────────────────
    K_layers      = 15
    lambda_dist   = 25.0
    lambda_angle  = 50.0
    lambda_fsp    = 5.0     # FSP — convergence rate signal
    lambda_hint   = 50.0    # Hint — primary loss for small batches
                             # higher than RKD because per-sample signal is denser

    t_start_early = 20                # teacher Zone 2 [20, 35)
    t_start_late  = I_T - K_layers   # teacher [105, 120) → student [45, 60)

    assert t_start_early + K_layers <= I_T
    assert K_layers <= I_S

    print(f'Early: teacher [{t_start_early},{t_start_early+K_layers}) → student [0,{K_layers})')
    print(f'Late:  teacher [{t_start_late},{I_T}) → student [{I_S-K_layers},{I_S})')
    print(f'λ_hint={lambda_hint}  λ_fsp={lambda_fsp}  λ_dist={lambda_dist}\n')

    # ── Teacher ──────────────────────────────────────────────────────────────
    model_teacher = PGA_Unfold_J20_Teacher(step_size_UPGA_J20).to(device)
    model_teacher.load_state_dict(
        torch.load(model_file_name_UPGA_J20, map_location=device))
    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad_(False)
    print(f'Teacher: {list(model_teacher.step_size.shape)}')

    # ── Student ───────────────────────────────────────────────────────────────
    model_student = PGA_Unfold_J20_60(step_size_student_J20_60).to(device)
    optimizer     = torch.optim.Adam(model_student.parameters(), lr=learning_rate)
    print(f'Student: {list(model_student.step_size.shape)}\n')

    for i_epoch in range(n_epoch):
        start_time  = time.time()
        e_loss = e_task = e_rkd = e_fsp = e_hint = e_teacher = 0.0
        num_batches = 0

        H_shuffled = torch.transpose(H_train, 0, 1)[
            np.random.permutation(len(H_train[0]))]

        for i_batch in range(0, len(H_train), batch_size):
            H = torch.transpose(
                H_shuffled[i_batch:i_batch + batch_size], 0, 1).to(device)
            B = H.shape[1]

            snr_dB_train = np.random.choice(snr_dB_list)
            snr_train    = 10 ** (snr_dB_train / 10)
            R = get_R(snr_dB_train, B)

            # ── Teacher: two window passes (frozen, no backward cost) ─────────
            with torch.no_grad():
                F_t, W_t, F_T_early = model_teacher.execute_PGA_window(
                    H, R, snr_train, I_T, n_iter_inner_J20,
                    t_start=t_start_early, K_layers=K_layers)
                teacher_task_loss = get_sum_loss(F_t, W_t, H, R, snr_train, B)

                _, _, F_T_late = model_teacher.execute_PGA_window(
                    H, R, snr_train, I_T, n_iter_inner_J20,
                    t_start=t_start_late, K_layers=K_layers)

            # ── Student: SINGLE 60-iter pass, both windows ────────────────────
            (rate, _, F_s, W_s,
             F_S_first, F_S_last) = model_student.execute_PGA_single_pass(
                H, R, snr_train, I_S, n_iter_inner_J20, K_layers)

            # ── Losses ────────────────────────────────────────────────────────
            loss_task = get_sum_loss(F_s, W_s, H, R, snr_train, B)

            loss_rkd  = (ci_rkd_loss(F_T_early, F_S_first, K_layers, B,
                                      lambda_dist, lambda_angle) +
                         ci_rkd_loss(F_T_late,  F_S_last,  K_layers, B,
                                      lambda_dist, lambda_angle))

            loss_fsp  = (ci_fsp_loss(F_T_early, F_S_first, B) +
                         ci_fsp_loss(F_T_late,  F_S_last,  B))

            loss_hint = (hint_loss(F_T_early, F_S_first, B) +
                         hint_loss(F_T_late,  F_S_last,  B))

            total_loss = (loss_task
                          + loss_rkd
                          + lambda_fsp  * loss_fsp
                          + lambda_hint * loss_hint)

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

            e_loss    += total_loss.item()
            e_task    += loss_task.item()
            e_rkd     += loss_rkd.item()
            e_fsp     += loss_fsp.item()
            e_hint    += loss_hint.item()
            e_teacher += teacher_task_loss.item()
            num_batches += 1

            if device.type == 'cuda':
                torch.cuda.empty_cache()

        nb = max(num_batches, 1)
        print(f"Epoch {i_epoch:4d} | "
              f"Time: {time.time()-start_time:.1f}s | "
              f"Loss: {e_loss/nb:.4f} | "
              f"Task: {e_task/nb:.4f} | "
              f"Teacher: {e_teacher/nb:.4f} | "
              f"RKD: {e_rkd/nb:.4f} | "
              f"FSP: {lambda_fsp*e_fsp/nb:.4f} | "
              f"Hint: {lambda_hint*e_hint/nb:.4f}")

    save_path = (model_file_name_UPGA_J20.replace('J20', 'J20_60outer') +
                 f'_CI_RKD_hint_Kl{K_layers}_win{t_start_early}.pth')
    torch.save(model_student.state_dict(), save_path)
    print(f'\nStudent saved → {save_path}')


# ══════════════════════════════════════════════════════════════════════════════
# 6.  PURE STUDENT — J20/60outer, NO RKD (baseline)
# ══════════════════════════════════════════════════════════════════════════════

if run_J20_I60 == 1:

    print('Pure Student: J20/60outer, task loss only')

    model_student = PGA_Unfold_J20_60(step_size_student_J20_60).to(device)
    optimizer     = torch.optim.Adam(model_student.parameters(), lr=learning_rate)
    print(f'Shape: {list(model_student.step_size.shape)}\n')

    for i_epoch in range(n_epoch):
        start_time  = time.time()
        epoch_loss  = 0.0
        num_batches = 0

        H_shuffled = torch.transpose(H_train, 0, 1)[
            np.random.permutation(len(H_train[0]))]

        for i_batch in range(0, len(H_train), batch_size):
            H = torch.transpose(
                H_shuffled[i_batch:i_batch + batch_size], 0, 1).to(device)
            B = H.shape[1]
            snr_dB_train = np.random.choice(snr_dB_list)
            snr_train    = 10 ** (snr_dB_train / 10)
            R = get_R(snr_dB_train, B)

            (rate, _, F_s, W_s, _, _) = model_student.execute_PGA_single_pass(
                H, R, snr_train, I_S, n_iter_inner_J20, K_layers=0)

            loss_task = get_sum_loss(F_s, W_s, H, R, snr_train, B)

            optimizer.zero_grad()
            loss_task.backward()
            if torch.isnan(loss_task) or torch.isinf(loss_task):
                optimizer.zero_grad(); continue

            torch.nn.utils.clip_grad_norm_(model_student.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                model_student.step_size.data.clamp_(min=1e-8)

            epoch_loss  += loss_task.item()
            num_batches += 1

        nb = max(num_batches, 1)
        print(f"Epoch {i_epoch:4d} [PURE] | "
              f"Time: {time.time()-start_time:.1f}s | "
              f"Loss: {epoch_loss/nb:.4f}")

    save_path = model_file_name_UPGA_J20.replace('J20', 'J20_60outer_PURE') + '.pth'
    torch.save(model_student.state_dict(), save_path)
    print(f'\nPure student saved → {save_path}')

# ══════════════════════════════════════════════════════════════════════════════
# 8.  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

if run_RKD_Distillation == 0:

    K_layers      = 15
    t_start_early = 20

    save_path = (model_file_name_UPGA_J20.replace('J20', 'J20_60outer') +
                 f'_CI_RKD_hint_Kl{K_layers}_win{t_start_early}.pth')

    model_test = PGA_Unfold_J20_60(step_size_student_J20_60).to(device)
    model_test.load_state_dict(torch.load(save_path, map_location=device))
    model_test.eval()

    with torch.no_grad():
        (rate_iter, beam_iter, F_out, W_out, _, _) = model_test.execute_PGA_single_pass(
            H_test, Rtest, snr, I_S, n_iter_inner_J20, K_layers)

    rate_RKD = [r.detach().cpu().numpy()
                for r in (sum(rate_iter) / len(H_test[0]))]
    beam_RKD = [r.detach().cpu().numpy()
                for r in (sum(beam_iter) / len(H_test[0]))]
    iter_number = np.array(list(range(I_S + 1)))

    print(f'\nJ20/60outer CI-RKD+Hint | K_layers={K_layers} | win={t_start_early}')
    print(f'  Final R-wt : {rate_RKD[-1]:.4f}')
    print(f'  Final tau  : {beam_RKD[-1]:.4f}')
