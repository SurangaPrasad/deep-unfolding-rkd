"""
CI-RKD: J20-inner / 60-outer Student distilled from J20-inner / 120-outer Teacher
====================================================================================
Teacher : PGA_Unfold_J20  (120 outer × 20 inner)  — pretrained, frozen
Student : PGA_Unfold_J20_60  (60 outer × 20 inner) — trained with CI-RKD

Research goal
--------------
Achieve similar or better performance to teacher at I=120
using only I=60 outer iterations — halving inference compute.

Why this student is stronger than J10
---------------------------------------
- J=20 inner steps per outer iter — same inner quality as teacher
- RKD only needs to accelerate outer convergence, not compensate
  for weaker inner updates
- Cleaner ablation: only ONE variable reduced (outer iters)

Windowing strategy
-------------------
Teacher window [t_start, t_start+K_layers) captures the steepest
convergence region — matched against student first K_layers outer iters.
t_start=20 recommended from convergence plot analysis.

Symmetric distillation
------------------------
Early loss  : teacher [t_start, t_start+K_layers)  → student [0, K_layers)
Late loss   : teacher [I_T-K_layers, I_T)          → student [I_S-K_layers, I_S)
Both losses train step_sizes across the full 60-iteration budget.

Variable naming
---------------
K        : number of users  (from system_config)
K_layers : number of outer iteration pairs distilled per window
I_T      : teacher outer iterations = 120
I_S      : student outer iterations = 60
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
I_T = n_iter_outer        # teacher: 120 outer iterations
I_S = n_iter_outer // 2   # student:  60 outer iterations
run_J20_I60 =1

# ── Student step_size initialisation ─────────────────────────────────────────
# Shape [J20, I_S, K+1] = [20, 60, K+1]
# Warm-start from teacher's first I_S outer iterations — much better than zeros
#step_size_student_J20_60 = torch.zeros(n_iter_inner_J20, I_S, K + 1)
step_size_student_J20_60 = torch.full([n_iter_inner_J20, I_S, K + 1], step_size_fixed, requires_grad=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def flatten_F(F_mat, B):
    """F_mat : [K, B, Nt, Nrf] complex → [B, K*Nt*Nrf] complex"""
    return F_mat.permute(1, 0, 2, 3).reshape(B, -1)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  RKD LOSSES  (exact implementations, unchanged)
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


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CI-RKD LOSS — reversed weights
# ══════════════════════════════════════════════════════════════════════════════

def ci_rkd_loss(F_T_window, F_S_window, K_layers, B,
                lambda_dist=25.0, lambda_angle=50.0):
    """
    CI-RKD with reversed weight schedule.
    w_l = (l+1) / sum(1..K_layers)
    Higher l = more converged teacher = stronger weight.
    """
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


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TEACHER MODEL
# ══════════════════════════════════════════════════════════════════════════════

class PGA_Unfold_J20_Teacher(PGA_Unfold_J20):
    """
    Teacher — 120 outer × 20 inner.
    execute_PGA_window() extracts any [t_start, t_start+K_layers) window.
    t_start=None → last K_layers (plateau region).
    """

    def execute_PGA_window(self, H, R, Pt, n_iter_outer, n_iter_inner,
                            t_start, K_layers):
        if t_start is None:
            t_start = n_iter_outer - K_layers
        t_end = t_start + K_layers
        assert t_end <= n_iter_outer, \
            f"Window [{t_start},{t_end}) exceeds n_iter_outer={n_iter_outer}"

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
# 5.  STUDENT MODEL — J=20 inner, 60 outer
# ══════════════════════════════════════════════════════════════════════════════

class PGA_Unfold_J20_60(PGA_Unfold_J20):
    """
    Student — 60 outer × 20 inner.

    step_size shape: [20, 60, K+1]   (I_S=60)

    Inference cost:  60 × 20 = 1200 steps
    Teacher cost:   120 × 20 = 2400 steps
    → 2× cheaper at inference

    Two trajectory methods:
      execute_PGA_with_first_Kl : stores first K_layers outer iters (early RKD)
      execute_PGA_with_last_Kl  : stores last  K_layers outer iters (late RKD)
    Both needed for symmetric distillation.
    """

    def execute_PGA_with_first_Kl(self, H, R, Pt,
                                    n_iter_outer, n_iter_inner, K_layers):
        """Store F for first K_layers outer iters. Rest run without cloning."""
        rate_init, tau_init, F, W = initialize(H, R, Pt, initial_normalization)
        rate_over_iters = torch.zeros(n_iter_outer, H.shape[1], device=H.device)
        tau_over_iters  = torch.zeros(n_iter_outer, H.shape[1], device=H.device)
        F_traj_first = []

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

            if ii < K_layers:
                F_traj_first.append(F.clone())   # in-graph ✓

            rate_over_iters[ii] = get_sum_rate(H, F, W, Pt)
            tau_over_iters[ii]  = get_beam_error(H, F, W, R, Pt)

        rates = torch.cat([rate_init, rate_over_iters], dim=0)
        taus  = torch.cat([tau_init,  tau_over_iters],  dim=0)
        return torch.transpose(rates, 0, 1), torch.transpose(taus, 0, 1), F, W, F_traj_first

    def execute_PGA_with_last_Kl(self, H, R, Pt,
                                   n_iter_outer, n_iter_inner, K_layers):
        """
        Store F for last K_layers outer iters. Rest run without cloning.
        In-graph — gradients flow to step_size for late iterations.
        Used for symmetric late distillation.
        """
        rate_init, tau_init, F, W = initialize(H, R, Pt, initial_normalization)
        rate_over_iters = torch.zeros(n_iter_outer, H.shape[1], device=H.device)
        tau_over_iters  = torch.zeros(n_iter_outer, H.shape[1], device=H.device)
        F_traj_last = []
        I = n_iter_outer

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

            if ii >= I - K_layers:
                F_traj_last.append(F.clone())   # in-graph ✓

            rate_over_iters[ii] = get_sum_rate(H, F, W, Pt)
            tau_over_iters[ii]  = get_beam_error(H, F, W, R, Pt)

        rates = torch.cat([rate_init, rate_over_iters], dim=0)
        taus  = torch.cat([tau_init,  tau_over_iters],  dim=0)
        return torch.transpose(rates, 0, 1), torch.transpose(taus, 0, 1), F, W, F_traj_last


# ══════════════════════════════════════════════════════════════════════════════
# 6.  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

if run_RKD_Distillation == 1:

    print('CI-RKD: J20-inner/60-outer student  ←  J20-inner/120-outer teacher')
    print(f'Teacher: I_T={I_T} outer × J=20 inner = {I_T*n_iter_inner_J20} steps/batch')
    print(f'Student: I_S={I_S} outer × J=20 inner = {I_S*n_iter_inner_J20} steps/batch')
    print(f'Inference speedup: {I_T//I_S}×  (60 vs 120 outer iters)')
    print(f'SNR list={snr_dB_list}  |  K (users)={K}  |  device={device}\n')

    # ── Hyperparameters ───────────────────────────────────────────────────────
    K_layers     = 15      # pairs distilled per window (early + late)
    lambda_dist  = 25.0
    lambda_angle = 50.0
    lambda_late  = 1.0     # weight on late symmetric loss (tune: 0.5–2.0)
    in_warmup    = 0       # no warmup — kept as you had it

    # Teacher early window: steepest convergence region from plot
    t_start_early = 20     # teacher iters [20, 35)
    # Teacher late window: converged region matched to student's last K_layers
    t_start_late  = I_T - K_layers   # teacher iters [105, 120)

    assert t_start_early + K_layers <= I_T
    assert K_layers <= I_S

    print(f'Early window: teacher [{t_start_early}, {t_start_early+K_layers}) '
          f'→ student [0, {K_layers})')
    print(f'Late  window: teacher [{t_start_late}, {I_T}) '
          f'→ student [{I_S-K_layers}, {I_S})')
    print(f'K_layers={K_layers}  |  batch={batch_size}\n')

    # ── Teacher ──────────────────────────────────────────────────────────────
    model_teacher = PGA_Unfold_J20_Teacher(step_size_UPGA_J20).to(device)
    model_teacher.load_state_dict(
        torch.load(model_file_name_UPGA_J20, map_location=device))
    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad_(False)
    print(f'Teacher loaded: {model_file_name_UPGA_J20}')

    # ── Student — warm-start step_size from teacher first I_S layers ──────────
    # This gives the student a much better starting point than zeros.
    # Teacher step_size: [20, 120, K+1] → take first 60 outer iters
    #with torch.no_grad():
      #  step_size_student_J20_60.copy_(
       #     model_teacher.step_size.data[:, :I_S, :].cpu())
   # print(f'Student step_size warm-started from teacher iters [0, {I_S})')

    model_student = PGA_Unfold_J20_60(step_size_student_J20_60).to(device)
    optimizer     = torch.optim.Adam(model_student.parameters(), lr=learning_rate)
    print(f'Student: PGA_Unfold_J20_60  step_size shape = '
          f'{list(model_student.step_size.shape)}\n')

    for i_epoch in range(n_epoch):
        start_time         = time.time()
        epoch_loss         = 0.0
        epoch_rkd_early    = 0.0
        epoch_rkd_late     = 0.0
        epoch_task_student = 0.0
        epoch_task_teacher = 0.0
        num_batches        = 0

        H_shuffled = torch.transpose(H_train, 0, 1)[
            np.random.permutation(len(H_train[0]))]

        for i_batch in range(0, len(H_train), batch_size):
            H = torch.transpose(
                H_shuffled[i_batch:i_batch + batch_size], 0, 1).to(device)
            B = H.shape[1]

            snr_dB_train = np.random.choice(snr_dB_list)
            snr_train    = 10 ** (snr_dB_train / 10)
            R = get_R(snr_dB_train, B)

            # ── Teacher: two windows ──────────────────────────────────────────
            with torch.no_grad():
                # Early window — steepest region
                F_t, W_t, F_T_early = model_teacher.execute_PGA_window(
                    H, R, snr_train, I_T, n_iter_inner_J20,
                    t_start=t_start_early, K_layers=K_layers)
                teacher_task_loss = get_sum_loss(F_t, W_t, H, R, snr_train, B)

                # Late window — converged region (matches student last K_layers)
                _, _, F_T_late = model_teacher.execute_PGA_window(
                    H, R, snr_train, I_T, n_iter_inner_J20,
                    t_start=t_start_late, K_layers=K_layers)

            # ── Student: first K_layers AND last K_layers ─────────────────────
            # Run two forward passes: one for early, one for late
            # early pass — stores first K_layers, computes task loss
            rate, _, F_s, W_s, F_S_first = model_student.execute_PGA_with_first_Kl(
                H, R, snr_train, I_S, n_iter_inner_J20, K_layers)

            loss_task = get_sum_loss(F_s, W_s, H, R, snr_train, B)

            # late pass — stores last K_layers  (separate forward, no grad overlap)
            _, _, _, _, F_S_last = model_student.execute_PGA_with_last_Kl(
                H, R, snr_train, I_S, n_iter_inner_J20, K_layers)

            # ── CI-RKD losses ─────────────────────────────────────────────────
            loss_rkd_early = ci_rkd_loss(
                F_T_early, F_S_first,
                K_layers=K_layers, B=B,
                lambda_dist=lambda_dist, lambda_angle=lambda_angle)

            loss_rkd_late = ci_rkd_loss(
                F_T_late, F_S_last,
                K_layers=K_layers, B=B,
                lambda_dist=lambda_dist, lambda_angle=lambda_angle)

            total_loss = loss_task + loss_rkd_early + lambda_late * loss_rkd_late

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
            epoch_rkd_early    += loss_rkd_early.item()
            epoch_rkd_late     += loss_rkd_late.item()
            epoch_task_student += loss_task.item()
            epoch_task_teacher += teacher_task_loss.item()
            num_batches        += 1

            if device.type == 'cuda':
                torch.cuda.empty_cache()

        nb = max(num_batches, 1)
        print(
            f"Epoch {i_epoch:4d} [CI-RKD] | "
            f"Time: {time.time()-start_time:.1f}s | "
            f"Loss: {epoch_loss/nb:.4f} | "
            f"Student: {epoch_task_student/nb:.4f} | "
            f"Teacher: {epoch_task_teacher/nb:.4f} | "
            f"RKD_early: {epoch_rkd_early/nb:.4f} | "
            f"RKD_late: {epoch_rkd_late/nb:.4f}"
        )

    save_path = (model_file_name_UPGA_J20.replace('J20', 'J20_60outer') +
                 f'_CI_RKD_sym_Kl{K_layers}_win{t_start_early}_noinitial.pth')
    torch.save(model_student.state_dict(), save_path)
    print(f'\nStudent saved → {save_path}')

if run_J20_I60 == 0:

    print('Pure Student Training: J20-inner / 60-outer (NO RKD)')

    # ── Model ─────────────────────────────────────────────
    model_student = PGA_Unfold_J20_60(step_size_student_J20_60).to(device)
    optimizer = torch.optim.Adam(model_student.parameters(), lr=learning_rate)

    print(f'Student model initialized with shape: {model_student.step_size.shape}')

    # ── Training loop ─────────────────────────────────────
    for i_epoch in range(n_epoch):

        start_time = time.time()
        epoch_loss = 0.0
        epoch_task = 0.0
        num_batches = 0

        H_shuffled = torch.transpose(H_train, 0, 1)[
            np.random.permutation(len(H_train[0]))]

        for i_batch in range(0, len(H_train), batch_size):

            H = torch.transpose(
                H_shuffled[i_batch:i_batch + batch_size], 0, 1
            ).to(device)

            B = H.shape[1]

            snr_dB_train = np.random.choice(snr_dB_list)
            snr_train = 10 ** (snr_dB_train / 10)
            R = get_R(snr_dB_train, B)

            # ── Forward pass (student only) ──────────────────
            rate, _, F_s, W_s, _ = model_student.execute_PGA_with_first_Kl(
                H, R, snr_train,
                n_iter_outer=I_S,
                n_iter_inner=n_iter_inner_J20,
                K_layers=0  # no distillation windows needed
            )

            loss_task = get_sum_loss(F_s, W_s, H, R, snr_train, B)

            # ── Backprop ─────────────────────────────────────
            optimizer.zero_grad()
            loss_task.backward()

            if torch.isnan(loss_task) or torch.isinf(loss_task):
                optimizer.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(model_student.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                model_student.step_size.data.clamp_(min=1e-8)

            epoch_loss += loss_task.item()
            epoch_task += loss_task.item()
            num_batches += 1

        nb = max(num_batches, 1)

        print(
            f"Epoch {i_epoch:4d} [PURE STUDENT] | "
            f"Time: {time.time()-start_time:.1f}s | "
            f"Loss: {epoch_loss/nb:.4f} | "
            f"Task: {epoch_task/nb:.4f}"
        )

    # ── Save model ───────────────────────────────────────
    save_path = model_file_name_UPGA_J20.replace(
        'J20',
        'J20_60outer_PURE'
    ) + '.pth'

    torch.save(model_student.state_dict(), save_path)
    print(f'\nPure student saved → {save_path}')

# ══════════════════════════════════════════════════════════════════════════════
# 7.  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

if run_RKD_Distillation == 0:

    K_layers   = 15
    t_start_early = 20

    save_path = (model_file_name_UPGA_J20.replace('J20', 'J20_60outer') +
                 f'_CI_RKD_sym_Kl{K_layers}_win{t_start_early}.pth')

    model_test = PGA_Unfold_J20_60(step_size_student_J20_60).to(device)
    model_test.load_state_dict(torch.load(save_path, map_location=device))
    model_test.eval()

    with torch.no_grad():
        (rate_iter, beam_iter, F_out, W_out, _) = model_test.execute_PGA_with_first_Kl(
            H_test, Rtest, snr, I_S, n_iter_inner_J20, K_layers)

    rate_RKD = [r.detach().cpu().numpy()
                for r in (sum(rate_iter) / len(H_test[0]))]
    beam_RKD = [r.detach().cpu().numpy()
                for r in (sum(beam_iter) / len(H_test[0]))]
    iter_number = np.array(list(range(I_S + 1)))

    print(f'\nStudent J20/60outer | K_layers={K_layers} | t_start={t_start_early}')
    print(f'  Final sum rate   : {rate_RKD[-1]:.4f} bps/Hz')
    print(f'  Final beam error : {beam_RKD[-1]:.4f}')
