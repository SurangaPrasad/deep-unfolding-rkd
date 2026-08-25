"""
Ablation: J5/I120 — NO INIT, INNER RKD ONLY
=============================================
Teacher : PGA_Unfold_J20        [20 inner × 120 outer]  2400 steps
Student : PGA_Unfold_J5_I120_CI [ 5 inner × 120 outer]   600 steps  → 4× cheaper

Key difference from J5/I60:
- Student runs FULL 120 outer iterations (same as teacher)
- Only the inner depth differs (5 vs 20)
- No outer depth mismatch — RKD outer windows are directly comparable
- step_size shape: [5, 120, K+1]

This is the cleanest test of inner RKD because the outer trajectory
length is identical between teacher and student.
"""

import time
import torch
import torch.nn as nn
import numpy as np

from system_config import *
from utility import *
from PGA_models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

H_train, H_test0 = get_data_tensor(data_source)
H_test           = H_test0[:, :test_size, :, :].to(device)
torch.manual_seed(3407)

radar_cache = {}
for _snr_db in snr_dB_list:
    _R, _, _, _ = get_radar_data(_snr_db, H_train[:, :1, :, :])
    radar_cache[_snr_db] = _R.to(device)

Rtest, at, theta, ideal_beam = get_radar_data(snr_dB, H_test.cpu())
Rtest = Rtest.to(device)

def get_R(snr_db, B):
    return radar_cache[snr_db].expand(-1, B, -1, -1)

# ── Iteration counts ──────────────────────────────────────────────────────────
I_T           = n_iter_outer        # teacher: 120 outer
I_S           = n_iter_outer        # student: 120 outer (SAME as teacher)
N_INNER_T     = n_iter_inner_J20    # teacher: 20 inner
N_INNER_S     = 5                   # student:  5 inner
INNER_START_T = N_INNER_T - N_INNER_S  # = 15
# student inner j ←→ teacher inner (15+j) — direct 1-to-1 mapping


# ══════════════════════════════════════════════════════════════════════════════
# INNER RKD LOSS
# ══════════════════════════════════════════════════════════════════════════════

def flatten_F(F_mat, B):
    return F_mat.permute(1, 0, 2, 3).reshape(B, -1)

def _to_real(x):
    if torch.is_complex(x):
        return torch.view_as_real(x).flatten(-2)
    return x

def inner_rkd_loss(F_T_inner, F_S_inner, n_inner, B,
                   lambda_dist=12.0, lambda_angle=25.0):
    n_pairs    = len(F_T_inner)
    weight_sum = n_inner * (n_inner + 1) / 2

    T_feats = torch.stack(
        [_to_real(flatten_F(F_T_inner[l], B)).detach()
         for l in range(n_pairs)], dim=0)
    S_feats = torch.stack(
        [_to_real(flatten_F(F_S_inner[l], B))
         for l in range(n_pairs)], dim=0)

    with torch.no_grad():
        T_dist = torch.cdist(T_feats, T_feats, p=2)
        T_dist = T_dist / T_dist.mean(
            dim=(-2,-1), keepdim=True).clamp(min=1e-12)

    S_dist = torch.cdist(S_feats, S_feats, p=2)
    S_dist = S_dist / S_dist.mean(
        dim=(-2,-1), keepdim=True).clamp(min=1e-12)

    inner_positions = torch.tensor(
        [(l % n_inner) + 1 for l in range(n_pairs)],
        dtype=S_dist.dtype, device=S_dist.device)
    weights = (inner_positions / weight_sum).view(n_pairs, 1, 1)

    dist_loss = nn.functional.smooth_l1_loss(
        S_dist * weights, T_dist * weights, reduction='mean')

    angle_loss = 0.0
    for l in range(n_pairs):
        w_l = ((l % n_inner) + 1) / weight_sum
        t, s = T_feats[l], S_feats[l]
        with torch.no_grad():
            t_e   = t.unsqueeze(0) - t.unsqueeze(1)
            t_e   = nn.functional.normalize(t_e, p=2, dim=-1)
            t_cos = torch.bmm(t_e, t_e.permute(0, 2, 1))
        s_e   = s.unsqueeze(0) - s.unsqueeze(1)
        s_e   = nn.functional.normalize(s_e, p=2, dim=-1)
        s_cos = torch.bmm(s_e, s_e.permute(0, 2, 1))
        angle_loss += w_l * nn.functional.smooth_l1_loss(s_cos, t_cos)

    return lambda_dist * dist_loss + lambda_angle * angle_loss


# ══════════════════════════════════════════════════════════════════════════════
# TEACHER MODEL
# ══════════════════════════════════════════════════════════════════════════════

class PGA_Unfold_J20_Teacher(PGA_Unfold_J20):

    def execute_PGA_inner_windows(self, H, R, Pt, n_iter_outer, n_iter_inner,
                                   t_start_steep, K_outer):
        t_end_steep = t_start_steep + K_outer
        assert t_end_steep <= n_iter_outer

        _, _, F, W = initialize(H, R, Pt, initial_normalization)
        F_inner_T  = []

        for ii in range(n_iter_outer):
            collect_this_outer = (t_start_steep <= ii < t_end_steep)
            for jj in range(n_iter_inner):
                grad_F_com  = get_grad_F_com(H, F, W)
                grad_F_rad  = get_grad_F_rad(F, W, R)
                delta_F_com = self.step_size[jj][ii][0] * grad_F_com
                delta_F_rad = self.step_size[jj][ii][0] * grad_F_rad
                F = F + delta_F_com * WEIGHT_F_COM - delta_F_rad * WEIGHT_F_RAD
                if sum(torch.abs(F[0, :, 0, 0])) > 1e3:
                    F = normalize_power(F, W, H, Pt)
                if collect_this_outer and jj >= INNER_START_T:
                    F_inner_T.append(F.detach().clone())
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

        return F, W, F_inner_T


# ══════════════════════════════════════════════════════════════════════════════
# STUDENT MODEL — J=5 inner, 120 outer
# ══════════════════════════════════════════════════════════════════════════════

class PGA_Unfold_J5_I120_CI(nn.Module):
    """
    Student — 120 outer × 5 inner iterations.
    step_size shape: [5, 120, K+1]
    4× cheaper than teacher (600 vs 2400 steps).

    Same outer depth as teacher — no outer schedule mismatch.
    Inner RKD mapping: student inner j ←→ teacher inner (15+j).
    """

    def __init__(self, step_size_init):
        super().__init__()
        if isinstance(step_size_init, torch.Tensor) and step_size_init.dim() == 3:
            self.step_size = nn.Parameter(step_size_init.float().clone())
        else:
            self.step_size = nn.Parameter(
                step_size_init * torch.ones(N_INNER_S, I_S, K + 1))

    def execute_PGA_inner_windows(self, H, R, Pt,
                                   n_iter_outer, n_iter_inner, K_outer):
        rate_init, tau_init, F, W = initialize(H, R, Pt, initial_normalization)
        rate_over_iters = torch.zeros(n_iter_outer, H.shape[1], device=H.device)
        tau_over_iters  = torch.zeros(n_iter_outer, H.shape[1], device=H.device)
        F_inner_S       = []

        for ii in range(n_iter_outer):
            collect_this_outer = (ii < K_outer)
            for jj in range(n_iter_inner):
                grad_F_com  = get_grad_F_com(H, F, W)
                grad_F_rad  = get_grad_F_rad(F, W, R)
                delta_F_com = self.step_size[jj][ii][0] * grad_F_com
                delta_F_rad = self.step_size[jj][ii][0] * grad_F_rad
                F = F + delta_F_com * WEIGHT_F_COM - delta_F_rad * WEIGHT_F_RAD
                F = normalize_power(F, W, H, Pt)
                if collect_this_outer:
                    F_inner_S.append(F.clone())
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

            rate_over_iters[ii] = get_sum_rate(H, F, W, Pt)
            tau_over_iters[ii]  = get_beam_error(H, F, W, R, Pt)

        rates = torch.cat([rate_init, rate_over_iters], dim=0)
        taus  = torch.cat([tau_init,  tau_over_iters],  dim=0)
        return (torch.transpose(rates, 0, 1),
                torch.transpose(taus,  0, 1),
                F, W, F_inner_S)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

if run_RKD_Distillation == 1:

    print('=' * 70)
    print('J5/I120 — NO INIT, INNER RKD ONLY')
    print(f'Teacher : {I_T} outer × {N_INNER_T} inner = {I_T*N_INNER_T} steps/sample')
    print(f'Student : {I_S} outer ×  {N_INNER_S} inner =  {I_S*N_INNER_S} steps/sample')
    print(f'Speedup : {(I_T*N_INNER_T)//(I_S*N_INNER_S)}×')
    print(f'NOTE: same outer depth as teacher — cleanest inner RKD test')
    print('=' * 70 + '\n')

    K_outer            = 10
    t_start_steep      = 20
    lambda_dist_inner  = 12.0
    lambda_angle_inner = 25.0
    min_step_size      = 1e-8
    max_step_size      = 0.35
    stabilised_lr      = learning_rate / 2.0

    model_teacher = PGA_Unfold_J20_Teacher(step_size_UPGA_J20).to(device)
    model_teacher.load_state_dict(
        torch.load(model_file_name_UPGA_J20, map_location=device))
    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad_(False)
    print(f'Teacher loaded : {model_file_name_UPGA_J20}')

    # Random scalar init — no AGT
    model_student = PGA_Unfold_J5_I120_CI(step_size_UPGA_J5_I120).to(device)
    optimizer     = torch.optim.Adam(model_student.parameters(),
                                     lr=stabilised_lr)
    print(f'Student step_size shape : {list(model_student.step_size.shape)}')
    print(f'Init range : [{model_student.step_size.min():.4e},'
          f' {model_student.step_size.max():.4e}]  (scalar 0.01)\n')

    best_student_loss = -float('inf')

    for i_epoch in range(n_epoch):
        start_time      = time.time()
        epoch_loss      = 0.0
        epoch_rkd_inner = 0.0
        epoch_task_s    = 0.0
        epoch_task_t    = 0.0
        num_batches     = 0

        if torch.isnan(model_student.step_size.data).any():
            print(f"  [WARNING] NaN at epoch {i_epoch}. Resetting.")
            with torch.no_grad():
                model_student.step_size.data.fill_(0.01)
            optimizer.state.clear()

        H_shuffled = torch.transpose(H_train, 0, 1)[
            np.random.permutation(len(H_train[0]))]

        for i_batch in range(0, len(H_train), batch_size):
            H = torch.transpose(
                H_shuffled[i_batch:i_batch+batch_size], 0, 1).to(device)
            B = H.shape[1]
            if B < 2:
                continue

            snr_dB_train = np.random.choice(snr_dB_list)
            snr_train    = 10 ** (snr_dB_train / 10)
            R = get_R(snr_dB_train, B)

            with torch.no_grad():
                F_t, W_t, F_inner_T = \
                    model_teacher.execute_PGA_inner_windows(
                        H, R, snr_train, I_T, N_INNER_T,
                        t_start_steep=t_start_steep,
                        K_outer=K_outer)
                teacher_task_loss = get_sum_loss(F_t, W_t, H, R, snr_train, B)

            rate, _, F_s, W_s, F_inner_S = \
                model_student.execute_PGA_inner_windows(
                    H, R, snr_train, I_S, N_INNER_S, K_outer)

            assert len(F_inner_T) == len(F_inner_S) == K_outer * N_INNER_S

            loss_task      = get_sum_loss(F_s, W_s, H, R, snr_train, B)
            loss_rkd_inner = inner_rkd_loss(
                F_inner_T, F_inner_S,
                n_inner=N_INNER_S, B=B,
                lambda_dist=lambda_dist_inner,
                lambda_angle=lambda_angle_inner)

            total_loss = loss_task + loss_rkd_inner

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model_student.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                model_student.step_size.data.clamp_(
                    min=min_step_size, max=max_step_size)

            epoch_loss      += total_loss.item()
            epoch_rkd_inner += loss_rkd_inner.item()
            epoch_task_s    += loss_task.item()
            epoch_task_t    += teacher_task_loss.item()
            num_batches     += 1

        nb = max(num_batches, 1)
        current_student = epoch_task_s / nb

        print(f"Epoch {i_epoch:4d} [J5/I120 | inner_rkd] | "
              f"Time: {time.time()-start_time:.1f}s | "
              f"Student: {current_student:.4f} | "
              f"Teacher: {epoch_task_t/nb:.4f} | "
              f"RKD_inner: {epoch_rkd_inner/nb:.4f}")

        with torch.no_grad():
            ss = model_student.step_size.data
            ss_at_ceil = (ss >= max_step_size - 1e-6).sum().item()
            print(f"             step_size : min={ss.min():.4e}  "
                  f"max={ss.max():.4e}  mean={ss.mean():.4e}  "
                  f"at_ceil={ss_at_ceil}")

        if current_student > best_student_loss:
            best_student_loss = current_student
            torch.save(model_student.state_dict(),
                       model_file_name_UPGA_J10 +
                       '_J5_I120_INNER_RKD_ONLY_best_old.pth')
            print(f"  [BEST] epoch {i_epoch} → saved")

    torch.save(model_student.state_dict(),
               model_file_name_UPGA_J10 + '_J5_I120_INNER_RKD_ONLY_old.pth')
    print(f'\nSaved.')
