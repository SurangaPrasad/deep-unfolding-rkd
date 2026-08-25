"""
Full: J5/I60 Student — LAST-5 INIT + INNER RKD + OUTER FW RKD
==============================================================
Teacher : PGA_Unfold_J20       [20 inner × 120 outer]  pretrained, frozen
Student : PGA_Unfold_J5_I60_CI [ 5 inner ×  60 outer]  trained with CI-RKD

Two complementary RKD signals
------------------------------
1. Inner RKD (from document 14):
   Matches relational geometry of F-iterates collected during the inner
   loop. Teacher inner steps [15:19] matched 1-to-1 with student [0:4].
   Supervises within-outer-iteration refinement quality.

2. Outer FW RKD (from reference code):
   Matches relational geometry of the effective beamformer FW at the
   final outer iteration. Teacher and student FW are normalised and
   then distance + angle losses are computed on the batch.
   Supervises the final beamformer quality — anchors the endpoint.

Total loss
----------
L_total = L_task + L_inner_RKD + lambda_fw * (lambda_dist * L_dist_fw
                                             + lambda_angle * L_angle_fw)
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

I_T           = 120
I_S           = 120 // 2
N_INNER_T     = 20
N_INNER_S     = 3
INNER_START_T = N_INNER_T - N_INNER_S   # = 15


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LAST-5 INIT
# ══════════════════════════════════════════════════════════════════════════════

def build_last5_init(teacher_model, n_outer_S=60):
    with torch.no_grad():
        ss_T        = teacher_model.step_size.data
        ss_last5    = ss_T[INNER_START_T:].clone()
        fingerprint = ss_last5.mean(dim=1, keepdim=True)
        ss_init     = fingerprint.expand(-1, n_outer_S, -1).clone()

    print(f"\n[J5 LAST-5 init]")
    print(f"  Source : teacher inner steps [{INNER_START_T}:{N_INNER_T}]")
    print(f"  teacher {list(ss_T.shape)} → fingerprint {list(fingerprint.shape)}"
          f" → student init {list(ss_init.shape)}")
    print(f"  Range  : [{ss_init.min():.4e}, {ss_init.max():.4e}]\n")
    return ss_init


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def flatten_F(F_mat, B):
    return F_mat.permute(1, 0, 2, 3).reshape(B, -1)

def _to_real(x):
    if torch.is_complex(x):
        return torch.view_as_real(x).flatten(-2)
    return x


# ══════════════════════════════════════════════════════════════════════════════
# 3.  INNER-LAYER RKD LOSS  (from document 14 — unchanged)
# ══════════════════════════════════════════════════════════════════════════════

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
# 4.  OUTER FW RKD LOSS  (from reference code — on effective beamformer FW)
#
#     Operates on the final effective beamformer FW at the last outer
#     iteration. Both teacher and student FW are flattened and L2-normalised
#     before computing distance and angle relational losses across the batch.
#
#     This is complementary to inner RKD:
#       inner RKD  → supervises how F evolves within each outer iteration
#       outer FW RKD → supervises the quality of the final FW product
#
#     Using FW rather than F alone captures the joint effect of both
#     precoders on the transmitted signal, which is the quantity that
#     directly determines sum-rate and beampattern quality.
# ══════════════════════════════════════════════════════════════════════════════

def rkd_distance_loss(teacher_repr, student_repr):
    """
    Pairwise distance loss on normalised FW representations.
    teacher_repr, student_repr : [B, D] L2-normalised real tensors.
    """
    with torch.no_grad():
        t_dist = torch.cdist(teacher_repr.unsqueeze(0),
                             teacher_repr.unsqueeze(0), p=2).squeeze(0)  # [B, B]
        t_dist = t_dist / t_dist.mean().clamp(min=1e-12)

    s_dist = torch.cdist(student_repr.unsqueeze(0),
                         student_repr.unsqueeze(0), p=2).squeeze(0)      # [B, B]
    s_dist = s_dist / s_dist.mean().clamp(min=1e-12)

    return nn.functional.smooth_l1_loss(s_dist, t_dist, reduction='mean')


def rkd_angle_loss(teacher_repr, student_repr):
    """
    Triplet angle loss on normalised FW representations.
    teacher_repr, student_repr : [B, D] L2-normalised real tensors.
    """
    with torch.no_grad():
        t_e   = teacher_repr.unsqueeze(0) - teacher_repr.unsqueeze(1)   # [B, B, D]
        t_e   = nn.functional.normalize(t_e, p=2, dim=-1)
        t_cos = torch.bmm(t_e, t_e.permute(0, 2, 1))                    # [B, B, B]

    s_e   = student_repr.unsqueeze(0) - student_repr.unsqueeze(1)
    s_e   = nn.functional.normalize(s_e, p=2, dim=-1)
    s_cos = torch.bmm(s_e, s_e.permute(0, 2, 1))

    return nn.functional.smooth_l1_loss(s_cos, t_cos, reduction='mean')


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TEACHER MODEL
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
# 6.  STUDENT MODEL
# ══════════════════════════════════════════════════════════════════════════════

class PGA_Unfold_J5_I60_CI(nn.Module):

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
# 7.  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

if run_RKD_Distillation == 1:

    print('=' * 70)
    print('INIT + INNER RKD + OUTER FW RKD  J5/I60  ←  J20/I120')
    print(f'Teacher : {I_T} outer × {N_INNER_T} inner = {I_T*N_INNER_T} steps/sample')
    print(f'Student :  {I_S} outer ×  {N_INNER_S} inner =  {I_S*N_INNER_S} steps/sample')
    print(f'Speedup : {(I_T*N_INNER_T)//(I_S*N_INNER_S)}×')
    print('=' * 70 + '\n')

    K_outer       = 10
    t_start_steep = 20

    # Inner RKD weights
    lambda_dist_inner  = 12.0
    lambda_angle_inner = 25.0

    # Outer FW RKD weights — start small so it complements rather than
    # dominates the inner RKD signal
    lambda_dist_fw  = 25.0
    lambda_angle_fw = 50.0
    lambda_fw       = 0.5   # overall scale for outer FW RKD term

    min_step_size = 1e-8
    max_step_size = 0.35
    stabilised_lr = learning_rate / 2.0

    print(f'Inner RKD  : lambda_dist={lambda_dist_inner}'
          f'  lambda_angle={lambda_angle_inner}')
    print(f'Outer FW RKD: lambda_dist={lambda_dist_fw}'
          f'  lambda_angle={lambda_angle_fw}  scale={lambda_fw}')
    print(f'step_size ∈ [{min_step_size}, {max_step_size}]'
          f'  |  lr={stabilised_lr}  |  batch={batch_size}\n')

    model_teacher = PGA_Unfold_J20_Teacher(step_size_UPGA_J20).to(device)
    model_teacher.load_state_dict(
        torch.load(model_file_name_UPGA_J20, map_location=device))
    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad_(False)
    print(f'Teacher loaded : {model_file_name_UPGA_J20}')

    ss_init = build_last5_init(model_teacher, n_outer_S=I_S)

    model_student = PGA_Unfold_J5_I60_CI(ss_init).to(device)
    optimizer     = torch.optim.Adam(model_student.parameters(), lr=stabilised_lr)
    print(f'Student step_size shape : {list(model_student.step_size.shape)}\n')

    best_student_loss = -float('inf')

    for i_epoch in range(n_epoch):
        start_time       = time.time()
        epoch_loss       = 0.0
        epoch_rkd_inner  = 0.0
        epoch_rkd_fw     = 0.0
        epoch_task_s     = 0.0
        epoch_task_t     = 0.0
        num_batches      = 0

        if torch.isnan(model_student.step_size.data).any():
            print(f"  [WARNING] NaN in step_size at epoch {i_epoch}. Resetting.")
            with torch.no_grad():
                model_student.step_size.data.copy_(ss_init.to(device))
            optimizer.state.clear()

        H_shuffled = torch.transpose(H_train, 0, 1)[
            np.random.permutation(len(H_train[0]))]

        for i_batch in range(0, len(H_train[0]), batch_size):
            H = torch.transpose(
                H_shuffled[i_batch:i_batch+batch_size], 0, 1).to(device)
            B = H.shape[1]
            if B < 2:
                continue

            snr_dB_train = np.random.choice(snr_dB_list)
            snr_train    = 10 ** (snr_dB_train / 10)
            R = get_R(snr_dB_train, B)

            # ── Teacher forward ───────────────────────────────────────────────
            with torch.no_grad():
                F_t, W_t, F_inner_T = \
                    model_teacher.execute_PGA_inner_windows(
                        H, R, snr_train, I_T, N_INNER_T,
                        t_start_steep=t_start_steep,
                        K_outer=K_outer)
                teacher_task_loss = get_sum_loss(F_t, W_t, H, R, snr_train, B)

                # Effective beamformer FW for outer FW RKD
                # F_t shape: [K_u, B, Nt, Nrf], W_t shape: [K_u, B, Nrf, K]
                # matmul → [K_u, B, Nt, K] → flatten → normalise
                FW_t = torch.matmul(F_t, W_t)                        # [K_u,B,Nt,K]
                FW_t_flat = FW_t.permute(1,0,2,3).reshape(B, -1)     # [B, D]
                if torch.is_complex(FW_t_flat):
                    FW_t_flat = torch.view_as_real(FW_t_flat).flatten(-2)
                teacher_repr = nn.functional.normalize(
                    FW_t_flat, dim=1).detach()                        # [B, D_real]

            # ── Student forward ───────────────────────────────────────────────
            rate, _, F_s, W_s, F_inner_S = \
                model_student.execute_PGA_inner_windows(
                    H, R, snr_train, I_S, N_INNER_S, K_outer)

            # Effective beamformer FW for outer FW RKD
            FW_s      = torch.matmul(F_s, W_s)
            FW_s_flat = FW_s.permute(1,0,2,3).reshape(B, -1)
            if torch.is_complex(FW_s_flat):
                FW_s_flat = torch.view_as_real(FW_s_flat).flatten(-2)
            student_repr = nn.functional.normalize(FW_s_flat, dim=1) # [B, D_real]

            # ── Losses ────────────────────────────────────────────────────────
            loss_task      = get_sum_loss(F_s, W_s, H, R, snr_train, B)

            loss_rkd_inner = inner_rkd_loss(
                F_inner_T, F_inner_S,
                n_inner=N_INNER_S, B=B,
                lambda_dist=lambda_dist_inner,
                lambda_angle=lambda_angle_inner)

            loss_rkd_fw = (lambda_dist_fw * rkd_distance_loss(teacher_repr, student_repr)
                         + lambda_angle_fw * rkd_angle_loss(teacher_repr, student_repr))

            total_loss = loss_task + loss_rkd_inner + lambda_fw * loss_rkd_fw

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"  [WARNING] NaN/Inf at epoch {i_epoch} "
                      f"batch {i_batch} — skipping.")
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
            epoch_rkd_fw    += loss_rkd_fw.item()
            epoch_task_s    += loss_task.item()
            epoch_task_t    += teacher_task_loss.item()
            num_batches     += 1

        nb = max(num_batches, 1)
        current_student = epoch_task_s / nb

        print(f"Epoch {i_epoch:4d} [J5/I60 | init+inner+fw_rkd] | "
              f"Time: {time.time()-start_time:.1f}s | "
              f"Loss: {epoch_loss/nb:.4f} | "
              f"Student: {current_student:.4f} | "
              f"Teacher: {epoch_task_t/nb:.4f} | "
              f"RKD_inner: {epoch_rkd_inner/nb:.4f} | "
              f"RKD_fw: {epoch_rkd_fw/nb:.4f}")

        with torch.no_grad():
            ss = model_student.step_size.data
            ss_at_ceil = (ss >= max_step_size - 1e-6).sum().item()
            ss_at_floor= (ss <= min_step_size + 1e-9).sum().item()
            print(f"             step_size : "
                  f"min={ss.min():.4e}  max={ss.max():.4e}  "
                  f"mean={ss.mean():.4e}  std={ss.std():.4e}  "
                  f"at_ceil={ss_at_ceil}  at_floor={ss_at_floor}")

        if current_student > best_student_loss:
            best_student_loss = current_student
            torch.save(model_student.state_dict(),
                       model_file_name_UPGA_J10 +
                       '_J5_I60_INIT_INNER_FW_RKD_best.pth')
            print(f"  [BEST] epoch {i_epoch} → saved")

    save_path = model_file_name_UPGA_J10 + '_J3_I60_INIT_INNER_FW_RKD_LR001.pth'
    torch.save(model_student.state_dict(), save_path)
    print(f'\nStudent saved → {save_path}')


# ══════════════════════════════════════════════════════════════════════════════
# 8.  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

if run_RKD_Distillation == 0:

    save_path = model_file_name_UPGA_J10 + '_J5_I60_INIT_INNER_FW_RKD_best.pth'

    model_test = PGA_Unfold_J5_I60_CI(0.01).to(device)
    model_test.load_state_dict(torch.load(save_path, map_location=device))
    model_test.eval()

    with torch.no_grad():
        rate_iter, beam_iter, _, _, _ = \
            model_test.execute_PGA_inner_windows(
                H_test, Rtest, snr, I_S, N_INNER_S, K_outer=0)

    rate_out = [r.detach().cpu().numpy()
                for r in (sum(rate_iter) / len(H_test[0]))]
    tau_out  = [r.detach().cpu().numpy()
                for r in (sum(beam_iter) / len(H_test[0]))]

    print(f'\nJ5/I60 | INIT + INNER RKD + FW RKD')
    print(f'  Final sum rate   : {rate_out[-1]:.4f} bps/Hz')
    print(f'  Final beam error : {tau_out[-1]:.4f}')