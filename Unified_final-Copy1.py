"""
CI-RKD + Log-Weighted Iter Distillation — 2×2 Ablation
=======================================================
Set ABLATION_CELL to train one model at a time:

  'cell_1_1' → Flat init,  No RKD          (R_flat)
  'cell_1_2' → Flat init,  +CI-RKD+LogIter (R_flat+RKD)
  'cell_2_1' → AGT  init,  No RKD          (R_AGT)
  'cell_2_2' → AGT  init,  +CI-RKD+LogIter (R_AGT+RKD)  ← proposed
"""

import time
import torch
import torch.nn as nn
import numpy as np

from system_config import *
from utility_gpu import *      # single import — GPU-safe versions of all utility functions
from PGA_models import *

# ══════════════════════════════════════════════════════════════════════════════
# CHOOSE WHICH CELL TO TRAIN — change this one variable
# ══════════════════════════════════════════════════════════════════════════════
ABLATION_CELL = 'cell_1_2'   # 'cell_1_1' | 'cell_1_2' | 'cell_2_1' | 'cell_2_2'
# ══════════════════════════════════════════════════════════════════════════════

assert ABLATION_CELL in ('cell_1_1', 'cell_1_2', 'cell_2_1', 'cell_2_2'), \
    f"Unknown ABLATION_CELL {ABLATION_CELL!r}"

USE_AGT = ABLATION_CELL in ('cell_2_1', 'cell_2_2')
USE_RKD = ABLATION_CELL in ('cell_1_2', 'cell_2_2')

# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"\n{'='*70}")
print(f"  ABLATION CELL : {ABLATION_CELL}")
print(f"  AGT init      : {USE_AGT}")
print(f"  CI-RKD+Iter   : {USE_RKD}")
print(f"{'='*70}\n")

# ── Data ───────────────────────────────────────────────────────────────────────
H_train, H_test0 = get_data_tensor(data_source)
H_test           = H_test0[:, :test_size, :, :].to(device)
torch.manual_seed(3407)

# ── Radar cache ────────────────────────────────────────────────────────────────
radar_cache = {}
for _snr_db in snr_dB_list:
    _R, _, _, _ = get_radar_data(_snr_db, H_train[:, :1, :, :])
    radar_cache[_snr_db] = _R.to(device)

Rtest, at, theta, ideal_beam = get_radar_data(snr_dB, H_test.cpu())
Rtest = Rtest.to(device)

def get_R(snr_db, B):
    return radar_cache[snr_db].expand(-1, B, -1, -1)

# ── Iteration counts ───────────────────────────────────────────────────────────
I_T       = n_iter_outer
I_S       = n_iter_outer // 2
N_INNER_T = n_iter_inner_J20
N_INNER_S = n_iter_inner_J10


# ══════════════════════════════════════════════════════════════════════════════
# 1.  AGT INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def build_agt_init(teacher_model, n_outer_S=60, inner_strategy='avg_pairs'):
    with torch.no_grad():
        ss_T      = teacher_model.step_size.data
        n_inner_t = ss_T.shape[0]

        if inner_strategy == 'avg_pairs':
            assert n_inner_t == 2 * N_INNER_S, (
                f"avg_pairs requires J_T == 2*J_S (got {n_inner_t} vs {N_INNER_S})")
            ss_compressed = ss_T.view(N_INNER_S, 2, I_T, K + 1).mean(dim=1)
            desc = "avg_pairs"
        elif inner_strategy == 'first':
            ss_compressed = ss_T[:N_INNER_S].clone()
            desc = "first"
        elif inner_strategy == 'last':
            ss_compressed = ss_T[-N_INNER_S:].clone()
            desc = "last"
        elif inner_strategy == 'subsample':
            indices = torch.linspace(0, n_inner_t - 1, N_INNER_S).long()
            ss_compressed = ss_T[indices].clone()
            desc = "subsample"
        else:
            raise ValueError(f"Unknown inner_strategy {inner_strategy!r}")

        fingerprint = ss_compressed.mean(dim=1, keepdim=True)
        ss_init     = fingerprint.expand(-1, n_outer_S, -1).clone()

    print(f"[AGT init | strategy='{desc}']")
    print(f"  teacher {list(ss_T.shape)}"
          f" → compressed {list(ss_compressed.shape)}"
          f" → fingerprint {list(fingerprint.shape)}"
          f" → student init {list(ss_init.shape)}")
    print(f"  Range: [{ss_init.min():.4e}, {ss_init.max():.4e}]\n")
    return ss_init


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def flatten_F(F_mat, B):
    return F_mat.permute(1, 0, 2, 3).reshape(B, -1)

def _to_real(x):
    if torch.is_complex(x):
        return torch.view_as_real(x).flatten(-2)
    return x


# ══════════════════════════════════════════════════════════════════════════════
# 3.  COMBINED LOSS: CI-RKD + LOG-WEIGHTED DIRECT SUPERVISION
# ══════════════════════════════════════════════════════════════════════════════

def combined_rkd_iter_loss(F_T_window, F_S_window, K_layers, B,
                            lambda_dist=25.0, lambda_angle=50.0,
                            lambda_iter=0.1):
    """
    Combined loss for one trajectory window:
        L = lambda_dist  * L_dist   (RKD pairwise distance)
          + lambda_angle * L_angle  (RKD pairwise angle)
          + lambda_iter  * L_iter   (log-weighted direct supervision)
    """
    weight_sum = K_layers * (K_layers + 1) / 2

    T_feats = torch.stack(
        [_to_real(flatten_F(F_T_window[l], B)).detach()
         for l in range(K_layers)], dim=0)
    S_feats = torch.stack(
        [_to_real(flatten_F(F_S_window[l], B))
         for l in range(K_layers)], dim=0)

    target_device = S_feats.device
    T_feats = T_feats.to(target_device)
    S_feats = S_feats.to(target_device)

    # ── RKD distance loss ─────────────────────────────────────────────────────
    with torch.no_grad():
        T_dist = torch.cdist(T_feats, T_feats, p=2)
        T_dist = T_dist / T_dist.mean(
            dim=(-2, -1), keepdim=True).clamp(min=1e-12)

    S_dist = torch.cdist(S_feats, S_feats, p=2)
    S_dist = S_dist / S_dist.mean(
        dim=(-2, -1), keepdim=True).clamp(min=1e-12)

    rkd_weights = (torch.arange(1, K_layers + 1,
                                dtype=S_dist.dtype,
                                device=target_device)
                   / weight_sum).view(K_layers, 1, 1)

    dist_loss = nn.functional.smooth_l1_loss(
        S_dist * rkd_weights, T_dist * rkd_weights, reduction='mean')

    # ── RKD angle loss ────────────────────────────────────────────────────────
    angle_loss = 0.0
    for l in range(K_layers):
        w_l = (l + 1) / weight_sum
        t   = T_feats[l]
        s   = S_feats[l]
        with torch.no_grad():
            t_e   = t.unsqueeze(0) - t.unsqueeze(1)
            t_e   = nn.functional.normalize(t_e, p=2, dim=-1)
            t_cos = torch.bmm(t_e, t_e.permute(0, 2, 1))
        s_e   = s.unsqueeze(0) - s.unsqueeze(1)
        s_e   = nn.functional.normalize(s_e, p=2, dim=-1)
        s_cos = torch.bmm(s_e, s_e.permute(0, 2, 1))
        angle_loss += w_l * nn.functional.smooth_l1_loss(s_cos, t_cos)

    # ── Log-weighted direct supervision ───────────────────────────────────────
    log_weights = torch.log(
        torch.arange(2, K_layers + 2,
                     dtype=S_feats.dtype,
                     device=target_device))

    diff      = S_feats - T_feats
    iter_loss = (log_weights.view(K_layers, 1, 1)
                 * diff.pow(2).mean(dim=(-2, -1))).mean()

    rkd_loss = lambda_dist * dist_loss + lambda_angle * angle_loss
    total    = rkd_loss + lambda_iter * iter_loss

    return total, rkd_loss.detach(), iter_loss.detach()


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TEACHER MODEL
# ══════════════════════════════════════════════════════════════════════════════

class PGA_Unfold_J20_Teacher(PGA_Unfold_J20):

    def execute_PGA_two_windows(self, H, R, Pt, n_iter_outer, n_iter_inner,
                                 t_start_early, t_start_late, K_layers):
        t_end_early = t_start_early + K_layers
        t_end_late  = t_start_late  + K_layers
        assert t_end_early <= n_iter_outer, "Early window exceeds teacher depth"
        assert t_end_late  <= n_iter_outer, "Late window exceeds teacher depth"

        _, _, F, W = safe_initialize(H, R, Pt, initial_normalization, device)

        F_traj_early, F_traj_late = [], []

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

            if t_start_early <= ii < t_end_early:
                F_traj_early.append(F.detach().clone())
            if t_start_late  <= ii < t_end_late:
                F_traj_late.append(F.detach().clone())

        return F, W, F_traj_early, F_traj_late


# ══════════════════════════════════════════════════════════════════════════════
# 5.  STUDENT MODEL
# ══════════════════════════════════════════════════════════════════════════════

class PGA_Unfold_J10_I60_CI(nn.Module):

    def __init__(self, step_size_init):
        super().__init__()
        if isinstance(step_size_init, torch.Tensor) and step_size_init.dim() == 3:
            self.step_size = nn.Parameter(step_size_init.float().clone())
        else:
            self.step_size = nn.Parameter(
                step_size_init * torch.ones(N_INNER_S, I_S, K + 1))

    def execute_PGA_with_both_windows(self, H, R, Pt,
                                       n_iter_outer, n_iter_inner, K_layers):

        rate_init, tau_init, F, W = safe_initialize(
            H, R, Pt, initial_normalization, device)

        rate_over_iters = torch.zeros(n_iter_outer, H.shape[1], device=device)
        tau_over_iters  = torch.zeros(n_iter_outer, H.shape[1], device=device)
        F_traj_first, F_traj_last = [], []

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
                F_traj_first.append(F.clone())
            if ii >= n_iter_outer - K_layers:
                F_traj_last.append(F.clone())

            rate_over_iters[ii] = get_sum_rate(H, F, W, Pt)
            tau_over_iters[ii]  = get_beam_error(H, F, W, R, Pt)

        rates = torch.cat([rate_init, rate_over_iters], dim=0)
        taus  = torch.cat([tau_init,  tau_over_iters],  dim=0)
        return (torch.transpose(rates, 0, 1),
                torch.transpose(taus,  0, 1),
                F, W, F_traj_first, F_traj_last)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

if run_RKD_Distillation == 1:

    inner_strategy = 'avg_pairs'
    step_size_seed = 0.01

    K_layers      = 15
    lambda_dist   = 25.0
    lambda_angle  = 50.0
    lambda_iter   = 0.1
    lambda_late   = 1.0
    t_start_early = 20
    t_start_late  = I_T - K_layers

    min_step_size = 1e-8
    max_step_size = 0.5

    assert t_start_early + K_layers <= I_T, "Early window exceeds teacher depth"
    assert K_layers <= I_S // 2, (
        f"Student windows overlap — reduce K_layers to ≤ {I_S // 2}")

    print(f'K_layers={K_layers}  lambda_dist={lambda_dist}'
          f'  lambda_angle={lambda_angle}  lambda_iter={lambda_iter}'
          f'  lambda_late={lambda_late}')
    print(f'step_size ∈ [{min_step_size}, {max_step_size}]'
          f'  batch={batch_size}')
    print(f'Early : teacher [{t_start_early}, {t_start_early+K_layers})'
          f'  →  student [0, {K_layers})')
    print(f'Late  : teacher [{t_start_late}, {I_T})'
          f'  →  student [{I_S-K_layers}, {I_S})\n')

    # ── Load teacher ───────────────────────────────────────────────────────────
    model_teacher = PGA_Unfold_J20_Teacher(step_size_UPGA_J20).to(device)
    model_teacher.load_state_dict(
        torch.load(model_file_name_UPGA_J20, map_location=device))
    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad_(False)
    print(f'Teacher loaded : {model_file_name_UPGA_J20}\n')

    # ── Build step-size init ───────────────────────────────────────────────────
    if USE_AGT:
        ss_init  = build_agt_init(
            model_teacher, n_outer_S=I_S, inner_strategy=inner_strategy)
        init_tag = f'AGT_{inner_strategy}'
    else:
        ss_init  = torch.full((N_INNER_S, I_S, K + 1), step_size_seed)
        init_tag = f'flat{step_size_seed}'
        print(f'[Flat init | seed={step_size_seed}]')
        print(f'  step_size shape : {list(ss_init.shape)}\n')

    rkd_tag = 'CI_RKD_LogIter' if USE_RKD else 'noRKD'

    # ── Instantiate student ────────────────────────────────────────────────────
    model_student = PGA_Unfold_J10_I60_CI(ss_init).to(device)
    optimizer     = torch.optim.Adam(model_student.parameters(), lr=learning_rate)
    print(f'Student step_size shape : {list(model_student.step_size.shape)}\n')

    # ── Training loop ──────────────────────────────────────────────────────────
    for i_epoch in range(n_epoch):
        start_time       = time.time()
        epoch_loss       = 0.0
        epoch_rkd_early  = 0.0
        epoch_rkd_late   = 0.0
        epoch_iter_early = 0.0
        epoch_iter_late  = 0.0
        epoch_task_s     = 0.0
        epoch_task_t     = 0.0
        num_batches      = 0

        # ── NaN health check ───────────────────────────────────────────────────
        if torch.isnan(model_student.step_size.data).any():
            print(f"  [WARNING] NaN in step_size at epoch {i_epoch}. Resetting.")
            with torch.no_grad():
                model_student.step_size.data.copy_(ss_init.to(device))
            optimizer.state.clear()

        H_shuffled = torch.transpose(H_train, 0, 1)[
            np.random.permutation(len(H_train[0]))]

        for i_batch in range(0, len(H_train[0]), batch_size):
            H = torch.transpose(
                H_shuffled[i_batch:i_batch + batch_size], 0, 1).to(device)
            B = H.shape[1]
            if USE_RKD and B < 2:
                continue

            snr_dB_train = np.random.choice(snr_dB_list)
            snr_train    = 10 ** (snr_dB_train / 10)
            R = get_R(snr_dB_train, B)

            # ── Teacher pass (only when RKD active) ───────────────────────────
            if USE_RKD:
                with torch.no_grad():
                    F_t, W_t, F_T_early, F_T_late = \
                        model_teacher.execute_PGA_two_windows(
                            H, R, snr_train, I_T, N_INNER_T,
                            t_start_early=t_start_early,
                            t_start_late=t_start_late,
                            K_layers=K_layers)
                    epoch_task_t += get_sum_loss(
                        F_t, W_t, H, R, snr_train, B).item()

            # ── Student pass ───────────────────────────────────────────────────
            eff_K = K_layers if USE_RKD else 0
            rate, _, F_s, W_s, F_S_first, F_S_last = \
                model_student.execute_PGA_with_both_windows(
                    H, R, snr_train, I_S, N_INNER_S, eff_K)

            # ── Task loss (always active) ──────────────────────────────────────
            loss_task  = get_sum_loss(F_s, W_s, H, R, snr_train, B)
            total_loss = loss_task

            # ── RKD + Iter loss (only when USE_RKD) ───────────────────────────
            if USE_RKD:
                loss_early, rkd_e, iter_e = combined_rkd_iter_loss(
                    F_T_early, F_S_first, K_layers, B,
                    lambda_dist=lambda_dist,
                    lambda_angle=lambda_angle,
                    lambda_iter=lambda_iter)

                loss_late, rkd_l, iter_l = combined_rkd_iter_loss(
                    F_T_late, F_S_last, K_layers, B,
                    lambda_dist=lambda_dist,
                    lambda_angle=lambda_angle,
                    lambda_iter=lambda_iter)

                total_loss = loss_task + loss_early + lambda_late * loss_late

                epoch_rkd_early  += rkd_e.item()
                epoch_rkd_late   += rkd_l.item()
                epoch_iter_early += iter_e.item()
                epoch_iter_late  += iter_l.item()

            # ── NaN/Inf guard BEFORE backward ──────────────────────────────────
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"  [WARNING] NaN/Inf at epoch {i_epoch} "
                      f"batch {i_batch} — skipping.")
                optimizer.zero_grad()
                continue

            # ── Backward + update ──────────────────────────────────────────────
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model_student.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                model_student.step_size.data.clamp_(
                    min=min_step_size, max=max_step_size)

            epoch_loss   += total_loss.item()
            epoch_task_s += loss_task.item()
            num_batches  += 1

        nb = max(num_batches, 1)

        log = (f"Epoch {i_epoch:4d} [{ABLATION_CELL}] | "
               f"Time: {time.time()-start_time:.1f}s | "
               f"Loss: {epoch_loss/nb:.4f} | "
               f"Task_S: {epoch_task_s/nb:.4f}")
        if USE_RKD:
            log += (f" | Task_T: {epoch_task_t/nb:.4f}"
                    f" | RKD_e: {epoch_rkd_early/nb:.4f}"
                    f" | RKD_l: {epoch_rkd_late/nb:.4f}"
                    f" | Iter_e: {epoch_iter_early/nb:.4f}"
                    f" | Iter_l: {epoch_iter_late/nb:.4f}")
        print(log)

        with torch.no_grad():
            ss = model_student.step_size.data
            print(f"             step_size : "
                  f"min={ss.min():.4e}  max={ss.max():.4e}  "
                  f"mean={ss.mean():.4e}  std={ss.std():.4e}")

    # ── Save ───────────────────────────────────────────────────────────────────
    save_path = (model_file_name_UPGA_J10 +
                 f'_I60_{ABLATION_CELL}_{init_tag}_{rkd_tag}'
                 f'_Kl{K_layers}_win{t_start_early}.pth')
    torch.save(model_student.state_dict(), save_path)
    print(f'\nStudent saved → {save_path}')


# ══════════════════════════════════════════════════════════════════════════════
# 7.  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

if run_RKD_Distillation == 1:

    ABLATION_CELL  = 'cell_2_2'   # ← change to evaluate a different cell
    USE_AGT        = ABLATION_CELL in ('cell_2_1', 'cell_2_2')
    USE_RKD        = ABLATION_CELL in ('cell_1_2', 'cell_2_2')

    inner_strategy = 'avg_pairs'
    step_size_seed = 0.01
    K_layers       = 15
    lambda_dist    = 25.0
    lambda_angle   = 50.0
    lambda_iter    = 0.1
    lambda_late    = 1.0
    t_start_early  = 20

    init_tag = (f'AGT_{inner_strategy}' if USE_AGT
                else f'flat{step_size_seed}')
    rkd_tag  = 'CI_RKD_LogIter' if USE_RKD else 'noRKD'

    save_path = (model_file_name_UPGA_J10 +
                 f'_I60_{ABLATION_CELL}_{init_tag}_{rkd_tag}'
                 f'_Kl{K_layers}_win{t_start_early}.pth')

    print(f'Loading : {save_path}')

    model_test = PGA_Unfold_J10_I60_CI(step_size_UPGA_J10).to(device)
    model_test.load_state_dict(torch.load(save_path, map_location=device))
    model_test.eval()

    with torch.no_grad():
        rate_iter, beam_iter, _, _, _, _ = \
            model_test.execute_PGA_with_both_windows(
                H_test, Rtest, snr, I_S, N_INNER_S, K_layers=0)

    rate_out = (sum(rate_iter) / len(H_test[0])).detach().cpu().numpy()
    beam_out = (sum(beam_iter) / len(H_test[0])).detach().cpu().numpy()

    print(f'\n{ABLATION_CELL} | {init_tag} | {rkd_tag}')
    print(f'  Final sum rate   : {rate_out[-1]:.4f} bps/Hz')
    print(f'  Final beam error : {beam_out[-1]:.4f}')
