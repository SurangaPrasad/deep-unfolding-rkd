"""
Unified_v2_30layers.py
======================
J20/I120 Teacher  →  J10/I60 Student
RKD coverage: 30/60 layers (original two-window design)

Four ablation cells:
  cell_1: flat init, task loss only
  cell_2: flat init, L_task + L_log + CI-RKD
  cell_3: AGT  init, L_task + L_log  (no CI-RKD)
  cell_4: AGT  init, L_task + L_log + CI-RKD  (proposed)
"""

import time
import torch
import torch.nn as nn
import numpy as np

from system_config import *
from utility_gpu import *
from PGA_models import *

# ══════════════════════════════════════════════════════════════════════════════
# CHOOSE WHICH CELL TO TRAIN
# ══════════════════════════════════════════════════════════════════════════════
ABLATION_CELL = 'cell_3'   # 'cell_1' | 'cell_2' | 'cell_3' | 'cell_4'
# ══════════════════════════════════════════════════════════════════════════════
run_RKD_Distillation = 1

assert ABLATION_CELL in ('cell_1', 'cell_2', 'cell_3', 'cell_4'), \
    f"Unknown ABLATION_CELL {ABLATION_CELL!r}"

USE_AGT = ABLATION_CELL in ('cell_3', 'cell_4')
USE_RKD = ABLATION_CELL in ('cell_2', 'cell_4')
USE_LOG = ABLATION_CELL in ('cell_2', 'cell_3', 'cell_4')  # cell_3 now uses L_log
USE_KD  = USE_RKD or USE_LOG

# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"\n{'='*70}")
print(f"  Script        : Unified_v2_30layers (30/60 RKD coverage)")
print(f"  ABLATION CELL : {ABLATION_CELL}")
print(f"  AGT init      : {USE_AGT}")
print(f"  L_log         : {USE_LOG}")
print(f"  CI-RKD        : {USE_RKD}")
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
I_T       = n_iter_outer        # teacher: 120 outer
I_S       = n_iter_outer // 2   # student:  60 outer
N_INNER_T = n_iter_inner_J20    # teacher:  20 inner
N_INNER_S = n_iter_inner_J10    # student:  10 inner


# ══════════════════════════════════════════════════════════════════════════════
# 1.  AGT INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def build_agt_init(teacher_model, n_outer_S=60, inner_strategy='avg_pairs'):
    """
    Algorithm Geometry Transfer:
      Stage 1: compress inner J_T=20 -> J_S=10 by averaging pairs
      Stage 2: collapse outer (average over I_T=120) -> inner fingerprint
      Stage 3: broadcast fingerprint to all I_S=60 student outer slots
    """
    with torch.no_grad():
        ss_T      = teacher_model.step_size.data   # [20, 120, K+1]
        n_inner_t = ss_T.shape[0]

        if inner_strategy == 'avg_pairs':
            assert n_inner_t == 2 * N_INNER_S
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
    print(f"  {list(ss_T.shape)} -> {list(ss_compressed.shape)}"
          f" -> {list(fingerprint.shape)} -> {list(ss_init.shape)}")
    print(f"  Range: [{ss_init.min():.4e}, {ss_init.max():.4e}]\n")
    return ss_init


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_effective_beamformer(F, W, B):
    FW = torch.matmul(F, W)
    FW = FW.permute(1, 0, 2, 3).reshape(B, -1)
    if torch.is_complex(FW):
        FW = torch.view_as_real(FW).flatten(-2)
    return FW


# ══════════════════════════════════════════════════════════════════════════════
# 3.  LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def log_weighted_rkd_loss(F_T_window, W_T_window,
                           F_S_window, W_S_window,
                           K_layers, B,
                           lambda_dist=25.0, lambda_angle=50.0):
    log_w = torch.log(
        torch.arange(1, K_layers + 1, dtype=torch.float32,
                     device=device).clamp(min=1e-8))

    total_dist  = 0.0
    total_angle = 0.0

    for l in range(K_layers):
        w_l = log_w[l].item()

        phi_T = get_effective_beamformer(
            F_T_window[l].detach(), W_T_window[l].detach(), B)
        phi_S = get_effective_beamformer(
            F_S_window[l], W_S_window[l], B)

        with torch.no_grad():
            d_T  = torch.cdist(phi_T.unsqueeze(0),
                               phi_T.unsqueeze(0), p=2).squeeze(0)
            d_T  = d_T / d_T.mean().clamp(min=1e-12)
        d_S  = torch.cdist(phi_S.unsqueeze(0),
                           phi_S.unsqueeze(0), p=2).squeeze(0)
        d_S  = d_S / d_S.mean().clamp(min=1e-12)
        total_dist += w_l * nn.functional.smooth_l1_loss(d_S, d_T)

        with torch.no_grad():
            e_T   = phi_T.unsqueeze(0) - phi_T.unsqueeze(1)
            e_T   = nn.functional.normalize(e_T, p=2, dim=-1)
            cos_T = torch.bmm(e_T, e_T.permute(0, 2, 1))
        e_S   = phi_S.unsqueeze(0) - phi_S.unsqueeze(1)
        e_S   = nn.functional.normalize(e_S, p=2, dim=-1)
        cos_S = torch.bmm(e_S, e_S.permute(0, 2, 1))
        total_angle += w_l * nn.functional.smooth_l1_loss(cos_S, cos_T)

    return lambda_dist * total_dist + lambda_angle * total_angle


def log_weighted_deep_loss(rate_over_iters, tau_over_iters,
                            J_T_final, lambda_log=0.0001):
    device_  = rate_over_iters.device
    omega_t  = torch.as_tensor(OMEGA, device=device_, dtype=torch.float32)
    J_S      = rate_over_iters - omega_t * tau_over_iters
    log_w    = torch.log(
        torch.arange(1, I_S + 1, dtype=torch.float32,
                     device=device_).clamp(min=1e-8)).unsqueeze(1)
    sq_diff  = (J_S - J_T_final.detach()).pow(2)
    loss     = (log_w * sq_diff.mean(dim=1)).sum()
    return lambda_log * loss


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TEACHER MODEL
# ══════════════════════════════════════════════════════════════════════════════

class PGA_Unfold_J20_Teacher(PGA_Unfold_J20):

    def execute_PGA_two_windows(self, H, R, Pt,
                                 n_iter_outer, n_iter_inner,
                                 t_start_early, t_start_late, K_layers):
        t_end_early = t_start_early + K_layers
        t_end_late  = t_start_late  + K_layers

        _, _, F, W = safe_initialize(H, R, Pt, initial_normalization, device)

        F_traj_early, W_traj_early = [], []
        F_traj_late,  W_traj_late  = [], []

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
                W_traj_early.append(W.detach().clone())
            if t_start_late <= ii < t_end_late:
                F_traj_late.append(F.detach().clone())
                W_traj_late.append(W.detach().clone())

        R_final   = get_sum_rate(H, F, W, Pt)
        tau_final = get_beam_error(H, F, W, R, Pt)
        omega_t   = torch.as_tensor(OMEGA, device=device, dtype=torch.float32)
        J_T_final = (R_final - omega_t * tau_final).detach()

        return (F, W,
                F_traj_early, W_traj_early,
                F_traj_late,  W_traj_late,
                J_T_final)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  STUDENT MODEL
# ══════════════════════════════════════════════════════════════════════════════

class PGA_Unfold_J10_I60_v2(nn.Module):

    def __init__(self, step_size_init):
        super().__init__()
        if isinstance(step_size_init, torch.Tensor) and step_size_init.dim() == 3:
            self.step_size = nn.Parameter(step_size_init.float().clone())
        else:
            self.step_size = nn.Parameter(
                step_size_init * torch.ones(N_INNER_S, I_S, K + 1))

    def execute_PGA_with_windows(self, H, R, Pt,
                                  n_iter_outer, n_iter_inner,
                                  K_layers, collect_windows=True):
        rate_init, tau_init, F, W = safe_initialize(
            H, R, Pt, initial_normalization, device)

        rate_over_iters = torch.zeros(n_iter_outer, H.shape[1], device=device)
        tau_over_iters  = torch.zeros(n_iter_outer, H.shape[1], device=device)

        F_traj_first, W_traj_first = [], []
        F_traj_last,  W_traj_last  = [], []

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

            if collect_windows:
                if ii < K_layers:
                    F_traj_first.append(F.clone())
                    W_traj_first.append(W.clone().detach())
                if ii >= n_iter_outer - K_layers:
                    F_traj_last.append(F.clone())
                    W_traj_last.append(W.clone().detach())

            rate_over_iters[ii] = get_sum_rate(H, F, W, Pt)
            tau_over_iters[ii]  = get_beam_error(H, F, W, R, Pt)

        rates = torch.cat([rate_init, rate_over_iters], dim=0)
        taus  = torch.cat([tau_init,  tau_over_iters],  dim=0)

        return (torch.transpose(rates, 0, 1),
                torch.transpose(taus,  0, 1),
                F, W,
                F_traj_first, W_traj_first,
                F_traj_last,  W_traj_last,
                rate_over_iters, tau_over_iters)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

if run_RKD_Distillation == 1:

    inner_strategy = 'avg_pairs'
    step_size_seed = 0.01
    K_layers       = 15
    t_start_early  = 20
    t_start_late   = I_T - K_layers

    lambda_dist    = 25.0
    lambda_angle   = 50.0
    lambda_log     = 0.0001
    lambda_late    = 1.0
    min_step_size  = 1e-8
    max_step_size  = 0.5

    assert t_start_early + K_layers <= I_T
    assert K_layers <= I_S // 2

    print(f'K_layers={K_layers}  t_early=[{t_start_early},{t_start_early+K_layers})'
          f'  t_late=[{t_start_late},{I_T})')
    print(f'lambda_dist={lambda_dist}  lambda_angle={lambda_angle}'
          f'  lambda_log={lambda_log}  lambda_late={lambda_late}')
    print(f'step_size in [{min_step_size}, {max_step_size}]'
          f'  batch={batch_size}\n')

    # ── Load teacher ───────────────────────────────────────────────────────────
    model_teacher = PGA_Unfold_J20_Teacher(step_size_UPGA_J20).to(device)
    model_teacher.load_state_dict(
        torch.load('UPGA_J20_I120_w_030.pth', map_location=device))
    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad_(False)
    print('Teacher loaded: UPGA_J20_I120_w_030.pth\n')

    # ── Initialisation ─────────────────────────────────────────────────────────
    if USE_AGT:
        ss_init  = build_agt_init(model_teacher, n_outer_S=I_S,
                                   inner_strategy=inner_strategy)
        init_tag = f'AGT_{inner_strategy}'
    else:
        ss_init  = torch.full((N_INNER_S, I_S, K + 1), step_size_seed)
        init_tag = f'flat{step_size_seed}'
        print(f'[Flat init | seed={step_size_seed}  shape={list(ss_init.shape)}]\n')

    if USE_RKD:
        rkd_tag = 'RKDlog_30layers'
    elif USE_LOG:
        rkd_tag = 'logOnly_30layers'
    else:
        rkd_tag = 'noRKD'

    # ── Student ────────────────────────────────────────────────────────────────
    model_student = PGA_Unfold_J10_I60_v2(ss_init).to(device)
    optimizer     = torch.optim.Adam(model_student.parameters(), lr=learning_rate)
    print(f'Student step_size shape: {list(model_student.step_size.shape)}\n')

    # ── Training loop ──────────────────────────────────────────────────────────
    for i_epoch in range(n_epoch):
        start_time      = time.time()
        epoch_loss      = 0.0
        epoch_task_s    = 0.0
        epoch_task_t    = 0.0
        epoch_rkd_early = 0.0
        epoch_rkd_late  = 0.0
        epoch_log       = 0.0
        num_batches     = 0

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
            if USE_KD and B < 2:
                continue

            snr_dB_train = np.random.choice(snr_dB_list)
            snr_train    = 10 ** (snr_dB_train / 10)
            R = get_R(snr_dB_train, B)

            # ── Teacher pass ───────────────────────────────────────────────────
            # Needed for L_log (all KD cells) and CI-RKD windows (cells 2,4)
            if USE_KD:
                with torch.no_grad():
                    (_, _,
                     F_T_early, W_T_early,
                     F_T_late,  W_T_late,
                     J_T_final) = model_teacher.execute_PGA_two_windows(
                        H, R, snr_train, I_T, N_INNER_T,
                        t_start_early=t_start_early,
                        t_start_late=t_start_late,
                        K_layers=K_layers)
                    epoch_task_t += get_sum_loss(
                        F_T_late[-1], W_T_late[-1], H, R, snr_train, B).item()

            # ── Student pass ───────────────────────────────────────────────────
            (_, _,
             F_s, W_s,
             F_S_first, W_S_first,
             F_S_last,  W_S_last,
             rate_over_iters,
             tau_over_iters) = model_student.execute_PGA_with_windows(
                H, R, snr_train, I_S, N_INNER_S,
                K_layers=K_layers,
                collect_windows=USE_RKD)

            # ── Task loss ──────────────────────────────────────────────────────
            loss_task  = get_sum_loss(F_s, W_s, H, R, snr_train, B)
            total_loss = loss_task

            # ── L_log (cells 2, 3, 4) ─────────────────────────────────────────
            if USE_LOG:
                loss_log = log_weighted_deep_loss(
                    rate_over_iters, tau_over_iters,
                    J_T_final, lambda_log=lambda_log)
                total_loss = total_loss + loss_log
                epoch_log += loss_log.item()

            # ── CI-RKD (cells 2, 4) ───────────────────────────────────────────
            if USE_RKD:
                loss_rkd_early = log_weighted_rkd_loss(
                    F_T_early, W_T_early,
                    F_S_first, W_S_first,
                    K_layers, B,
                    lambda_dist=lambda_dist,
                    lambda_angle=lambda_angle)

                loss_rkd_late = log_weighted_rkd_loss(
                    F_T_late,  W_T_late,
                    F_S_last,  W_S_last,
                    K_layers, B,
                    lambda_dist=lambda_dist,
                    lambda_angle=lambda_angle)

                total_loss = (total_loss
                              + loss_rkd_early
                              + lambda_late * loss_rkd_late)

                epoch_rkd_early += loss_rkd_early.item()
                epoch_rkd_late  += loss_rkd_late.item()

            # ── NaN guard ──────────────────────────────────────────────────────
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"  [WARNING] NaN/Inf epoch={i_epoch} "
                      f"batch={i_batch} — skipping.")
                optimizer.zero_grad()
                continue

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
        if USE_LOG:
            log += f" | Log: {epoch_log/nb:.4f}"
        if USE_RKD:
            log += (f" | RKD_e: {epoch_rkd_early/nb:.4f}"
                    f" | RKD_l: {epoch_rkd_late/nb:.4f}")
        if USE_KD:
            log += f" | Task_T: {epoch_task_t/nb:.4f}"
        print(log)

        with torch.no_grad():
            ss = model_student.step_size.data
            print(f"             step_size: "
                  f"min={ss.min():.4e}  max={ss.max():.4e}  "
                  f"mean={ss.mean():.4e}  std={ss.std():.4e}")

    # ── Save ───────────────────────────────────────────────────────────────────
    save_path = (model_file_name_UPGA_J10 +
                 f'_I60_{ABLATION_CELL}_{init_tag}_{rkd_tag}'
                 f'_Kl{K_layers}_win{t_start_early}_correct.pth')
    torch.save(model_student.state_dict(), save_path)
    print(f'\nStudent saved -> {save_path}')


# ══════════════════════════════════════════════════════════════════════════════
# 7.  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

if run_RKD_Distillation == 0:

    ABLATION_CELL  = 'cell_4'
    USE_AGT        = ABLATION_CELL in ('cell_3', 'cell_4')
    USE_RKD        = ABLATION_CELL in ('cell_2', 'cell_4')
    USE_LOG        = ABLATION_CELL in ('cell_2', 'cell_3', 'cell_4')

    inner_strategy = 'avg_pairs'
    step_size_seed = 0.01
    K_layers       = 15
    t_start_early  = 20

    if USE_RKD:
        rkd_tag = 'RKDlog_30layers'
    elif USE_LOG:
        rkd_tag = 'logOnly_30layers'
    else:
        rkd_tag = 'noRKD'

    init_tag = (f'AGT_{inner_strategy}' if USE_AGT
                else f'flat{step_size_seed}')

    save_path = (model_file_name_UPGA_J10 +
                 f'_I60_{ABLATION_CELL}_{init_tag}_{rkd_tag}'
                 f'_Kl{K_layers}_win{t_start_early}_corrected.pth')

    print(f'Loading: {save_path}')
    model_test = PGA_Unfold_J10_I60_v2(step_size_UPGA_J10).to(device)
    model_test.load_state_dict(torch.load(save_path, map_location=device))
    model_test.eval()

    with torch.no_grad():
        (rate_iter, beam_iter, _, _,
         _, _, _, _,
         rate_over_iters, tau_over_iters) = \
            model_test.execute_PGA_with_windows(
                H_test, Rtest, snr, I_S, N_INNER_S,
                K_layers=0, collect_windows=False)

    rate_out = (sum(rate_iter) / len(H_test[0])).detach().cpu().numpy()
    beam_out = (sum(beam_iter) / len(H_test[0])).detach().cpu().numpy()

    print(f'\n{ABLATION_CELL} | {init_tag} | {rkd_tag}')
    print(f'  Final sum rate   : {rate_out[-1]:.4f} bps/Hz')
    print(f'  Final beam error : {beam_out[-1]:.4f}')
