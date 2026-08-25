"""
Train_J10_I20_v2.py
====================
J20/I80 Teacher  →  J10/I20 Student
Compression ratio: 8x (20x80 -> 10x20)

Four ablation cells:
  cell_1: flat init, task loss only
  cell_2: flat init, L_task + L_log + CI-RKD
  cell_3: AGT  init, task loss only
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
# OVERRIDE SYSTEM CONFIG
# ══════════════════════════════════════════════════════════════════════════════
I_T = 80          # teacher outer iterations
I_S = 40          # student outer iterations
J_T = n_iter_inner_J20   # 20
J_S = n_iter_inner_J10   # 10
p   = I_T//I_S # outer compression factor = 4

print(f"\n{'='*70}")
print(f"  Teacher : J={J_T}, I={I_T}  ({J_T*I_T} steps)")
print(f"  Student : J={J_S}, I={I_S}  ({J_S*I_S} steps)")
print(f"  Compression ratio : {J_T*I_T}/{J_S*I_S} = {J_T*I_T//(J_S*I_S)}x")
print(f"  Outer compression factor p = {p}")
print(f"{'='*70}\n")

# ══════════════════════════════════════════════════════════════════════════════
# CHOOSE CELL
# ══════════════════════════════════════════════════════════════════════════════
ABLATION_CELL = 'cell_3'   # 'cell_1'|'cell_2'|'cell_3'|'cell_4'
# ══════════════════════════════════════════════════════════════════════════════

assert ABLATION_CELL in ('cell_1', 'cell_2', 'cell_3', 'cell_4')

USE_AGT = ABLATION_CELL in ('cell_3', 'cell_4')
USE_RKD = ABLATION_CELL in ('cell_2', 'cell_4')
USE_LOG = ABLATION_CELL in ('cell_2', 'cell_4')  # L_log only when RKD present
USE_KD  = USE_RKD or USE_LOG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device         : {device}")
print(f"ABLATION CELL  : {ABLATION_CELL}")
print(f"AGT init       : {USE_AGT}")
print(f"L_log          : {USE_LOG}")
print(f"CI-RKD         : {USE_RKD}\n")


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# AGT INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def build_agt_init(teacher_model):
    """
    Stage 1: compress inner J_T=20 -> J_S=10 by averaging pairs
    Stage 2: compress outer I_T=80  -> I_S=20 by averaging groups of p=4
    """
    with torch.no_grad():
        ss_T = teacher_model.step_size.data   # [20, 80, K+1]

        # Stage 1: inner compression [20,80,K+1] -> [10,80,K+1]
        assert ss_T.shape[0] == 2 * J_S, \
            f"Expected J_T={2*J_S}, got {ss_T.shape[0]}"
        ss_inner = ss_T.view(J_S, 2, I_T, K+1).mean(dim=1)

        # Stage 2: outer compression [10,80,K+1] -> [10,20,K+1]
        assert I_T % I_S == 0, \
            f"I_T={I_T} must be divisible by I_S={I_S}"
        ss_init = ss_inner.view(J_S, I_S, p, K+1).mean(dim=2)

    print(f"[AGT init]")
    print(f"  {list(ss_T.shape)} -> {list(ss_inner.shape)} "
          f"-> {list(ss_init.shape)}")
    print(f"  Range: [{ss_init.min():.4e}, {ss_init.max():.4e}]\n")
    return ss_init


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════

def get_effective_beamformer(F, W, B):
    FW = torch.matmul(F, W)
    FW = FW.permute(1, 0, 2, 3).reshape(B, -1)
    if torch.is_complex(FW):
        FW = torch.view_as_real(FW).flatten(-2)
    return FW


# ══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
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
            d_T = torch.cdist(phi_T.unsqueeze(0),
                              phi_T.unsqueeze(0), p=2).squeeze(0)
            d_T = d_T / d_T.mean().clamp(min=1e-12)
        d_S = torch.cdist(phi_S.unsqueeze(0),
                          phi_S.unsqueeze(0), p=2).squeeze(0)
        d_S = d_S / d_S.mean().clamp(min=1e-12)
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
                            J_T_final, lambda_log=0.01):
    device_  = rate_over_iters.device
    omega_t  = torch.as_tensor(OMEGA, device=device_, dtype=torch.float32)
    J_S_vals = rate_over_iters - omega_t * tau_over_iters
    log_w    = torch.log(
        torch.arange(1, I_S + 1, dtype=torch.float32,
                     device=device_).clamp(min=1e-8)).unsqueeze(1)
    sq_diff  = (J_S_vals - J_T_final.detach()).pow(2)
    loss     = (log_w * sq_diff.mean(dim=1)).sum() / I_S
    return lambda_log * loss


# ══════════════════════════════════════════════════════════════════════════════
# TEACHER MODEL
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
# STUDENT MODEL
# ══════════════════════════════════════════════════════════════════════════════

class PGA_Unfold_J10_IS_v2(nn.Module):
    """
    J=10, I=I_S student with trajectory window collection.
    step_size shape: [J_S, I_S, K+1]
    """

    def __init__(self, step_size_init):
        super().__init__()
        if isinstance(step_size_init, torch.Tensor) and step_size_init.dim() == 3:
            self.step_size = nn.Parameter(step_size_init.float().clone())
        else:
            self.step_size = nn.Parameter(
                step_size_init * torch.ones(J_S, I_S, K + 1))

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
# HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

step_size_seed = 0.01
K_layers       = 5              # window length (shorter since I_S=20)
t_start_early  = 10             # teacher early window start
t_start_late   = I_T - K_layers # teacher late window start

lambda_dist    = 25.0
lambda_angle   = 50.0
lambda_log     = 1e-5
lambda_late    = 1.0
min_step_size  = 1e-8
max_step_size  = 0.5

assert t_start_early + K_layers <= I_T
assert K_layers <= I_S // 2

print(f'K_layers={K_layers}  '
      f't_early=[{t_start_early},{t_start_early+K_layers})  '
      f't_late=[{t_start_late},{I_T})')
print(f'lambda_dist={lambda_dist}  lambda_angle={lambda_angle}  '
      f'lambda_log={lambda_log}  lambda_late={lambda_late}\n')


# ══════════════════════════════════════════════════════════════════════════════
# LOAD TEACHER
# ══════════════════════════════════════════════════════════════════════════════

# Teacher must be pretrained with I_T=80 outer iterations
# Use a step_size tensor of shape [20, 80, K+1]
step_size_teacher = torch.full(
    [J_T, I_T, K + 1], step_size_seed,
    device=device, requires_grad=True)

model_teacher = PGA_Unfold_J20_Teacher(step_size_teacher).to(device)
model_teacher.load_state_dict(
    torch.load('UPGA_J20_I80_w_0_30.pth', map_location=device))
model_teacher.eval()
for p_param in model_teacher.parameters():
    p_param.requires_grad_(False)
print('Teacher loaded: UPGA_J20_I80_w_0_30.pth\n')


# ══════════════════════════════════════════════════════════════════════════════
# INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

if USE_AGT:
    ss_init  = build_agt_init(model_teacher)
    init_tag = 'AGT_avg_pairs'
else:
    ss_init  = torch.full((J_S, I_S, K + 1), step_size_seed)
    init_tag = f'flat{step_size_seed}'
    print(f'[Flat init | seed={step_size_seed}  '
          f'shape={list(ss_init.shape)}]\n')

if USE_RKD:
    rkd_tag = 'RKDlog'
elif USE_LOG:
    rkd_tag = 'logOnly'
else:
    rkd_tag = 'noRKD'


# ══════════════════════════════════════════════════════════════════════════════
# STUDENT
# ══════════════════════════════════════════════════════════════════════════════

model_student = PGA_Unfold_J10_IS_v2(ss_init).to(device)
optimizer     = torch.optim.Adam(model_student.parameters(), lr=learning_rate)
print(f'Student step_size shape: {list(model_student.step_size.shape)}\n')


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

for i_epoch in range(n_epoch):
    start_time      = time.time()
    epoch_loss      = 0.0
    epoch_task_s    = 0.0
    epoch_task_t    = 0.0
    epoch_rkd_early = 0.0
    epoch_rkd_late  = 0.0
    epoch_log       = 0.0
    num_batches     = 0

    # NaN guard
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

        # ── Teacher pass ───────────────────────────────────────────────────────
        if USE_KD:
            with torch.no_grad():
                (_, _,
                 F_T_early, W_T_early,
                 F_T_late,  W_T_late,
                 J_T_final) = model_teacher.execute_PGA_two_windows(
                    H, R, snr_train, I_T, J_T,
                    t_start_early=t_start_early,
                    t_start_late=t_start_late,
                    K_layers=K_layers)
                epoch_task_t += get_sum_loss(
                    F_T_late[-1], W_T_late[-1], H, R, snr_train, B).item()

        # ── Student pass ───────────────────────────────────────────────────────
        (_, _,
         F_s, W_s,
         F_S_first, W_S_first,
         F_S_last,  W_S_last,
         rate_over_iters,
         tau_over_iters) = model_student.execute_PGA_with_windows(
            H, R, snr_train, I_S, J_S,
            K_layers=K_layers,
            collect_windows=USE_RKD)

        # ── Task loss ──────────────────────────────────────────────────────────
        loss_task  = get_sum_loss(F_s, W_s, H, R, snr_train, B)
        total_loss = loss_task

        # ── L_log (cells 2 and 4 only) ────────────────────────────────────────
        if USE_LOG:
            loss_log = log_weighted_deep_loss(
                rate_over_iters, tau_over_iters,
                J_T_final, lambda_log=lambda_log)
            total_loss = total_loss + loss_log
            epoch_log += loss_log.item()

        # ── CI-RKD (cells 2 and 4 only) ───────────────────────────────────────
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

        # ── NaN guard ──────────────────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════

save_path = (model_file_name_UPGA_J10 +
             f'_IS{I_S}_IT{I_T}_{ABLATION_CELL}_{init_tag}_{rkd_tag}'
             f'_Kl{K_layers}_win{t_start_early}_030.pth')
torch.save(model_student.state_dict(), save_path)
print(f'\nStudent saved -> {save_path}')


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

model_test = PGA_Unfold_J10_IS_v2(ss_init).to(device)
model_test.load_state_dict(torch.load(save_path, map_location=device))
model_test.eval()

with torch.no_grad():
    (rate_iter, beam_iter, _, _,
     _, _, _, _,
     rate_over_iters, tau_over_iters) = \
        model_test.execute_PGA_with_windows(
            H_test, Rtest, snr, I_S, J_S,
            K_layers=0, collect_windows=False)

rate_mean = rate_iter.mean(dim=0).cpu().numpy()
beam_mean = beam_iter.mean(dim=0).cpu().numpy()
obj_mean  = rate_mean - OMEGA * beam_mean

print(f'\n{"="*60}')
print(f'  {ABLATION_CELL} | {init_tag} | {rkd_tag}')
print(f'  I_T={I_T}, I_S={I_S}, compression={J_T*I_T//(J_S*I_S)}x')
print(f'  Final sum rate   : {rate_mean[-1]:.4f} bps/Hz')
print(f'  Final beam error : {beam_mean[-1]:.4f}')
print(f'  Final joint obj  : {obj_mean[-1]:.4f}')
print(f'{"="*60}\n')