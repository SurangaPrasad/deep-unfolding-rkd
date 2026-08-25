"""
Ablation 1: J5/I60 — NO INIT, NO RKD
======================================
Random step_size initialisation. Task loss only.
This is the true lower bound — no teacher knowledge transferred.
Compare against all other conditions to measure total gain.
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

I_T       = n_iter_outer
I_S       = n_iter_outer // 2
N_INNER_T = n_iter_inner_J20
N_INNER_S = 5


# ══════════════════════════════════════════════════════════════════════════════
# STUDENT MODEL
# ══════════════════════════════════════════════════════════════════════════════

class PGA_Unfold_J3_I60_CI(nn.Module):

    def __init__(self, step_size_init):
        super().__init__()
        if isinstance(step_size_init, torch.Tensor) and step_size_init.dim() == 3:
            self.step_size = nn.Parameter(step_size_init.float().clone())
        else:
            self.step_size = nn.Parameter(
                step_size_init * torch.ones(N_INNER_S, I_S, K + 1))

    def execute_PGA(self, H, R, Pt, n_iter_outer, n_iter_inner):
        rate_init, tau_init, F, W = initialize(H, R, Pt, initial_normalization)
        rate_over_iters = torch.zeros(n_iter_outer, H.shape[1], device=H.device)
        tau_over_iters  = torch.zeros(n_iter_outer, H.shape[1], device=H.device)

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

            rate_over_iters[ii] = get_sum_rate(H, F, W, Pt)
            tau_over_iters[ii]  = get_beam_error(H, F, W, R, Pt)

        rates = torch.cat([rate_init, rate_over_iters], dim=0)
        taus  = torch.cat([tau_init,  tau_over_iters],  dim=0)
        return (torch.transpose(rates, 0, 1),
                torch.transpose(taus,  0, 1),
                F, W)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

if run_RKD_Distillation == 1:

    print('=' * 70)
    print('ABLATION 1: NO INIT, NO RKD  —  J5/I60  (lower bound)')
    print(f'Student :  {I_S} outer × {N_INNER_S} inner = {I_S*N_INNER_S} steps/sample')
    print(f'Speedup : {(I_T*N_INNER_T)//(I_S*N_INNER_S)}×')
    print('=' * 70 + '\n')

    min_step_size = 1e-8
    max_step_size = 0.35
    stabilised_lr = learning_rate / 2.0

    # Random initialisation using the default scalar seed
    model_student = PGA_Unfold_J3_I60_CI(step_size_UPGA_J5_I60).to(device)
    optimizer     = torch.optim.Adam(model_student.parameters(), lr=stabilised_lr)
    print(f'Student step_size shape : {list(model_student.step_size.shape)}')
    print(f'Init range : [{model_student.step_size.min():.4e},'
          f' {model_student.step_size.max():.4e}]  (random/default)\n')

    best_student_loss = -float('inf')

    for i_epoch in range(n_epoch):
        start_time   = time.time()
        epoch_loss   = 0.0
        epoch_task_s = 0.0
        num_batches  = 0

        if torch.isnan(model_student.step_size.data).any():
            print(f"  [WARNING] NaN at epoch {i_epoch}. Resetting.")
            model_student = PGA_Unfold_J3_I60_CI(step_size_UPGA_J5_I60).to(device)
            optimizer = torch.optim.Adam(model_student.parameters(), lr=stabilised_lr)

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

            _, _, F_s, W_s = model_student.execute_PGA(
                H, R, snr_train, I_S, N_INNER_S)

            total_loss = get_sum_loss(F_s, W_s, H, R, snr_train, B)

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

            epoch_loss   += total_loss.item()
            epoch_task_s += total_loss.item()
            num_batches  += 1

        nb = max(num_batches, 1)
        current_student = epoch_task_s / nb

        print(f"Epoch {i_epoch:4d} [no_init_no_rkd] | "
              f"Time: {time.time()-start_time:.1f}s | "
              f"Student: {current_student:.4f}")

        with torch.no_grad():
            ss = model_student.step_size.data
            print(f"             step_size : min={ss.min():.4e}  "
                  f"max={ss.max():.4e}  mean={ss.mean():.4e}")

        if current_student > best_student_loss:
            best_student_loss = current_student
            torch.save(model_student.state_dict(),
                       model_file_name_UPGA_J10 +
                       '_J3_I60_NO_INIT_NO_RKD_best.pth')
            print(f"  [BEST] epoch {i_epoch} → saved")

    torch.save(model_student.state_dict(),
               model_file_name_UPGA_J10 + '_J5_I60_NO_INIT_NO_RKD_old.pth')
    print(f'\nSaved.')


