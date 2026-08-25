"""
CI-RKD Distillation — J10/I60 Student  [OUTER-ONLY INITIALISATION]
====================================================================
Teacher : PGA_Unfold_J20        [20 inner × 120 outer]  pretrained, frozen
Student : PGA_Unfold_J10_I60_CI [10 inner ×  60 outer]  trained with CI-RKD

What 'outer-only' means
------------------------
We transfer the teacher's OUTER convergence schedule only.
The inner step geometry (how step sizes vary across inner refinement
steps within a single outer iteration) is explicitly discarded by
averaging over the inner axis.

Pipeline:
  Step 1  Average over all 20 inner iterations
          Result: [1, 120, K+1]  ← removes inner geometry entirely
  Step 2  Extract n_outer_S outer slots using the chosen strategy
          Result: [1, 60, K+1]   ← the 'outer schedule'
  Step 3  Broadcast uniformly to [10, 60, K+1]
          Every inner slot starts identical.
          The student's inner geometry is completely free to learn.

Three outer strategies available:
  'first'    copy teacher outer iters [0:60]
             The teacher's active descent phase.
  'last'     copy teacher outer iters [60:120]
             The teacher's near-plateau phase.
  'steepest' copy teacher outer iters [t_start:t_start+60]
             The teacher's steepest convergence region.

Transfer learning analogy
--------------------------
This is the structural COMPLEMENT of inner-only initialisation.
Instead of transferring local optimization geometry (inner axis),
we transfer global convergence schedule (outer axis).

The outer schedule encodes how step sizes should evolve as the
solution converges across 120 outer iterations.  This knowledge
IS architecture-specific: the teacher's outer schedule was learned
assuming 120 steps are available.  Transferring it to a 60-step
student may mislead: the teacher's step size at iter 30 was designed
for a point halfway through a 120-step trajectory, not for the
endpoint of a 60-step one.

Ablation role
-------------
Expected result: outer-only ≈ random init (outer schedule does not
transfer across architectures of different depth).  This would
confirm that the inner axis carries the transferable knowledge and
validate the specificity hierarchy claim.

Compare against:
  inner_only  → if inner_only >> outer_only, inner transfers, outer doesn't
  full_copy   → upper bound with both axes transferred
  random_init → lower bound baseline
"""

import time
import torch
import torch.nn as nn
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

# ── Radar cache ───────────────────────────────────────────────────────────────
radar_cache = {}
for _snr_db in snr_dB_list:
    _R, _, _, _ = get_radar_data(_snr_db, H_train[:, :1, :, :])
    radar_cache[_snr_db] = _R.to(device)

Rtest, at, theta, ideal_beam = get_radar_data(snr_dB, H_test.cpu())
Rtest = Rtest.to(device)

def get_R(snr_db, B):
    return radar_cache[snr_db].expand(-1, B, -1, -1)

# ── Iteration counts ──────────────────────────────────────────────────────────
I_T       = n_iter_outer
I_S       = n_iter_outer // 2
N_INNER_T = n_iter_inner_J20
N_INNER_S = n_iter_inner_J10


# ══════════════════════════════════════════════════════════════════════════════
# 1.  OUTER-ONLY INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def build_outer_only_init(teacher_model, n_outer_S=60,
                           outer_strategy='first',
                           t_start_steepest=20):
    """
    Transfer only the teacher's outer convergence schedule.

    Parameters
    ----------
    teacher_model     : loaded frozen PGA_Unfold_J20_Teacher on device
    n_outer_S         : student outer iterations (60)
    outer_strategy    : which part of the teacher's outer schedule to copy
        'first'    copy teacher outer iters [0 : n_outer_S]
                   Active descent phase — most gradient signal.
        'last'     copy teacher outer iters [-n_outer_S :]
                   Plateau phase — low gradient signal.
        'steepest' copy teacher outer iters [t_start : t_start+n_outer_S]
                   Steepest descent region from convergence plot.
    t_start_steepest  : start of steepest window (default 20, used only
                        when outer_strategy='steepest')

    Returns
    -------
    ss_init : Tensor [10, 60, K+1]
        All 10 inner slots are identical (uniform inner geometry).
        Only the outer axis carries teacher knowledge.

    Design note
    -----------
    Step 1 averages over the inner axis: [20, 120, K+1] → [1, 120, K+1].
    This is the complement of inner-only Step 2 (which averages over outer).
    What remains is a profile of how step sizes evolve across outer
    iterations, averaged over all inner refinement stages.

    Step 2 extracts the chosen n_outer_S outer slots.
    Step 3 broadcasts to all 10 inner slots: [1, 60, K+1] → [10, 60, K+1].
    Every inner step starts with the same outer schedule; the student
    learns to differentiate the inner steps during training.
    """
    with torch.no_grad():
        ss_T      = teacher_model.step_size.data   # [20, 120, K+1]
        n_outer_t = ss_T.shape[1]                  # 120

        # ── Step 1: average over inner axis → outer profile ───────────────────
        # This is the key operation: it collapses the inner geometry entirely.
        # What remains is purely how step sizes evolve across the 120 outer
        # iterations, averaged over all inner refinement stages.
        outer_profile = ss_T.mean(dim=0, keepdim=True)  # [1, 120, K+1]

        # ── Step 2: extract n_outer_S outer slots ─────────────────────────────
        if outer_strategy == 'first':
            # Teacher's active descent phase: outer iters [0, 60).
            # These step sizes were learned when the algorithm was still
            # far from convergence and taking large corrective steps.
            assert n_outer_S <= n_outer_t
            outer_selected = outer_profile[:, :n_outer_S, :]
            outer_desc = f"first: teacher outer [0:{n_outer_S}] (descent phase)"

        elif outer_strategy == 'last':
            # Teacher's plateau phase: outer iters [60, 120).
            # These step sizes were learned near convergence — small and
            # conservative.  Expected to perform poorly as initialization
            # since the student starts far from convergence.
            assert n_outer_S <= n_outer_t
            outer_selected = outer_profile[:, -n_outer_S:, :]
            outer_desc = (f"last: teacher outer "
                          f"[{n_outer_t-n_outer_S}:{n_outer_t}] (plateau phase)")

        elif outer_strategy == 'steepest':
            # Teacher's steepest convergence region: outer iters
            # [t_start_steepest, t_start_steepest + n_outer_S).
            # These step sizes come from where the teacher's loss curve
            # drops most steeply — maximum gradient signal.
            t_end = t_start_steepest + n_outer_S
            assert t_end <= n_outer_t, (
                f"Steepest window [{t_start_steepest},{t_end}) "
                f"exceeds teacher depth {n_outer_t}")
            outer_selected = outer_profile[:, t_start_steepest:t_end, :]
            outer_desc = (f"steepest: teacher outer "
                          f"[{t_start_steepest}:{t_end}]")

        else:
            raise ValueError(f"Unknown outer_strategy {outer_strategy!r}. "
                             f"Choose: 'first', 'last', 'steepest'.")

        # outer_selected : [1, 60, K+1]

        # ── Step 3: broadcast uniformly to all inner slots ────────────────────
        # All 10 inner steps start with the same outer schedule profile.
        # The inner geometry is completely free to be learned.
        ss_init = outer_selected.expand(N_INNER_S, -1, -1).clone()
        # ss_init : [10, 60, K+1]

    print(f"\n[OUTER-ONLY init | strategy='{outer_strategy}']")
    print(f"  Inner : averaged → discarded (uniform broadcast)")
    print(f"  Outer : {outer_desc}")
    print(f"  teacher {list(ss_T.shape)}"
          f" → outer_profile {list(outer_profile.shape)}"
          f" → selected {list(outer_selected.shape)}"
          f" → student init {list(ss_init.shape)}")
    print(f"  Range : [{ss_init.min():.4e}, {ss_init.max():.4e}]")
    print(f"  Inner variation : {ss_init.std(dim=0).mean():.4e} "
          f"(should be ~0 — confirms uniform inner geometry)\n")
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
# 3.  RKD LOSSES  (vectorised — identical to inner_only file)
# ══════════════════════════════════════════════════════════════════════════════

def ci_rkd_loss(F_T_window, F_S_window, K_layers, B,
                lambda_dist=25.0, lambda_angle=50.0):
    weight_sum = K_layers * (K_layers + 1) / 2

    T_feats = torch.stack(
        [_to_real(flatten_F(F_T_window[l], B)).detach() for l in range(K_layers)],
        dim=0)
    S_feats = torch.stack(
        [_to_real(flatten_F(F_S_window[l], B)) for l in range(K_layers)],
        dim=0)

    with torch.no_grad():
        T_dist = torch.cdist(T_feats, T_feats, p=2)
        T_dist = T_dist / T_dist.mean(dim=(-2,-1), keepdim=True).clamp(min=1e-12)

    S_dist = torch.cdist(S_feats, S_feats, p=2)
    S_dist = S_dist / S_dist.mean(dim=(-2,-1), keepdim=True).clamp(min=1e-12)

    weights = (torch.arange(1, K_layers+1, dtype=S_dist.dtype, device=S_dist.device)
               / weight_sum).view(K_layers, 1, 1)

    dist_loss = nn.functional.smooth_l1_loss(
        S_dist * weights, T_dist * weights, reduction='mean')

    angle_loss = 0.0
    for l in range(K_layers):
        w_l = (l + 1) / weight_sum
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
# 4.  TEACHER MODEL  (identical to inner_only file)
# ══════════════════════════════════════════════════════════════════════════════

class PGA_Unfold_J20_Teacher(PGA_Unfold_J20):

    def execute_PGA_two_windows(self, H, R, Pt, n_iter_outer, n_iter_inner,
                                 t_start_early, t_start_late, K_layers):
        t_end_early = t_start_early + K_layers
        t_end_late  = t_start_late  + K_layers
        assert t_end_early <= n_iter_outer
        assert t_end_late  <= n_iter_outer

        _, _, F, W = initialize(H, R, Pt, initial_normalization)
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
# 5.  STUDENT MODEL  (identical to inner_only file)
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
        rate_init, tau_init, F, W = initialize(H, R, Pt, initial_normalization)
        rate_over_iters = torch.zeros(n_iter_outer, H.shape[1], device=H.device)
        tau_over_iters  = torch.zeros(n_iter_outer, H.shape[1], device=H.device)
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

    print('=' * 70)
    print('CI-RKD  J10/I60  [OUTER-ONLY INIT]')
    print(f'Transfers: outer convergence schedule only — inner geometry is FREE')
    print(f'Teacher : {I_T} outer × {N_INNER_T} inner = {I_T*N_INNER_T} steps/sample')
    print(f'Student :  {I_S} outer × {N_INNER_S} inner =  {I_S*N_INNER_S} steps/sample')
    print(f'Speedup : {(I_T*N_INNER_T)//(I_S*N_INNER_S)}×')
    print('=' * 70 + '\n')

    # ── Change this to compare the three outer strategies ─────────────────────
    outer_strategy   = 'first'   # 'first' | 'last' | 'steepest'
    t_start_steepest = 20        # only used when outer_strategy='steepest'

    K_layers      = 15
    lambda_dist   = 25.0
    lambda_angle  = 50.0
    lambda_late   = 1.0
    t_start_early = 20
    t_start_late  = I_T - K_layers
    min_step_size = 1e-8
    max_step_size = 0.2

    assert t_start_early + K_layers <= I_T
    assert K_layers <= I_S // 2

    print(f'outer_strategy : {outer_strategy!r}')
    print(f'Early : teacher [{t_start_early}, {t_start_early+K_layers})'
          f'  →  student [0, {K_layers})')
    print(f'Late  : teacher [{t_start_late}, {I_T})'
          f'  →  student [{I_S-K_layers}, {I_S})')
    print(f'K_layers={K_layers}  |  lambda_late={lambda_late}'
          f'  |  step_size ∈ [{min_step_size}, {max_step_size}]'
          f'  |  batch={batch_size}\n')

    model_teacher = PGA_Unfold_J20_Teacher(step_size_UPGA_J20).to(device)
    model_teacher.load_state_dict(
        torch.load(model_file_name_UPGA_J20, map_location=device))
    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad_(False)
    print(f'Teacher loaded : {model_file_name_UPGA_J20}')

    ss_init = build_outer_only_init(
        model_teacher,
        n_outer_S        = I_S,
        outer_strategy   = outer_strategy,
        t_start_steepest = t_start_steepest)

    model_student = PGA_Unfold_J10_I60_CI(ss_init).to(device)
    optimizer     = torch.optim.Adam(model_student.parameters(), lr=learning_rate)
    print(f'Student step_size shape : {list(model_student.step_size.shape)}\n')

    for i_epoch in range(n_epoch):
        start_time      = time.time()
        epoch_loss      = epoch_rkd_early = epoch_rkd_late = 0.0
        epoch_task_s    = epoch_task_t    = 0.0
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
                H_shuffled[i_batch:i_batch+batch_size], 0, 1).to(device)
            B = H.shape[1]
            if B < 2:
                continue

            snr_dB_train = np.random.choice(snr_dB_list)
            snr_train    = 10 ** (snr_dB_train / 10)
            R = get_R(snr_dB_train, B)

            with torch.no_grad():
                F_t, W_t, F_T_early, F_T_late = \
                    model_teacher.execute_PGA_two_windows(
                        H, R, snr_train, I_T, N_INNER_T,
                        t_start_early=t_start_early,
                        t_start_late=t_start_late,
                        K_layers=K_layers)
                teacher_task_loss = get_sum_loss(F_t, W_t, H, R, snr_train, B)

            rate, _, F_s, W_s, F_S_first, F_S_last = \
                model_student.execute_PGA_with_both_windows(
                    H, R, snr_train, I_S, N_INNER_S, K_layers)

            loss_task      = get_sum_loss(F_s, W_s, H, R, snr_train, B)
            loss_rkd_early = ci_rkd_loss(F_T_early, F_S_first,
                                          K_layers, B, lambda_dist, lambda_angle)
            loss_rkd_late  = ci_rkd_loss(F_T_late,  F_S_last,
                                          K_layers, B, lambda_dist, lambda_angle)
            total_loss = loss_task + loss_rkd_early + lambda_late * loss_rkd_late

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
            epoch_rkd_early += loss_rkd_early.item()
            epoch_rkd_late  += loss_rkd_late.item()
            epoch_task_s    += loss_task.item()
            epoch_task_t    += teacher_task_loss.item()
            num_batches     += 1

        nb = max(num_batches, 1)
        print(f"Epoch {i_epoch:4d} [outer_only | {outer_strategy}] | "
              f"Time: {time.time()-start_time:.1f}s | "
              f"Loss: {epoch_loss/nb:.4f} | "
              f"Student: {epoch_task_s/nb:.4f} | "
              f"Teacher: {epoch_task_t/nb:.4f} | "
              f"RKD_early: {epoch_rkd_early/nb:.4f} | "
              f"RKD_late: {epoch_rkd_late/nb:.4f}")
        with torch.no_grad():
            ss = model_student.step_size.data
            print(f"             step_size : "
                  f"min={ss.min():.4e}  max={ss.max():.4e}  "
                  f"mean={ss.mean():.4e}  std={ss.std():.4e}")

    save_path = (model_file_name_UPGA_J10 +
                 f'_I60_CI_RKD_OUTER_ONLY_{outer_strategy}'
                 f'_Kl{K_layers}_win{t_start_early}.pth')
    torch.save(model_student.state_dict(), save_path)
    print(f'\nStudent saved → {save_path}')


# ══════════════════════════════════════════════════════════════════════════════
# 7.  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

if run_RKD_Distillation == 0:

    K_layers       = 15
    t_start_early  = 20
    outer_strategy = 'first'   # must match training run

    save_path = (model_file_name_UPGA_J10 +
                 f'_I60_CI_RKD_OUTER_ONLY_{outer_strategy}'
                 f'_Kl{K_layers}_win{t_start_early}.pth')

    model_test = PGA_Unfold_J10_I60_CI(step_size_UPGA_J10).to(device)
    model_test.load_state_dict(torch.load(save_path, map_location=device))
    model_test.eval()

    with torch.no_grad():
        rate_iter, beam_iter, _, _, _, _ = \
            model_test.execute_PGA_with_both_windows(
                H_test, Rtest, snr, I_S, N_INNER_S, K_layers=0)

    rate_RKD = [r.detach().cpu().numpy()
                for r in (sum(rate_iter) / len(H_test[0]))]
    beam_RKD = [r.detach().cpu().numpy()
                for r in (sum(beam_iter) / len(H_test[0]))]

    print(f'\nJ10/I60 | OUTER_ONLY_{outer_strategy} | K_layers={K_layers}')
    print(f'  Final sum rate   : {rate_RKD[-1]:.4f} bps/Hz')
    print(f'  Final beam error : {beam_RKD[-1]:.4f}')