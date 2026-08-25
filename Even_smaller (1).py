"""
CI-RKD Distillation — J5/I60 Student
[LAST-5 INIT + INNER-LAYER RKD]
======================================================================
Teacher : PGA_Unfold_J20       [20 inner × 120 outer]  pretrained, frozen
Student : PGA_Unfold_J5_I60_CI [ 5 inner ×  60 outer]  trained with CI-RKD

Inference cost
--------------
Teacher : 120 × 20 = 2400 steps/sample
Student :  60 ×  5 =  300 steps/sample  →  8× cheaper

Initialisation — LAST 5
------------------------
Student inner step j is initialised from teacher inner step (15+j):
  student step 0  ←  teacher step 15
  student step 1  ←  teacher step 16
  student step 2  ←  teacher step 17
  student step 3  ←  teacher step 18
  student step 4  ←  teacher step 19

These are the teacher's fine/polishing step sizes — small and conservative.
The student will initially under-step because its inner loop starts from
a rough F (no preceding inner steps), whereas the teacher's steps 15-19
were calibrated for an F already refined by 15 preceding steps.

RKD supervision — INNER LAYERS
--------------------------------
RKD is applied on the inner-layer F-iterates rather than outer boundaries.

For each selected outer iteration in the steepest window:
  Teacher collects F after each of inner steps [15, 16, 17, 18, 19]
  Student collects F after each of inner steps  [0,  1,  2,  3,  4]

Matched pairs (direct 1-to-1 — no subsampling needed):
  student inner j  ←→  teacher inner (15+j)

The RKD gradient pushes student step sizes to produce F-iterates with
similar relational geometry to the teacher's late inner steps — but since
the student starts from a rougher F, it must learn larger step sizes to
achieve this, naturally compensating for the conservative initialisation.

This is the compensation mechanism:
  init  → conservative (under-stepping)
  RKD   → pushes toward teacher inner geometry
  net   → step sizes grow to compensate during training

Total loss
----------
L_total = L_task + L_inner_RKD

L_inner_RKD supervises K_outer × 5 inner iterate pairs per batch.
No outer-boundary RKD. No late window.

The student's outer convergence is entirely determined by the task loss.
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
I_T        = n_iter_outer           # teacher: 120 outer iterations
I_S        = n_iter_outer // 2      # student:  60 outer iterations
N_INNER_T  = n_iter_inner_J20       # teacher:  20 inner iterations
N_INNER_S  = 3                      # student:   5 inner iterations
# Teacher inner steps used for init and RKD reference
INNER_START_T = N_INNER_T - N_INNER_S   # = 15
# Direct mapping: student inner j  ←→  teacher inner (INNER_START_T + j)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LAST-5 INNER INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def build_last5_init(teacher_model, n_outer_S=60):
    """
    Initialise student step_size [5, 60, K+1] from teacher's last 5
    inner steps [15, 16, 17, 18, 19].

    Step 1 — extract teacher inner steps [15:20] → [5, 120, K+1]
    Step 2 — average over 120 outer iterations → [5, 1, K+1] fingerprint
              Removes teacher's outer schedule which is depth-specific.
    Step 3 — broadcast uniformly to [5, 60, K+1]

    The resulting step sizes are small/conservative — the teacher's
    polishing steps. The inner RKD loss will push them to grow during
    training to compensate for operating on rougher F.

    This creates a direct 1-to-1 RKD correspondence:
      student inner j  ←→  teacher inner (15+j)
    which is the cleanest possible inner RKD alignment.
    """
    with torch.no_grad():
        ss_T = teacher_model.step_size.data     # [20, 120, K+1]

        # Step 1: extract last 5 teacher inner steps
        ss_last5      = ss_T[INNER_START_T:].clone()    # [5, 120, K+1]

        # Step 2: collapse outer schedule → fingerprint
        fingerprint   = ss_last5.mean(dim=1, keepdim=True)  # [5, 1, K+1]

        # Step 3: broadcast uniformly
        ss_init       = fingerprint.expand(-1, n_outer_S, -1).clone()  # [5, 60, K+1]

    print(f"\n[J5 LAST-5 init]")
    print(f"  Source : teacher inner steps [{INNER_START_T}:{N_INNER_T}]"
          f" (fine/polishing)")
    print(f"  Mapping: student step j  ←→  teacher step ({INNER_START_T}+j)")
    print(f"  teacher {list(ss_T.shape)}"
          f" → last5 {list(ss_last5.shape)}"
          f" → fingerprint {list(fingerprint.shape)}"
          f" → student init {list(ss_init.shape)}")
    print(f"  Range : [{ss_init.min():.4e}, {ss_init.max():.4e}]")
    print(f"  NOTE  : conservative init — RKD will push step sizes up\n")
    return ss_init


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def flatten_F(F_mat, B):
    """[K_u, B, Nt, Nrf] complex → [B, K_u*Nt*Nrf] flattened."""
    return F_mat.permute(1, 0, 2, 3).reshape(B, -1)

def _to_real(x):
    """Complex → real by interleaving Re/Im. Unchanged if already real."""
    if torch.is_complex(x):
        return torch.view_as_real(x).flatten(-2)
    return x


# ══════════════════════════════════════════════════════════════════════════════
# 3.  INNER-LAYER RKD LOSS
#
#     Operates on F-iterates collected DURING the inner loop rather than
#     at outer boundaries. For each selected outer iteration, we have
#     N_INNER_S=5 matched (teacher, student) F pairs.
#
#     The reversal weight schedule w_j = (j+1)/sum(1..N_INNER_S) weights
#     later inner steps more heavily — they are more refined and stable
#     within the inner trajectory, analogous to the outer reversed schedule.
#
#     Distance loss is vectorised over all K_outer × N_INNER_S pairs.
#     Angle loss loops over pairs (avoids [K*B*B*D] memory spike).
# ══════════════════════════════════════════════════════════════════════════════

def inner_rkd_loss(F_T_inner, F_S_inner, n_inner, B,
                   lambda_dist=12.0, lambda_angle=25.0):
    """
    RKD loss on inner-layer F-iterates.

    Parameters
    ----------
    F_T_inner : list of n_inner teacher F-matrices
                collected at inner steps [INNER_START_T, N_INNER_T)
                within selected outer iterations. Length = K_outer * n_inner.
    F_S_inner : list of n_inner student F-matrices
                collected at inner steps [0, N_INNER_S)
                within the same outer iterations. Length = K_outer * n_inner.
    n_inner   : N_INNER_S = 5 (number of inner steps matched per outer iter)
    B         : batch size

    The total number of matched pairs = K_outer * n_inner.
    Each pair (teacher inner j+k*n_inner, student inner j+k*n_inner)
    corresponds to outer iteration k, inner step j.

    Reversed weight schedule over inner steps:
    w_j = (j % n_inner + 1) / sum(1..n_inner)
    Later inner steps within each outer iteration get higher weight
    because they represent more refined F within that outer iteration.

    lambda_dist=12.0 and lambda_angle=25.0 are halved from J=10 values
    because J=5 iterates have larger raw distances.
    """
    n_pairs    = len(F_T_inner)   # K_outer * n_inner
    weight_sum = n_inner * (n_inner + 1) / 2

    T_feats = torch.stack(
        [_to_real(flatten_F(F_T_inner[l], B)).detach()
         for l in range(n_pairs)], dim=0)   # [n_pairs, B, D_real]

    S_feats = torch.stack(
        [_to_real(flatten_F(F_S_inner[l], B))
         for l in range(n_pairs)], dim=0)   # [n_pairs, B, D_real] — in-graph

    # ── Vectorised distance loss ──────────────────────────────────────────────
    with torch.no_grad():
        T_dist = torch.cdist(T_feats, T_feats, p=2)    # [n_pairs, B, B]
        T_dist = T_dist / T_dist.mean(
            dim=(-2,-1), keepdim=True).clamp(min=1e-12)

    S_dist = torch.cdist(S_feats, S_feats, p=2)        # [n_pairs, B, B]
    S_dist = S_dist / S_dist.mean(
        dim=(-2,-1), keepdim=True).clamp(min=1e-12)

    # Weight by inner step position within each outer iteration
    # pair l corresponds to inner step (l % n_inner)
    inner_positions = torch.tensor(
        [(l % n_inner) + 1 for l in range(n_pairs)],
        dtype=S_dist.dtype, device=S_dist.device)
    weights = (inner_positions / weight_sum).view(n_pairs, 1, 1)

    dist_loss = nn.functional.smooth_l1_loss(
        S_dist * weights, T_dist * weights, reduction='mean')

    # ── Angle loss ────────────────────────────────────────────────────────────
    angle_loss = 0.0
    for l in range(n_pairs):
        w_l = ((l % n_inner) + 1) / weight_sum
        t   = T_feats[l]   # [B, D_real] — detached
        s   = S_feats[l]   # [B, D_real] — in-graph

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
# 4.  TEACHER MODEL — INNER TRAJECTORY COLLECTION
# ══════════════════════════════════════════════════════════════════════════════

class PGA_Unfold_J20_Teacher(PGA_Unfold_J20):
    """
    Teacher wrapper collecting inner-step F-iterates at steps [15:20]
    within selected outer iterations (the steepest descent window).

    For outer iter ii in [t_start_steep, t_start_steep + K_outer):
        collect F after normalize_power at each of inner steps [15, 16, 17, 18, 19]

    This matches exactly the student's inner steps [0, 1, 2, 3, 4]
    via the correspondence: student j  ←→  teacher (15+j).

    Note: F is collected BEFORE the unit-modulus projection F = F/|F|
    so that the inner trajectory is comparable to the student's inner
    trajectory, both of which use normalize_power (not unit-modulus)
    during the inner loop.
    """

    def execute_PGA_inner_windows(self, H, R, Pt, n_iter_outer, n_iter_inner,
                                   t_start_steep, K_outer):
        """
        Single 120-outer pass collecting inner F-iterates.

        Parameters
        ----------
        t_start_steep : int
            First outer iteration in the steepest window.
        K_outer : int
            Number of outer iterations to collect inner trajectories from.
            Total F collected: K_outer × N_INNER_S.

        Returns
        -------
        F, W        : final beamformers (for task loss)
        F_inner_T   : list of K_outer * N_INNER_S teacher F-matrices
                      ordered as [outer0_inner0, outer0_inner1, ...,
                                  outer0_inner4, outer1_inner0, ...]
        """
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

                # Collect inner F at steps [INNER_START_T, N_INNER_T)
                # i.e. the last N_INNER_S=5 teacher inner steps only
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
# 5.  STUDENT MODEL — J=5 inner, 60 outer
# ══════════════════════════════════════════════════════════════════════════════

class PGA_Unfold_J5_I60_CI(nn.Module):
    """
    Student — 60 outer × 5 inner iterations.
    step_size shape: [5, 60, K+1]
    8× cheaper than teacher at inference.

    Collects inner F-iterates at ALL 5 student inner steps within
    selected outer iterations (steepest student window [0, K_outer)).

    Direct mapping to teacher:
      student inner step jj  ←→  teacher inner step (INNER_START_T + jj)
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
        """
        Single 60-outer × 5-inner forward pass.

        Collects F after normalize_power at each of the 5 inner steps
        within outer iterations [0, K_outer) — the student's steepest region.

        All 60 outer iterations run for task loss computation.
        Inner F collection only happens in [0, K_outer).

        Returns
        -------
        rates, taus  : convergence curves
        F, W         : final beamformers
        F_inner_S    : list of K_outer * N_INNER_S student F-matrices
                       in-graph — gradients flow to step_size
        """
        rate_init, tau_init, F, W = initialize(H, R, Pt, initial_normalization)
        rate_over_iters = torch.zeros(n_iter_outer, H.shape[1], device=H.device)
        tau_over_iters  = torch.zeros(n_iter_outer, H.shape[1], device=H.device)
        F_inner_S       = []   # inner trajectory — in computation graph

        for ii in range(n_iter_outer):
            collect_this_outer = (ii < K_outer)

            for jj in range(n_iter_inner):
                grad_F_com  = get_grad_F_com(H, F, W)
                grad_F_rad  = get_grad_F_rad(F, W, R)
                delta_F_com = self.step_size[jj][ii][0] * grad_F_com
                delta_F_rad = self.step_size[jj][ii][0] * grad_F_rad
                F = F + delta_F_com * WEIGHT_F_COM - delta_F_rad * WEIGHT_F_RAD
                F = normalize_power(F, W, H, Pt)

                # Collect ALL 5 student inner steps within K_outer outer iters
                if collect_this_outer:
                    F_inner_S.append(F.clone())   # in-graph ✓

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
# 6.  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

if run_RKD_Distillation == 1:

    print('=' * 70)
    print('CI-RKD [LAST-5 INIT + INNER RKD]  J5/I60  ←  J20/I120')
    print(f'Teacher : {I_T} outer × {N_INNER_T} inner = {I_T*N_INNER_T} steps/sample')
    print(f'Student :  {I_S} outer ×  {N_INNER_S} inner =  {I_S*N_INNER_S} steps/sample')
    print(f'Speedup : {(I_T*N_INNER_T)//(I_S*N_INNER_S)}×')
    print('=' * 70 + '\n')

    # ── Hyperparameters ───────────────────────────────────────────────────────
    # K_outer: how many outer iterations to collect inner trajectories from.
    # Total matched pairs per batch: K_outer * N_INNER_S = 10 * 5 = 50.
    # Larger K_outer → more RKD signal but more memory and compute.
    # Start with K_outer=10 (student outer iters [0,10),
    # teacher outer iters [20,30)).
    K_outer       = 10
    t_start_steep = 20      # teacher steepest window start

    lambda_dist   = 12.0    # halved from J=10 — J5 iterates have larger distances
    lambda_angle  = 25.0
    min_step_size = 1e-8
    max_step_size = 0.35    # slightly relaxed — init is conservative so
                            # step sizes need room to grow during training
    stabilised_lr = learning_rate / 2.0

    assert t_start_steep + K_outer <= I_T
    assert K_outer <= I_S

    print(f'Init           : last-5 teacher inner steps [{INNER_START_T}:{N_INNER_T}]')
    print(f'RKD mapping    : student inner j  ←→  teacher inner ({INNER_START_T}+j)')
    print(f'Outer window   : student [0, {K_outer})  ←  teacher'
          f' [{t_start_steep}, {t_start_steep+K_outer})')
    print(f'Pairs/batch    : {K_outer} outer × {N_INNER_S} inner = '
          f'{K_outer*N_INNER_S} matched F pairs')
    print(f'lambda_dist={lambda_dist}  |  lambda_angle={lambda_angle}')
    print(f'step_size ∈ [{min_step_size}, {max_step_size}]'
          f'  |  lr={stabilised_lr}  |  batch={batch_size}\n')

    # ── Load teacher ──────────────────────────────────────────────────────────
    model_teacher = PGA_Unfold_J20_Teacher(step_size_UPGA_J20).to(device)
    model_teacher.load_state_dict(
        torch.load(model_file_name_UPGA_J20, map_location=device))
    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad_(False)
    print(f'Teacher loaded : {model_file_name_UPGA_J20}')

    # ── Build student initialisation (last 5 teacher inner steps) ─────────────
    ss_init = build_last5_init(model_teacher, n_outer_S=I_S)

    # ── Instantiate student ───────────────────────────────────────────────────
    model_student = PGA_Unfold_J5_I60_CI(ss_init).to(device)
    optimizer     = torch.optim.Adam(model_student.parameters(),
                                     lr=stabilised_lr)
    print(f'Student step_size shape : {list(model_student.step_size.shape)}\n')

    # ── Training loop ─────────────────────────────────────────────────────────
    for i_epoch in range(n_epoch):
        start_time        = time.time()
        epoch_loss        = 0.0
        epoch_rkd_inner   = 0.0
        epoch_task_s      = 0.0
        epoch_task_t      = 0.0
        num_batches       = 0

        # ── Parameter health check ────────────────────────────────────────────
        if torch.isnan(model_student.step_size.data).any():
            print(f"  [WARNING] NaN in step_size at epoch {i_epoch}. "
                  f"Resetting parameters and Adam state.")
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

            # ── Teacher: collect inner F at last 5 inner steps ────────────────
            # within steepest outer window [t_start_steep, t_start_steep+K_outer)
            with torch.no_grad():
                F_t, W_t, F_inner_T = \
                    model_teacher.execute_PGA_inner_windows(
                        H, R, snr_train, I_T, N_INNER_T,
                        t_start_steep=t_start_steep,
                        K_outer=K_outer)
                teacher_task_loss = get_sum_loss(F_t, W_t, H, R, snr_train, B)

            # ── Student: collect inner F at all 5 inner steps ─────────────────
            # within steepest student window [0, K_outer)
            rate, _, F_s, W_s, F_inner_S = \
                model_student.execute_PGA_inner_windows(
                    H, R, snr_train, I_S, N_INNER_S, K_outer)

            # Sanity check — both lists must have same length
            assert len(F_inner_T) == len(F_inner_S) == K_outer * N_INNER_S, (
                f"Inner trajectory length mismatch: "
                f"teacher={len(F_inner_T)}, student={len(F_inner_S)}, "
                f"expected={K_outer*N_INNER_S}")

            # ── Compute losses ────────────────────────────────────────────────
            loss_task       = get_sum_loss(F_s, W_s, H, R, snr_train, B)
            loss_rkd_inner  = inner_rkd_loss(
                F_inner_T, F_inner_S,
                n_inner=N_INNER_S, B=B,
                lambda_dist=lambda_dist, lambda_angle=lambda_angle)

            total_loss = loss_task + loss_rkd_inner

            # ── NaN/Inf check BEFORE backward ─────────────────────────────────
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"  [WARNING] NaN/Inf at epoch {i_epoch} "
                      f"batch {i_batch} — skipping.")
                optimizer.zero_grad()
                continue

            # ── Backward + update ─────────────────────────────────────────────
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model_student.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                model_student.step_size.data.clamp_(
                    min=min_step_size, max=max_step_size)

            epoch_loss       += total_loss.item()
            epoch_rkd_inner  += loss_rkd_inner.item()
            epoch_task_s     += loss_task.item()
            epoch_task_t     += teacher_task_loss.item()
            num_batches      += 1

        nb = max(num_batches, 1)

        print(f"Epoch {i_epoch:4d} [J5/I60 | last5+inner_rkd] | "
              f"Time: {time.time()-start_time:.1f}s | "
              f"Loss: {epoch_loss/nb:.4f} | "
              f"Student: {epoch_task_s/nb:.4f} | "
              f"Teacher: {epoch_task_t/nb:.4f} | "
              f"RKD_inner: {epoch_rkd_inner/nb:.4f}")

        with torch.no_grad():
            ss = model_student.step_size.data
            ss_at_ceil = (ss >= max_step_size - 1e-6).sum().item()
            ss_at_floor= (ss <= min_step_size + 1e-9).sum().item()
            print(f"             step_size : "
                  f"min={ss.min():.4e}  max={ss.max():.4e}  "
                  f"mean={ss.mean():.4e}  std={ss.std():.4e}  "
                  f"at_ceil={ss_at_ceil}  at_floor={ss_at_floor}")
            # Expected behaviour over training:
            # - mean should INCREASE from init (~0.010-0.015) as RKD
            #   pushes step sizes to be more aggressive
            # - at_ceil should stay near 0 — if it grows, reduce lr
            # - at_floor counts step sizes pinned at min — some is normal
            # - RKD_inner loss should decrease steadily over epochs

    save_path = (model_file_name_UPGA_J10 +
                 f'_J3_I60_LAST5_INNER_RKD'
                 f'_Ko{K_outer}_tsteep{t_start_steep}.pth')
    torch.save(model_student.state_dict(), save_path)
    print(f'\nStudent saved → {save_path}')

