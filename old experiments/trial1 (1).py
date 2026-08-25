"""
CI-RKD Distillation — J10-inner / 60-outer Student
====================================================
Teacher : PGA_Unfold_J20        [20 inner × 120 outer]  pretrained, frozen
Student : PGA_Unfold_J10_I60_CI [10 inner ×  60 outer]  trained with CI-RKD

Inference cost
--------------
Teacher : 120 × 20 = 2400 steps/sample
Student :  60 × 10 =  600 steps/sample  →  4× cheaper

Inner-only initialisation strategies (all four available)
----------------------------------------------------------
All strategies compress the teacher's 20 inner steps down to 10, then
average across the 120 outer iterations to obtain the 'inner fingerprint'
[10, 1, K+1], and broadcast it uniformly to [10, 60, K+1].  The outer
dimension is left free for the student to learn under RKD supervision.

  'first'      take teacher steps [0:10]  — coarse/aggressive corrections
  'last'       take teacher steps [10:20] — fine/polishing corrections
  'subsample'  take steps [0,2,4,...,18]  — evenly spaced across full range
  'avg_pairs'  average pairs mean([0,1]), mean([2,3]), ... — most info-preserving

Training scheme: SYMMETRIC CI-RKD (single pass each)
------------------------------------------------------
Early RKD : teacher [t_start_early, t_start_early+K_layers) → student [0, K_layers)
Late  RKD : teacher [I_T-K_layers,  I_T)                    → student [I_S-K_layers, I_S)

Both teacher windows collected in ONE teacher pass.
Both student windows collected in ONE student pass.
Single backward() handles all three loss terms simultaneously.

Fixes vs earlier versions
--------------------------
1. NaN check moved BEFORE backward() to prevent Adam moment corruption.
2. step_size clamped with both min AND max to prevent divergence.
3. torch.cuda.empty_cache() removed from the batch loop (was forcing
   a full CUDA sync every batch — the main cause of slowness).
4. RKD distance loss vectorised over K_layers via batched torch.cdist.
5. import time moved to top of file.
6. Parameter health check at epoch start to catch silent corruption.
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
I_T       = n_iter_outer        # teacher: 120 outer iterations
I_S       = n_iter_outer // 2   # student:  60 outer iterations
N_INNER_T = n_iter_inner_J20    # teacher:  20 inner iterations
N_INNER_S = n_iter_inner_J10    # student:  10 inner iterations


# ══════════════════════════════════════════════════════════════════════════════
# 1.  INNER-ONLY STEP-SIZE INITIALISATION — ALL FOUR STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

def build_inner_only_step_size_J10(teacher_model, n_outer_S=60,
                                    inner_strategy='avg_pairs'):
    """
    Derive student step_size [10, 60, K+1] from teacher [20, 120, K+1].

    The pipeline has three stages regardless of strategy:

      Step 1 — compress inner axis 20 → 10 (strategy-dependent).
      Step 2 — average across the 120 outer iterations → [10, 1, K+1].
               This removes the teacher's outer schedule, which was
               tuned for I=120 and would mislead a student ending at I=60.
      Step 3 — broadcast uniformly to [10, 60, K+1].
               Every outer slot starts with the same inner profile;
               the student learns to differentiate them during training.
    """
    with torch.no_grad():
        ss_T      = teacher_model.step_size.data    # [20, 120, K+1]
        n_inner_t = ss_T.shape[0]                   # 20

        # ── Step 1: compress inner 20 → 10 ───────────────────────────────────

        if inner_strategy == 'first':
            # The first 10 teacher inner steps are the large, coarse-correction
            # steps.  A student with only 10 inner steps never has the budget to
            # polish, so these are the most directly relevant.
            ss_compressed = ss_T[:N_INNER_S].clone()              # [10, 120, K+1]
            desc = f"teacher inner [0:{N_INNER_S}] (coarse/aggressive)"

        elif inner_strategy == 'last':
            # The final 10 teacher inner steps are the fine, polishing steps
            # designed for iterates already near a local optimum.  Potentially
            # too conservative — run as a diagnostic lower-bound baseline.
            ss_compressed = ss_T[-N_INNER_S:].clone()             # [10, 120, K+1]
            desc = (f"teacher inner [{n_inner_t-N_INNER_S}:{n_inner_t}]"
                    f" (fine/polishing)")

        elif inner_strategy == 'subsample':
            # Evenly spaced indices spanning the full inner range.
            # For 10 from 20: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18].
            # Covers both coarse and fine scales without blending.
            indices = torch.linspace(0, n_inner_t - 1, N_INNER_S).long()
            ss_compressed = ss_T[indices].clone()                  # [10, 120, K+1]
            desc = f"teacher inner {indices.tolist()} (evenly spaced)"

        elif inner_strategy == 'avg_pairs':
            # Average each consecutive pair: mean([0,1]), mean([2,3]), ...
            # Requires n_inner_T == 2 * N_INNER_S, which is exactly satisfied
            # here (20 == 2*10).  Most information-preserving: every teacher
            # inner step contributes, nothing is discarded.
            assert n_inner_t == 2 * N_INNER_S, (
                f"avg_pairs requires n_inner_T == 2*N_INNER_S "
                f"(got {n_inner_t} vs {N_INNER_S})")
            ss_paired     = ss_T.view(N_INNER_S, 2, I_T, K + 1)   # [10, 2, 120, K+1]
            ss_compressed = ss_paired.mean(dim=1)                   # [10, 120, K+1]
            desc = "avg pairs mean([0,1]), mean([2,3]), ..., mean([18,19])"

        else:
            raise ValueError(
                f"Unknown inner_strategy {inner_strategy!r}. "
                f"Choose from: 'first', 'last', 'subsample', 'avg_pairs'.")

        # ── Step 2: collapse outer schedule → inner fingerprint ───────────────
        # keepdim=True so expand() in Step 3 works correctly.
        fingerprint = ss_compressed.mean(dim=1, keepdim=True)         # [10,  1, K+1]

        # ── Step 3: broadcast uniformly across student's 60 outer slots ──────
        ss_init = fingerprint.expand(-1, n_outer_S, -1).clone()       # [10, 60, K+1]

    print(f"\n[J10 inner-only init | strategy='{inner_strategy}']")
    print(f"  Inner mapping  : {desc}")
    print(f"  teacher {list(ss_T.shape)}"
          f" → compressed {list(ss_compressed.shape)}"
          f" → fingerprint {list(fingerprint.shape)}"
          f" → student init {list(ss_init.shape)}")
    print(f"  Step-size range: [{ss_init.min().item():.4e}, "
          f"{ss_init.max().item():.4e}]\n")
    return ss_init


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def flatten_F(F_mat, B):
    """[K_u, B, Nt, Nrf] complex  →  [B, K_u*Nt*Nrf] complex."""
    return F_mat.permute(1, 0, 2, 3).reshape(B, -1)


def _to_real(x):
    """Convert a possibly-complex tensor to real by interleaving Re/Im.
    Returns unchanged if already real."""
    if torch.is_complex(x):
        return torch.view_as_real(x).flatten(-2)   # (..., 2*D)
    return x


# ══════════════════════════════════════════════════════════════════════════════
# 3.  VECTORISED RKD LOSSES
#
#     The distance loss is fully vectorised over K_layers: all K feature
#     tensors are stacked into [K, B, D_real] and torch.cdist is called
#     ONCE in batched mode, producing [K, B, B].  This replaces K separate
#     cdist calls, reducing CUDA kernel launches by K_layers-fold.
#
#     The angle loss creates a [B, B, D] intermediate per layer.  Batching
#     over K would produce [K, B, B, D] which can be many gigabytes when D
#     is large (K_u × Nt × Nrf × 2 for complex).  So the angle loss keeps
#     its loop, but features are pre-converted to real OUTSIDE the loop to
#     avoid repeated is_complex overhead.
# ══════════════════════════════════════════════════════════════════════════════

def ci_rkd_loss(F_T_window, F_S_window, K_layers, B,
                lambda_dist=25.0, lambda_angle=50.0):
    """
    Reversed-weight CI-RKD: w_l = (l+1) / sum(1..K_layers).
    Higher l = more-converged teacher iterate = stronger supervision weight.

    Teacher features are detached (set during the no_grad teacher pass).
    Student features remain in-graph so gradients flow to step_size.
    """
    weight_sum = K_layers * (K_layers + 1) / 2

    # Pre-process all K_layers feature tensors outside any loop.
    # Stack into [K, B, D_real] for teacher (detached) and student (in-graph).
    T_feats = torch.stack(
        [_to_real(flatten_F(F_T_window[l], B)).detach() for l in range(K_layers)],
        dim=0)   # [K, B, D_real]

    S_feats = torch.stack(
        [_to_real(flatten_F(F_S_window[l], B)) for l in range(K_layers)],
        dim=0)   # [K, B, D_real] — in-graph

    # ── Vectorised distance loss ──────────────────────────────────────────────
    # One batched cdist call instead of K_layers separate calls.
    with torch.no_grad():
        T_dist = torch.cdist(T_feats, T_feats, p=2)             # [K, B, B]
        # Normalise each layer's distance matrix by its own mean so that
        # layers at different convergence stages are on a comparable scale.
        T_dist = T_dist / T_dist.mean(
            dim=(-2, -1), keepdim=True).clamp(min=1e-12)

    S_dist = torch.cdist(S_feats, S_feats, p=2)                 # [K, B, B]
    S_dist = S_dist / S_dist.mean(
        dim=(-2, -1), keepdim=True).clamp(min=1e-12)

    # Per-layer weights as a [K, 1, 1] tensor so they broadcast over [K, B, B].
    weights = (torch.arange(1, K_layers + 1,
                            dtype=S_dist.dtype, device=S_dist.device)
               / weight_sum).view(K_layers, 1, 1)

    # Multiply weights in before smooth_l1 so each layer contributes
    # proportionally to the final scalar.
    dist_loss = nn.functional.smooth_l1_loss(
        S_dist * weights, T_dist * weights, reduction='mean')

    # ── Angle loss — loop over K_layers ──────────────────────────────────────
    # T_feats[l] and S_feats[l] are already real [B, D_real] — no conversion
    # overhead inside the loop since _to_real was called above during stacking.
    angle_loss = 0.0
    for l in range(K_layers):
        w_l = (l + 1) / weight_sum

        t = T_feats[l]   # [B, D_real] — detached
        s = S_feats[l]   # [B, D_real] — in-graph

        with torch.no_grad():
            t_e   = t.unsqueeze(0) - t.unsqueeze(1)             # [B, B, D]
            t_e   = nn.functional.normalize(t_e, p=2, dim=-1)
            t_cos = torch.bmm(t_e, t_e.permute(0, 2, 1))        # [B, B, B]

        s_e   = s.unsqueeze(0) - s.unsqueeze(1)
        s_e   = nn.functional.normalize(s_e, p=2, dim=-1)
        s_cos = torch.bmm(s_e, s_e.permute(0, 2, 1))

        angle_loss += w_l * nn.functional.smooth_l1_loss(s_cos, t_cos)

    return lambda_dist * dist_loss + lambda_angle * angle_loss


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TEACHER MODEL
# ══════════════════════════════════════════════════════════════════════════════

class PGA_Unfold_J20_Teacher(PGA_Unfold_J20):
    """
    Teacher wrapper.  A single forward pass collects both the early and late
    trajectory windows simultaneously, halving teacher compute per batch.
    All stored tensors are detached — the teacher is a frozen reference.
    """

    def execute_PGA_two_windows(self, H, R, Pt, n_iter_outer, n_iter_inner,
                                 t_start_early, t_start_late, K_layers):
        """
        Single 120-outer pass collecting:
          Early window : [t_start_early, t_start_early + K_layers)
          Late  window : [t_start_late,  t_start_late  + K_layers)
        Both conditions are checked inside the same outer loop body so there
        is truly zero additional compute compared to collecting one window.
        """
        t_end_early = t_start_early + K_layers
        t_end_late  = t_start_late  + K_layers
        assert t_end_early <= n_iter_outer, "Early window exceeds teacher depth"
        assert t_end_late  <= n_iter_outer, "Late window exceeds teacher depth"

        _, _, F, W = initialize(H, R, Pt, initial_normalization)
        F_traj_early = []
        F_traj_late  = []

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
# 5.  STUDENT MODEL — J=10 inner, 60 outer
# ══════════════════════════════════════════════════════════════════════════════

class PGA_Unfold_J10_I60_CI(nn.Module):
    """
    Student — 60 outer × 10 inner iterations.
    step_size shape: [10, 60, K+1]

    Accepts either a pre-built [10, 60, K+1] tensor (from the init function,
    any strategy) or a scalar seed broadcast to that shape (baseline fallback).
    The model class itself is entirely strategy-agnostic.

    The single forward method execute_PGA_with_both_windows collects the early
    and late CI-RKD windows in one sweep, so both lists share the same
    computation graph and a single backward() handles all three losses.

    Note on normalisation: J10 calls normalize_power at every inner step,
    matching the original PGA_Unfold_J10 behaviour.  J20 uses an overflow
    threshold check instead.  This distinction is preserved exactly.
    """

    def __init__(self, step_size_init):
        super().__init__()
        if isinstance(step_size_init, torch.Tensor) and step_size_init.dim() == 3:
            self.step_size = nn.Parameter(step_size_init.float().clone())
        else:
            self.step_size = nn.Parameter(
                step_size_init * torch.ones(N_INNER_S, I_S, K + 1))

    def execute_PGA_with_both_windows(self, H, R, Pt,
                                       n_iter_outer, n_iter_inner, K_layers):
        """
        Single 60-outer × 10-inner forward pass.

        Collects F_traj_first (early: iters [0, K_layers)) and
        F_traj_last (late: iters [I_S-K_layers, I_S)) in the same sweep.

        With K_layers=15 and I_S=60 the windows are [0,15) and [45,60),
        with a 30-iteration gap between them where the student develops
        freely without any RKD supervision.
        """
        rate_init, tau_init, F, W = initialize(H, R, Pt, initial_normalization)
        rate_over_iters = torch.zeros(n_iter_outer, H.shape[1], device=H.device)
        tau_over_iters  = torch.zeros(n_iter_outer, H.shape[1], device=H.device)
        F_traj_first = []   # early window — in computation graph
        F_traj_last  = []   # late  window — in computation graph

        for ii in range(n_iter_outer):
            for jj in range(n_iter_inner):
                grad_F_com  = get_grad_F_com(H, F, W)
                grad_F_rad  = get_grad_F_rad(F, W, R)
                delta_F_com = self.step_size[jj][ii][0] * grad_F_com
                delta_F_rad = self.step_size[jj][ii][0] * grad_F_rad
                F = F + delta_F_com * WEIGHT_F_COM - delta_F_rad * WEIGHT_F_RAD
                # J10 normalises power at every inner step (original behaviour)
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
                F_traj_first.append(F.clone())              # early — in-graph ✓
            if ii >= n_iter_outer - K_layers:
                F_traj_last.append(F.clone())               # late  — in-graph ✓

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
    print('CI-RKD  J10/I60 student  ←  J20/I120 teacher  [INNER-ONLY INIT]')
    print(f'Teacher : {I_T} outer × {N_INNER_T} inner = {I_T * N_INNER_T} steps/sample')
    print(f'Student :  {I_S} outer × {N_INNER_S} inner =  {I_S * N_INNER_S} steps/sample')
    print(f'Speedup : {(I_T * N_INNER_T) // (I_S * N_INNER_S)}×')
    print('=' * 70 + '\n')

    # ── Strategy and hyperparameters ──────────────────────────────────────────
    # Change inner_strategy to switch between all four initialisations.
    # Everything else — model, losses, training loop — is identical.
    inner_strategy = 'avg_pairs'    # 'avg_pairs' | 'first' | 'last' | 'subsample'

    K_layers      = 15      # student windows: [0,15) and [45,60) — 30-step gap
    lambda_dist   = 25.0
    lambda_angle  = 50.0
    lambda_late   = 1.0
    t_start_early = 20                  # steepest teacher region: [20, 35)
    t_start_late  = I_T - K_layers      # teacher plateau: [105, 120)

    # step_size bounds.  The lower bound prevents negative values.  The upper
    # bound is the critical addition: without it, step sizes can grow without
    # bound across epochs, eventually causing the PGA inner loop to produce
    # values that overflow to Inf, which propagates to NaN in the loss.
    # Set max_step_size to a value well above your init range (here ~0.025)
    # but low enough to prevent divergence.  0.5 is a safe conservative choice.
    min_step_size = 1e-8
    max_step_size = 0.1

    assert t_start_early + K_layers <= I_T, "Early window exceeds teacher depth"
    assert K_layers <= I_S // 2, (
        f"Windows overlap: [0,{K_layers}) ∩ [{I_S-K_layers},{I_S}) "
        f"— reduce K_layers to ≤ {I_S // 2}")

    print(f'inner_strategy : {inner_strategy!r}')
    print(f'Early : teacher [{t_start_early}, {t_start_early + K_layers})'
          f'  →  student [0, {K_layers})')
    print(f'Late  : teacher [{t_start_late}, {I_T})'
          f'  →  student [{I_S - K_layers}, {I_S})')
    print(f'K_layers={K_layers}  |  lambda_late={lambda_late}'
          f'  |  step_size ∈ [{min_step_size}, {max_step_size}]'
          f'  |  batch={batch_size}\n')

    # ── Load teacher — must happen before the init call ───────────────────────
    model_teacher = PGA_Unfold_J20_Teacher(step_size_UPGA_J20).to(device)
    model_teacher.load_state_dict(
        torch.load(model_file_name_UPGA_J20, map_location=device))
    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad_(False)
    print(f'Teacher loaded : {model_file_name_UPGA_J20}')

    # ── Build student initialisation ──────────────────────────────────────────
    ss_init = build_inner_only_step_size_J10(
        model_teacher, n_outer_S=I_S, inner_strategy=inner_strategy)

    # ── Instantiate student ───────────────────────────────────────────────────
    model_student = PGA_Unfold_J10_I60_CI(ss_init).to(device)
    optimizer     = torch.optim.Adam(model_student.parameters(), lr=learning_rate)
    print(f'Student step_size shape : {list(model_student.step_size.shape)}\n')

    # ── Training loop ─────────────────────────────────────────────────────────
    for i_epoch in range(n_epoch):
        start_time      = time.time()
        epoch_loss      = 0.0
        epoch_rkd_early = 0.0
        epoch_rkd_late  = 0.0
        epoch_task_s    = 0.0
        epoch_task_t    = 0.0
        num_batches     = 0

        # ── Parameter health check ────────────────────────────────────────────
        # If step_size contains NaN at the start of an epoch it means a
        # previous backward() wrote NaN into a gradient, which then corrupted
        # Adam's internal moment buffers (exp_avg, exp_avg_sq).  Simply calling
        # zero_grad() does NOT clear Adam's moments — only optimizer.state.clear()
        # does.  We detect this early and reset rather than silently training on
        # a broken model for a full epoch.
        if torch.isnan(model_student.step_size.data).any():
            print(f"  [WARNING] NaN in step_size at epoch {i_epoch} start. "
                  f"Resetting parameters and Adam state.")
            with torch.no_grad():
                model_student.step_size.data.copy_(ss_init.to(device))
            # Clearing Adam's moment buffers is essential — NaN moments would
            # corrupt every subsequent update even with clean gradients.
            optimizer.state.clear()

        H_shuffled = torch.transpose(H_train, 0, 1)[
            np.random.permutation(len(H_train[0]))]

        for i_batch in range(0, len(H_train[0]), batch_size):
            H = torch.transpose(
                H_shuffled[i_batch:i_batch + batch_size], 0, 1).to(device)
            B = H.shape[1]
            if B < 2:
                continue   # pairwise RKD losses require at least 2 samples

            snr_dB_train = np.random.choice(snr_dB_list)
            snr_train    = 10 ** (snr_dB_train / 10)
            R = get_R(snr_dB_train, B)

            # ── Teacher: ONE pass, BOTH windows ──────────────────────────────
            with torch.no_grad():
                F_t, W_t, F_T_early, F_T_late = \
                    model_teacher.execute_PGA_two_windows(
                        H, R, snr_train, I_T, N_INNER_T,
                        t_start_early=t_start_early,
                        t_start_late=t_start_late,
                        K_layers=K_layers)
                teacher_task_loss = get_sum_loss(F_t, W_t, H, R, snr_train, B)

            # ── Student: ONE pass, BOTH windows + task output ─────────────────
            rate, _, F_s, W_s, F_S_first, F_S_last = \
                model_student.execute_PGA_with_both_windows(
                    H, R, snr_train, I_S, N_INNER_S, K_layers)

            # ── Compute all three losses ──────────────────────────────────────
            loss_task      = get_sum_loss(F_s, W_s, H, R, snr_train, B)
            loss_rkd_early = ci_rkd_loss(F_T_early, F_S_first,
                                          K_layers, B, lambda_dist, lambda_angle)
            loss_rkd_late  = ci_rkd_loss(F_T_late,  F_S_last,
                                          K_layers, B, lambda_dist, lambda_angle)
            total_loss = loss_task + loss_rkd_early + lambda_late * loss_rkd_late

            # ── NaN/Inf check BEFORE backward ─────────────────────────────────
            # This is the critical ordering fix.  In the old code the check came
            # AFTER backward(), which meant NaN gradients had already been written
            # into Adam's moment buffers before the batch was skipped.  Adam's
            # moments are not cleared by zero_grad() — they persist and corrupt
            # every subsequent update.  By checking first we prevent that entirely.
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"  [WARNING] NaN/Inf loss at epoch {i_epoch} "
                      f"batch {i_batch} — skipping batch.")
                optimizer.zero_grad()
                continue

            # ── Backward + update ─────────────────────────────────────────────
            optimizer.zero_grad()
            total_loss.backward()   # single call handles all three loss terms

            torch.nn.utils.clip_grad_norm_(model_student.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                # Clamp with BOTH min and max.  The min prevents negative step
                # sizes; the max prevents unbounded growth that would eventually
                # cause the PGA inner loop to overflow → Inf → NaN loss.
                model_student.step_size.data.clamp_(
                    min=min_step_size, max=max_step_size)

            epoch_loss      += total_loss.item()
            epoch_rkd_early += loss_rkd_early.item()
            epoch_rkd_late  += loss_rkd_late.item()
            epoch_task_s    += loss_task.item()
            epoch_task_t    += teacher_task_loss.item()
            num_batches     += 1

            # ── NO torch.cuda.empty_cache() here ─────────────────────────────
            # Removed entirely from the batch loop.  It forced a full CUDA
            # context synchronisation every batch, which was the primary source
            # of slowness.  PyTorch's caching allocator handles memory reuse
            # automatically and efficiently without any manual intervention.
            # Only add it back (once per epoch, outside the batch loop) if you
            # genuinely encounter OOM errors.

        nb = max(num_batches, 1)
        print(f"Epoch {i_epoch:4d} [J10/I60 | {inner_strategy}] | "
              f"Time: {time.time() - start_time:.1f}s | "
              f"Loss: {epoch_loss / nb:.4f} | "
              f"Student: {epoch_task_s / nb:.4f} | "
              f"Teacher: {epoch_task_t / nb:.4f} | "
              f"RKD_early: {epoch_rkd_early / nb:.4f} | "
              f"RKD_late: {epoch_rkd_late / nb:.4f}")

        # Step-size statistics once per epoch so you can monitor for divergence.
        # If max_val starts climbing rapidly toward max_step_size, it is a sign
        # that the clamp is actively preventing divergence and you may want to
        # reduce the learning rate or tighten max_step_size.
        with torch.no_grad():
            ss = model_student.step_size.data
            print(f"             step_size : "
                  f"min={ss.min():.4e}  max={ss.max():.4e}  "
                  f"mean={ss.mean():.4e}")

    # Strategy embedded in save path so four runs never overwrite each other
    save_path = (model_file_name_UPGA_J10 +
                 f'_I60_CI_RKD_sym_inner_{inner_strategy}'
                 f'_Kl{K_layers}_win{t_start_early}_today_both.pth')
    torch.save(model_student.state_dict(), save_path)
    print(f'\nStudent saved → {save_path}')


