"""
CI-RKD Distillation — J10-inner / 60-outer Student
====================================================
Teacher : PGA_Unfold_J20        [20 inner × 120 outer]  pretrained, frozen
Student : PGA_Unfold_J10_I60_CI [10 inner ×  60 outer]  trained with CI-RKD

Inference cost
--------------
Teacher : 120 × 20 = 2400 steps/sample
Student :  60 × 10 =  600 steps/sample  →  4× cheaper

═══════════════════════════════════════════════════════════════
CONFIGURABLE STRATEGIES — change these two variables to run
different experiments and compare results:

  inner_strategy  (controls step_size INITIALISATION, inner axis)
  ─────────────────────────────────────────────────────────────
  'avg_pairs'  → average consecutive pairs mean([0,1]),mean([2,3]),...
                 Most information-preserving.  RECOMMENDED DEFAULT.
  'first'      → take teacher inner steps [0:10] (coarse/aggressive)
  'last'       → take teacher inner steps [10:20] (fine/polishing)
  'subsample'  → take evenly spaced [0,2,4,...,18]

  outer_init_strategy  (controls step_size INITIALISATION, outer axis)
  ─────────────────────────────────────────────────────────────────────
  'fingerprint' → average teacher outer axis → broadcast uniformly
                  Removes teacher outer schedule entirely. RECOMMENDED.
                  Student outer schedule is completely free to learn.
  'first'       → copy teacher outer iters [0:60] directly
                  Imports teacher's early-phase outer schedule.
  'last'        → copy teacher outer iters [60:120] directly
                  Imports teacher's plateau-phase outer schedule.
  'steepest'    → copy teacher outer iters [t_start:t_start+60]
                  Imports teacher's steepest-descent outer schedule.

  rkd_window_strategy  (controls which teacher iters supervise training)
  ──────────────────────────────────────────────────────────────────────
  'symmetric'   → early window + late window (RECOMMENDED)
                  Anchors both start and end of student trajectory.
  'early_only'  → teacher steepest [t_start_early, t_start_early+K_layers)
                  → student [0, K_layers). Good start, free end.
  'late_only'   → teacher plateau [I_T-K_layers, I_T)
                  → student [I_S-K_layers, I_S). Free start, good end.
  'first_only'  → teacher [0, K_layers) → student [0, K_layers)
                  Teacher's raw early iters, not steepest region.

═══════════════════════════════════════════════════════════════

Research framing: Algorithm Geometry Transfer (AGT)
----------------------------------------------------
Standard transfer learning transfers learned feature representations.
This code transfers learned *algorithm geometry* — the step sizes that
encode how PGA should descend on the JCAS optimization landscape.

The inner-only initialisation ('fingerprint' outer_init_strategy) makes
a principled decomposition:
  inner axis → local per-step refinement geometry (transferable)
               encodes curvature of objective within one outer iter
  outer axis → global convergence schedule (architecture-specific)
               encodes how fast algorithm converges over 120 steps
               SHOULD NOT transfer to a 60-step student

This mirrors the specificity hierarchy in neural network transfer:
lower layers (general) transfer better than higher layers (specific).

Stability fixes vs earlier versions
------------------------------------
1. NaN check BEFORE backward() to prevent Adam moment corruption.
2. step_size clamped with BOTH min AND max (critical — prevents divergence
   caused by the late RKD loss pushing step sizes without a ceiling).
3. torch.cuda.empty_cache() removed from batch loop (main slowness cause).
4. RKD distance loss vectorised over K_layers via batched torch.cdist.
5. Parameter health check at epoch start.
6. Per-epoch step_size statistics to monitor for instability early.
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
# 1.  STEP-SIZE INITIALISATION
#     Two axes, each independently configurable.
#     Inner axis: how to compress teacher's 20 inner steps → 10.
#     Outer axis: how to handle teacher's 120 outer slots → 60.
# ══════════════════════════════════════════════════════════════════════════════

def build_step_size_init(teacher_model,
                          n_outer_S          = 60,
                          inner_strategy     = 'avg_pairs',
                          outer_init_strategy = 'fingerprint',
                          t_start_steepest   = 20):
    """
    Build student step_size [10, 60, K+1] from teacher [20, 120, K+1].

    The function is divided into two clearly separated stages:
      Stage A — inner axis transformation (20 → 10)
      Stage B — outer axis transformation (120 → 60)

    Both stages are independently configurable, giving you a 4×4 grid
    of initialisation combinations to compare experimentally.

    Parameters
    ----------
    teacher_model       : loaded, frozen teacher already on device
    n_outer_S           : student outer iterations (default 60)
    inner_strategy      : 'avg_pairs' | 'first' | 'last' | 'subsample'
    outer_init_strategy : 'fingerprint' | 'first' | 'last' | 'steepest'
    t_start_steepest    : first outer iter of steepest window (default 20)
    """
    with torch.no_grad():
        ss_T      = teacher_model.step_size.data    # [20, 120, K+1]
        n_inner_t = ss_T.shape[0]                   # 20
        n_outer_t = ss_T.shape[1]                   # 120

        # ── Stage A: compress inner axis 20 → 10 ─────────────────────────────

        if inner_strategy == 'avg_pairs':
            # Average consecutive pairs: mean([0,1]), mean([2,3]), ..., mean([18,19]).
            # Every teacher inner step contributes — no information discarded.
            # The resulting profile is a smoothed version of the full inner schedule.
            # Requires n_inner_T == 2 * N_INNER_S, exactly satisfied here.
            assert n_inner_t == 2 * N_INNER_S, (
                f"avg_pairs requires n_inner_T == 2*N_INNER_S "
                f"(got {n_inner_t} vs {N_INNER_S})")
            ss_inner = ss_T.view(N_INNER_S, 2, n_outer_t, K + 1).mean(dim=1)
            inner_desc = "avg_pairs: mean([0,1]), mean([2,3]), ..., mean([18,19])"

        elif inner_strategy == 'first':
            # Take the first 10 teacher inner steps (coarse, aggressive).
            # Rationale: a 10-inner student never has budget to polish,
            # so it operates analogously to the teacher's early inner phase.
            ss_inner = ss_T[:N_INNER_S].clone()
            inner_desc = f"first: teacher inner [0:{N_INNER_S}]"

        elif inner_strategy == 'last':
            # Take the last 10 teacher inner steps (fine, polishing).
            # These are designed for iterates near a local optimum —
            # potentially too conservative for a fresh-start student.
            # Useful as a diagnostic lower-bound baseline.
            ss_inner = ss_T[-N_INNER_S:].clone()
            inner_desc = (f"last: teacher inner "
                          f"[{n_inner_t-N_INNER_S}:{n_inner_t}]")

        elif inner_strategy == 'subsample':
            # Evenly spaced indices across the full inner range.
            # For 10 from 20: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18].
            # Samples both coarse and fine scales without blending.
            indices = torch.linspace(0, n_inner_t - 1, N_INNER_S).long()
            ss_inner = ss_T[indices].clone()
            inner_desc = f"subsample: indices {indices.tolist()}"

        else:
            raise ValueError(
                f"Unknown inner_strategy {inner_strategy!r}. "
                f"Choose: 'avg_pairs', 'first', 'last', 'subsample'.")

        # ss_inner is now [N_INNER_S, 120, K+1] = [10, 120, K+1]

        # ── Stage B: transform outer axis 120 → 60 ───────────────────────────

        if outer_init_strategy == 'fingerprint':
            # Average across ALL 120 outer iterations to get a [10, 1, K+1]
            # inner fingerprint, then broadcast uniformly to [10, 60, K+1].
            #
            # This is the principled 'inner-only' approach:
            # - Removes the teacher's outer schedule entirely (it was tuned
            #   for I=120 and would mislead a student terminating at I=60)
            # - Preserves only the RELATIVE SHAPE of the inner step profile
            # - Student outer schedule is completely free to be learned
            #
            # Research framing: we transfer algorithm geometry (inner step
            # structure) while letting the student discover its own convergence
            # trajectory — analogous to transferring feature representations
            # while re-learning task-specific heads in neural net transfer learning.
            fingerprint = ss_inner.mean(dim=1, keepdim=True)         # [10, 1, K+1]
            ss_init     = fingerprint.expand(-1, n_outer_S, -1).clone()
            outer_desc  = "fingerprint: avg all outer → broadcast uniformly"

        elif outer_init_strategy == 'first':
            # Copy teacher's first n_outer_S outer iterations directly.
            # These correspond to the teacher's active descent phase (iters 0–59).
            # Risk: the teacher's outer schedule at iter 30 was designed
            # assuming 90 more outer steps remain — student has no such buffer.
            assert n_outer_S <= n_outer_t
            ss_init    = ss_inner[:, :n_outer_S, :].clone()
            outer_desc = f"first: teacher outer [0:{n_outer_S}]"

        elif outer_init_strategy == 'last':
            # Copy teacher's last n_outer_S outer iterations.
            # These are the plateau/fine-tuning steps.
            # Risk: the student starts at iter 0 but inherits step sizes
            # designed for a nearly-converged iterate at iter 90+.
            assert n_outer_S <= n_outer_t
            ss_init    = ss_inner[:, -n_outer_S:, :].clone()
            outer_desc = f"last: teacher outer [{n_outer_t-n_outer_S}:{n_outer_t}]"

        elif outer_init_strategy == 'steepest':
            # Copy teacher's steepest convergence window [t_start:t_start+n_outer_S].
            # This gives the student the step sizes from the teacher's most
            # informative descent region, where gradients are largest.
            t_end_steep = t_start_steepest + n_outer_S
            assert t_end_steep <= n_outer_t, (
                f"Steepest window [{t_start_steepest},{t_end_steep}) "
                f"exceeds teacher depth {n_outer_t}")
            ss_init    = ss_inner[:, t_start_steepest:t_end_steep, :].clone()
            outer_desc = (f"steepest: teacher outer "
                          f"[{t_start_steepest}:{t_end_steep}]")

        else:
            raise ValueError(
                f"Unknown outer_init_strategy {outer_init_strategy!r}. "
                f"Choose: 'fingerprint', 'first', 'last', 'steepest'.")

        # ss_init is now [N_INNER_S, n_outer_S, K+1] = [10, 60, K+1]

    print(f"\n[Step-size init]")
    print(f"  inner : {inner_desc}")
    print(f"  outer : {outer_desc}")
    print(f"  teacher {list(ss_T.shape)}"
          f" → inner-compressed {list(ss_inner.shape)}"
          f" → student init {list(ss_init.shape)}")
    print(f"  Range : [{ss_init.min().item():.4e}, "
          f"{ss_init.max().item():.4e}]\n")
    return ss_init


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def flatten_F(F_mat, B):
    """[K_u, B, Nt, Nrf] complex  →  [B, K_u*Nt*Nrf] complex."""
    return F_mat.permute(1, 0, 2, 3).reshape(B, -1)


def _to_real(x):
    """Convert complex tensor to real by stacking Re/Im.  No-op if real."""
    if torch.is_complex(x):
        return torch.view_as_real(x).flatten(-2)
    return x


# ══════════════════════════════════════════════════════════════════════════════
# 3.  VECTORISED RKD LOSSES
# ══════════════════════════════════════════════════════════════════════════════

def ci_rkd_loss(F_T_window, F_S_window, K_layers, B,
                lambda_dist=25.0, lambda_angle=50.0):
    """
    Reversed-weight CI-RKD: w_l = (l+1) / sum(1..K_layers).
    Higher l = more converged teacher iterate = stronger supervision.

    Distance loss is fully vectorised over K_layers via batched cdist,
    replacing K separate kernel launches with one.  The angle loss stays
    as a loop because its [B, B, D] intermediate would become [K, B, B, D]
    if batched, which is prohibitively large for typical D values.
    """
    weight_sum = K_layers * (K_layers + 1) / 2

    # Stack all K_layers features into [K, B, D_real] in one pass.
    # Pre-converting to real outside any loop avoids repeated is_complex checks.
    T_feats = torch.stack(
        [_to_real(flatten_F(F_T_window[l], B)).detach() for l in range(K_layers)],
        dim=0)   # [K, B, D_real] — detached

    S_feats = torch.stack(
        [_to_real(flatten_F(F_S_window[l], B)) for l in range(K_layers)],
        dim=0)   # [K, B, D_real] — in-graph

    # ── Vectorised distance loss ──────────────────────────────────────────────
    with torch.no_grad():
        T_dist = torch.cdist(T_feats, T_feats, p=2)             # [K, B, B]
        T_dist = T_dist / T_dist.mean(
            dim=(-2, -1), keepdim=True).clamp(min=1e-12)

    S_dist = torch.cdist(S_feats, S_feats, p=2)                 # [K, B, B]
    S_dist = S_dist / S_dist.mean(
        dim=(-2, -1), keepdim=True).clamp(min=1e-12)

    weights = (torch.arange(1, K_layers + 1,
                            dtype=S_dist.dtype, device=S_dist.device)
               / weight_sum).view(K_layers, 1, 1)                # [K, 1, 1]

    dist_loss = nn.functional.smooth_l1_loss(
        S_dist * weights, T_dist * weights, reduction='mean')

    # ── Angle loss — loop but features already real ───────────────────────────
    angle_loss = 0.0
    for l in range(K_layers):
        w_l = (l + 1) / weight_sum
        t   = T_feats[l]   # [B, D_real] detached
        s   = S_feats[l]   # [B, D_real] in-graph
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
# 4.  TEACHER MODEL
#     A single forward pass collects both trajectory windows simultaneously.
#     Under torch.no_grad() there is no graph cost to storing extra tensors,
#     so collecting two windows costs exactly the same as collecting one.
# ══════════════════════════════════════════════════════════════════════════════

class PGA_Unfold_J20_Teacher(PGA_Unfold_J20):

    def execute_PGA_two_windows(self, H, R, Pt, n_iter_outer, n_iter_inner,
                                 t_start_early, t_start_late, K_layers,
                                 collect_early=True, collect_late=True):
        """
        Single 120-outer pass.  Collects up to two trajectory windows.

        collect_early / collect_late flags allow single-window modes
        (for 'early_only', 'late_only', 'first_only' rkd_window_strategy)
        without duplicating the forward loop code.

        Both windows are always detached — teacher is a frozen reference.
        """
        t_end_early = t_start_early + K_layers
        t_end_late  = t_start_late  + K_layers
        if collect_early:
            assert t_end_early <= n_iter_outer, "Early window exceeds teacher depth"
        if collect_late:
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

            if collect_early and (t_start_early <= ii < t_end_early):
                F_traj_early.append(F.detach().clone())
            if collect_late and (t_start_late <= ii < t_end_late):
                F_traj_late.append(F.detach().clone())

        return F, W, F_traj_early, F_traj_late


# ══════════════════════════════════════════════════════════════════════════════
# 5.  STUDENT MODEL — J=10 inner, 60 outer
# ══════════════════════════════════════════════════════════════════════════════

class PGA_Unfold_J10_I60_CI(nn.Module):
    """
    Student — 60 outer × 10 inner iterations.
    step_size shape: [10, 60, K+1]

    Accepts either a pre-built [10, 60, K+1] tensor from build_step_size_init()
    (any combination of inner/outer strategies) or a scalar seed broadcast to
    that shape (random-init baseline).

    The forward method execute_PGA_with_both_windows collects both the early
    and late trajectory windows in a single 60-outer sweep.  Both lists share
    the same computation graph, so a single backward() on total_loss handles
    all gradient flows simultaneously — no doubled compute.

    J10 normalises power at every inner step (original PGA_Unfold_J10
    behaviour), unlike J20 which uses an overflow-threshold check.
    """

    def __init__(self, step_size_init):
        super().__init__()
        if isinstance(step_size_init, torch.Tensor) and step_size_init.dim() == 3:
            # Pre-built tensor: wrap directly without further broadcast
            self.step_size = nn.Parameter(step_size_init.float().clone())
        else:
            # Scalar fallback: broadcast to [10, 60, K+1]
            self.step_size = nn.Parameter(
                step_size_init * torch.ones(N_INNER_S, I_S, K + 1))

    def execute_PGA_with_both_windows(self, H, R, Pt,
                                       n_iter_outer, n_iter_inner, K_layers):
        """
        Single 60-outer × 10-inner pass collecting:
          F_traj_first : iters [0,            K_layers)   early window
          F_traj_last  : iters [I_S-K_layers, I_S)        late  window

        With K_layers=15, I_S=60: windows are [0,15) and [45,60).
        The 30-iteration gap between them is deliberate — the student
        develops the middle of its trajectory freely, without supervision.

        When rkd_window_strategy is not 'symmetric', one list will simply
        remain empty (K_layers condition never triggers for that side).
        The calling code handles this by only computing losses for
        non-empty lists.
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
                # J10 normalises power at every inner step
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

    # ╔══════════════════════════════════════════════════════════════╗
    # ║           STRATEGY CONFIGURATION — CHANGE HERE              ║
    # ╠══════════════════════════════════════════════════════════════╣
    # ║                                                              ║
    # ║  inner_strategy     : how to compress teacher inner 20→10   ║
    # ║    'avg_pairs'  → RECOMMENDED (preserves all info)          ║
    # ║    'first'      → coarse/aggressive teacher steps           ║
    # ║    'last'       → fine/polishing teacher steps              ║
    # ║    'subsample'  → evenly spaced across full range           ║
    # ║                                                              ║
    # ║  outer_init_strategy: how to handle teacher outer 120→60    ║
    # ║    'fingerprint'→ RECOMMENDED (free outer schedule)         ║
    # ║    'first'      → teacher outer iters [0:60]                ║
    # ║    'last'       → teacher outer iters [60:120]              ║
    # ║    'steepest'   → teacher outer iters [t_start:t_start+60]  ║
    # ║                                                              ║
    # ║  rkd_window_strategy: which teacher iters supervise RKD     ║
    # ║    'symmetric'  → RECOMMENDED (early + late windows)        ║
    # ║    'early_only' → teacher steepest → student start only     ║
    # ║    'late_only'  → teacher plateau  → student end only       ║
    # ║    'first_only' → teacher [0,K)    → student [0,K)          ║
    # ║                                                              ║
    # ╚══════════════════════════════════════════════════════════════╝

    inner_strategy      = 'avg_pairs'    # ← change for inner axis experiments
    outer_init_strategy = 'fingerprint'  # ← change for outer axis experiments
    rkd_window_strategy = 'symmetric'    # ← change for training window experiments

    # ── Hyperparameters ───────────────────────────────────────────────────────
    K_layers      = 15
    lambda_dist   = 25.0
    lambda_angle  = 50.0
    lambda_late   = 1.0
    t_start_early = 20                   # steepest teacher descent: [20, 35)
    t_start_late  = I_T - K_layers       # teacher plateau: [105, 120)

    # step_size bounds — critical for stability.
    # min prevents negative values; max prevents unbounded growth from the
    # late RKD loss (the primary cause of NaN in the J10/60 case).
    # The max=0.5 gives ample room to learn while preventing divergence.
    min_step_size = 1e-8
    max_step_size = 0.5

    # ── Validate window settings ──────────────────────────────────────────────
    assert t_start_early + K_layers <= I_T, "Early window exceeds teacher depth"
    assert K_layers <= I_S // 2, (
        f"Student windows overlap: [0,{K_layers}) ∩ "
        f"[{I_S-K_layers},{I_S}) — reduce K_layers to ≤ {I_S//2}")

    # ── Determine which windows are active ────────────────────────────────────
    # This dict-based dispatch keeps the training loop clean regardless of
    # which window strategy is selected.
    _window_modes = {
        'symmetric'  : (True,  True),    # collect early AND late
        'early_only' : (True,  False),   # collect early only
        'late_only'  : (False, True),    # collect late only
        'first_only' : (True,  False),   # early window but from teacher iter 0
    }
    assert rkd_window_strategy in _window_modes, (
        f"Unknown rkd_window_strategy {rkd_window_strategy!r}. "
        f"Choose: {list(_window_modes.keys())}")
    collect_early, collect_late = _window_modes[rkd_window_strategy]

    # For 'first_only', the early window starts from teacher iter 0, not t_start_early
    _teacher_early_start = 0 if rkd_window_strategy == 'first_only' else t_start_early

    print('=' * 70)
    print('CI-RKD  J10/I60 student  ←  J20/I120 teacher')
    print(f'Teacher : {I_T} outer × {N_INNER_T} inner = {I_T*N_INNER_T} steps/sample')
    print(f'Student :  {I_S} outer × {N_INNER_S} inner =  {I_S*N_INNER_S} steps/sample')
    print(f'Speedup : {(I_T*N_INNER_T)//(I_S*N_INNER_S)}×')
    print('=' * 70)
    print(f'inner_strategy      : {inner_strategy!r}')
    print(f'outer_init_strategy : {outer_init_strategy!r}')
    print(f'rkd_window_strategy : {rkd_window_strategy!r}')
    if collect_early:
        print(f'  Early : teacher [{_teacher_early_start}, '
              f'{_teacher_early_start+K_layers})  →  student [0, {K_layers})')
    if collect_late:
        print(f'  Late  : teacher [{t_start_late}, {I_T})'
              f'  →  student [{I_S-K_layers}, {I_S})')
    print(f'K_layers={K_layers}  |  lambda_late={lambda_late}'
          f'  |  step_size ∈ [{min_step_size}, {max_step_size}]'
          f'  |  batch={batch_size}\n')

    # ── Load teacher ──────────────────────────────────────────────────────────
    model_teacher = PGA_Unfold_J20_Teacher(step_size_UPGA_J20).to(device)
    model_teacher.load_state_dict(
        torch.load(model_file_name_UPGA_J20, map_location=device))
    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad_(False)
    print(f'Teacher loaded : {model_file_name_UPGA_J20}')

    # ── Build step-size initialisation ────────────────────────────────────────
    ss_init = build_step_size_init(
        model_teacher,
        n_outer_S           = I_S,
        inner_strategy      = inner_strategy,
        outer_init_strategy = outer_init_strategy,
        t_start_steepest    = t_start_early)   # reuse same steepest start

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
        # NaN in step_size means a previous backward() wrote NaN into a
        # gradient, which corrupted Adam's internal moment buffers
        # (exp_avg, exp_avg_sq).  zero_grad() does NOT clear these —
        # only optimizer.state.clear() does.  We detect and reset early
        # rather than silently training on a broken model for a full epoch.
        if torch.isnan(model_student.step_size.data).any():
            print(f"  [WARNING] NaN in step_size at epoch {i_epoch} start. "
                  f"Resetting parameters and Adam state.")
            with torch.no_grad():
                model_student.step_size.data.copy_(ss_init.to(device))
            optimizer.state.clear()   # clears corrupted moment buffers

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

            # ── Teacher: ONE pass, collects requested windows ─────────────────
            with torch.no_grad():
                F_t, W_t, F_T_early, F_T_late = \
                    model_teacher.execute_PGA_two_windows(
                        H, R, snr_train, I_T, N_INNER_T,
                        t_start_early  = _teacher_early_start,
                        t_start_late   = t_start_late,
                        K_layers       = K_layers,
                        collect_early  = collect_early,
                        collect_late   = collect_late)
                teacher_task_loss = get_sum_loss(F_t, W_t, H, R, snr_train, B)

            # ── Student: ONE pass, collects BOTH windows in same graph ────────
            rate, _, F_s, W_s, F_S_first, F_S_last = \
                model_student.execute_PGA_with_both_windows(
                    H, R, snr_train, I_S, N_INNER_S, K_layers)

            # ── Compute losses for active windows ─────────────────────────────
            loss_task = get_sum_loss(F_s, W_s, H, R, snr_train, B)

            loss_rkd_early = (ci_rkd_loss(F_T_early, F_S_first,
                                           K_layers, B, lambda_dist, lambda_angle)
                              if collect_early else
                              torch.tensor(0.0, device=device))

            loss_rkd_late  = (ci_rkd_loss(F_T_late,  F_S_last,
                                           K_layers, B, lambda_dist, lambda_angle)
                              if collect_late else
                              torch.tensor(0.0, device=device))

            total_loss = loss_task + loss_rkd_early + lambda_late * loss_rkd_late

            # ── NaN/Inf check BEFORE backward ─────────────────────────────────
            # Critical ordering: checking AFTER backward() (as in the old code)
            # allows NaN gradients to corrupt Adam's moment buffers before the
            # batch is skipped.  Those corrupted moments persist silently and
            # produce NaN updates on every subsequent batch even with clean losses.
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"  [WARNING] NaN/Inf at epoch {i_epoch} "
                      f"batch {i_batch} — skipping.")
                optimizer.zero_grad()
                continue

            # ── Backward + update ─────────────────────────────────────────────
            optimizer.zero_grad()
            total_loss.backward()   # one call handles all active loss terms

            torch.nn.utils.clip_grad_norm_(model_student.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                # Both min AND max clamp are essential.
                # min: prevents negative step sizes.
                # max: prevents the late RKD loss from driving step sizes to
                #      infinity — the primary cause of NaN in J10/60 training
                #      (not an issue in J20/60 because 20 inner steps dilute
                #      each step size's impact more than 10 inner steps do).
                model_student.step_size.data.clamp_(
                    min=min_step_size, max=max_step_size)

            epoch_loss      += total_loss.item()
            epoch_rkd_early += loss_rkd_early.item()
            epoch_rkd_late  += loss_rkd_late.item()
            epoch_task_s    += loss_task.item()
            epoch_task_t    += teacher_task_loss.item()
            num_batches     += 1

            # torch.cuda.empty_cache() is intentionally absent here.
            # Calling it every batch forces a full CUDA context sync and
            # memory defragmentation, which was the primary source of
            # slowness in earlier versions.  PyTorch's caching allocator
            # handles memory reuse automatically and efficiently.

        nb = max(num_batches, 1)
        print(f"Epoch {i_epoch:4d} "
              f"[{inner_strategy}|{outer_init_strategy}|{rkd_window_strategy}] | "
              f"Time: {time.time()-start_time:.1f}s | "
              f"Loss: {epoch_loss/nb:.4f} | "
              f"Student: {epoch_task_s/nb:.4f} | "
              f"Teacher: {epoch_task_t/nb:.4f} | "
              f"RKD_early: {epoch_rkd_early/nb:.4f} | "
              f"RKD_late: {epoch_rkd_late/nb:.4f}")

        # Step-size statistics once per epoch.  Watch for max trending toward
        # max_step_size — that means the clamp is actively working and you
        # may want to reduce lambda_late or the learning rate.
        with torch.no_grad():
            ss = model_student.step_size.data
            print(f"             step_size : "
                  f"min={ss.min():.4e}  max={ss.max():.4e}  "
                  f"mean={ss.mean():.4e}  std={ss.std():.4e}")

    # All three strategy names are embedded in the save path so experiments
    # never overwrite each other and results stay fully reproducible.
    save_path = (model_file_name_UPGA_J10 +
                 f'_I60_CI_RKD'
                 f'_inner{inner_strategy}'
                 f'_outer{outer_init_strategy}'
                 f'_win{rkd_window_strategy}'
                 f'_Kl{K_layers}.pth')
    torch.save(model_student.state_dict(), save_path)
    print(f'\nStudent saved → {save_path}')


# ══════════════════════════════════════════════════════════════════════════════
# 7.  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

if run_RKD_Distillation == 0:

    # These must match exactly what was used during training
    inner_strategy      = 'avg_pairs'
    outer_init_strategy = 'fingerprint'
    rkd_window_strategy = 'symmetric'
    K_layers            = 15

    save_path = (model_file_name_UPGA_J10 +
                 f'_I60_CI_RKD'
                 f'_inner{inner_strategy}'
                 f'_outer{outer_init_strategy}'
                 f'_win{rkd_window_strategy}'
                 f'_Kl{K_layers}.pth')

    # Scalar init is fine at eval time — load_state_dict overwrites step_size
    # entirely from the checkpoint.  Only the shape [10, 60, K+1] must match.
    model_test = PGA_Unfold_J10_I60_CI(step_size_UPGA_J10).to(device)
    model_test.load_state_dict(torch.load(save_path, map_location=device))
    model_test.eval()

    with torch.no_grad():
        # K_layers=0 means neither window condition fires — both trajectory
        # lists remain empty.  We only need rate_iter and beam_iter here.
        rate_iter, beam_iter, _, _, _, _ = \
            model_test.execute_PGA_with_both_windows(
                H_test, Rtest, snr, I_S, N_INNER_S, K_layers=0)

    rate_RKD = [r.detach().cpu().numpy()
                for r in (sum(rate_iter) / len(H_test[0]))]
    beam_RKD = [r.detach().cpu().numpy()
                for r in (sum(beam_iter) / len(H_test[0]))]

    print(f'\nJ10/I60 | inner={inner_strategy} | '
          f'outer_init={outer_init_strategy} | '
          f'win={rkd_window_strategy} | K_layers={K_layers}')
    print(f'  Final sum rate   : {rate_RKD[-1]:.4f} bps/Hz')
    print(f'  Final beam error : {beam_RKD[-1]:.4f}')