"""
PGA_Conv AGT Initialisation — Three Experiments
================================================
All three models are conventional PGA (J=1, no inner loop).
The goal is to isolate how much convergence acceleration comes
from step-size initialisation alone, without any inner iteration
structure or RKD supervision.

Experiment 1 — PGA_Conv I=60, init from J10/I60 student (last inner step)
---------------------------------------------------------------------------
Source  : trained J10/I60 student,  step_size [10, 60, K+1]
Extract : last inner step j=9  →  ss_student[9, :, :]   shape [60, K+1]
Result  : PGA_Conv step_size [60, K+1] — outer schedule fully preserved

Experiment 2 — PGA_Conv I=120, init from J20 teacher (avg all inner)
----------------------------------------------------------------------
Source  : trained J20/I120 teacher, step_size [20, 120, K+1]
Extract : mean over 20 inner steps  →  ss_T.mean(dim=0)   shape [120, K+1]
Result  : PGA_Conv step_size [120, K+1] — outer schedule fully preserved

Experiment 3 — PGA_Conv I=120, first 60 from student, last 60 flat
--------------------------------------------------------------------
Source  : trained J10/I60 student,  step_size [10, 60, K+1]
Extract : last inner step j=9  →  ss_student[9, :, :]   shape [60, K+1]
First 60 outer slots  : from student  (teacher-informed warm start)
Last  60 outer slots  : flat 0.01     (no teacher knowledge)
Result  : PGA_Conv step_size [120, K+1] — hybrid init
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
I_T  = n_iter_outer          # teacher : 120
I_S  = n_iter_outer // 2     # student :  60
J_T  = n_iter_inner_J20      # teacher :  20 inner
J_S  = n_iter_inner_J10      # student :  10 inner

FLAT_SEED     = 0.01
MIN_STEP_SIZE = 1e-8
MAX_STEP_SIZE = 0.5

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
# 1.  LOAD TEACHER AND STUDENT
# ══════════════════════════════════════════════════════════════════════════════

# ── Teacher J20/I120 ──────────────────────────────────────────────────────────
model_teacher = PGA_Unfold_J20(step_size_UPGA_J20).to(device)
model_teacher.load_state_dict(
    torch.load(model_file_name_UPGA_J20, map_location=device))
model_teacher.eval()
for p in model_teacher.parameters():
    p.requires_grad_(False)
print(f"Teacher loaded  : {model_file_name_UPGA_J20}")
print(f"  step_size shape : {list(model_teacher.step_size.shape)}")   # [20, 120, K+1]

# ── Student J10/I60 ───────────────────────────────────────────────────────────
# Use the best trained student (AGT + CI-RKD).
# Any of the four ablation checkpoints can be substituted here.
student_path  = (directory_model +
                 'UPGA_J10_320.pth_I60_CI_RKD_sym_inner_avg_pairs_Kl15_win20_basic.pth')
dummy_init    = torch.zeros(J_S, I_S, K + 1)   # [10, 60, 2]
model_student = PGA_Unfold_J10_I60_CI(dummy_init)
model_student.load_state_dict(
    torch.load(student_path, map_location=device))
model_student.eval()
for p in model_student.parameters():
    p.requires_grad_(False)
print(f"Student loaded  : {student_path}")
print(f"  step_size shape : {list(model_student.step_size.shape)}")   # [10, 60, K+1]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  INITIALISATION BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_init_exp1():
    """
    Experiment 1 — PGA_Conv I=60
    Source: J10/I60 student, last inner step (j=9).

    ss_student[9, :, :]  →  [60, K+1]
    - Row 9 (0-indexed) is the 10th = last inner step.
    - The full outer schedule of the student (60 slots) is preserved as-is.
    - No averaging, no collapsing — direct extraction.
    """
    with torch.no_grad():
        ss_S   = model_student.step_size.data     # [10, 60, K+1]
        ss_init = ss_S[J_S - 1, :, :].clone()    # [60, K+1]  — last inner row

    print(f"\n[Exp 1 | PGA_Conv I=60 | source: student last inner (j={J_S-1})]")
    print(f"  student {list(ss_S.shape)} → init {list(ss_init.shape)}")
    print(f"  F  step-size range : [{ss_init[:,0].min():.4e}, "
          f"{ss_init[:,0].max():.4e}]")
    print(f"  W  step-size range : [{ss_init[:,1:].min():.4e}, "
          f"{ss_init[:,1:].max():.4e}]")
    return ss_init   # [60, K+1]


def build_init_exp2():
    """
    Experiment 2 — PGA_Conv I=120
    Source: J20/I120 teacher, average over all 20 inner steps.

    mean(ss_T, dim=0)  →  [120, K+1]
    - Every inner step contributes equally to the mean.
    - The full outer schedule of the teacher (120 slots) is preserved.
    - No outer averaging — only the inner axis is collapsed.
    """
    with torch.no_grad():
        ss_T    = model_teacher.step_size.data    # [20, 120, K+1]
        ss_init = ss_T.mean(dim=0).clone()        # [120, K+1]

    print(f"\n[Exp 2 | PGA_Conv I=120 | source: teacher avg all inner]")
    print(f"  teacher {list(ss_T.shape)} → init {list(ss_init.shape)}")
    print(f"  F  step-size range : [{ss_init[:,0].min():.4e}, "
          f"{ss_init[:,0].max():.4e}]")
    print(f"  W  step-size range : [{ss_init[:,1:].min():.4e}, "
          f"{ss_init[:,1:].max():.4e}]")
    return ss_init   # [120, K+1]


def build_init_exp3():
    """
    Experiment 3 — PGA_Conv I=120, hybrid init
    First 60 outer slots  : from J10/I60 student last inner step (j=9)
    Last  60 outer slots  : flat FLAT_SEED (no teacher knowledge)

    This lets us see whether the warm-start benefit is localised to the
    early iterations or propagates through to the later ones.
    If the model converges well despite the uninitialised tail, it confirms
    that early-phase geometry is the key transfer mechanism.
    """
    with torch.no_grad():
        ss_S = model_student.step_size.data        # [10, 60, K+1]

        # First 60: student last inner step
        first_60 = ss_S[J_S - 1, :, :].clone()    # [60, K+1]

        # Last 60: flat scalar broadcast to [60, K+1]
        last_60  = torch.full((I_S, K + 1), FLAT_SEED)

        ss_init  = torch.cat([first_60, last_60], dim=0)   # [120, K+1]

    print(f"\n[Exp 3 | PGA_Conv I=120 | hybrid: student[0:60] + flat[60:120]]")
    print(f"  First 60 F range : [{first_60[:,0].min():.4e}, "
          f"{first_60[:,0].max():.4e}]  (student-informed)")
    print(f"  Last  60 F value : {FLAT_SEED}  (flat uninitialised)")
    print(f"  Full init shape  : {list(ss_init.shape)}")
    return ss_init   # [120, K+1]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  GENERIC TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_conv_pga(ss_init, n_outer, label, save_suffix):
    """
    Train a PGA_Conv model from a given step_size initialisation.

    Args:
        ss_init     : torch.Tensor [n_outer, K+1]  — initial step-sizes
        n_outer     : int — number of outer iterations for this model
        label       : str — printed in epoch log
        save_suffix : str — appended to the checkpoint filename

    Returns:
        save_path   : str — path where the model was saved
        model       : trained PGA_Conv (eval mode, on device)
        ss_init_dev : the init tensor on device (for NaN reset)
    """
    ss_init_dev = ss_init.float().to(device)
    model       = PGA_Conv(ss_init_dev.clone()).to(device)
    optimizer   = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\n{'='*65}")
    print(f"Training : {label}")
    print(f"  step_size shape : {list(model.step_size.shape)}")
    print(f"  outer iters     : {n_outer}")
    print(f"  steps/sample    : {n_outer}  (J=1)")
    print(f"{'='*65}\n")

    for i_epoch in range(n_epoch):
        start_time  = time.time()
        epoch_loss  = 0.0
        num_batches = 0

        # ── Parameter health check ────────────────────────────────────────────
        if torch.isnan(model.step_size.data).any():
            print(f"  [WARNING] NaN detected at epoch {i_epoch}. "
                  f"Resetting to init.")
            with torch.no_grad():
                model.step_size.data.copy_(ss_init_dev)
            optimizer.state.clear()

        H_shuffled = torch.transpose(H_train, 0, 1)[
            np.random.permutation(len(H_train[0]))]

        for i_batch in range(0, len(H_train[0]), batch_size):
            H = torch.transpose(
                H_shuffled[i_batch:i_batch + batch_size], 0, 1).to(device)
            B = H.shape[1]

            snr_dB_train = np.random.choice(snr_dB_list)
            snr_train    = 10 ** (snr_dB_train / 10)
            R = get_R(snr_dB_train, B)

            # ── Forward ───────────────────────────────────────────────────────
            _, _, F_s, W_s = model.execute_PGA(
                H, R, snr_train, n_outer)

            # ── Task loss ─────────────────────────────────────────────────────
            loss = get_sum_loss(F_s, W_s, H, R, snr_train, B)

            # ── NaN guard BEFORE backward ─────────────────────────────────────
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [WARNING] NaN/Inf loss epoch {i_epoch} "
                      f"batch {i_batch} — skipping.")
                optimizer.zero_grad()
                continue

            # ── Backward + update ─────────────────────────────────────────────
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                model.step_size.data.clamp_(
                    min=MIN_STEP_SIZE, max=MAX_STEP_SIZE)

            epoch_loss  += loss.item()
            num_batches += 1

        nb = max(num_batches, 1)
        print(f"Epoch {i_epoch:4d} [{label}] | "
              f"Time: {time.time()-start_time:.1f}s | "
              f"Loss: {epoch_loss/nb:.4f}")

        with torch.no_grad():
            ss = model.step_size.data
            print(f"             step_size : "
                  f"min={ss.min():.4e}  max={ss.max():.4e}  "
                  f"mean={ss.mean():.4e}")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_path = (model_file_name_UPGA_J1 + save_suffix + '.pth')
    torch.save(model.state_dict(), save_path)
    print(f"\n{label} saved → {save_path}")

    model.eval()
    return save_path, model, ss_init_dev


# ══════════════════════════════════════════════════════════════════════════════
# 4.  RUN ALL THREE EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Experiment 1 ──────────────────────────────────────────────────────────────
ss1 = build_init_exp1()
save1, model1, ss1_dev = train_conv_pga(
    ss_init     = ss1,
    n_outer     = I_S,                          # 60
    label       = 'Conv I=60 | student last inner',
    save_suffix = '_I60_AGT_student_last_inner')

# ── Experiment 2 ──────────────────────────────────────────────────────────────
ss2 = build_init_exp2()
save2, model2, ss2_dev = train_conv_pga(
    ss_init     = ss2,
    n_outer     = I_T,                          # 120
    label       = 'Conv I=120 | teacher avg all inner',
    save_suffix = '_I120_AGT_teacher_avg_inner')

# ── Experiment 3 ──────────────────────────────────────────────────────────────
ss3 = build_init_exp3()
save3, model3, ss3_dev = train_conv_pga(
    ss_init     = ss3,
    n_outer     = I_T,                          # 120
    label       = 'Conv I=120 | student[0:60] + flat[60:120]',
    save_suffix = '_I120_AGT_hybrid_student60_flat60')


# ══════════════════════════════════════════════════════════════════════════════
# 5.  EVALUATION — all three models + flat baselines
# ══════════════════════════════════════════════════════════════════════════════

def eval_conv(model, n_outer, label):
    """Run model on test set and return per-iteration rate and beam error."""
    with torch.no_grad():
        rate, tau, _, _ = model.execute_PGA(
            H_test, Rtest, snr, n_outer)
    rate_iter = [r.detach().cpu().numpy()
                 for r in (sum(rate) / len(H_test[0]))]
    tau_iter  = [t.detach().cpu().numpy()
                 for t in (sum(tau)  / len(H_test[0]))]
    print(f"\n{label}")
    print(f"  Final sum rate   : {rate_iter[-1]:.4f} bps/Hz")
    print(f"  Final beam error : {tau_iter[-1]:.6f}")
    return rate_iter, tau_iter


print('\n' + '='*65)
print('EVALUATION')
print('='*65)

rate1, tau1 = eval_conv(model1, I_S, 'Exp1 | Conv I=60  | student last inner')
rate2, tau2 = eval_conv(model2, I_T, 'Exp2 | Conv I=120 | teacher avg inner')
rate3, tau3 = eval_conv(model3, I_T, 'Exp3 | Conv I=120 | hybrid init')

# ── Flat baselines for comparison ─────────────────────────────────────────────
# I=60 flat baseline
ss_flat60  = torch.full((I_S, K + 1), FLAT_SEED)
model_flat60 = PGA_Conv(ss_flat60).to(device)
# Note: flat baselines are evaluated WITHOUT fine-tuning to show
# the pure initialisation effect.  Train them first if you want
# a fair fine-tuned comparison — just call train_conv_pga() with
# the flat ss_init and the appropriate n_outer.
rate_flat60, tau_flat60 = eval_conv(
    model_flat60, I_S, 'Baseline | Conv I=60  | flat 0.01 (no training)')

# I=120 flat baseline
ss_flat120 = torch.full((I_T, K + 1), FLAT_SEED)
model_flat120 = PGA_Conv(ss_flat120).to(device)
rate_flat120, tau_flat120 = eval_conv(
    model_flat120, I_T, 'Baseline | Conv I=120 | flat 0.01 (no training)')


# ══════════════════════════════════════════════════════════════════════════════
# 6.  SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*65)
print(f"{'Model':<40} {'Rate':>8} {'Beam err':>10} {'Steps':>7}")
print('-'*65)
print(f"{'Conv I=60  | student last inner':<40} "
      f"{rate1[-1]:>8.4f} {tau1[-1]:>10.6f} {I_S:>7}")
print(f"{'Conv I=120 | teacher avg inner':<40} "
      f"{rate2[-1]:>8.4f} {tau2[-1]:>10.6f} {I_T:>7}")
print(f"{'Conv I=120 | hybrid init':<40} "
      f"{rate3[-1]:>8.4f} {tau3[-1]:>10.6f} {I_T:>7}")
print(f"{'Conv I=60  | flat 0.01':<40} "
      f"{rate_flat60[-1]:>8.4f} {tau_flat60[-1]:>10.6f} {I_S:>7}")
print(f"{'Conv I=120 | flat 0.01':<40} "
      f"{rate_flat120[-1]:>8.4f} {tau_flat120[-1]:>10.6f} {I_T:>7}")
print('='*65)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use('Agg')          # headless — safe on cluster/server
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ── Iteration axes ────────────────────────────────────────────────────────────
# rate_iter includes the init point (iter 0), so length = n_outer + 1.
x60  = list(range(len(rate1)))    # 0 … 60
x120 = list(range(len(rate2)))    # 0 … 120

# ── Colour / style map ────────────────────────────────────────────────────────
STYLES = {
    'exp1'    : dict(color='#E8593C', ls='--',  lw=1.8,
                     label='Conv I=60  | student last inner'),
    'exp2'    : dict(color='#1D9E75', ls='--',  lw=1.8,
                     label='Conv I=120 | teacher avg inner'),
    'exp3'    : dict(color='#7F77DD', ls='-.',  lw=1.8,
                     label='Conv I=120 | hybrid (student[0:60] + flat[60:120])'),
    'flat60'  : dict(color='#888780', ls=':',   lw=1.4,
                     label='Conv I=60  | flat 0.01'),
    'flat120' : dict(color='#444441', ls=':',   lw=1.4,
                     label='Conv I=120 | flat 0.01'),
}

# Iteration index where Exp 3 switches from warm to flat
# (+1 accounts for the init point at index 0)
HYBRID_SWITCH = I_S + 1


def _add_hybrid_marker(ax, y_frac=0.08):
    """Thin vertical line showing where flat init kicks in for Exp 3."""
    ylo, yhi = ax.get_ylim()
    ax.axvline(x=HYBRID_SWITCH, color=STYLES['exp3']['color'],
               lw=0.9, ls=':', alpha=0.55)
    ax.text(HYBRID_SWITCH + 0.6, ylo + (yhi - ylo) * y_frac,
            'flat\nstarts', fontsize=7.5,
            color=STYLES['exp3']['color'], alpha=0.75, va='bottom')


# ── Figure 1: Sum Rate ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x60,  rate1,        **STYLES['exp1'])
ax.plot(x120, rate2,        **STYLES['exp2'])
ax.plot(x120, rate3,        **STYLES['exp3'])
ax.plot(x60,  rate_flat60,  **STYLES['flat60'])
ax.plot(x120, rate_flat120, **STYLES['flat120'])
_add_hybrid_marker(ax, y_frac=0.05)
ax.set_xlabel('Iterations / Layers (I)', fontsize=12)
ax.set_ylabel(r'$R - \omega\bar{\tau}$', fontsize=12)
ax.set_title('Sum Rate vs Iterations — Conv PGA AGT Init Comparison',
             fontsize=12)
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
rate_fig = 'conv_pga_agt_rate.png'
plt.savefig(rate_fig, dpi=150)
plt.close()
print(f"\nRate figure saved     → {rate_fig}")


# ── Figure 2: Beam Error ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x60,  tau1,        **STYLES['exp1'])
ax.plot(x120, tau2,        **STYLES['exp2'])
ax.plot(x120, tau3,        **STYLES['exp3'])
ax.plot(x60,  tau_flat60,  **STYLES['flat60'])
ax.plot(x120, tau_flat120, **STYLES['flat120'])
_add_hybrid_marker(ax, y_frac=0.88)
ax.set_xlabel('Iterations / Layers (I)', fontsize=12)
ax.set_ylabel(r'Beam Error $\tau$', fontsize=12)
ax.set_title('Beam Error vs Iterations — Conv PGA AGT Init Comparison',
             fontsize=12)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
beam_fig = 'conv_pga_agt_beam.png'
plt.savefig(beam_fig, dpi=150)
plt.close()
print(f"Beam figure saved     → {beam_fig}")


# ── Figure 3: Combined side-by-side ──────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

keys = ['exp1', 'exp2', 'exp3', 'flat60', 'flat120']
panels = [
    (ax1,
     [(x60,  rate1), (x120, rate2), (x120, rate3),
      (x60,  rate_flat60), (x120, rate_flat120)],
     r'$R - \omega\bar{\tau}$', 'Sum Rate',    0.05),
    (ax2,
     [(x60,  tau1),  (x120, tau2),  (x120, tau3),
      (x60,  tau_flat60),  (x120, tau_flat120)],
     r'Beam Error $\tau$',     'Beam Error',   0.88),
]

for ax, data_list, ylabel, title, hfrac in panels:
    for (x, y), key in zip(data_list, keys):
        ax.plot(x, y, **STYLES[key])
    _add_hybrid_marker(ax, y_frac=hfrac)
    ax.set_xlabel('Iterations / Layers (I)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)

# Shared legend below both subplots
handles = [
    mlines.Line2D([], [],
                  color=STYLES[k]['color'],
                  ls=STYLES[k]['ls'],
                  lw=STYLES[k]['lw'],
                  label=STYLES[k]['label'])
    for k in keys
]
fig.legend(handles=handles, loc='lower center', ncol=2,
           fontsize=9, bbox_to_anchor=(0.5, -0.14))
plt.suptitle('Conv PGA — AGT Initialisation Comparison', fontsize=13, y=1.01)
plt.tight_layout()
combined_fig = 'conv_pga_agt_combined.png'
plt.savefig(combined_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"Combined figure saved → {combined_fig}")
