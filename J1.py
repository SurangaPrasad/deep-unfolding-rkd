"""
Standalone Evaluation + Plotting — Conv PGA AGT Experiments
=============================================================
Loads the three trained PGA_Conv checkpoints, runs a custom eval-only
forward pass that computes the joint ISAC objective directly at each
outer iteration WITHOUT calling get_sum_rate or get_beam_error
internally — eliminating the double-normalize spike at the last
iteration.

Objective: R - OMEGA * tau   (OMEGA from system_config)

Outputs
-------
  conv_pga_agt_obj.png      — objective vs iterations, all curves
  conv_pga_agt_combined.png — publication-ready single panel
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from system_config import *
from utility import *
from PGA_models import *

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")
print(f"OMEGA  : {OMEGA}")
print(f"K+1    : {K+1}")

# ── Test data ─────────────────────────────────────────────────────────────────
H_train, H_test0 = get_data_tensor(data_source)
H_test           = H_test0[:, :test_size, :, :].to(device)
Rtest, _, _, _   = get_radar_data(snr_dB, H_test.cpu())
Rtest            = Rtest.to(device)

# ── Iteration counts ──────────────────────────────────────────────────────────
I_T = n_iter_outer        # 120
I_S = n_iter_outer // 2   # 60
J_S = n_iter_inner_J10    # 10

FLAT_SEED = 0.01


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DIRECT OBJECTIVE — no internal normalize
# ══════════════════════════════════════════════════════════════════════════════

def compute_objective_direct(H, F, W, R, Pt):
    """
    Compute R - OMEGA*tau using F and W that are already normalized
    by the PGA loop.  Does NOT call normalize internally, so there is
    no double-scaling of W and no spike at the last iteration.
    """
    # ── Sum rate ──────────────────────────────────────────────────────────────
    F_H  = torch.transpose(F, 2, 3).conj()
    W_H  = torch.transpose(W, 2, 3).conj()
    V    = W @ W_H
    rate = torch.zeros(len(H[0]), device=H.device, dtype=torch.float32)

    for m in range(M):
        W_m             = W.clone()
        W_m[:, :, :, m] = 0.0
        V_m             = W_m @ torch.transpose(W_m, 2, 3).conj()
        h_mk0           = torch.unsqueeze(H[:, :, m, :], dim=2)
        h_mk            = torch.transpose(h_mk0, 2, 3)
        h_mk_H          = torch.transpose(h_mk, 2, 3).conj()
        Htilde_mk       = h_mk @ h_mk_H
        trace_1         = get_trace(F @ V   @ F_H @ Htilde_mk)
        trace_2         = get_trace(F @ V_m @ F_H @ Htilde_mk)
        rate            = rate + (torch.log2(trace_1 + sigma2)
                                - torch.log2(trace_2 + sigma2)).real

    sum_rate = torch.mean(rate)

    # ── Beam error ────────────────────────────────────────────────────────────
    X   = F @ W
    X_H = torch.transpose(X, 2, 3).conj()
    if normalize_tau == 1:
        error = (torch.linalg.matrix_norm(X @ X_H - R, ord='fro') ** 2
                 / torch.linalg.matrix_norm(R, ord='fro') ** 2)
    else:
        error = torch.linalg.matrix_norm(X @ X_H - R, ord='fro') ** 2
    sum_error = torch.mean(error)

    return (sum_rate - OMEGA * sum_error).item()


# ══════════════════════════════════════════════════════════════════════════════
# 2.  EVAL-ONLY FORWARD PASS FOR PGA_Conv
# ══════════════════════════════════════════════════════════════════════════════

def eval_objective_conv(model, n_outer, label):
    """
    Custom eval forward pass for PGA_Conv (J=1).
    Returns rate_iter, tau_iter, obj_iter — all per outer iteration.
    Computes metrics directly without internal normalize to avoid spike.
    step_size shape: [n_outer, K+1]
    """
    rate_iter = []
    tau_iter  = []
    obj_iter  = []

    def _rate_direct(H, F, W, Pt):
        F_H  = torch.transpose(F, 2, 3).conj()
        V    = W @ torch.transpose(W, 2, 3).conj()
        rate = torch.zeros(len(H[0]), device=H.device, dtype=torch.float32)
        for m in range(M):
            W_m             = W.clone()
            W_m[:, :, :, m] = 0.0
            V_m             = W_m @ torch.transpose(W_m, 2, 3).conj()
            h_mk0           = torch.unsqueeze(H[:, :, m, :], dim=2)
            h_mk            = torch.transpose(h_mk0, 2, 3)
            h_mk_H          = torch.transpose(h_mk, 2, 3).conj()
            Htilde_mk       = h_mk @ h_mk_H
            trace_1         = get_trace(F @ V   @ F_H @ Htilde_mk)
            trace_2         = get_trace(F @ V_m @ F_H @ Htilde_mk)
            rate            = rate + (torch.log2(trace_1 + sigma2)
                                    - torch.log2(trace_2 + sigma2)).real
        return torch.mean(rate).item()

    def _tau_direct(F, W, R):
        X   = F @ W
        X_H = torch.transpose(X, 2, 3).conj()
        if normalize_tau == 1:
            error = (torch.linalg.matrix_norm(X @ X_H - R, ord='fro') ** 2
                     / torch.linalg.matrix_norm(R, ord='fro') ** 2)
        else:
            error = torch.linalg.matrix_norm(X @ X_H - R, ord='fro') ** 2
        return torch.mean(error).item()

    with torch.no_grad():
        _, _, F, W = initialize(H_test, Rtest, snr, initial_normalization)
        r = _rate_direct(H_test, F, W, snr)
        t = _tau_direct(F, W, Rtest)
        rate_iter.append(r); tau_iter.append(t)
        obj_iter.append(r - OMEGA * t)

        for ii in range(n_outer):
            grad_F_com = get_grad_F_com(H_test, F, W)
            grad_F_rad = get_grad_F_rad(F, W, Rtest)
            F = (F
                 + model.step_size[ii][0] * grad_F_com * WEIGHT_F_COM
                 - model.step_size[ii][0] * grad_F_rad * WEIGHT_F_RAD)
            if sum(torch.abs(F[0, :, 0, 0])) > 1e1:
                F = normalize_power(F, W, H_test, snr)
            F = F / torch.abs(F)

            W_new      = W.clone().detach()
            grad_W_com = get_grad_W_com(H_test, F, W)
            grad_W_rad = get_grad_W_rad(F, W, Rtest)
            for k in range(K):
                W_new[k] = (W[k].clone().detach()
                            + model.step_size[ii][k+1] * grad_W_com[k] * WEIGHT_W_COM
                            - model.step_size[ii][k+1] * grad_W_rad[k] * WEIGHT_W_RAD)

            F, W = normalize(F, W_new, H_test, snr)
            r = _rate_direct(H_test, F, W, snr)
            t = _tau_direct(F, W, Rtest)
            rate_iter.append(r); tau_iter.append(t)
            obj_iter.append(r - OMEGA * t)

    print(f"\n{label}")
    print(f"  Iterations      : {n_outer}")
    print(f"  Final rate      : {rate_iter[-1]:.4f} bps/Hz")
    print(f"  Final tau       : {tau_iter[-1]:.6f}")
    print(f"  Final objective : {obj_iter[-1]:.4f}")
    return rate_iter, tau_iter, obj_iter



# ══════════════════════════════════════════════════════════════════════════════
# 3.  CHECKPOINT PATHS
# ══════════════════════════════════════════════════════════════════════════════

path_exp1 = model_file_name_UPGA_J1 + '_I60_AGT_student_last_inner.pth'
path_exp2 = model_file_name_UPGA_J1 + '_I120_AGT_teacher_avg_inner.pth'
path_exp3 = model_file_name_UPGA_J1 + '_I120_AGT_hybrid_student60_flat60.pth'


# ══════════════════════════════════════════════════════════════════════════════
# 4.  LOAD MODELS
# ══════════════════════════════════════════════════════════════════════════════

def load_conv(path, n_outer, label):
    ss_dummy = torch.zeros(n_outer, K + 1)
    model    = PGA_Conv(ss_dummy).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Loaded : {label}")
    print(f"  path            : {path}")
    print(f"  step_size shape : {list(model.step_size.shape)}")
    return model

print('\n' + '=' * 65)
print('LOADING CHECKPOINTS')
print('=' * 65)
model1 = load_conv(path_exp1, I_S, 'Exp1 Conv I=60  student last inner')
model2 = load_conv(path_exp2, I_T, 'Exp2 Conv I=120 teacher avg inner')
model3 = load_conv(path_exp3, I_T, 'Exp3 Conv I=120 hybrid init')

# Flat baselines — untrained models
model_flat60  = PGA_Conv(torch.full((I_S, K+1), FLAT_SEED)).to(device)
model_flat60.eval()
model_flat120 = PGA_Conv(torch.full((I_T, K+1), FLAT_SEED)).to(device)
model_flat120.eval()
print("\nFlat baselines ready (no training)")

print('=' * 65)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 65)
print(f'EVALUATION  (objective = R - {OMEGA} * tau, direct computation)')
print('=' * 65)

rate1,       tau1,       obj1        = eval_objective_conv(model1,      I_S, 'Exp1 | Conv I=60  | student last inner')
rate2,       tau2,       obj2        = eval_objective_conv(model2,      I_T, 'Exp2 | Conv I=120 | teacher avg inner')
rate3,       tau3,       obj3        = eval_objective_conv(model3,      I_T, 'Exp3 | Conv I=120 | hybrid init')
rate_flat60, tau_flat60, obj_flat60  = eval_objective_conv(model_flat60,  I_S, 'Baseline | Conv I=60  | flat 0.01')
rate_flat120,tau_flat120,obj_flat120 = eval_objective_conv(model_flat120, I_T, 'Baseline | Conv I=120 | flat 0.01')

print('\n' + '=' * 65)
print(f"{'Model':<42} {'Rate':>8} {'Tau':>10} {'Obj':>8} {'Steps':>6}")
print('-' * 65)
print(f"{'Conv I=60  | student last inner':<42} {rate1[-1]:>8.4f} {tau1[-1]:>10.4f} {obj1[-1]:>8.4f} {I_S:>6}")
print(f"{'Conv I=120 | teacher avg inner':<42} {rate2[-1]:>8.4f} {tau2[-1]:>10.4f} {obj2[-1]:>8.4f} {I_T:>6}")
print(f"{'Conv I=120 | hybrid init':<42} {obj3[-1]:>8.4f} {tau3[-1]:>10.4f} {obj3[-1]:>8.4f} {I_T:>6}")
print(f"{'Conv I=60  | flat 0.01':<42} {rate_flat60[-1]:>8.4f} {tau_flat60[-1]:>10.4f} {obj_flat60[-1]:>8.4f} {I_S:>6}")
print(f"{'Conv I=120 | flat 0.01':<42} {rate_flat120[-1]:>8.4f} {tau_flat120[-1]:>10.4f} {obj_flat120[-1]:>8.4f} {I_T:>6}")
print('=' * 65)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  PLOTTING — three separate IEEE-quality figures
# ══════════════════════════════════════════════════════════════════════════════

# x-axes (include init point at index 0)
x60  = list(range(len(obj1)))    # 0 … 60
x120 = list(range(len(obj2)))    # 0 … 120

HYBRID_SWITCH = I_S + 1   # iteration where Exp3 switches warm → flat

# ── IEEE colorblind-safe palette, thick lines, distinct markers ───────────────
STYLES = {
    'exp1'    : dict(color='#0072B2', ls='-',   lw=2.2, marker='o',
                     markevery=8,  markersize=6,
                     label='Conv I=60  | student last inner'),
    'exp2'    : dict(color='#D55E00', ls='--',  lw=2.2, marker='s',
                     markevery=8,  markersize=6,
                     label='Conv I=120 | teacher avg inner'),
    'exp3'    : dict(color='#009E73', ls='-.',  lw=2.2, marker='^',
                     markevery=8,  markersize=6,
                     label='Conv I=120 | hybrid  (student[0:60] + flat[60:120])'),
    'flat60'  : dict(color='#8B2FC9', ls=':',   lw=1.8, marker='x',
                     markevery=8,  markersize=6,
                     label='Conv I=60  | flat 0.01  (no training)'),
    'flat120' : dict(color='#000000', ls=':',   lw=1.8, marker='d',
                     markevery=8,  markersize=6,
                     label='Conv I=120 | flat 0.01  (no training)'),
}
KEYS = ['exp1', 'exp2', 'exp3', 'flat60', 'flat120']
FONT = dict(xlabel=13, ylabel=13, title=12, legend=10, tick=11, annot=9)

plt.rcParams.update({
    'font.family'       : 'serif',
    'font.size'         : 11,
    'axes.linewidth'    : 1.2,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.grid'         : True,
    'grid.alpha'        : 0.35,
    'grid.linestyle'    : '--',
    'grid.linewidth'    : 0.7,
    'xtick.major.width' : 1.2,
    'ytick.major.width' : 1.2,
    'xtick.major.size'  : 5,
    'ytick.major.size'  : 5,
})


def _add_hybrid_marker(ax):
    ylo, yhi = ax.get_ylim()
    ax.axvline(x=HYBRID_SWITCH,
               color=STYLES['exp3']['color'],
               lw=1.2, ls=':', alpha=0.6)
    ax.text(HYBRID_SWITCH + 0.8,
            ylo + (yhi - ylo) * 0.06,
            'flat\nstarts',
            fontsize=FONT['annot'],
            color=STYLES['exp3']['color'],
            alpha=0.85, va='bottom')


def _make_handles():
    return [mlines.Line2D([], [],
                          color=STYLES[k]['color'],
                          ls=STYLES[k]['ls'],
                          lw=STYLES[k]['lw'],
                          marker=STYLES[k]['marker'],
                          markersize=STYLES[k]['markersize'],
                          label=STYLES[k]['label'])
            for k in KEYS]


def _save(fig, fname):
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {fname}")


def _make_fig():
    """Figure with extra bottom space for the outside legend."""
    fig, ax = plt.subplots(figsize=(8, 5.8))
    fig.subplots_adjust(bottom=0.28)
    return fig, ax


def _outside_legend(fig):
    """Place a 2-column legend below the axes, outside the plot area."""
    fig.legend(handles=_make_handles(),
               loc='lower center',
               ncol=2,
               fontsize=FONT['legend'],
               framealpha=0.95,
               edgecolor='#aaaaaa',
               fancybox=False,
               frameon=True,
               bbox_to_anchor=(0.5, 0.0))


# ── Figure 1: Sum Rate ────────────────────────────────────────────────────────
fig, ax = _make_fig()
ax.plot(x60,  rate1,        **STYLES['exp1'])
ax.plot(x120, rate2,        **STYLES['exp2'])
ax.plot(x120, rate3,        **STYLES['exp3'])
ax.plot(x60,  rate_flat60,  **STYLES['flat60'])
ax.plot(x120, rate_flat120, **STYLES['flat120'])
_add_hybrid_marker(ax)
ax.set_xlabel('Iterations / Layers (I)', fontsize=FONT['xlabel'])
ax.set_ylabel(r'Sum Rate $R$ (bps/Hz)',  fontsize=FONT['ylabel'])
ax.set_title('Sum Rate vs Iterations — Conv PGA AGT Init', fontsize=FONT['title'])
ax.tick_params(labelsize=FONT['tick'])
_outside_legend(fig)
_save(fig, 'conv_pga_agt_rate.png')


# ── Figure 2: Beam Error ──────────────────────────────────────────────────────
fig, ax = _make_fig()
ax.plot(x60,  tau1,         **STYLES['exp1'])
ax.plot(x120, tau2,         **STYLES['exp2'])
ax.plot(x120, tau3,         **STYLES['exp3'])
ax.plot(x60,  tau_flat60,   **STYLES['flat60'])
ax.plot(x120, tau_flat120,  **STYLES['flat120'])
_add_hybrid_marker(ax)
ax.set_xlabel('Iterations / Layers (I)', fontsize=FONT['xlabel'])
ax.set_ylabel(r'Beam Error $\tau$',       fontsize=FONT['ylabel'])
ax.set_title('Beam Error vs Iterations — Conv PGA AGT Init', fontsize=FONT['title'])
ax.tick_params(labelsize=FONT['tick'])
_outside_legend(fig)
_save(fig, 'conv_pga_agt_beam.png')


# ── Figure 3: Joint ISAC Objective ───────────────────────────────────────────
fig, ax = _make_fig()
ax.plot(x60,  obj1,        **STYLES['exp1'])
ax.plot(x120, obj2,        **STYLES['exp2'])
ax.plot(x120, obj3,        **STYLES['exp3'])
ax.plot(x60,  obj_flat60,  **STYLES['flat60'])
ax.plot(x120, obj_flat120, **STYLES['flat120'])
_add_hybrid_marker(ax)
ax.set_xlabel('Iterations / Layers (I)', fontsize=FONT['xlabel'])
ax.set_ylabel(r'$R - \omega\bar{\tau}$',  fontsize=FONT['ylabel'])
ax.set_title(f'Joint ISAC Objective ($\\omega={OMEGA}$) — Conv PGA AGT Init',
             fontsize=FONT['title'])
ax.tick_params(labelsize=FONT['tick'])
_outside_legend(fig)
_save(fig, 'conv_pga_agt_obj.png')