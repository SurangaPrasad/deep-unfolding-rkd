"""
Ablation Plotting Script — Final Version
=========================================
Figures produced (PNG + EPS each):
  1. ablation_obj_vs_iter   — joint ISAC objective vs iterations
  2. ablation_rate_vs_iter  — sum rate vs iterations
  3. ablation_beam_vs_iter  — beam error vs iterations
  4. ablation_rate_vs_SNR   — sum rate vs SNR
  5. ablation_beam_vs_SNR   — radar beampattern MSE [dB] vs SNR

Ablation cells (J=10 inner, I=60 outer):
  Cell 1-1 : Flat init | No RKD   → R_flat
  Cell 1-2 : Flat init | +CI-RKD  → R_flat+RKD
  Cell 2-1 : AGT init  | No RKD   → R_AGT
  Cell 2-2 : AGT init  | +CI-RKD  → R_AGT+RKD

References:
  Teacher : PGA_Unfold_J20  — J=20, I=120
  J1      : PGA_Conv        — J=1,  I=120
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from system_config import *
from utility import *
from PGA_models import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")

# ── Test data ─────────────────────────────────────────────────────────────────
H_train, H_test0 = get_data_tensor(data_source)
H_test           = H_test0[:, :test_size, :, :].to(device)

# ── Iteration budgets ─────────────────────────────────────────────────────────
I_T = n_iter_outer        # 120  — teacher / J1
I_S = n_iter_outer // 2   # 60   — students
J_T = n_iter_inner_J20    # 20
J_S = n_iter_inner_J10    # 10

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT PATHS
# ══════════════════════════════════════════════════════════════════════════════

path_teacher  = './model/UPGA_J20_320.pth'
path_J1       = './model/UPGA_J1.pth'

path_cell_1_1 = './model/UPGA_J10.pth_I60_basic_sym_inner_avg_pairs_Kl15_win20.pth'
path_cell_1_2 = './model/UPGA_J10_all.pth_I60_noinit_CI_RKD_dist25.0_angle50.0_Kl15_win20.pth'
path_cell_2_1 = './model/UPGA_J10.pth_I60_init_sym_inner_avg_pairs_Kl15_win20.pth'
path_cell_2_2 = './model/UPGA_J10.pth_I60_CI_RKD_sym_inner_avg_pairs_Kl15_win20_b10.pth'

# ══════════════════════════════════════════════════════════════════════════════
# STYLES
# ══════════════════════════════════════════════════════════════════════════════

# Color palette — J1 is orange to match request
COLORS = {
    'teacher'  : '#2CA02C',   # green
    'J1'       : '#FF7F0E',   # orange
    'cell_1_1' : '#1F77B4',   # blue
    'cell_1_2' : '#E377C2',   # pink
    'cell_2_1' : '#D62728',   # red
    'cell_2_2' : '#9467BD',   # purple
}

def _make_styles(me_long, me_short):
    return {
        'teacher'  : dict(color=COLORS['teacher'], ls='-',       lw=2.5,
                          marker='o', markevery=me_long,  markersize=7,
                          label=r'$R_T$'),
        'J1'       : dict(color=COLORS['J1'],      ls='--',      lw=2.2,
                          marker='s', markevery=me_long,  markersize=7,
                          label=r'$R_{J_1}$'),
        'cell_1_1' : dict(color=COLORS['cell_1_1'],ls='-.',      lw=2.0,
                          marker='^', markevery=me_short, markersize=7,
                          label=r'$R_\mathrm{flat}$'),
        'cell_1_2' : dict(color=COLORS['cell_1_2'],ls=':',       lw=2.0,
                          marker='v', markevery=me_short, markersize=7,
                          label=r'$R_\mathrm{flat+RKD}$'),
        'cell_2_1' : dict(color=COLORS['cell_2_1'],ls=(0,(5,2)), lw=2.0,
                          marker='D', markevery=me_short, markersize=7,
                          label=r'$R_\mathrm{AGT}$'),
        'cell_2_2' : dict(color=COLORS['cell_2_2'],ls='-',       lw=2.8,
                          marker='*', markevery=me_short, markersize=9,
                          label=r'$R_\mathrm{AGT+RKD}$'),
    }

STYLES_ITER = _make_styles(me_long=8, me_short=6)
STYLES_SNR  = _make_styles(me_long=1, me_short=1)
STYLES      = STYLES_ITER

PLOT_ORDER = ['teacher', 'J1', 'cell_1_1', 'cell_1_2', 'cell_2_1', 'cell_2_2']

FONT = dict(xlabel=13, ylabel=13, legend=10, tick=11)

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

# ── Figure / legend helpers ───────────────────────────────────────────────────

def _make_fig():
    fig, ax = plt.subplots(figsize=(8, 5.8))
    fig.subplots_adjust(bottom=0.30)
    return fig, ax


def _make_handles(style_dict, keys=None):
    if keys is None:
        keys = PLOT_ORDER
    handles = []
    for k in keys:
        s = style_dict[k]
        h = mlines.Line2D([], [],
                          color=s['color'], ls=s['ls'], lw=s['lw'],
                          marker=s['marker'], markersize=s['markersize'],
                          label=s['label'])
        handles.append(h)
    return handles


def _outside_legend(fig, style_dict, keys=None):
    fig.legend(handles=_make_handles(style_dict, keys),
               loc='lower center', ncol=2,
               fontsize=FONT['legend'],
               framealpha=0.95,
               edgecolor='#aaaaaa',
               fancybox=False, frameon=True,
               bbox_to_anchor=(0.5, 0.0))


def _save(fig, stem):
    for ext in ('.png', '.eps'):
        path = os.path.join(directory_result, stem + ext)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved → {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# EVAL FORWARD PASSES
# get_sum_rate / get_beam_error call normalize() internally — correct.
# ══════════════════════════════════════════════════════════════════════════════

def eval_conv_J1(model, n_outer, H, R, Pt, label=''):
    """PGA_Conv (J=1). step_size: [n_outer, K+1]"""
    rate_iter, tau_iter, obj_iter = [], [], []
    with torch.no_grad():
        _, _, F, W = initialize(H, R, Pt, initial_normalization)
        r = get_sum_rate(H, F, W, Pt).item()
        t = get_beam_error(H, F, W, R, Pt).item()
        rate_iter.append(r); tau_iter.append(t); obj_iter.append(r - OMEGA * t)
        for ii in range(n_outer):
            grad_F_com = get_grad_F_com(H, F, W)
            grad_F_rad = get_grad_F_rad(F, W, R)
            F = (F + model.step_size[ii][0] * grad_F_com * WEIGHT_F_COM
                   - model.step_size[ii][0] * grad_F_rad * WEIGHT_F_RAD)
            if sum(torch.abs(F[0, :, 0, 0])) > 1e1:
                F = normalize_power(F, W, H, Pt)
            F = F / torch.abs(F)
            W_new = W.clone().detach()
            grad_W_com = get_grad_W_com(H, F, W)
            grad_W_rad = get_grad_W_rad(F, W, R)
            for k in range(K):
                W_new[k] = (W[k].clone().detach()
                            + model.step_size[ii][k+1] * grad_W_com[k] * WEIGHT_W_COM
                            - model.step_size[ii][k+1] * grad_W_rad[k] * WEIGHT_W_RAD)
            F, W = normalize(F, W_new, H, Pt)
            r = get_sum_rate(H, F, W, Pt).item()
            t = get_beam_error(H, F, W, R, Pt).item()
            rate_iter.append(r); tau_iter.append(t); obj_iter.append(r - OMEGA * t)
    if label:
        print(f"  {label:45s}  rate={rate_iter[-1]:.4f}  tau={tau_iter[-1]:.6f}  obj={obj_iter[-1]:.4f}")
    return rate_iter, tau_iter, obj_iter


def eval_teacher(model, n_outer, n_inner, H, R, Pt, label=''):
    """PGA_Unfold_J20. step_size: [J, I, K+1]"""
    rate_iter, tau_iter, obj_iter = [], [], []
    with torch.no_grad():
        _, _, F, W = initialize(H, R, Pt, initial_normalization)
        r = get_sum_rate(H, F, W, Pt).item()
        t = get_beam_error(H, F, W, R, Pt).item()
        rate_iter.append(r); tau_iter.append(t); obj_iter.append(r - OMEGA * t)
        for ii in range(n_outer):
            for jj in range(n_inner):
                grad_F_com = get_grad_F_com(H, F, W)
                grad_F_rad = get_grad_F_rad(F, W, R)
                F = (F + model.step_size[jj][ii][0] * grad_F_com * WEIGHT_F_COM
                       - model.step_size[jj][ii][0] * grad_F_rad * WEIGHT_F_RAD)
                if sum(torch.abs(F[0, :, 0, 0])) > 1e3:
                    F = normalize_power(F, W, H, Pt)
            F = F / torch.abs(F)
            W_new = W.clone().detach()
            grad_W_com = get_grad_W_com(H, F, W)
            grad_W_rad = get_grad_W_rad(F, W, R)
            for k in range(K):
                W_new[k] = (W[k].clone().detach()
                            + model.step_size[0][ii][k+1] * grad_W_com[k] * WEIGHT_W_COM
                            - model.step_size[0][ii][k+1] * grad_W_rad[k] * WEIGHT_W_RAD)
            F, W = normalize(F, W_new, H, Pt)
            r = get_sum_rate(H, F, W, Pt).item()
            t = get_beam_error(H, F, W, R, Pt).item()
            rate_iter.append(r); tau_iter.append(t); obj_iter.append(r - OMEGA * t)
    if label:
        print(f"  {label:45s}  rate={rate_iter[-1]:.4f}  tau={tau_iter[-1]:.6f}  obj={obj_iter[-1]:.4f}")
    return rate_iter, tau_iter, obj_iter


def eval_student_J10(model, n_outer, n_inner, H, R, Pt, label=''):
    """PGA_Unfold_J10_I60. step_size: [J, I, K+1]"""
    ss_sum = model.step_size.data.sum().item()
    rate_iter, tau_iter, obj_iter = [], [], []
    with torch.no_grad():
        _, _, F, W = initialize(H, R, Pt, initial_normalization)
        r = get_sum_rate(H, F, W, Pt).item()
        t = get_beam_error(H, F, W, R, Pt).item()
        rate_iter.append(r); tau_iter.append(t); obj_iter.append(r - OMEGA * t)
        for ii in range(n_outer):
            for jj in range(n_inner):
                grad_F_com = get_grad_F_com(H, F, W)
                grad_F_rad = get_grad_F_rad(F, W, R)
                mu = model.step_size[jj][ii][0]
                F = (F + mu * grad_F_com * WEIGHT_F_COM
                       - mu * grad_F_rad * WEIGHT_F_RAD)
                if sum(torch.abs(F[0, :, 0, 0])) > 1e3:
                    F = normalize_power(F, W, H, Pt)
            F = F / torch.abs(F)
            W_new = W.clone().detach()
            grad_W_com = get_grad_W_com(H, F, W)
            grad_W_rad = get_grad_W_rad(F, W, R)
            for k in range(K):
                lam = model.step_size[0][ii][k+1]
                W_new[k] = (W[k].clone().detach()
                            + lam * grad_W_com[k] * WEIGHT_W_COM
                            - lam * grad_W_rad[k] * WEIGHT_W_RAD)
            F, W = normalize(F, W_new, H, Pt)
            r = get_sum_rate(H, F, W, Pt).item()
            t = get_beam_error(H, F, W, R, Pt).item()
            rate_iter.append(r); tau_iter.append(t); obj_iter.append(r - OMEGA * t)
    if label:
        print(f"  {label:45s}  rate={rate_iter[-1]:.4f}  tau={tau_iter[-1]:.6f}"
              f"  obj={obj_iter[-1]:.4f}  [ss_sum={ss_sum:.4f}]")
    return rate_iter, tau_iter, obj_iter


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*65)
print('LOADING MODELS')
print('='*65)

model_teacher = PGA_Unfold_J20(step_size_UPGA_J20).to(device)
model_teacher.load_state_dict(torch.load(path_teacher, map_location=device))
model_teacher.eval()
print(f"  Teacher loaded   : {path_teacher}")

model_J1 = PGA_Conv(step_size_UPGA_J1).to(device)
model_J1.load_state_dict(torch.load(path_J1, map_location=device))
model_J1.eval()
print(f"  J1 loaded        : {path_J1}")

_ss_student = torch.zeros(J_S, I_S, K + 1)

def load_student(path, label):
    ckpt = torch.load(path, map_location=device)
    m = PGA_Unfold_J10_I60(_ss_student).to(device)
    missing, unexpected = m.load_state_dict(ckpt, strict=False)
    m.eval()
    ss = m.step_size.data
    print(f"  {label:25s}: {path}")
    print(f"    step_size shape={list(ss.shape)}  "
          f"min={ss.min():.4e}  max={ss.max():.4e}  "
          f"mean={ss.mean():.4e}  sum={ss.sum():.4f}")
    if missing:    print(f"    [WARN] missing    : {missing}")
    if unexpected: print(f"    [WARN] unexpected : {unexpected}")
    return m

model_1_1 = load_student(path_cell_1_1, 'Cell 1-1 Flat|NoRKD')
model_1_2 = load_student(path_cell_1_2, 'Cell 1-2 Flat|RKD')
model_2_1 = load_student(path_cell_2_1, 'Cell 2-1 AGT|NoRKD')
model_2_2 = load_student(path_cell_2_2, 'Cell 2-2 AGT|RKD')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: CONVERGENCE
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*65)
print(f'CONVERGENCE EVAL  (SNR = {snr_dB} dB)')
print('='*65)

Rtest, _, _, _ = get_radar_data(snr_dB, H_test.cpu())
Rtest = Rtest.to(device)

rate_T,  tau_T,  obj_T  = eval_teacher(    model_teacher, I_T, J_T, H_test, Rtest, snr, 'Teacher (J20, I120)')
rate_J1, tau_J1, obj_J1 = eval_conv_J1(    model_J1,      I_T,      H_test, Rtest, snr, 'J1 (I120)')
rate_11, tau_11, obj_11 = eval_student_J10(model_1_1, I_S, J_S, H_test, Rtest, snr, 'Cell 1-1 Flat|NoRKD (I60)')
rate_12, tau_12, obj_12 = eval_student_J10(model_1_2, I_S, J_S, H_test, Rtest, snr, 'Cell 1-2 Flat|RKD   (I60)')
rate_21, tau_21, obj_21 = eval_student_J10(model_2_1, I_S, J_S, H_test, Rtest, snr, 'Cell 2-1 AGT|NoRKD  (I60)')
rate_22, tau_22, obj_22 = eval_student_J10(model_2_2, I_S, J_S, H_test, Rtest, snr, 'Cell 2-2 AGT|RKD    (I60)')

x_long  = list(range(I_T + 1))   # 0 … 120
x_short = list(range(I_S + 1))   # 0 … 60

print('\n--- Convergence figures ---')

# ── Helper: compute a tight but sensible y-axis range ────────────────────────
def _iter_ylim(curves, pad_frac=0.08):
    """
    Compute ylim that zooms in on the interesting part of the curves.
    Discards the first 2 points (wild init transient) for range computation,
    then adds a small pad so curves aren't clipped.
    """
    all_vals = []
    for c in curves:
        all_vals.extend(c[2:])          # skip init point and first iter
    lo, hi = min(all_vals), max(all_vals)
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


# ── Objective vs iterations ───────────────────────────────────────────────────
fig, ax = _make_fig()
ax.plot(x_long,  obj_T,   **STYLES_ITER['teacher'])
ax.plot(x_long,  obj_J1,  **STYLES_ITER['J1'])
ax.plot(x_short, obj_11,  **STYLES_ITER['cell_1_1'])
ax.plot(x_short, obj_12,  **STYLES_ITER['cell_1_2'])
ax.plot(x_short, obj_21,  **STYLES_ITER['cell_2_1'])
ax.plot(x_short, obj_22,  **STYLES_ITER['cell_2_2'])
ax.set_xlabel(r'Number of iterations/layers $(I)$', fontsize=FONT['xlabel'])
ax.set_ylabel(r'$R - \omega\bar{\tau}$ [bits/s/Hz]', fontsize=FONT['ylabel'])
ax.set_xlim(0, I_T)
ylo, yhi = _iter_ylim([obj_T, obj_J1, obj_11, obj_12, obj_21, obj_22])
ax.set_ylim(ylo, yhi)
ax.tick_params(labelsize=FONT['tick'])
_outside_legend(fig, STYLES_ITER)
_save(fig, 'ablation_obj_vs_iter')

# ── Rate vs iterations ────────────────────────────────────────────────────────
fig, ax = _make_fig()
ax.plot(x_long,  rate_T,  **STYLES_ITER['teacher'])
ax.plot(x_long,  rate_J1, **STYLES_ITER['J1'])
ax.plot(x_short, rate_11, **STYLES_ITER['cell_1_1'])
ax.plot(x_short, rate_12, **STYLES_ITER['cell_1_2'])
ax.plot(x_short, rate_21, **STYLES_ITER['cell_2_1'])
ax.plot(x_short, rate_22, **STYLES_ITER['cell_2_2'])
ax.set_xlabel(r'Number of iterations/layers $(I)$', fontsize=FONT['xlabel'])
ax.set_ylabel(r'$R$ [bits/s/Hz]', fontsize=FONT['ylabel'])
ax.set_xlim(0, I_T)
ylo, yhi = _iter_ylim([rate_T, rate_J1, rate_11, rate_12, rate_21, rate_22])
ax.set_ylim(ylo, yhi)
ax.tick_params(labelsize=FONT['tick'])
_outside_legend(fig, STYLES_ITER)
_save(fig, 'ablation_rate_vs_iter')

# ── Beam error vs iterations ──────────────────────────────────────────────────
fig, ax = _make_fig()
ax.plot(x_long,  tau_T,  **STYLES_ITER['teacher'])
ax.plot(x_long,  tau_J1, **STYLES_ITER['J1'])
ax.plot(x_short, tau_11, **STYLES_ITER['cell_1_1'])
ax.plot(x_short, tau_12, **STYLES_ITER['cell_1_2'])
ax.plot(x_short, tau_21, **STYLES_ITER['cell_2_1'])
ax.plot(x_short, tau_22, **STYLES_ITER['cell_2_2'])
ax.set_xlabel(r'Number of iterations/layers $(I)$', fontsize=FONT['xlabel'])
ax.set_ylabel(r'$\bar{\tau}$', fontsize=FONT['ylabel'])
ax.set_xlim(0, I_T)
ylo, yhi = _iter_ylim([tau_T, tau_J1, tau_11, tau_12, tau_21, tau_22])
ax.set_ylim(ylo, yhi)
ax.tick_params(labelsize=FONT['tick'])
_outside_legend(fig, STYLES_ITER)
_save(fig, 'ablation_beam_vs_iter')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES 4 & 5: SNR SWEEP
# get_MSE uses steering vectors and returns dB directly.
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*65)
print('SNR SWEEP EVAL')
print('='*65)

rate_vs_snr = {k: np.zeros(len(snr_dB_list)) for k in PLOT_ORDER}
mse_vs_snr  = {k: np.zeros(len(snr_dB_list)) for k in PLOT_ORDER}


def _final_FW_J1(model, n_outer, H, R, Pt):
    with torch.no_grad():
        _, _, F, W = initialize(H, R, Pt, initial_normalization)
        for ii in range(n_outer):
            grad_F_com = get_grad_F_com(H, F, W)
            grad_F_rad = get_grad_F_rad(F, W, R)
            F = (F + model.step_size[ii][0] * grad_F_com * WEIGHT_F_COM
                   - model.step_size[ii][0] * grad_F_rad * WEIGHT_F_RAD)
            if sum(torch.abs(F[0, :, 0, 0])) > 1e1:
                F = normalize_power(F, W, H, Pt)
            F = F / torch.abs(F)
            W_new = W.clone().detach()
            grad_W_com = get_grad_W_com(H, F, W)
            grad_W_rad = get_grad_W_rad(F, W, R)
            for k in range(K):
                W_new[k] = (W[k].clone().detach()
                            + model.step_size[ii][k+1] * grad_W_com[k] * WEIGHT_W_COM
                            - model.step_size[ii][k+1] * grad_W_rad[k] * WEIGHT_W_RAD)
            F, W = normalize(F, W_new, H, Pt)
    return F, W


def _final_FW_teacher(model, n_outer, n_inner, H, R, Pt):
    with torch.no_grad():
        _, _, F, W = initialize(H, R, Pt, initial_normalization)
        for ii in range(n_outer):
            for jj in range(n_inner):
                grad_F_com = get_grad_F_com(H, F, W)
                grad_F_rad = get_grad_F_rad(F, W, R)
                F = (F + model.step_size[jj][ii][0] * grad_F_com * WEIGHT_F_COM
                       - model.step_size[jj][ii][0] * grad_F_rad * WEIGHT_F_RAD)
                if sum(torch.abs(F[0, :, 0, 0])) > 1e3:
                    F = normalize_power(F, W, H, Pt)
            F = F / torch.abs(F)
            W_new = W.clone().detach()
            grad_W_com = get_grad_W_com(H, F, W)
            grad_W_rad = get_grad_W_rad(F, W, R)
            for k in range(K):
                W_new[k] = (W[k].clone().detach()
                            + model.step_size[0][ii][k+1] * grad_W_com[k] * WEIGHT_W_COM
                            - model.step_size[0][ii][k+1] * grad_W_rad[k] * WEIGHT_W_RAD)
            F, W = normalize(F, W_new, H, Pt)
    return F, W


def _final_FW_student(model, n_outer, n_inner, H, R, Pt):
    with torch.no_grad():
        _, _, F, W = initialize(H, R, Pt, initial_normalization)
        for ii in range(n_outer):
            for jj in range(n_inner):
                grad_F_com = get_grad_F_com(H, F, W)
                grad_F_rad = get_grad_F_rad(F, W, R)
                mu = model.step_size[jj][ii][0]
                F = (F + mu * grad_F_com * WEIGHT_F_COM
                       - mu * grad_F_rad * WEIGHT_F_RAD)
                if sum(torch.abs(F[0, :, 0, 0])) > 1e3:
                    F = normalize_power(F, W, H, Pt)
            F = F / torch.abs(F)
            W_new = W.clone().detach()
            grad_W_com = get_grad_W_com(H, F, W)
            grad_W_rad = get_grad_W_rad(F, W, R)
            for k in range(K):
                lam = model.step_size[0][ii][k+1]
                W_new[k] = (W[k].clone().detach()
                            + lam * grad_W_com[k] * WEIGHT_W_COM
                            - lam * grad_W_rad[k] * WEIGHT_W_RAD)
            F, W = normalize(F, W_new, H, Pt)
    return F, W


for ss, snr_dB_ss in enumerate(snr_dB_list):
    snr_ss = 10 ** (snr_dB_ss / 10)
    print(f'\n  SNR = {snr_dB_ss} dB')

    R_ss, at_ss, _, _ = get_radar_data(snr_dB_ss, H_test.cpu())
    R_ss  = R_ss.to(device)
    at_ss = at_ss[:, :test_size, :, :].to(device)

    F, W = _final_FW_teacher(model_teacher, I_T, J_T, H_test, R_ss, snr_ss)
    rate_vs_snr['teacher'][ss] = get_sum_rate(H_test, F, W, snr_ss).item()
    mse_vs_snr['teacher'][ss]  = get_MSE(F, W, at_ss, R_ss, snr_ss).item()
    print(f"  {'Teacher':30s}  rate={rate_vs_snr['teacher'][ss]:.4f}  MSE={mse_vs_snr['teacher'][ss]:.4f} dB")

    F, W = _final_FW_J1(model_J1, I_T, H_test, R_ss, snr_ss)
    rate_vs_snr['J1'][ss] = get_sum_rate(H_test, F, W, snr_ss).item()
    mse_vs_snr['J1'][ss]  = get_MSE(F, W, at_ss, R_ss, snr_ss).item()
    print(f"  {'J1':30s}  rate={rate_vs_snr['J1'][ss]:.4f}  MSE={mse_vs_snr['J1'][ss]:.4f} dB")

    for key, model_s, lbl in [
        ('cell_1_1', model_1_1, 'Cell 1-1 Flat|NoRKD'),
        ('cell_1_2', model_1_2, 'Cell 1-2 Flat|RKD'),
        ('cell_2_1', model_2_1, 'Cell 2-1 AGT|NoRKD'),
        ('cell_2_2', model_2_2, 'Cell 2-2 AGT|RKD'),
    ]:
        F, W = _final_FW_student(model_s, I_S, J_S, H_test, R_ss, snr_ss)
        rate_vs_snr[key][ss] = get_sum_rate(H_test, F, W, snr_ss).item()
        mse_vs_snr[key][ss]  = get_MSE(F, W, at_ss, R_ss, snr_ss).item()
        print(f"  {lbl:30s}  rate={rate_vs_snr[key][ss]:.4f}  MSE={mse_vs_snr[key][ss]:.4f} dB")

print('\n--- SNR figures ---')

# ── Rate vs SNR ───────────────────────────────────────────────────────────────
fig, ax = _make_fig()
for key in PLOT_ORDER:
    ax.plot(snr_dB_list, rate_vs_snr[key], **STYLES_SNR[key])
ax.set_xlabel('SNR [dB]', fontsize=FONT['xlabel'])
ax.set_ylabel(r'$R$ [bits/s/Hz]', fontsize=FONT['ylabel'])
ax.set_xticks(snr_dB_list)
ax.tick_params(labelsize=FONT['tick'])
_outside_legend(fig, STYLES_SNR)
_save(fig, 'ablation_rate_vs_SNR')

# ── Beam MSE vs SNR ───────────────────────────────────────────────────────────
fig, ax = _make_fig()
for key in PLOT_ORDER:
    ax.plot(snr_dB_list, mse_vs_snr[key], **STYLES_SNR[key])
ax.set_xlabel('SNR [dB]', fontsize=FONT['xlabel'])
ax.set_ylabel('Average radar beampattern MSE [dB]', fontsize=FONT['ylabel'])
ax.set_xticks(snr_dB_list)
ax.tick_params(labelsize=FONT['tick'])
_outside_legend(fig, STYLES_SNR)
_save(fig, 'ablation_beam_vs_SNR')

print('\nDone.')
