"""
Ablation Plotting Script v2
============================
7 curves per convergence figure, using execute_PGA directly.
Uses get_sum_rate / get_beam_error exactly as original code does.

Curves:
  R_T          : Teacher (J=20, I=120)
  R_{J1}       : UPGA J=1, I=120 (baseline)
  R_{J1,AGT}   : UPGA J=1 + AGT init (J=1 student)
  R_flat       : Flat init | No RKD  (J=10, I=60)
  R_flat+RKD   : Flat init | +CI-RKD (J=10, I=60)
  R_AGT        : AGT init  | No RKD  (J=10, I=60)
  R_AGT+RKD    : AGT init  | +CI-RKD (J=10, I=60)
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")

H_train, H_test0 = get_data_tensor(data_source)
H_test = H_test0[:, :test_size, :, :].to(device)

I_T = n_iter_outer        # 120
I_S = n_iter_outer // 2   # 60
J_T = n_iter_inner_J20    # 20
J_S = n_iter_inner_J10    # 10

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT PATHS
# ══════════════════════════════════════════════════════════════════════════════

path_teacher  = './model/UPGA_J20_320.pth'
path_J1       = './model/UPGA_J1.pth'
path_J1_AGT   = './model/UPGA_J1.pth_I120_AGT_teacher_avg_inner.pth'  # J1 + AGT, I=120

path_cell_1_1 = './model/UPGA_J10_I60.pth'
path_cell_1_2 = './model/UPGA_J10_all.pth_I60_noinit_CI_RKD_dist25.0_angle50.0_Kl15_win20.pth'
path_cell_2_1 = './model/UPGA_J10.pth_I60_init_sym_inner_avg_pairs_Kl15_win20.pth'
path_cell_2_2 = './model/UPGA_J10.pth_I60_CI_RKD_sym_inner_avg_pairs_Kl15_win20_b10.pth'

path_cell_1_1 = './model/UPGA_J10_I60.pth'  # Flat init | No RKD
path_cell_1_2 = './model/UPGA_J10_320.pth_I60_noinit_CI_RKD_dist25.0_angle50.0_Kl15_win20new.pth'         # Flat init | + CI-RKD
path_cell_2_1 = './model/UPGA_J10.pth_I60_init_sym_inner_avg_pairs_Kl15_win20.pth'             
# AGT init  | No RKD
path_cell_2_2 = './model/UPGA_J10.pth_I60_CI_RKD_sym_inner_avg_pairs_Kl15_win20_b10.pth'            # AGT init  | + CI-RKD



path_cell_1_2 = './model/UPGA_J10_all.pth_I60_noinit_CI_RKD_dist25.0_angle50.0_Kl15_win20.pth' # Flat init | + CI-RKD
path_cell_2_1 = './model/UPGA_J10.pth_I60_init_sym_inner_avg_pairs_Kl15_win20.pth'             # AGT init  | No RKD
path_cell_2_2 = './model/UPGA_J10.pth_I60_CI_RKD_sym_inner_avg_pairs_Kl15_win20_b10.pth'           # AGT init  | + CI-RKD
# ══════════════════════════════════════════════════════════════════════════════
# STYLES — 7 curves
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    'teacher'  : '#2CA02C',
    'J1'       : '#FF7F0E',
    'J1_AGT'   : '#17BECF',
    'cell_1_1' : '#1F77B4',
    'cell_1_2' : '#E377C2',
    'cell_2_1' : '#D62728',
    'cell_2_2' : '#9467BD',
}

def _make_styles(me_long, me_short):
    return {
        'teacher'  : dict(color=COLORS['teacher'],  ls='-',       lw=2.5,
                          marker='o', markevery=me_long,  markersize=7,
                          label=r'$R_T$'),
        'J1'       : dict(color=COLORS['J1'],        ls='--',      lw=2.2,
                          marker='s', markevery=me_long,  markersize=7,
                          label=r'$R_{J_1}$'),
        'J1_AGT'   : dict(color=COLORS['J1_AGT'],   ls='-.',      lw=2.2,
                          marker='p', markevery=me_long,  markersize=7,
                          label=r'$R_{J_1,\mathrm{AGT}}$'),
        'cell_1_1' : dict(color=COLORS['cell_1_1'], ls='-.',      lw=2.0,
                          marker='^', markevery=me_short, markersize=7,
                          label=r'$R_\mathrm{flat}$'),
        'cell_1_2' : dict(color=COLORS['cell_1_2'], ls=':',       lw=2.0,
                          marker='v', markevery=me_short, markersize=7,
                          label=r'$R_\mathrm{flat+RKD}$'),
        'cell_2_1' : dict(color=COLORS['cell_2_1'], ls=(0,(5,2)), lw=2.0,
                          marker='D', markevery=me_short, markersize=7,
                          label=r'$R_\mathrm{AGT}$'),
        'cell_2_2' : dict(color=COLORS['cell_2_2'], ls='-',       lw=2.8,
                          marker='*', markevery=me_short, markersize=9,
                          label=r'$R_\mathrm{AGT+RKD}$'),
    }

STYLES_ITER = _make_styles(me_long=8, me_short=6)
STYLES_SNR  = _make_styles(me_long=1, me_short=1)

PLOT_ORDER = ['teacher', 'J1', 'J1_AGT', 'cell_1_1', 'cell_1_2', 'cell_2_1', 'cell_2_2']

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

def _make_fig():
    fig, ax = plt.subplots(figsize=(8, 5.8))
    return fig, ax

def _make_handles(style_dict, keys=None):
    if keys is None:
        keys = PLOT_ORDER
    return [mlines.Line2D([], [],
                color=style_dict[k]['color'], ls=style_dict[k]['ls'],
                lw=style_dict[k]['lw'], marker=style_dict[k]['marker'],
                markersize=style_dict[k]['markersize'],
                label=style_dict[k]['label'])
            for k in keys]

def _inside_legend(ax, style_dict, keys=None, loc='best'):
    ax.legend(handles=_make_handles(style_dict, keys),
              loc=loc,
              fontsize=FONT['legend'],
              framealpha=0.85,
              edgecolor='#aaaaaa',
              fancybox=False,
              frameon=True,
              ncol=1)

def _save(fig, stem):
    for ext in ('.png', '.eps'):
        path = os.path.join(directory_result, stem + ext)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved → {path}")
    plt.close(fig)

def _iter_ylim(curves, skip=2, pad_frac=0.08):
    """Zoom y-axis: skip first `skip` points (init transient), add pad."""
    all_vals = []
    for c in curves:
        all_vals.extend(c[skip:])
    lo, hi = min(all_vals), max(all_vals)
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*65)
print('LOADING MODELS')
print('='*65)

model_teacher = PGA_Unfold_J20(step_size_UPGA_J20).to(device)
model_teacher.load_state_dict(torch.load(path_teacher, map_location=device))
model_teacher.eval()
print(f"  Teacher          : {path_teacher}")

model_J1 = PGA_Conv(step_size_UPGA_J1).to(device)
model_J1.load_state_dict(torch.load(path_J1, map_location=device))
model_J1.eval()
print(f"  J1               : {path_J1}")

# J1 + AGT: PGA_Conv with I_T=120 step_size
_ss_J1_AGT = torch.zeros(I_T, K + 1)
model_J1_AGT = PGA_Conv(_ss_J1_AGT).to(device)
model_J1_AGT.load_state_dict(torch.load(path_J1_AGT, map_location=device))
model_J1_AGT.eval()
print(f"  J1 AGT (I=120)   : {path_J1_AGT}")

_ss_student = torch.zeros(J_S, I_S, K + 1)

def load_student(path, label):
    m = PGA_Unfold_J10_I60(_ss_student).to(device)
    ckpt = torch.load(path, map_location=device)
    missing, unexpected = m.load_state_dict(ckpt, strict=False)
    m.eval()
    ss = m.step_size.data
    print(f"  {label:25s}: {path}")
    print(f"    min={ss.min():.4e}  max={ss.max():.4e}  sum={ss.sum():.4f}")
    return m

model_1_1 = load_student(path_cell_1_1, 'Cell 1-1 Flat|NoRKD')
model_1_2 = load_student(path_cell_1_2, 'Cell 1-2 Flat|RKD')
model_2_1 = load_student(path_cell_2_1, 'Cell 2-1 AGT|NoRKD')
model_2_2 = load_student(path_cell_2_2, 'Cell 2-2 AGT|RKD')

# ══════════════════════════════════════════════════════════════════════════════
# RUN CONVERGENCE — using execute_PGA directly (matches Doc 14)
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*65)
print(f'CONVERGENCE EVAL  (SNR = {snr_dB} dB)')
print('='*65)

R_conv, _, _, _ = get_radar_data(snr_dB, H_test.cpu())
R_conv = R_conv.to(device)

def _to_lists(rate_t, tau_t, n_samples):
    rate_list = [r.detach().cpu().numpy() for r in (sum(rate_t) / n_samples)]
    tau_list  = [e.detach().cpu().numpy() for e in (sum(tau_t)  / n_samples)]
    obj_list  = [r - OMEGA * t for r, t in zip(rate_list, tau_list)]
    return rate_list, tau_list, obj_list

n = len(H_test[0])

print('Running Teacher...')
rate_t, tau_t, _, _ = model_teacher.execute_PGA(H_test, R_conv, snr, I_T, J_T)
rate_T, tau_T, obj_T = _to_lists(rate_t, tau_t, n)

print('Running J1...')
rate_t, tau_t, _, _ = model_J1.execute_PGA(H_test, R_conv, snr, I_T)
rate_J1, tau_J1, obj_J1 = _to_lists(rate_t, tau_t, n)

print('Running J1 AGT...')
rate_t, tau_t, _, _ = model_J1_AGT.execute_PGA(H_test, R_conv, snr, I_S)
rate_J1_AGT, tau_J1_AGT, obj_J1_AGT = _to_lists(rate_t, tau_t, n)

print('Running Cell 1-1 (Flat|NoRKD)...')
rate_t, tau_t, _, _ = model_1_1.execute_PGA(H_test, R_conv, snr, I_S, J_S)
rate_11, tau_11, obj_11 = _to_lists(rate_t, tau_t, n)

print('Running Cell 1-2 (Flat|RKD)...')
rate_t, tau_t, _, _ = model_1_2.execute_PGA(H_test, R_conv, snr, I_S, J_S)
rate_12, tau_12, obj_12 = _to_lists(rate_t, tau_t, n)

print('Running Cell 2-1 (AGT|NoRKD)...')
rate_t, tau_t, _, _ = model_2_1.execute_PGA(H_test, R_conv, snr, I_S, J_S)
rate_21, tau_21, obj_21 = _to_lists(rate_t, tau_t, n)

print('Running Cell 2-2 (AGT|RKD)...')
rate_t, tau_t, _, _ = model_2_2.execute_PGA(H_test, R_conv, snr, I_S, J_S)
rate_22, tau_22, obj_22 = _to_lists(rate_t, tau_t, n)

for lbl, r, t, o in [
    ('Teacher',      rate_T,       tau_T,       obj_T),
    ('J1',           rate_J1,      tau_J1,       obj_J1),
    ('J1 AGT',       rate_J1_AGT,  tau_J1_AGT,   obj_J1_AGT),
    ('Flat|NoRKD',   rate_11,      tau_11,       obj_11),
    ('Flat|RKD',     rate_12,      tau_12,       obj_12),
    ('AGT|NoRKD',    rate_21,      tau_21,       obj_21),
    ('AGT|RKD',      rate_22,      tau_22,       obj_22),
]:
    print(f"  {lbl:20s}  rate={r[-1]:.4f}  tau={t[-1]:.6f}  obj={o[-1]:.4f}")

x_long  = list(range(I_T + 1))
x_short = list(range(I_S + 1))

print('\n--- Convergence figures ---')

# ── Objective vs iterations ───────────────────────────────────────────────────
fig, ax = _make_fig()
ax.plot(x_long,  obj_T,       **STYLES_ITER['teacher'])
ax.plot(x_long,  obj_J1,      **STYLES_ITER['J1'])
ax.plot(x_short, obj_J1_AGT,  **STYLES_ITER['J1_AGT'])
ax.plot(x_short, obj_11,      **STYLES_ITER['cell_1_1'])
ax.plot(x_short, obj_12,      **STYLES_ITER['cell_1_2'])
ax.plot(x_short, obj_21,      **STYLES_ITER['cell_2_1'])
ax.plot(x_short, obj_22,      **STYLES_ITER['cell_2_2'])
ax.set_xlabel(r'Number of iterations/layers $(I)$', fontsize=FONT['xlabel'])
ax.set_ylabel(r'$R - \omega\bar{\tau}$ [bits/s/Hz]', fontsize=FONT['ylabel'])
ax.set_xlim(0, I_T)
ylo, yhi = _iter_ylim([obj_T, obj_J1, obj_J1_AGT, obj_11, obj_12, obj_21, obj_22])
ax.set_ylim(ylo, yhi)
ax.tick_params(labelsize=FONT['tick'])
_inside_legend(ax, STYLES_ITER)
_save(fig, 'ablation_obj_vs_iter')

# ── Rate vs iterations ────────────────────────────────────────────────────────
fig, ax = _make_fig()
ax.plot(x_long,  rate_T,       **STYLES_ITER['teacher'])
ax.plot(x_long,  rate_J1,      **STYLES_ITER['J1'])
ax.plot(x_short, rate_J1_AGT,  **STYLES_ITER['J1_AGT'])
ax.plot(x_short, rate_11,      **STYLES_ITER['cell_1_1'])
ax.plot(x_short, rate_12,      **STYLES_ITER['cell_1_2'])
ax.plot(x_short, rate_21,      **STYLES_ITER['cell_2_1'])
ax.plot(x_short, rate_22,      **STYLES_ITER['cell_2_2'])
ax.set_xlabel(r'Number of iterations/layers $(I)$', fontsize=FONT['xlabel'])
ax.set_ylabel(r'$R$ [bits/s/Hz]', fontsize=FONT['ylabel'])
ax.set_xlim(0, I_T)
ylo, yhi = _iter_ylim([rate_T, rate_J1, rate_J1_AGT, rate_11, rate_12, rate_21, rate_22])
ax.set_ylim(ylo, yhi)
ax.tick_params(labelsize=FONT['tick'])
_inside_legend(ax, STYLES_ITER)
_save(fig, 'ablation_rate_vs_iter')

# ── Beam error vs iterations ──────────────────────────────────────────────────
fig, ax = _make_fig()
ax.plot(x_long,  tau_T,       **STYLES_ITER['teacher'])
ax.plot(x_long,  tau_J1,      **STYLES_ITER['J1'])
ax.plot(x_short, tau_J1_AGT,  **STYLES_ITER['J1_AGT'])
ax.plot(x_short, tau_11,      **STYLES_ITER['cell_1_1'])
ax.plot(x_short, tau_12,      **STYLES_ITER['cell_1_2'])
ax.plot(x_short, tau_21,      **STYLES_ITER['cell_2_1'])
ax.plot(x_short, tau_22,      **STYLES_ITER['cell_2_2'])
ax.set_xlabel(r'Number of iterations/layers $(I)$', fontsize=FONT['xlabel'])
ax.set_ylabel(r'$\bar{\tau}$', fontsize=FONT['ylabel'])
ax.set_xlim(0, I_T)
ylo, yhi = _iter_ylim([tau_T, tau_J1, tau_J1_AGT, tau_11, tau_12, tau_21, tau_22])
ax.set_ylim(ylo, yhi)
ax.tick_params(labelsize=FONT['tick'])
_inside_legend(ax, STYLES_ITER)
_save(fig, 'ablation_beam_vs_iter')

# ══════════════════════════════════════════════════════════════════════════════
# SNR SWEEP — get_MSE returns dB directly
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*65)
print('SNR SWEEP EVAL')
print('='*65)

rate_vs_snr = {k: np.zeros(len(snr_dB_list)) for k in PLOT_ORDER}
mse_vs_snr  = {k: np.zeros(len(snr_dB_list)) for k in PLOT_ORDER}

for ss, snr_dB_ss in enumerate(snr_dB_list):
    snr_ss = 10 ** (snr_dB_ss / 10)
    print(f'\n  SNR = {snr_dB_ss} dB')
    R_ss, at_ss, _, _ = get_radar_data(snr_dB_ss, H_test.cpu())
    R_ss  = R_ss.to(device)
    at_ss = at_ss[:, :test_size, :, :].to(device)

    def _snr_eval_J20(model):
        _, _, F, W = model.execute_PGA(H_test, R_ss, snr_ss, I_T, J_T)
        return get_sum_rate(H_test, F, W, snr_ss).item(), get_MSE(F, W, at_ss, R_ss, snr_ss).item()

    def _snr_eval_J1(model, n_outer):
        _, _, F, W = model.execute_PGA(H_test, R_ss, snr_ss, n_outer)
        return get_sum_rate(H_test, F, W, snr_ss).item(), get_MSE(F, W, at_ss, R_ss, snr_ss).item()

    def _snr_eval_J10(model):
        _, _, F, W = model.execute_PGA(H_test, R_ss, snr_ss, I_S, J_S)
        return get_sum_rate(H_test, F, W, snr_ss).item(), get_MSE(F, W, at_ss, R_ss, snr_ss).item()

    with torch.no_grad():
        rate_vs_snr['teacher'][ss], mse_vs_snr['teacher'][ss] = _snr_eval_J20(model_teacher)
        print(f"  {'Teacher':25s}  rate={rate_vs_snr['teacher'][ss]:.4f}  MSE={mse_vs_snr['teacher'][ss]:.4f} dB")

        rate_vs_snr['J1'][ss], mse_vs_snr['J1'][ss] = _snr_eval_J1(model_J1, I_T)
        print(f"  {'J1':25s}  rate={rate_vs_snr['J1'][ss]:.4f}  MSE={mse_vs_snr['J1'][ss]:.4f} dB")

        rate_vs_snr['J1_AGT'][ss], mse_vs_snr['J1_AGT'][ss] = _snr_eval_J1(model_J1_AGT, I_S)
        print(f"  {'J1 AGT':25s}  rate={rate_vs_snr['J1_AGT'][ss]:.4f}  MSE={mse_vs_snr['J1_AGT'][ss]:.4f} dB")

        for key, model_s, lbl in [
            ('cell_1_1', model_1_1, 'Flat|NoRKD'),
            ('cell_1_2', model_1_2, 'Flat|RKD'),
            ('cell_2_1', model_2_1, 'AGT|NoRKD'),
            ('cell_2_2', model_2_2, 'AGT|RKD'),
        ]:
            rate_vs_snr[key][ss], mse_vs_snr[key][ss] = _snr_eval_J10(model_s)
            print(f"  {lbl:25s}  rate={rate_vs_snr[key][ss]:.4f}  MSE={mse_vs_snr[key][ss]:.4f} dB")

print('\n--- SNR figures ---')

# ── Rate vs SNR ───────────────────────────────────────────────────────────────
fig, ax = _make_fig()
for key in PLOT_ORDER:
    ax.plot(snr_dB_list, rate_vs_snr[key], **STYLES_SNR[key])
ax.set_xlabel('SNR [dB]', fontsize=FONT['xlabel'])
ax.set_ylabel(r'$R$ [bits/s/Hz]', fontsize=FONT['ylabel'])
ax.set_xticks(snr_dB_list)
ax.tick_params(labelsize=FONT['tick'])
_inside_legend(ax, STYLES_SNR)
_save(fig, 'ablation_rate_vs_SNR')

# ── MSE vs SNR ────────────────────────────────────────────────────────────────
fig, ax = _make_fig()
for key in PLOT_ORDER:
    ax.plot(snr_dB_list, mse_vs_snr[key], **STYLES_SNR[key])
ax.set_xlabel('SNR [dB]', fontsize=FONT['xlabel'])
ax.set_ylabel('Average radar beampattern MSE [dB]', fontsize=FONT['ylabel'])
ax.set_xticks(snr_dB_list)
ax.tick_params(labelsize=FONT['tick'])
_inside_legend(ax, STYLES_SNR)
_save(fig, 'ablation_beam_vs_SNR')

print('\nDone.')