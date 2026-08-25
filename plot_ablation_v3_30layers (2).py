from PGA_models import *
import random
import numpy as np
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ── Load data ─────────────────────────────────────────────────────────────────
H_train, H_test0 = get_data_tensor(data_source)
H_test = H_test0[:, :test_size, :, :]

R, at0, theta, ideal_beam = get_radar_data(snr_dB, H_test)
at = at0[:, :test_size, :, :]

iter_number_UPGA_J20     = np.array(list(range(n_iter_outer + 1)))
iter_number_UPGA_J10_I60 = np.array(list(range(60 + 1)))
iter_number_J10_I120_trunc = np.array(list(range(60 + 1)))

# ── Define v2 student model class (from Unified_v2_30layers.py) ───────────────
# This class collects per-layer rate/tau and two trajectory windows.
# execute_PGA_with_windows returns:
#   rates, taus, F, W, F_first, W_first, F_last, W_last,
#   rate_over_iters, tau_over_iters
# For evaluation we only need rates, taus, F, W so we ignore the rest.
class PGA_Unfold_J10_I60_v2(nn.Module):
    def __init__(self, step_size_init):
        super().__init__()
        if isinstance(step_size_init, torch.Tensor) and step_size_init.dim() == 3:
            self.step_size = nn.Parameter(step_size_init.float().clone())
        else:
            self.step_size = nn.Parameter(
                step_size_init * torch.ones(n_iter_inner_J10, 60, K + 1))

    def execute_PGA_with_windows(self, H, R, Pt, n_iter_outer_run,
                                  n_iter_inner, K_layers=0,
                                  collect_windows=False):
        rate_init, tau_init, F, W = initialize(H, R, Pt, initial_normalization)
        rate_over_iters = torch.zeros(n_iter_outer_run, H.shape[1])
        tau_over_iters  = torch.zeros(n_iter_outer_run, H.shape[1])
        F_first, W_first, F_last, W_last = [], [], [], []

        for ii in range(n_iter_outer_run):
            for jj in range(n_iter_inner):
                grad_F_com = get_grad_F_com(H, F, W)
                grad_F_rad = get_grad_F_rad(F, W, R)
                F = F + self.step_size[jj][ii][0] * (
                    grad_F_com * WEIGHT_F_COM - grad_F_rad * WEIGHT_F_RAD)
                F = normalize_power(F, W, H, Pt)
            F = F / torch.abs(F)
            W_new = W.clone().detach()
            grad_W_com = get_grad_W_com(H, F, W)
            grad_W_rad = get_grad_W_rad(F, W, R)
            for k in range(K):
                W_new[k] = (W[k].clone().detach()
                            + self.step_size[0][ii][k+1]
                            * (grad_W_com[k] * WEIGHT_W_COM
                               - grad_W_rad[k] * WEIGHT_W_RAD))
            F, W = normalize(F, W_new, H, Pt)
            rate_over_iters[ii] = get_sum_rate(H, F, W, Pt)
            tau_over_iters[ii]  = get_beam_error(H, F, W, R, Pt)

        rates = torch.cat([rate_init, rate_over_iters], dim=0)
        taus  = torch.cat([tau_init,  tau_over_iters],  dim=0)
        return (torch.transpose(rates, 0, 1),
                torch.transpose(taus,  0, 1),
                F, W,
                F_first, W_first, F_last, W_last,
                rate_over_iters, tau_over_iters)

    def execute_PGA(self, H, R, Pt, n_iter_outer_run, n_iter_inner):
        """Convenience wrapper returning same format as original models."""
        out = self.execute_PGA_with_windows(
            H, R, Pt, n_iter_outer_run, n_iter_inner,
            K_layers=0, collect_windows=False)
        return out[0], out[1], out[2], out[3]

print('Loading models...')

# ── Teacher (J=20, I=120) ─────────────────────────────────────────────────────
model_UPGA_J20 = PGA_Unfold_J20(step_size_UPGA_J20)
model_UPGA_J20.load_state_dict(torch.load(model_file_name_UPGA_J20, map_location='cpu'))
model_UPGA_J20.eval()
with torch.no_grad():
    sum_rate_J20, tau_J20, _, _ = model_UPGA_J20.execute_PGA(
        H_test, R, snr, n_iter_outer, n_iter_inner_J20)
rate_iter_J20 = [r.detach().numpy() for r in (sum(sum_rate_J20) / len(H_test[0]))]
tau_iter_J20  = [e.detach().numpy() for e in (sum(tau_J20)      / len(H_test[0]))]

# ── Conventional PGA (J=1, I=120) ────────────────────────────────────────────
model_UPGA_J1 = PGA_Conv(step_size_UPGA_J1)
model_UPGA_J1.load_state_dict(torch.load('./model/UPGA_J1.pth', map_location='cpu'))
model_UPGA_J1.eval()
with torch.no_grad():
    sum_rate_J1, tau_J1, _, _ = model_UPGA_J1.execute_PGA(
        H_test, R, snr, n_iter_outer)
rate_iter_J1 = [r.detach().numpy() for r in (sum(sum_rate_J1) / len(H_test[0]))]
tau_iter_J1  = [e.detach().numpy() for e in (sum(tau_J1)      / len(H_test[0]))]

# ── Conventional PGA + GI ─────────────────────────────────────────────────────
_ckpt_J1_AGT     = torch.load('./model/UPGA_J1.pth_I120_AGT_teacher_avg_inner.pth',
                               map_location='cpu')
_actual_I_J1_AGT = _ckpt_J1_AGT['step_size'].shape[0]
_ss_J1_AGT       = torch.zeros(_actual_I_J1_AGT, K + 1)
model_J1_AGT     = PGA_Conv(_ss_J1_AGT)
model_J1_AGT.load_state_dict(_ckpt_J1_AGT)
model_J1_AGT.eval()
with torch.no_grad():
    sum_rate_J1_AGT, tau_J1_AGT, _, _ = model_J1_AGT.execute_PGA(
        H_test, R, snr, _actual_I_J1_AGT)
rate_iter_J1_AGT = [r.detach().numpy() for r in (sum(sum_rate_J1_AGT) / len(H_test[0]))]
tau_iter_J1_AGT  = [e.detach().numpy() for e in (sum(tau_J1_AGT)      / len(H_test[0]))]
iter_number_J1_AGT = np.array(list(range(_actual_I_J1_AGT + 1)))

# ── J10/I120 truncated at I=60 ────────────────────────────────────────────────
print('Loading J10/I120 model (truncated to I=60)...')
model_J10_I120 = PGA_Unfold_J10(step_size_UPGA_J10)
model_J10_I120.load_state_dict(torch.load(
    './model/UPGA_J10_all.pth',
    map_location='cpu'))
model_J10_I120.eval()
with torch.no_grad():
    sum_rate_J10_I120, tau_J10_I120, _, _ = model_J10_I120.execute_PGA(
        H_test, R, snr, n_iter_outer, n_iter_inner_J10)
rate_iter_J10_trunc = [r.detach().numpy()
                        for r in (sum(sum_rate_J10_I120) / len(H_test[0]))][:61]
tau_iter_J10_trunc  = [e.detach().numpy()
                        for e in (sum(tau_J10_I120)      / len(H_test[0]))][:61]

# ══════════════════════════════════════════════════════════════════════════════
# ORIGINAL ABLATION MODELS (v1)
# ══════════════════════════════════════════════════════════════════════════════

# Cell 2-2: AGT + CI-RKD (original proposed)
model_RKD = PGA_Unfold_J10_I60(step_size_UPGA_J10_I60)
model_RKD.load_state_dict(torch.load(
    './model/UPGA_J10.pth_I60_CI_RKD_sym_inner_avg_pairs_Kl15_win20.pth',
    map_location='cpu'))
model_RKD.eval()
with torch.no_grad():
    sum_rate_RKD, tau_RKD, _, _ = model_RKD.execute_PGA(
        H_test, R, snr, 60, n_iter_inner_J10)
rate_iter_RKD = [r.detach().numpy() for r in (sum(sum_rate_RKD) / len(H_test[0]))]
tau_iter_RKD  = [e.detach().numpy() for e in (sum(tau_RKD)      / len(H_test[0]))]

# Cell 2-1: AGT, no RKD
model_RKD1 = PGA_Unfold_J10_I60(step_size_UPGA_J10_I60)
model_RKD1.load_state_dict(torch.load(
    './model/UPGA_J10.pth_I60_init_sym_inner_avg_pairs_Kl15_win20.pth',
    map_location='cpu'))
model_RKD1.eval()
with torch.no_grad():
    sum_rate_RKD1, tau_RKD1, _, _ = model_RKD1.execute_PGA(
        H_test, R, snr, 60, n_iter_inner_J10)
rate_iter_RKD1 = [r.detach().numpy() for r in (sum(sum_rate_RKD1) / len(H_test[0]))]
tau_iter_RKD1  = [e.detach().numpy() for e in (sum(tau_RKD1)      / len(H_test[0]))]

# Cell 1-2: flat + CI-RKD
model_RKD2 = PGA_Unfold_J10_I60(step_size_UPGA_J10_I60)
model_RKD2.load_state_dict(torch.load(
    './model/UPGA_J10_all.pth_I60_noinit_CI_RKD_dist25.0_angle50.0_Kl15_win20.pth',
    map_location='cpu'))
model_RKD2.eval()
with torch.no_grad():
    sum_rate_RKD2, tau_RKD2, _, _ = model_RKD2.execute_PGA(
        H_test, R, snr, 60, n_iter_inner_J10)
rate_iter_RKD2 = [r.detach().numpy() for r in (sum(sum_rate_RKD2) / len(H_test[0]))]
tau_iter_RKD2  = [e.detach().numpy() for e in (sum(tau_RKD2)      / len(H_test[0]))]

# ══════════════════════════════════════════════════════════════════════════════
# NEW V2 30-LAYER MODELS
# ══════════════════════════════════════════════════════════════════════════════
# These use PGA_Unfold_J10_I60_v2 with the new combined loss:
#   L_total = L_task + L_log (all 60 layers) + L_CI (30 layers, two windows)
# ─────────────────────────────────────────────────────────────────────────────

# v2 cell_1: flat init, no RKD, no log (task loss only)
model_v2_c1 = PGA_Unfold_J10_I60_v2(step_size_UPGA_J10_I60)
model_v2_c1.load_state_dict(torch.load('./model/UPGA_J10_320.pth_I60_cell_1_flat0.01_noRKD_Kl15_win20.pth', map_location='cpu'))
model_v2_c1.eval()
with torch.no_grad():
    sum_rate_v2c1, tau_v2c1, _, _ = model_v2_c1.execute_PGA(
        H_test, R, snr, 60, n_iter_inner_J10)
rate_iter_v2c1 = [r.detach().numpy() for r in (sum(sum_rate_v2c1) / len(H_test[0]))]
tau_iter_v2c1  = [e.detach().numpy() for e in (sum(tau_v2c1)      / len(H_test[0]))]

# v2 cell_2: flat init + RKD + log
model_v2_c2 = PGA_Unfold_J10_I60_v2(step_size_UPGA_J10_I60)
model_v2_c2.load_state_dict(torch.load('./model/UPGA_J10_320.pth_I60_cell_2_flat0.01_RKDlog_30layers_Kl15_win20.pth',map_location='cpu'))
model_v2_c2.eval()
with torch.no_grad():
    sum_rate_v2c2, tau_v2c2, _, _ = model_v2_c2.execute_PGA(
        H_test, R, snr, 60, n_iter_inner_J10)
rate_iter_v2c2 = [r.detach().numpy() for r in (sum(sum_rate_v2c2) / len(H_test[0]))]
tau_iter_v2c2  = [e.detach().numpy() for e in (sum(tau_v2c2)      / len(H_test[0]))]

# v2 cell_3: AGT init, no RKD, no log (task loss only)
model_v2_c3 = PGA_Unfold_J10_I60_v2(step_size_UPGA_J10_I60)
model_v2_c3.load_state_dict(torch.load('./model/UPGA_J10_320.pth_I60_cell_3_AGT_avg_pairs_noRKD_Kl15_win20.pth',map_location='cpu'))
model_v2_c3.eval()
with torch.no_grad():
    sum_rate_v2c3, tau_v2c3, _, _ = model_v2_c3.execute_PGA(
        H_test, R, snr, 60, n_iter_inner_J10)
rate_iter_v2c3 = [r.detach().numpy() for r in (sum(sum_rate_v2c3) / len(H_test[0]))]
tau_iter_v2c3  = [e.detach().numpy() for e in (sum(tau_v2c3)      / len(H_test[0]))]

# v2 cell_4: AGT init + RKD + log (proposed v2)
model_v2_c4 = PGA_Unfold_J10_I60_v2(step_size_UPGA_J10_I60)
model_v2_c4.load_state_dict(torch.load('./model/UPGA_J10_320.pth_I60_cell_4_AGT_avg_pairs_RKDlog_30layers_Kl15_win20.pth', map_location='cpu'))
model_v2_c4.eval()
with torch.no_grad():
    sum_rate_v2c4, tau_v2c4, _, _ = model_v2_c4.execute_PGA(
        H_test, R, snr, 60, n_iter_inner_J10)
rate_iter_v2c4 = [r.detach().numpy() for r in (sum(sum_rate_v2c4) / len(H_test[0]))]
tau_iter_v2c4  = [e.detach().numpy() for e in (sum(tau_v2c4)      / len(H_test[0]))]

print('All models loaded.\n')

# ── Compute objectives ────────────────────────────────────────────────────────
obj_iter_J20       = [r - OMEGA * t for r, t in zip(rate_iter_J20,       tau_iter_J20)]
obj_iter_J1        = [r - OMEGA * t for r, t in zip(rate_iter_J1,        tau_iter_J1)]
obj_iter_J1_AGT    = [r - OMEGA * t for r, t in zip(rate_iter_J1_AGT,    tau_iter_J1_AGT)]
obj_iter_J10_trunc = [r - OMEGA * t for r, t in zip(rate_iter_J10_trunc, tau_iter_J10_trunc)]
obj_iter_RKD       = [r - OMEGA * t for r, t in zip(rate_iter_RKD,       tau_iter_RKD)]
obj_iter_RKD1      = [r - OMEGA * t for r, t in zip(rate_iter_RKD1,      tau_iter_RKD1)]
obj_iter_RKD2      = [r - OMEGA * t for r, t in zip(rate_iter_RKD2,      tau_iter_RKD2)]
obj_iter_v2c1      = [r - OMEGA * t for r, t in zip(rate_iter_v2c1,      tau_iter_v2c1)]
obj_iter_v2c2      = [r - OMEGA * t for r, t in zip(rate_iter_v2c2,      tau_iter_v2c2)]
obj_iter_v2c3      = [r - OMEGA * t for r, t in zip(rate_iter_v2c3,      tau_iter_v2c3)]
obj_iter_v2c4      = [r - OMEGA * t for r, t in zip(rate_iter_v2c4,      tau_iter_v2c4)]

x_long  = iter_number_UPGA_J20
x_short = iter_number_UPGA_J10_I60
x_trunc = iter_number_J10_I120_trunc

# ── Print final values ────────────────────────────────────────────────────────
print('\n' + '='*80)
print(f'Final values at SNR = {snr_dB} dB,  omega = {OMEGA}')
print('='*80)
print(f'  {"Model":<45} {"Obj":>8} {"R [bits/s/Hz]":>15} {"Beam Error":>10}')
print(f'  {"-"*80}')
results = {
    'Teacher (J=20, I=120)'                  : (obj_iter_J20[-1],       rate_iter_J20[-1],       tau_iter_J20[-1]),
    'Conventional PGA'                        : (obj_iter_J1[-1],        rate_iter_J1[-1],        tau_iter_J1[-1]),
    'Conventional PGA+GI'                     : (obj_iter_J1_AGT[-1],    rate_iter_J1_AGT[-1],    tau_iter_J1_AGT[-1]),
    'J10/I120 truncated @ I=60'               : (obj_iter_J10_trunc[-1], rate_iter_J10_trunc[-1], tau_iter_J10_trunc[-1]),
    '── v1 ──'                                : (None, None, None),
    'v1: Student+CI-RKD'                      : (obj_iter_RKD2[-1],      rate_iter_RKD2[-1],      tau_iter_RKD2[-1]),
    'v1: Student+GI'                          : (obj_iter_RKD1[-1],      rate_iter_RKD1[-1],      tau_iter_RKD1[-1]),
    'v1: Student+GI+CI-RKD'                   : (obj_iter_RKD[-1],       rate_iter_RKD[-1],       tau_iter_RKD[-1]),
    '── v2 (30-layer RKD + log loss) ──'      : (None, None, None),
    'v2 cell_1: flat, no KD'                  : (obj_iter_v2c1[-1],      rate_iter_v2c1[-1],      tau_iter_v2c1[-1]),
    'v2 cell_2: flat + RKD + log'             : (obj_iter_v2c2[-1],      rate_iter_v2c2[-1],      tau_iter_v2c2[-1]),
    'v2 cell_3: GI, no RKD'                   : (obj_iter_v2c3[-1],      rate_iter_v2c3[-1],      tau_iter_v2c3[-1]),
    'v2 cell_4: GI + RKD + log (proposed v2)' : (obj_iter_v2c4[-1],      rate_iter_v2c4[-1],      tau_iter_v2c4[-1]),
}
for name, (obj, rate, tau) in results.items():
    if obj is None:
        print(f'  {name}')
    else:
        print(f'  {name:<45} {obj:>8.4f} {rate:>15.4f} {tau:>10.4f}')
print('='*80 + '\n')

# ══════════════════════════════════════════════════════════════════════════════
# CONVERGENCE PLOTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Figure 1: Objective vs Iterations ────────────────────────────────────────
plt.figure(1)
plt.plot(x_long,             obj_iter_J20,    '-',          color='#2CA02C', linewidth=3, marker='o', markevery=8, markersize=6, label=r'Teacher $(J=20, I=120)$')
plt.plot(x_long,             obj_iter_J1,     '--',         color='#FF7F0E', linewidth=3, marker='s', markevery=8, markersize=6, label=r'Conventional PGA')
plt.plot(iter_number_J1_AGT, obj_iter_J1_AGT, '-.',         color='#17BECF', linewidth=3, marker='p', markevery=8, markersize=6, label=r'Conventional PGA+GI')
plt.plot(x_trunc,            obj_iter_J10_trunc, ':',       color='#8C564B', linewidth=3, marker='h', markevery=6, markersize=6, label=r'Student $(J=10, I=120)$ truncated at $I=60$')
# v1 baselines
plt.plot(x_short, obj_iter_RKD2, '-.',         color='#1F77B4', linewidth=2, marker='^', markevery=6, markersize=5, label=r'v1: Student+CI-RKD',      alpha=0.6)
plt.plot(x_short, obj_iter_RKD1, ls=(0,(5,2)), color='#D62728', linewidth=2, marker='D', markevery=6, markersize=5, label=r'v1: Student+GI',          alpha=0.6)
plt.plot(x_short, obj_iter_RKD,  '-',          color='#9467BD', linewidth=2, marker='*', markevery=6, markersize=6, label=r'v1: Student+GI+CI-RKD',  alpha=0.6)
# v2 new models
plt.plot(x_short, obj_iter_v2c1, ':',          color='#E377C2', linewidth=2.5, marker='x', markevery=6, markersize=6, label=r'v2: flat, no KD')
plt.plot(x_short, obj_iter_v2c2, '--',         color='#0047FF', linewidth=2.5, marker='P', markevery=6, markersize=6, label=r'v2: flat+RKD+log')
plt.plot(x_short, obj_iter_v2c3, ls=(0,(3,1)), color='#FF9500', linewidth=2.5, marker='X', markevery=6, markersize=6, label=r'v2: GI, no RKD')
plt.plot(x_short, obj_iter_v2c4, '-',          color='#00CC44', linewidth=3.5, marker='*', markevery=6, markersize=8, label=r'v2: GI+RKD+log (proposed)')
plt.xlabel(r'Number of iterations/layers $(I)$', fontsize=14)
plt.ylabel(r'$R - \omega\bar{\tau}$ [bits/s/Hz]', fontsize=14)
plt.xlim(0, 120)
plt.grid()
plt.legend(loc='best', fontsize=9)
plt.tight_layout()
plt.savefig(directory_result + 'v2_30l_obj_vs_iter.png')
plt.savefig(directory_result + 'v2_30l_obj_vs_iter.eps')

# ── Figure 2: Rate vs Iterations ─────────────────────────────────────────────
plt.figure(2)
plt.plot(x_long,             rate_iter_J20,    '-',          color='#2CA02C', linewidth=3, marker='o', markevery=8, markersize=6, label=r'Teacher $(J=20, I=120)$')
plt.plot(x_long,             rate_iter_J1,     '--',         color='#FF7F0E', linewidth=3, marker='s', markevery=8, markersize=6, label=r'Conventional PGA')
plt.plot(iter_number_J1_AGT, rate_iter_J1_AGT, '-.',         color='#17BECF', linewidth=3, marker='p', markevery=8, markersize=6, label=r'Conventional PGA+GI')
plt.plot(x_trunc,            rate_iter_J10_trunc, ':',       color='#8C564B', linewidth=3, marker='h', markevery=6, markersize=6, label=r'Student $(J=10, I=120)$ truncated at $I=60$')
plt.plot(x_short, rate_iter_RKD2, '-.',         color='#1F77B4', linewidth=2, marker='^', markevery=6, markersize=5, label=r'v1: Student+CI-RKD',     alpha=0.6)
plt.plot(x_short, rate_iter_RKD1, ls=(0,(5,2)), color='#D62728', linewidth=2, marker='D', markevery=6, markersize=5, label=r'v1: Student+GI',         alpha=0.6)
plt.plot(x_short, rate_iter_RKD,  '-',          color='#9467BD', linewidth=2, marker='*', markevery=6, markersize=6, label=r'v1: Student+GI+CI-RKD', alpha=0.6)
plt.plot(x_short, rate_iter_v2c1, ':',          color='#E377C2', linewidth=2.5, marker='x', markevery=6, markersize=6, label=r'v2: flat, no KD')
plt.plot(x_short, rate_iter_v2c2, '--',         color='#0047FF', linewidth=2.5, marker='P', markevery=6, markersize=6, label=r'v2: flat+RKD+log')
plt.plot(x_short, rate_iter_v2c3, ls=(0,(3,1)), color='#FF9500', linewidth=2.5, marker='X', markevery=6, markersize=6, label=r'v2: GI, no RKD')
plt.plot(x_short, rate_iter_v2c4, '-',          color='#00CC44', linewidth=3.5, marker='*', markevery=6, markersize=8, label=r'v2: GI+RKD+log (proposed)')
plt.xlabel(r'Number of iterations/layers $(I)$', fontsize=14)
plt.ylabel(r'$R$ [bits/s/Hz]', fontsize=14)
plt.xlim(0, 120)
plt.grid()
plt.legend(loc='best', fontsize=9)
plt.tight_layout()
plt.savefig(directory_result + 'v2_30l_rate_vs_iter.png')
plt.savefig(directory_result + 'v2_30l_rate_vs_iter.eps')

# ── Figure 3: Beam Error vs Iterations ───────────────────────────────────────
plt.figure(3)
plt.plot(x_long,             tau_iter_J20,    '-',          color='#2CA02C', linewidth=3, marker='o', markevery=8, markersize=6, label=r'Teacher $(J=20, I=120)$')
plt.plot(x_long,             tau_iter_J1,     '--',         color='#FF7F0E', linewidth=3, marker='s', markevery=8, markersize=6, label=r'Conventional PGA')
plt.plot(iter_number_J1_AGT, tau_iter_J1_AGT, '-.',         color='#17BECF', linewidth=3, marker='p', markevery=8, markersize=6, label=r'Conventional PGA+GI')
plt.plot(x_trunc,            tau_iter_J10_trunc, ':',       color='#8C564B', linewidth=3, marker='h', markevery=6, markersize=6, label=r'Student $(J=10, I=120)$ truncated at $I=60$')
plt.plot(x_short, tau_iter_RKD2, '-.',         color='#1F77B4', linewidth=2, marker='^', markevery=6, markersize=5, label=r'v1: Student+CI-RKD',     alpha=0.6)
plt.plot(x_short, tau_iter_RKD1, ls=(0,(5,2)), color='#D62728', linewidth=2, marker='D', markevery=6, markersize=5, label=r'v1: Student+GI',         alpha=0.6)
plt.plot(x_short, tau_iter_RKD,  '-',          color='#9467BD', linewidth=2, marker='*', markevery=6, markersize=6, label=r'v1: Student+GI+CI-RKD', alpha=0.6)
plt.plot(x_short, tau_iter_v2c1, ':',          color='#E377C2', linewidth=2.5, marker='x', markevery=6, markersize=6, label=r'v2: flat, no KD')
plt.plot(x_short, tau_iter_v2c2, '--',         color='#0047FF', linewidth=2.5, marker='P', markevery=6, markersize=6, label=r'v2: flat+RKD+log')
plt.plot(x_short, tau_iter_v2c3, ls=(0,(3,1)), color='#FF9500', linewidth=2.5, marker='X', markevery=6, markersize=6, label=r'v2: GI, no RKD')
plt.plot(x_short, tau_iter_v2c4, '-',          color='#00CC44', linewidth=3.5, marker='*', markevery=6, markersize=8, label=r'v2: GI+RKD+log (proposed)')
plt.xlabel(r'Number of iterations/layers $(I)$', fontsize=14)
plt.ylabel(r'$\bar{\tau}$', fontsize=14)
plt.xlim(0, 120)
plt.grid()
plt.legend(loc='best', fontsize=9)
plt.tight_layout()
plt.savefig(directory_result + 'v2_30l_beam_vs_iter.png')
plt.savefig(directory_result + 'v2_30l_beam_vs_iter.eps')

# ══════════════════════════════════════════════════════════════════════════════
# SNR SWEEP
# ══════════════════════════════════════════════════════════════════════════════
print('Running SNR sweep...')

rate_vs_snr = {'J20': [], 'J1': [], 'J1_AGT': [], 'J10_trunc': [],
               'v1_flat_rkd': [], 'v1_agt': [], 'v1_agt_rkd': [],
               'v2_c1': [], 'v2_c2': [], 'v2_c3': [], 'v2_c4': []}
mse_vs_snr  = {'J20': [], 'J1': [], 'J1_AGT': [], 'J10_trunc': [],
               'v1_flat_rkd': [], 'v1_agt': [], 'v1_agt_rkd': [],
               'v2_c1': [], 'v2_c2': [], 'v2_c3': [], 'v2_c4': []}

for ss in range(len(snr_dB_list)):
    snr_dB_ss = snr_dB_list[ss]
    snr_ss    = 10 ** (snr_dB_ss / 10)
    print(f'  SNR = {snr_dB_ss} dB')

    R_ss, at_ss, _, _ = get_radar_data(snr_dB_ss, H_test)
    at_ss = at_ss[:, :test_size, :, :]

    with torch.no_grad():
        _, _, F, W = model_UPGA_J20.execute_PGA(
            H_test, R_ss, snr_ss, n_iter_outer, n_iter_inner_J20)
        rate_vs_snr['J20'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['J20'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        _, _, F, W = model_UPGA_J1.execute_PGA(H_test, R_ss, snr_ss, n_iter_outer)
        rate_vs_snr['J1'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['J1'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        _, _, F, W = model_J1_AGT.execute_PGA(H_test, R_ss, snr_ss, _actual_I_J1_AGT)
        rate_vs_snr['J1_AGT'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['J1_AGT'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        _, _, F, W = model_J10_I120.execute_PGA(H_test, R_ss, snr_ss, 60, n_iter_inner_J10)
        rate_vs_snr['J10_trunc'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['J10_trunc'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        _, _, F, W = model_RKD2.execute_PGA(H_test, R_ss, snr_ss, 60, n_iter_inner_J10)
        rate_vs_snr['v1_flat_rkd'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['v1_flat_rkd'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        _, _, F, W = model_RKD1.execute_PGA(H_test, R_ss, snr_ss, 60, n_iter_inner_J10)
        rate_vs_snr['v1_agt'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['v1_agt'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        _, _, F, W = model_RKD.execute_PGA(H_test, R_ss, snr_ss, 60, n_iter_inner_J10)
        rate_vs_snr['v1_agt_rkd'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['v1_agt_rkd'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        _, _, F, W = model_v2_c1.execute_PGA(H_test, R_ss, snr_ss, 60, n_iter_inner_J10)
        rate_vs_snr['v2_c1'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['v2_c1'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        _, _, F, W = model_v2_c2.execute_PGA(H_test, R_ss, snr_ss, 60, n_iter_inner_J10)
        rate_vs_snr['v2_c2'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['v2_c2'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        _, _, F, W = model_v2_c3.execute_PGA(H_test, R_ss, snr_ss, 60, n_iter_inner_J10)
        rate_vs_snr['v2_c3'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['v2_c3'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        _, _, F, W = model_v2_c4.execute_PGA(H_test, R_ss, snr_ss, 60, n_iter_inner_J10)
        rate_vs_snr['v2_c4'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['v2_c4'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

snr_dB_list_np = np.array(snr_dB_list)

# ── Figure 4: Rate vs SNR ─────────────────────────────────────────────────────
plt.figure(4)
plt.plot(snr_dB_list_np, rate_vs_snr['J20'],         '-',          color='#2CA02C', linewidth=3, marker='o', markersize=6, label=r'Teacher $(J=20, I=120)$')
plt.plot(snr_dB_list_np, rate_vs_snr['J1'],          '--',         color='#FF7F0E', linewidth=3, marker='s', markersize=6, label=r'Conventional PGA')
plt.plot(snr_dB_list_np, rate_vs_snr['J1_AGT'],      '-.',         color='#17BECF', linewidth=3, marker='p', markersize=6, label=r'Conventional PGA+GI')
plt.plot(snr_dB_list_np, rate_vs_snr['J10_trunc'],   ':',          color='#8C564B', linewidth=3, marker='h', markersize=6, label=r'Student $(J=10, I=120)$ truncated at $I=60$')
plt.plot(snr_dB_list_np, rate_vs_snr['v1_flat_rkd'], '-.',         color='#1F77B4', linewidth=2, marker='^', markersize=5, label=r'v1: Student+CI-RKD',     alpha=0.6)
plt.plot(snr_dB_list_np, rate_vs_snr['v1_agt'],      ls=(0,(5,2)), color='#D62728', linewidth=2, marker='D', markersize=5, label=r'v1: Student+GI',         alpha=0.6)
plt.plot(snr_dB_list_np, rate_vs_snr['v1_agt_rkd'],  '-',          color='#9467BD', linewidth=2, marker='*', markersize=6, label=r'v1: Student+GI+CI-RKD', alpha=0.6)
plt.plot(snr_dB_list_np, rate_vs_snr['v2_c1'],       ':',          color='#E377C2', linewidth=2.5, marker='x', markersize=6, label=r'v2: flat, no KD')
plt.plot(snr_dB_list_np, rate_vs_snr['v2_c2'],       '--',         color='#0047FF', linewidth=2.5, marker='P', markersize=6, label=r'v2: flat+RKD+log')
plt.plot(snr_dB_list_np, rate_vs_snr['v2_c3'],       ls=(0,(3,1)), color='#FF9500', linewidth=2.5, marker='X', markersize=6, label=r'v2: GI, no RKD')
plt.plot(snr_dB_list_np, rate_vs_snr['v2_c4'],       '-',          color='#00CC44', linewidth=3.5, marker='*', markersize=8, label=r'v2: GI+RKD+log (proposed)')
plt.xlabel('SNR [dB]', fontsize=14)
plt.ylabel(r'$R$ [bits/s/Hz]', fontsize=14)
plt.xticks(snr_dB_list_np)
plt.grid()
plt.legend(loc='best', fontsize=9)
plt.tight_layout()
plt.savefig(directory_result + 'v2_30l_rate_vs_SNR.png')
plt.savefig(directory_result + 'v2_30l_rate_vs_SNR.eps')

# ── Figure 5: Beam MSE vs SNR ─────────────────────────────────────────────────
plt.figure(5)
plt.plot(snr_dB_list_np, mse_vs_snr['J20'],         '-',          color='#2CA02C', linewidth=3, marker='o', markersize=6, label=r'Teacher $(J=20, I=120)$')
plt.plot(snr_dB_list_np, mse_vs_snr['J1'],          '--',         color='#FF7F0E', linewidth=3, marker='s', markersize=6, label=r'Conventional PGA')
plt.plot(snr_dB_list_np, mse_vs_snr['J1_AGT'],      '-.',         color='#17BECF', linewidth=3, marker='p', markersize=6, label=r'Conventional PGA+GI')
plt.plot(snr_dB_list_np, mse_vs_snr['J10_trunc'],   ':',          color='#8C564B', linewidth=3, marker='h', markersize=6, label=r'Student $(J=10, I=120)$ truncated at $I=60$')
plt.plot(snr_dB_list_np, mse_vs_snr['v1_flat_rkd'], '-.',         color='#1F77B4', linewidth=2, marker='^', markersize=5, label=r'v1: Student+CI-RKD',     alpha=0.6)
plt.plot(snr_dB_list_np, mse_vs_snr['v1_agt'],      ls=(0,(5,2)), color='#D62728', linewidth=2, marker='D', markersize=5, label=r'v1: Student+GI',         alpha=0.6)
plt.plot(snr_dB_list_np, mse_vs_snr['v1_agt_rkd'],  '-',          color='#9467BD', linewidth=2, marker='*', markersize=6, label=r'v1: Student+GI+CI-RKD', alpha=0.6)
plt.plot(snr_dB_list_np, mse_vs_snr['v2_c1'],       ':',          color='#E377C2', linewidth=2.5, marker='x', markersize=6, label=r'v2: flat, no KD')
plt.plot(snr_dB_list_np, mse_vs_snr['v2_c2'],       '--',         color='#0047FF', linewidth=2.5, marker='P', markersize=6, label=r'v2: flat+RKD+log')
plt.plot(snr_dB_list_np, mse_vs_snr['v2_c3'],       ls=(0,(3,1)), color='#FF9500', linewidth=2.5, marker='X', markersize=6, label=r'v2: GI, no RKD')
plt.plot(snr_dB_list_np, mse_vs_snr['v2_c4'],       '-',          color='#00CC44', linewidth=3.5, marker='*', markersize=8, label=r'v2: GI+RKD+log (proposed)')
plt.xlabel('SNR [dB]', fontsize=14)
plt.ylabel('Average radar beampattern MSE [dB]', fontsize=14)
plt.xticks(snr_dB_list_np)
plt.grid()
plt.legend(loc='best', fontsize=9)
plt.tight_layout()
plt.savefig(directory_result + 'v2_30l_beam_vs_SNR.png')
plt.savefig(directory_result + 'v2_30l_beam_vs_SNR.eps')

plt.show()
print('\nDone.')
