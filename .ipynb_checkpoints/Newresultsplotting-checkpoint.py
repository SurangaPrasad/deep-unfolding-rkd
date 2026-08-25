from PGA_models import *
from utility_gpu import *
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

EVAL_ITERS = 30
OMEGA      = 0.3

# ── New config overrides ───────────────────────────────────────────────────────
I_T_new = 80
I_S_new = 40
J_T_new = n_iter_inner_J20   # 20
J_S_new = n_iter_inner_J10   # 10

# ── Load data ──────────────────────────────────────────────────────────────────
H_train, H_test0 = get_data_tensor(data_source)
H_test = H_test0[:, :test_size, :, :]

R, at0, theta, ideal_beam = get_radar_data(snr_dB, H_test)
at = at0[:, :test_size, :, :]

iter_number_new_student = np.array(list(range(I_S_new + 1)))


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

class PGA_Unfold_J10_IS_v2(nn.Module):
    def __init__(self, step_size_init):
        super().__init__()
        if isinstance(step_size_init, torch.Tensor) and step_size_init.dim() == 3:
            self.step_size = nn.Parameter(step_size_init.float().clone())
        else:
            self.step_size = nn.Parameter(
                step_size_init * torch.ones(J_S_new, I_S_new, K + 1))

    def execute_PGA(self, H, R, Pt, n_iter_outer_run, n_iter_inner):
        rate_init, tau_init, F, W = safe_initialize(
            H, R, Pt, initial_normalization, device)
        rate_over_iters = torch.zeros(
            n_iter_outer_run, H.shape[1], device=device)
        tau_over_iters  = torch.zeros(
            n_iter_outer_run, H.shape[1], device=device)
    
        for ii in range(n_iter_outer_run):
            for jj in range(n_iter_inner):
                grad_F_com  = get_grad_F_com(H, F, W)
                grad_F_rad  = get_grad_F_rad(F, W, R)
                F = F + self.step_size[jj][ii][0] * (
                    grad_F_com * WEIGHT_F_COM - grad_F_rad * WEIGHT_F_RAD)
                F = normalize_power(F, W, H, Pt)
            F = F / torch.abs(F)
            W_new      = W.clone().detach()
            grad_W_com = get_grad_W_com(H, F, W)
            grad_W_rad = get_grad_W_rad(F, W, R)   # <-- fixed
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
                F, W)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD HELPER
# ══════════════════════════════════════════════════════════════════════════════

def load_new_student(path):
    ckpt     = torch.load(path, map_location='cpu', weights_only=False)
    ss_shape = ckpt['step_size'].shape
    print(f'  step_size shape: {list(ss_shape)}')
    m = PGA_Unfold_J10_IS_v2(torch.ones(ss_shape) * 0.01)
    m.load_state_dict(ckpt)
    m.eval()
    return m


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════════════════════

print('Loading models...')

# ── Teacher J20/I80 ───────────────────────────────────────────────────────────
_ckpt_J20_I80  = torch.load('UPGA_J20_I80_w_0_30.pth', map_location='cpu',
                             weights_only=False)
_ss_shape_I80  = _ckpt_J20_I80['step_size'].shape
_ss_J20_I80    = torch.zeros(_ss_shape_I80)
model_UPGA_J20_I80 = PGA_Unfold_J20(_ss_J20_I80)
model_UPGA_J20_I80.load_state_dict(_ckpt_J20_I80)
model_UPGA_J20_I80.eval()
_actual_I_J20_I80 = _ss_shape_I80[1]
with torch.no_grad():
    sr_J20_I80, t_J20_I80, _, _ = model_UPGA_J20_I80.execute_PGA(
        H_test, R, snr, _actual_I_J20_I80, n_iter_inner_J20)
rate_iter_J20_I80 = [r.detach().numpy() for r in (sum(sr_J20_I80) / len(H_test[0]))]
tau_iter_J20_I80  = [e.detach().numpy() for e in (sum(t_J20_I80)  / len(H_test[0]))]
iter_number_new_teacher = np.array(list(range(_actual_I_J20_I80 + 1)))
print(f'  Teacher J20/I80 loaded — actual I={_actual_I_J20_I80}')

# ── nc1: J10/I120 truncated at I=I_S_new (no KD baseline) ────────────────────
print(f'Loading J10/I120 truncated at I={I_S_new}...')
_ckpt_J10_all  = torch.load('./model/UPGA_J10_all.pth',
                             map_location='cpu', weights_only=False)
_ss_shape_J10  = _ckpt_J10_all['step_size'].shape
_ss_J10        = torch.zeros(_ss_shape_J10)
model_J10_all  = PGA_Unfold_J10(_ss_J10)
model_J10_all.load_state_dict(_ckpt_J10_all)
model_J10_all.eval()
with torch.no_grad():
    sr_J10_trunc, t_J10_trunc, _, _ = model_J10_all.execute_PGA(
        H_test, R, snr, n_iter_outer, n_iter_inner_J10)
rate_iter_nc1 = [r.detach().numpy()
                  for r in (sum(sr_J10_trunc) / len(H_test[0]))][:I_S_new+1]
tau_iter_nc1  = [e.detach().numpy()
                  for e in (sum(t_J10_trunc)  / len(H_test[0]))][:I_S_new+1]
print(f'  J10/I120 truncated at I={I_S_new}')

# ── nc2: flat init + L_log + CI-RKD ──────────────────────────────────────────
print('Loading new cell_2...')
model_nc2 = load_new_student(
    model_file_name_UPGA_J10 +
    f'_IS{I_S_new}_IT{I_T_new}_cell_2_flat0.01_RKDlog'
    f'_Kl5_win10_030.pth')
with torch.no_grad():
    sr_nc2, t_nc2, _, _ = model_nc2.execute_PGA(
        H_test, R, snr, I_S_new, J_S_new)
rate_iter_nc2 = [r.detach().numpy() for r in (sum(sr_nc2) / len(H_test[0]))]
tau_iter_nc2  = [e.detach().numpy() for e in (sum(t_nc2)  / len(H_test[0]))]

# ── nc3: AGT init, task loss only ─────────────────────────────────────────────
print('Loading new cell_3...')
model_nc3 = load_new_student(
    model_file_name_UPGA_J10 +
    f'_IS{I_S_new}_IT{I_T_new}_cell_3_AGT_avg_pairs_noRKD'
    f'_Kl5_win10_030.pth')
with torch.no_grad():
    sr_nc3, t_nc3, _, _ = model_nc3.execute_PGA(
        H_test, R, snr, I_S_new, J_S_new)
rate_iter_nc3 = [r.detach().numpy() for r in (sum(sr_nc3) / len(H_test[0]))]
tau_iter_nc3  = [e.detach().numpy() for e in (sum(t_nc3)  / len(H_test[0]))]

# ── nc4: AGT init + L_log + CI-RKD (proposed) ────────────────────────────────
print('Loading new cell_4...')
model_nc4 = load_new_student(
    model_file_name_UPGA_J10 +
    f'_IS{I_S_new}_IT{I_T_new}_cell_4_AGT_avg_pairs_RKDlog'
    f'_Kl5_win10_030.pth')
with torch.no_grad():
    sr_nc4, t_nc4, _, _ = model_nc4.execute_PGA(
        H_test, R, snr, I_S_new, J_S_new)
rate_iter_nc4 = [r.detach().numpy() for r in (sum(sr_nc4) / len(H_test[0]))]
tau_iter_nc4  = [e.detach().numpy() for e in (sum(t_nc4)  / len(H_test[0]))]

print('All models loaded.\n')


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE OBJECTIVES
# ══════════════════════════════════════════════════════════════════════════════

obj_iter_J20_I80 = [r - OMEGA * t for r, t in zip(rate_iter_J20_I80, tau_iter_J20_I80)]
obj_iter_nc1     = [r - OMEGA * t for r, t in zip(rate_iter_nc1,     tau_iter_nc1)]
obj_iter_nc2     = [r - OMEGA * t for r, t in zip(rate_iter_nc2,     tau_iter_nc2)]
obj_iter_nc3     = [r - OMEGA * t for r, t in zip(rate_iter_nc3,     tau_iter_nc3)]
obj_iter_nc4     = [r - OMEGA * t for r, t in zip(rate_iter_nc4,     tau_iter_nc4)]


# ══════════════════════════════════════════════════════════════════════════════
# PRINT RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*78)
print(f'Final values at SNR = {snr_dB} dB,  omega = {OMEGA}')
print('='*78)
print(f'  {"Model":<55} {"Obj":>8} {"R":>10} {"Tau":>10}  {"Steps":>8}')
print(f'  {"-"*83}')
results_table = {
    'Teacher (J=20, I=80)'                       : (obj_iter_J20_I80[-1], rate_iter_J20_I80[-1], tau_iter_J20_I80[-1], 1600),
    f'J10/I120 truncated at I={I_S_new} (no KD)' : (obj_iter_nc1[-1],     rate_iter_nc1[-1],     tau_iter_nc1[-1],     J_S_new*n_iter_inner_J10),
    f'Student J10/I{I_S_new} + CIRS'              : (obj_iter_nc2[-1],     rate_iter_nc2[-1],     tau_iter_nc2[-1],     J_S_new*I_S_new),
    f'Student J10/I{I_S_new} + GI'                : (obj_iter_nc3[-1],     rate_iter_nc3[-1],     tau_iter_nc3[-1],     J_S_new*I_S_new),
    f'Student J10/I{I_S_new} + GI + CIRS'         : (obj_iter_nc4[-1],     rate_iter_nc4[-1],     tau_iter_nc4[-1],     J_S_new*I_S_new),
}
for name, (obj, rate, tau, steps) in results_table.items():
    print(f'  {name:<55} {obj:>8.4f} {rate:>10.4f} {tau:>10.4f}  {steps:>8}')
print('='*78 + '\n')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT STYLE
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    'teacher_I80' : '#1f7700',
    'nc1'         : '#FF7F0E',
    'nc2'         : '#CC79A7',
    'nc3'         : '#D62728',
    'nc4'         : '#9467BD',
}

LABELS = {
    'teacher_I80' : r'Teacher $(J=20, I=80)$',
    'nc1'         : rf'Student $(J=10, I={I_S_new})$ no KD',
    'nc2'         : rf'Student + CIRS $(J=10, I={I_S_new})$',
    'nc3'         : rf'Student + GI $(J=10, I={I_S_new})$',
    'nc4'         : rf'Student + GI + CIRS $(J=10, I={I_S_new})$',
}


# ══════════════════════════════════════════════════════════════════════════════
# CONVERGENCE PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def conv_plot(fig_num, data_dict, ylabel, filename,
              zoom_x=(20, 32), zoom_y=None):
    fig, ax = plt.subplots(figsize=(9, 6), num=fig_num)

    ax.plot(iter_number_new_teacher, data_dict['J20_I80'], '--',
            color=COLORS['teacher_I80'], linewidth=3,
            marker='o', markevery=8, markersize=6,
            label=LABELS['teacher_I80'])
    ax.plot(iter_number_new_student, data_dict['nc1'],     ':',
            color=COLORS['nc1'],        linewidth=2.5,
            marker='s', markevery=4, markersize=6,
            label=LABELS['nc1'])
    ax.plot(iter_number_new_student, data_dict['nc2'],     '-.',
            color=COLORS['nc2'],        linewidth=2.5,
            marker='x', markevery=4, markersize=6,
            label=LABELS['nc2'])
    ax.plot(iter_number_new_student, data_dict['nc3'],
            linestyle=(0, (5, 2)),
            color=COLORS['nc3'],        linewidth=2.5,
            marker='D', markevery=4, markersize=6,
            label=LABELS['nc3'])
    ax.plot(iter_number_new_student, data_dict['nc4'],     '-',
            color=COLORS['nc4'],        linewidth=3.5,
            marker='*', markevery=4, markersize=8,
            label=LABELS['nc4'])

    ax.set_xlabel(r'Number of iterations/layers $(I)$', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlim(0, _actual_I_J20_I80)
    ax.grid()
    ax.legend(loc='lower right', fontsize=10)
    ax.tick_params(labelsize=11)

    # Inset zoom
    if zoom_y is not None:
        axins = fig.add_axes([0.52, 0.12, 0.30, 0.32])
        axins.plot(iter_number_new_teacher, data_dict['J20_I80'], '--',
                   color=COLORS['teacher_I80'], linewidth=1.5,
                   marker='o', markevery=8, markersize=3)
        axins.plot(iter_number_new_student, data_dict['nc1'],     ':',
                   color=COLORS['nc1'],        linewidth=1.2,
                   marker='s', markevery=4, markersize=3)
        axins.plot(iter_number_new_student, data_dict['nc2'],     '-.',
                   color=COLORS['nc2'],        linewidth=1.2,
                   marker='x', markevery=4, markersize=3)
        axins.plot(iter_number_new_student, data_dict['nc3'],
                   linestyle=(0, (5, 2)),
                   color=COLORS['nc3'],        linewidth=1.2,
                   marker='D', markevery=4, markersize=3)
        axins.plot(iter_number_new_student, data_dict['nc4'],     '-',
                   color=COLORS['nc4'],        linewidth=2,
                   marker='*', markevery=4, markersize=4)
        axins.set_xlim(zoom_x[0], zoom_x[1])
        axins.set_ylim(zoom_y[0], zoom_y[1])
        axins.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
        axins.tick_params(labelsize=8)
        axins.xaxis.set_major_locator(ticker.MultipleLocator(5))
        axins.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
        axins.set_title('Zoom', fontsize=9)
        ax.indicate_inset_zoom(axins, edgecolor='gray', linewidth=0.8)

    plt.tight_layout()
    plt.savefig(directory_result + filename + '.png',
                dpi=300, bbox_inches='tight')
    plt.savefig(directory_result + filename + '.eps',
                bbox_inches='tight')
    print(f'Saved {filename}')


data_obj = {
    'J20_I80' : obj_iter_J20_I80,
    'nc1'     : obj_iter_nc1,
    'nc2'     : obj_iter_nc2,
    'nc3'     : obj_iter_nc3,
    'nc4'     : obj_iter_nc4,
}
data_rate = {
    'J20_I80' : rate_iter_J20_I80,
    'nc1'     : rate_iter_nc1,
    'nc2'     : rate_iter_nc2,
    'nc3'     : rate_iter_nc3,
    'nc4'     : rate_iter_nc4,
}
data_tau = {
    'J20_I80' : tau_iter_J20_I80,
    'nc1'     : tau_iter_nc1,
    'nc2'     : tau_iter_nc2,
    'nc3'     : tau_iter_nc3,
    'nc4'     : tau_iter_nc4,
}

conv_plot(1, data_obj,
          r'$R - \lambda\bar{\tau}$',
          'new_obj_vs_iter',
          zoom_x=(20, 32), zoom_y=None)

conv_plot(2, data_rate,
          r'$R$ [bits/s/Hz]',
          'new_rate_vs_iter',
          zoom_x=(20, 32), zoom_y=None)

conv_plot(3, data_tau,
          r'$\bar{\tau}$',
          'new_beam_vs_iter',
          zoom_x=(20, 32), zoom_y=None)


# ══════════════════════════════════════════════════════════════════════════════
# SNR SWEEP
# ══════════════════════════════════════════════════════════════════════════════

print(f'Running SNR sweep (EVAL_ITERS={EVAL_ITERS})...')

keys = ['J20_I80', 'nc1', 'nc2', 'nc3', 'nc4']
rate_vs_snr = {k: [] for k in keys}
mse_vs_snr  = {k: [] for k in keys}

for ss in range(len(snr_dB_list)):
    snr_dB_ss = snr_dB_list[ss]
    snr_ss    = 10 ** (snr_dB_ss / 10)
    print(f'  SNR = {snr_dB_ss} dB')

    R_ss, at_ss, _, _ = get_radar_data(snr_dB_ss, H_test)
    at_ss = at_ss[:, :test_size, :, :]

    with torch.no_grad():
        # Teacher J20/I80 — full depth
        _, _, F, W = model_UPGA_J20_I80.execute_PGA(
            H_test, R_ss, snr_ss, _actual_I_J20_I80, n_iter_inner_J20)
        rate_vs_snr['J20_I80'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['J20_I80'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        # nc1: J10/I120 truncated at EVAL_ITERS
        _, _, F, W = model_J10_all.execute_PGA(
            H_test, R_ss, snr_ss, EVAL_ITERS, n_iter_inner_J10)
        rate_vs_snr['nc1'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['nc1'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        # nc2: Student + CIRS
        _, _, F, W = model_nc2.execute_PGA(
            H_test, R_ss, snr_ss, EVAL_ITERS, J_S_new)
        rate_vs_snr['nc2'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['nc2'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        # nc3: Student + GI
        _, _, F, W = model_nc3.execute_PGA(
            H_test, R_ss, snr_ss, EVAL_ITERS, J_S_new)
        rate_vs_snr['nc3'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['nc3'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        # nc4: Student + GI + CIRS
        _, _, F, W = model_nc4.execute_PGA(
            H_test, R_ss, snr_ss, EVAL_ITERS, J_S_new)
        rate_vs_snr['nc4'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['nc4'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

snr_dB_list_np = np.array(snr_dB_list)


# ══════════════════════════════════════════════════════════════════════════════
# SNR SWEEP PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_snr_curves(ax, data_dict, show_legend=False):
    ax.plot(snr_dB_list_np, data_dict['J20_I80'], '--',
            color=COLORS['teacher_I80'], linewidth=3,
            marker='o', markersize=6, label=LABELS['teacher_I80'])
    ax.plot(snr_dB_list_np, data_dict['nc1'],     ':',
            color=COLORS['nc1'],        linewidth=2.5,
            marker='s', markersize=6, label=LABELS['nc1'])
    ax.plot(snr_dB_list_np, data_dict['nc2'],     '-.',
            color=COLORS['nc2'],        linewidth=2.5,
            marker='x', markersize=6, label=LABELS['nc2'])
    ax.plot(snr_dB_list_np, data_dict['nc3'],
            linestyle=(0, (5, 2)),
            color=COLORS['nc3'],        linewidth=2.5,
            marker='D', markersize=6, label=LABELS['nc3'])
    ax.plot(snr_dB_list_np, data_dict['nc4'],     '-',
            color=COLORS['nc4'],        linewidth=3.5,
            marker='*', markersize=8, label=LABELS['nc4'])
    ax.set_xticks(snr_dB_list_np)
    ax.grid()
    ax.tick_params(labelsize=11)
    if show_legend:
        ax.legend(loc='best', fontsize=10)


# Rate vs SNR — no legend
fig4, ax4 = plt.subplots(figsize=(7, 5), num=4)
plot_snr_curves(ax4, rate_vs_snr, show_legend=False)
ax4.set_xlabel('SNR [dB]', fontsize=14)
ax4.set_ylabel(r'$R$ [bits/s/Hz]', fontsize=14)
plt.tight_layout()
plt.savefig(directory_result + f'new_rate_vs_SNR_I{EVAL_ITERS}.png',
            dpi=300, bbox_inches='tight')
plt.savefig(directory_result + f'new_rate_vs_SNR_I{EVAL_ITERS}.eps',
            bbox_inches='tight')
print('Figure 4 saved.')

# Beam MSE vs SNR — legend here
fig5, ax5 = plt.subplots(figsize=(7, 5), num=5)
plot_snr_curves(ax5, mse_vs_snr, show_legend=False)
ax5.set_xlabel('SNR [dB]', fontsize=14)
ax5.set_ylabel('Average radar beampattern MSE [dB]', fontsize=14)
ax5.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig(directory_result + f'new_beam_vs_SNR_I{EVAL_ITERS}.png',
            dpi=300, bbox_inches='tight')
plt.savefig(directory_result + f'new_beam_vs_SNR_I{EVAL_ITERS}.eps',
            bbox_inches='tight')
print('Figure 5 saved.')

plt.show()
print('\nDone.')