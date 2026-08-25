from PGA_models import *
from utility_gpu import *
import random
import numpy as np
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

EVAL_ITERS = 30
OMEGA      = 0.3

# ── Load data ──────────────────────────────────────────────────────────────────
H_train, H_test0 = get_data_tensor(data_source)
H_test = H_test0[:, :test_size, :, :]

R, at0, theta, ideal_beam = get_radar_data(snr_dB, H_test)
at = at0[:, :test_size, :, :]

iter_number_UPGA_J20     = np.array(list(range(n_iter_outer + 1)))
iter_number_UPGA_J10_I60 = np.array(list(range(60 + 1)))


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

class PGA_Unfold_J10_I60_v2(nn.Module):
    def __init__(self, step_size_init):
        super().__init__()
        if isinstance(step_size_init, torch.Tensor) and step_size_init.dim() == 3:
            self.step_size = nn.Parameter(step_size_init.float().clone())
        else:
            self.step_size = nn.Parameter(
                step_size_init * torch.ones(n_iter_inner_J10, 60, K + 1))

    def execute_PGA(self, H, R, Pt, n_iter_outer_run, n_iter_inner):
        rate_init, tau_init, F, W = initialize(H, R, Pt, initial_normalization)
        rate_over_iters = torch.zeros(n_iter_outer_run, H.shape[1])
        tau_over_iters  = torch.zeros(n_iter_outer_run, H.shape[1])

        for ii in range(n_iter_outer_run):
            for jj in range(n_iter_inner):
                grad_F_com = get_grad_F_com(H, F, W)
                grad_F_rad = get_grad_F_rad(F, W, R)
                F = F + self.step_size[jj][ii][0] * (
                    grad_F_com * WEIGHT_F_COM - grad_F_rad * WEIGHT_F_RAD)
                F = normalize_power(F, W, H, Pt)
            F = F / torch.abs(F)
            W_new      = W.clone().detach()
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
                F, W)


class PGA_Conv_v2(nn.Module):
    def __init__(self, step_size_init):
        super().__init__()
        if isinstance(step_size_init, torch.Tensor) and step_size_init.dim() == 3:
            self.step_size = nn.Parameter(step_size_init.float().clone())
        else:
            self.step_size = nn.Parameter(
                step_size_init * torch.ones(1, n_iter_outer, K + 1))

    def execute_PGA(self, H, R, Pt, n_iter_outer_run, n_iter_inner=1):
        rate_init, tau_init, F, W = safe_initialize(
            H, R, Pt, initial_normalization, device)
        rate_over_iters = torch.zeros(
            n_iter_outer_run, H.shape[1], device=device)
        tau_over_iters  = torch.zeros(
            n_iter_outer_run, H.shape[1], device=device)

        for ii in range(n_iter_outer_run):
            grad_F_com = get_grad_F_com(H, F, W)
            grad_F_rad = get_grad_F_rad(F, W, R)
            F = F + self.step_size[0][ii][0] * (
                grad_F_com * WEIGHT_F_COM - grad_F_rad * WEIGHT_F_RAD)
            F = normalize_power(F, W, H, Pt)
            F = F / torch.abs(F)
            W_new      = W.clone().detach()
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
                F, W)


# ══════════════════════════════════════════════════════════════════════════════
# AGT HELPER
# ══════════════════════════════════════════════════════════════════════════════

def build_agt_init_for_eval(teacher_model, I_S):
    with torch.no_grad():
        ss_T          = teacher_model.step_size.data
        ss_compressed = ss_T.mean(dim=0, keepdim=True)
        fingerprint   = ss_compressed.mean(dim=1, keepdim=True)
        ss_init       = fingerprint.expand(-1, I_S, -1).clone()
    print(f"[AGT eval init] shape: {list(ss_init.shape)}  "
          f"range: [{ss_init.min():.4e}, {ss_init.max():.4e}]")
    return ss_init


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════════════════════

print('Loading models...')

# ── Teacher (J=20, I=120) ─────────────────────────────────────────────────────
model_UPGA_J20 = PGA_Unfold_J20(step_size_UPGA_J20)
model_UPGA_J20.load_state_dict(
    torch.load(model_file_name_UPGA_J20, map_location='cpu'))
model_UPGA_J20.eval()
with torch.no_grad():
    sum_rate_J20, tau_J20, _, _ = model_UPGA_J20.execute_PGA(
        H_test, R, snr, n_iter_outer, n_iter_inner_J20)
rate_iter_J20 = [r.detach().numpy() for r in (sum(sum_rate_J20) / len(H_test[0]))]
tau_iter_J20  = [e.detach().numpy() for e in (sum(tau_J20)      / len(H_test[0]))]

# ── Conventional PGA (fixed step size, no training) ───────────────────────────
model_conv_PGA = PGA_Conv(step_size_conv_PGA)
model_conv_PGA.eval()
with torch.no_grad():
    sum_rate_conv, tau_conv, _, _ = model_conv_PGA.execute_PGA(
        H_test, R, snr, n_iter_outer)
rate_iter_conv = [r.detach().numpy() for r in (sum(sum_rate_conv) / len(H_test[0]))]
tau_iter_conv  = [e.detach().numpy() for e in (sum(tau_conv)      / len(H_test[0]))]

# ── Conventional PGA + GI (no training) ───────────────────────────────────────
ss_conv_gi            = build_agt_init_for_eval(model_UPGA_J20, n_iter_outer)
model_conv_GI_notrain = PGA_Conv_v2(ss_conv_gi)
model_conv_GI_notrain.eval()
with torch.no_grad():
    sum_rate_conv_gi, tau_conv_gi, _, _ = model_conv_GI_notrain.execute_PGA(
        H_test, R, snr, n_iter_outer)
rate_iter_conv_gi   = [r.detach().numpy() for r in (sum(sum_rate_conv_gi) / len(H_test[0]))]
tau_iter_conv_gi    = [e.detach().numpy() for e in (sum(tau_conv_gi)      / len(H_test[0]))]
iter_number_conv_gi = np.array(list(range(n_iter_outer + 1)))

# ── Unfolded PGA J=1 (trained, no KD) ─────────────────────────────────────────
model_UPGA_J1 = PGA_Conv(step_size_UPGA_J1)
model_UPGA_J1.load_state_dict(
    torch.load('./model/UPGA_J1.pth', map_location='cpu'))
model_UPGA_J1.eval()
with torch.no_grad():
    sum_rate_J1, tau_J1, _, _ = model_UPGA_J1.execute_PGA(
        H_test, R, snr, n_iter_outer)
rate_iter_J1 = [r.detach().numpy() for r in (sum(sum_rate_J1) / len(H_test[0]))]
tau_iter_J1  = [e.detach().numpy() for e in (sum(tau_J1)      / len(H_test[0]))]

# ── Unfolded PGA J=1 + GI + CIRS (trained) ────────────────────────────────────
_ckpt_J1_AGT     = torch.load('./model/UPGA_J1.pth_I120_cell_4_AGT_mean_inner_RKDlog_30layers_Kl15_win20.pth',map_location='cpu') 

_actual_I_J1_AGT = _ckpt_J1_AGT['step_size'].shape[1]
_ss_J1_AGT       = torch.zeros(1, _actual_I_J1_AGT, K + 1)
model_J1_AGT     = PGA_Conv_v2(_ss_J1_AGT)
model_J1_AGT.load_state_dict(_ckpt_J1_AGT)
model_J1_AGT.eval()
with torch.no_grad():
    sum_rate_J1_AGT, tau_J1_AGT, _, _ = model_J1_AGT.execute_PGA(
        H_test, R, snr, _actual_I_J1_AGT)
rate_iter_J1_AGT   = [r.detach().numpy() for r in (sum(sum_rate_J1_AGT) / len(H_test[0]))]
tau_iter_J1_AGT    = [e.detach().numpy() for e in (sum(tau_J1_AGT)      / len(H_test[0]))]
iter_number_J1_AGT = np.array(list(range(_actual_I_J1_AGT + 1)))

# ── v2c1: Student J=10, I=60 (UPGA_J10_all truncated, no KD) ─────────────────
model_v2_c1 = PGA_Unfold_J10(step_size_UPGA_J10)
model_v2_c1.load_state_dict(torch.load(
    './model/UPGA_J10_all.pth', map_location='cpu'))
model_v2_c1.eval()
with torch.no_grad():
    sum_rate_v2c1, tau_v2c1, _, _ = model_v2_c1.execute_PGA(
        H_test, R, snr, n_iter_outer, n_iter_inner_J10)
rate_iter_v2c1 = [r.detach().numpy()
                   for r in (sum(sum_rate_v2c1) / len(H_test[0]))][:61]
tau_iter_v2c1  = [e.detach().numpy()
                   for e in (sum(tau_v2c1)      / len(H_test[0]))][:61]

# ── v2c2: Student + CIRS (flat init + CI-RKD) ─────────────────────────────────
model_v2_c2 = PGA_Unfold_J10_I60_v2(step_size_UPGA_J10_I60)
model_v2_c2.load_state_dict(torch.load(
    './model/UPGA_J10_320.pth_I60_cell_2_flat0.01_RKDlog_30layers_Kl15_win20_030.pth',
    map_location='cpu'))
model_v2_c2.eval()
with torch.no_grad():
    sum_rate_v2c2, tau_v2c2, _, _ = model_v2_c2.execute_PGA(
        H_test, R, snr, 60, n_iter_inner_J10)
rate_iter_v2c2 = [r.detach().numpy() for r in (sum(sum_rate_v2c2) / len(H_test[0]))]
tau_iter_v2c2  = [e.detach().numpy() for e in (sum(tau_v2c2)      / len(H_test[0]))]

# ── v2c3: Student + GI (AGT init, no CI-RKD) ──────────────────────────────────
model_v2_c3 = PGA_Unfold_J10_I60_v2(step_size_UPGA_J10_I60)
model_v2_c3.load_state_dict(torch.load(
    './model/UPGA_J10_320.pth_I60_cell_3_AGT_avg_pairs_noRKD_Kl15_win20_030.pth',map_location='cpu'))
model_v2_c3.eval()
with torch.no_grad():
    sum_rate_v2c3, tau_v2c3, _, _ = model_v2_c3.execute_PGA(
        H_test, R, snr, 60, n_iter_inner_J10)
rate_iter_v2c3 = [r.detach().numpy() for r in (sum(sum_rate_v2c3) / len(H_test[0]))]
tau_iter_v2c3  = [e.detach().numpy() for e in (sum(tau_v2c3)      / len(H_test[0]))]

# ── v2c4: Student + GI + CIRS (proposed) ──────────────────────────────────────
model_v2_c4 = PGA_Unfold_J10_I60_v2(step_size_UPGA_J10_I60)
model_v2_c4.load_state_dict(torch.load(
    './model/UPGA_J10_320.pth_I60_cell_4_AGT_avg_pairs_RKDlog_30layers_Kl15_win20_030.pth',
    map_location='cpu'))
model_v2_c4.eval()
with torch.no_grad():
    sum_rate_v2c4, tau_v2c4, _, _ = model_v2_c4.execute_PGA(
        H_test, R, snr, 60, n_iter_inner_J10)
rate_iter_v2c4 = [r.detach().numpy() for r in (sum(sum_rate_v2c4) / len(H_test[0]))]
tau_iter_v2c4  = [e.detach().numpy() for e in (sum(tau_v2c4)      / len(H_test[0]))]

print('All models loaded.\n')


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE OBJECTIVES
# ══════════════════════════════════════════════════════════════════════════════

obj_iter_J20      = [r - OMEGA * t for r, t in zip(rate_iter_J20,      tau_iter_J20)]
obj_iter_conv     = [r - OMEGA * t for r, t in zip(rate_iter_conv,     tau_iter_conv)]
obj_iter_conv_gi  = [r - OMEGA * t for r, t in zip(rate_iter_conv_gi,  tau_iter_conv_gi)]
obj_iter_J1       = [r - OMEGA * t for r, t in zip(rate_iter_J1,       tau_iter_J1)]
obj_iter_J1_AGT   = [r - OMEGA * t for r, t in zip(rate_iter_J1_AGT,   tau_iter_J1_AGT)]
obj_iter_v2c1     = [r - OMEGA * t for r, t in zip(rate_iter_v2c1,     tau_iter_v2c1)]
obj_iter_v2c2     = [r - OMEGA * t for r, t in zip(rate_iter_v2c2,     tau_iter_v2c2)]
obj_iter_v2c3     = [r - OMEGA * t for r, t in zip(rate_iter_v2c3,     tau_iter_v2c3)]
obj_iter_v2c4     = [r - OMEGA * t for r, t in zip(rate_iter_v2c4,     tau_iter_v2c4)]

x_long  = iter_number_UPGA_J20
x_short = iter_number_UPGA_J10_I60


# ══════════════════════════════════════════════════════════════════════════════
# PRINT RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*78)
print(f'Final values at SNR = {snr_dB} dB,  omega = {OMEGA}')
print('='*78)
print(f'  {"Model":<50} {"Obj":>8} {"R":>10} {"Tau":>10}')
print(f'  {"-"*78}')
results_table = {
    'Teacher (J=20, I=120)'                   : (obj_iter_J20[-1],    rate_iter_J20[-1],    tau_iter_J20[-1]),
    'Conventional PGA'                         : (obj_iter_conv[-1],   rate_iter_conv[-1],   tau_iter_conv[-1]),
    'Conventional PGA + GI'                    : (obj_iter_conv_gi[-1],rate_iter_conv_gi[-1],tau_iter_conv_gi[-1]),
    'Unfolded PGA (J=1, I=120)'                : (obj_iter_J1[-1],     rate_iter_J1[-1],     tau_iter_J1[-1]),
    'Unfolded PGA + GI (J=1, I=120)'   : (obj_iter_J1_AGT[-1], rate_iter_J1_AGT[-1], tau_iter_J1_AGT[-1]),
    'Student (J=10, I=60)'                     : (obj_iter_v2c1[-1],   rate_iter_v2c1[-1],   tau_iter_v2c1[-1]),
    'Student + CIRS'                           : (obj_iter_v2c2[-1],   rate_iter_v2c2[-1],   tau_iter_v2c2[-1]),
    'Student + GI'                             : (obj_iter_v2c3[-1],   rate_iter_v2c3[-1],   tau_iter_v2c3[-1]),
    'Student + GI + CIRS (proposed)'           : (obj_iter_v2c4[-1],   rate_iter_v2c4[-1],   tau_iter_v2c4[-1]),
}
for name, (obj, rate, tau) in results_table.items():
    print(f'  {name:<50} {obj:>8.4f} {rate:>10.4f} {tau:>10.4f}')
print('='*78 + '\n')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT STYLE
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    'teacher'    : '#2CA02C',
    'conv'       : '#FF7F0E',
    'conv_gi'    : '#8C564B',
    'upga_j1'    : '#BCBD22',
    'upga_j1_gi' : '#17BECF',
    'v2c1'       : '#0047FF',
    'v2c2'       : '#CC79A7',
    'v2c3'       : '#D62728',
    'v2c4'       : '#9467BD',
}

LABELS = {
    'teacher'    : r'Teacher $(J=20, I=120)$',
    'conv'       : r'Conventional PGA',
    'conv_gi'    : r'Conventional PGA + GI',
    'upga_j1'    : r'Unfolded PGA $(J=1, I=120)$',
    'upga_j1_gi' : r'Unfolded PGA + GI',
    'v2c1'       : r'Student $(J=10, I=60)$',
    'v2c2'       : r'Student + LRD',
    'v2c3'       : r'Student + GI',
    'v2c4'       : r'Student + GI + LRD',
}


# ══════════════════════════════════════════════════════════════════════════════
# CONVERGENCE PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def conv_plot(fig_num, data_dict, ylabel, filename):
    plt.figure(fig_num)
    plt.plot(x_long,              data_dict['J20'],        '-',              color=COLORS['teacher'],    linewidth=3,   marker='o', markevery=8, markersize=6, label=LABELS['teacher'])
    plt.plot(x_long,              data_dict['conv'],       '--',             color=COLORS['conv'],       linewidth=3,   marker='s', markevery=8, markersize=6, label=LABELS['conv'])
    plt.plot(iter_number_conv_gi, data_dict['conv_gi'],    ':',              color=COLORS['conv_gi'],    linewidth=2.5, marker='v', markevery=8, markersize=6, label=LABELS['conv_gi'])
    plt.plot(x_long,              data_dict['upga_j1'],    '-.',             color=COLORS['upga_j1'],    linewidth=3,   marker='^', markevery=8, markersize=6, label=LABELS['upga_j1'])
    plt.plot(iter_number_J1_AGT,  data_dict['upga_j1_gi'], ls=(0,(3,1,1,1)),color=COLORS['upga_j1_gi'], linewidth=2.5, marker='p', markevery=8, markersize=6, label=LABELS['upga_j1_gi'])
    plt.plot(x_short,             data_dict['v2c1'],       ':',              color=COLORS['v2c1'],       linewidth=3,   marker='h', markevery=6, markersize=6, label=LABELS['v2c1'])
    plt.plot(x_short,             data_dict['v2c2'],       '-.',             color=COLORS['v2c2'],       linewidth=2.5, marker='x', markevery=6, markersize=6, label=LABELS['v2c2'])
    plt.plot(x_short,             data_dict['v2c3'],       ls=(0,(5,2)),     color=COLORS['v2c3'],       linewidth=2.5, marker='D', markevery=6, markersize=6, label=LABELS['v2c3'])
    plt.plot(x_short,             data_dict['v2c4'],       '-',              color=COLORS['v2c4'],       linewidth=3.5, marker='*', markevery=6, markersize=8, label=LABELS['v2c4'])
    plt.xlabel(r'Number of iterations/layers $(I)$', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xlim(0, 120)
    plt.grid()
    plt.legend(loc='best', fontsize=11)
    plt.tight_layout()
    plt.savefig(directory_result + filename + 'n.png')
    plt.savefig(directory_result + filename + 'n.eps')

conv_data_obj = {
    'J20'        : obj_iter_J20,
    'conv'       : obj_iter_conv,
    'conv_gi'    : obj_iter_conv_gi,
    'upga_j1'    : obj_iter_J1,
    'upga_j1_gi' : obj_iter_J1_AGT,
    'v2c1'       : obj_iter_v2c1,
    'v2c2'       : obj_iter_v2c2,
    'v2c3'       : obj_iter_v2c3,
    'v2c4'       : obj_iter_v2c4,
}
conv_data_rate = {
    'J20'        : rate_iter_J20,
    'conv'       : rate_iter_conv,
    'conv_gi'    : rate_iter_conv_gi,
    'upga_j1'    : rate_iter_J1,
    'upga_j1_gi' : rate_iter_J1_AGT,
    'v2c1'       : rate_iter_v2c1,
    'v2c2'       : rate_iter_v2c2,
    'v2c3'       : rate_iter_v2c3,
    'v2c4'       : rate_iter_v2c4,
}
conv_data_tau = {
    'J20'        : tau_iter_J20,
    'conv'       : tau_iter_conv,
    'conv_gi'    : tau_iter_conv_gi,
    'upga_j1'    : tau_iter_J1,
    'upga_j1_gi' : tau_iter_J1_AGT,
    'v2c1'       : tau_iter_v2c1,
    'v2c2'       : tau_iter_v2c2,
    'v2c3'       : tau_iter_v2c3,
    'v2c4'       : tau_iter_v2c4,
}

conv_plot(1, conv_data_obj,  r'$R - \lambda\bar{\tau}$', 'v2_30l_obj_vs_iter_n')
conv_plot(2, conv_data_rate, r'$R$ [bits/s/Hz]',         'v2_30l_rate_vs_iter_n')
conv_plot(3, conv_data_tau,  r'$\bar{\tau}$',             'v2_30l_beam_vs_iter_n')


# ══════════════════════════════════════════════════════════════════════════════
# SNR SWEEP
# ══════════════════════════════════════════════════════════════════════════════

print(f'Running SNR sweep (student models evaluated at I={EVAL_ITERS})...')

keys = ['J20', 'conv', 'conv_gi', 'upga_j1', 'upga_j1_gi',
        'v2_c1', 'v2_c2', 'v2_c3', 'v2_c4']
rate_vs_snr = {k: [] for k in keys}
mse_vs_snr  = {k: [] for k in keys}

for ss in range(len(snr_dB_list)):
    snr_dB_ss = snr_dB_list[ss]
    snr_ss    = 10 ** (snr_dB_ss / 10)
    print(f'  SNR = {snr_dB_ss} dB')

    R_ss, at_ss, _, _ = get_radar_data(snr_dB_ss, H_test)
    at_ss = at_ss[:, :test_size, :, :]

    with torch.no_grad():
        # Teacher
        _, _, F, W = model_UPGA_J20.execute_PGA(
            H_test, R_ss, snr_ss, n_iter_outer, n_iter_inner_J20)
        rate_vs_snr['J20'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['J20'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        # Conventional PGA
        _, _, F, W = model_conv_PGA.execute_PGA(
            H_test, R_ss, snr_ss, EVAL_ITERS)
        rate_vs_snr['conv'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['conv'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        # Conv PGA + GI
        _, _, F, W = model_conv_GI_notrain.execute_PGA(
            H_test, R_ss, snr_ss, EVAL_ITERS)
        rate_vs_snr['conv_gi'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['conv_gi'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        # Unfolded PGA J=1
        _, _, F, W = model_UPGA_J1.execute_PGA(
            H_test, R_ss, snr_ss, EVAL_ITERS)
        rate_vs_snr['upga_j1'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['upga_j1'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        # Unfolded PGA J=1 + GI + CIRS
        _, _, F, W = model_J1_AGT.execute_PGA(
            H_test, R_ss, snr_ss, EVAL_ITERS)
        rate_vs_snr['upga_j1_gi'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['upga_j1_gi'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        # Student J=10, I=60 (truncated, no KD)
        _, _, F, W = model_v2_c1.execute_PGA(
            H_test, R_ss, snr_ss, EVAL_ITERS, n_iter_inner_J10)
        rate_vs_snr['v2_c1'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['v2_c1'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        # Student + CIRS
        _, _, F, W = model_v2_c2.execute_PGA(
            H_test, R_ss, snr_ss, EVAL_ITERS, n_iter_inner_J10)
        rate_vs_snr['v2_c2'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['v2_c2'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        # Student + GI
        _, _, F, W = model_v2_c3.execute_PGA(
            H_test, R_ss, snr_ss, EVAL_ITERS, n_iter_inner_J10)
        rate_vs_snr['v2_c3'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['v2_c3'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        # Student + GI + CIRS (proposed)
        _, _, F, W = model_v2_c4.execute_PGA(
            H_test, R_ss, snr_ss, EVAL_ITERS, n_iter_inner_J10)
        rate_vs_snr['v2_c4'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['v2_c4'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

snr_dB_list_np = np.array(snr_dB_list)



# plot without legend in beam error vs snr

# ══════════════════════════════════════════════════════════════════════════════
# SNR SWEEP PLOTS (4-5) — legend only in beam MSE plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_snr_curves(ax, data_dict, show_legend=False):
    ax.plot(snr_dB_list_np, data_dict['J20'],        '-',              color=COLORS['teacher'],    linewidth=3,   marker='o', markersize=6, label=LABELS['teacher'])
    ax.plot(snr_dB_list_np, data_dict['conv'],       '--',             color=COLORS['conv'],       linewidth=3,   marker='s', markersize=6, label=LABELS['conv'])
    ax.plot(snr_dB_list_np, data_dict['conv_gi'],    ':',              color=COLORS['conv_gi'],    linewidth=2.5, marker='v', markersize=6, label=LABELS['conv_gi'])
    ax.plot(snr_dB_list_np, data_dict['upga_j1'],    '-.',             color=COLORS['upga_j1'],    linewidth=3,   marker='^', markersize=6, label=LABELS['upga_j1'])
    ax.plot(snr_dB_list_np, data_dict['upga_j1_gi'], ls=(0,(3,1,1,1)),color=COLORS['upga_j1_gi'], linewidth=2.5, marker='p', markersize=6, label=LABELS['upga_j1_gi'])
    ax.plot(snr_dB_list_np, data_dict['v2_c1'],      ':',              color=COLORS['v2c1'],       linewidth=3,   marker='h', markersize=6, label=LABELS['v2c1'])
    ax.plot(snr_dB_list_np, data_dict['v2_c2'],      '-.',             color=COLORS['v2c2'],       linewidth=2.5, marker='x', markersize=6, label=LABELS['v2c2'])
    ax.plot(snr_dB_list_np, data_dict['v2_c3'],      ls=(0,(5,2)),     color=COLORS['v2c3'],       linewidth=2.5, marker='D', markersize=6, label=LABELS['v2c3'])
    ax.plot(snr_dB_list_np, data_dict['v2_c4'],      '-',              color=COLORS['v2c4'],       linewidth=3.5, marker='*', markersize=8, label=LABELS['v2c4'])
    ax.set_xticks(snr_dB_list_np)
    ax.grid()
    ax.tick_params(labelsize=11)
    if show_legend:
        ax.legend(loc='best', fontsize=11)

# ── Figure 4: Rate vs SNR — no legend ────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(7, 5), num=4)
plot_snr_curves(ax4, rate_vs_snr, show_legend=True)
ax4.set_xlabel('SNR [dB]', fontsize=14)
ax4.set_ylabel(r'$R$ [bits/s/Hz]', fontsize=14)
ax4.legend(loc='upper left', fontsize=12)
plt.tight_layout()
plt.savefig(directory_result + f'v2_30l_rate_vs_SNR_I{EVAL_ITERS}_sub.png',
            dpi=300, bbox_inches='tight')
plt.savefig(directory_result + f'v2_30l_rate_vs_SNR_I{EVAL_ITERS}_sub.eps',
            bbox_inches='tight')
print('Figure 4 saved.')

# ── Figure 5: Beam MSE vs SNR — legend here ───────────────────────────────────
fig5, ax5 = plt.subplots(figsize=(7, 5), num=5)
plot_snr_curves(ax5, mse_vs_snr, show_legend=True)
ax5.set_xlabel('SNR [dB]', fontsize=14)
ax5.set_ylabel('Average radar beampattern MSE [dB]', fontsize=14)
ax5.legend(loc='lower left', fontsize=12)
plt.tight_layout()
plt.savefig(directory_result + f'v2_30l_beam_vs_SNR_I{EVAL_ITERS}_sub.png',
            dpi=300, bbox_inches='tight')
plt.savefig(directory_result + f'v2_30l_beam_vs_SNR_I{EVAL_ITERS}_sub.eps',
            bbox_inches='tight')
print('Figure 5 saved.')


# ══════════════════════════════════════════════════════════════════════════════
# SNR SWEEP PLOTS
# ══════════════════════════════════════════════════════════════════════════════

#def snr_plot(fig_num, data_dict, ylabel, filename):
#    plt.figure(fig_num)
#    plt.plot(snr_dB_list_np, data_dict['J20'],        '-',              color=COLORS['teacher'],    linewidth=3,   marker='o', markersize=6, label=LABELS['teacher'])
#    plt.plot(snr_dB_list_np, data_dict['conv'],       '--',             color=COLORS['conv'],       linewidth=3,   marker='s', markersize=6, label=LABELS['conv'])
#    plt.plot(snr_dB_list_np, data_dict['conv_gi'],    ':',              color=COLORS['conv_gi'],    linewidth=2.5, marker='v', markersize=6, label=LABELS['conv_gi'])
#    plt.plot(snr_dB_list_np, data_dict['upga_j1'],    '-.',             color=COLORS['upga_j1'],    linewidth=3,   marker='^', markersize=6, label=LABELS['upga_j1'])
#    plt.plot(snr_dB_list_np, data_dict['upga_j1_gi'], ls=(0,(3,1,1,1)),color=COLORS['upga_j1_gi'], linewidth=2.5, marker='p', markersize=6, label=LABELS['upga_j1_gi'])
#    plt.plot(snr_dB_list_np, data_dict['v2_c1'],      ':',              color=COLORS['v2c1'],       linewidth=3,   marker='h', markersize=6, label=LABELS['v2c1'])
#    plt.plot(snr_dB_list_np, data_dict['v2_c2'],      '-.',             color=COLORS['v2c2'],       linewidth=2.5, marker='x', markersize=6, label=LABELS['v2c2'])
#    plt.plot(snr_dB_list_np, data_dict['v2_c3'],      ls=(0,(5,2)),     color=COLORS['v2c3'],       linewidth=2.5, marker='D', markersize=6, label=LABELS['v2c3'])
#    plt.plot(snr_dB_list_np, data_dict['v2_c4'],      '-',              color=COLORS['v2c4'],       linewidth=3.5, marker='*', markersize=8, label=LABELS['v2c4'])
#    plt.xlabel('SNR [dB]', fontsize=14)
#    plt.ylabel(ylabel, fontsize=14)
#    plt.xticks(snr_dB_list_np)
#    plt.grid()
#    plt.legend(loc='best', fontsize=11)
#    plt.tight_layout()
#    plt.savefig(directory_result + filename + '.png')
 #   plt.savefig(directory_result + filename + '.eps')

#snr_plot(4, rate_vs_snr,
#         r'$R$ [bits/s/Hz]',
 #        f'v2_30l_rate_vs_SNR_I{EVAL_ITERS}_neww')

#snr_plot(5, mse_vs_snr,
 #        'Average radar beampattern MSE [dB]',
  #       f'v2_30l_beam_vs_SNR_I{EVAL_ITERS}_neww')


# ── Figure 6: Rate vs Beam MSE (Pareto frontier, full iterations) ─────────────
print('Computing full-depth Pareto sweep...')

rate_pareto = {k: [] for k in keys}
mse_pareto  = {k: [] for k in keys}

for ss in range(len(snr_dB_list)):
    snr_dB_ss = snr_dB_list[ss]
    snr_ss    = 10 ** (snr_dB_ss / 10)

    R_ss, at_ss, _, _ = get_radar_data(snr_dB_ss, H_test)
    at_ss = at_ss[:, :test_size, :, :]

    with torch.no_grad():
        # Teacher — full depth
        _, _, F, W = model_UPGA_J20.execute_PGA(
            H_test, R_ss, snr_ss, n_iter_outer, n_iter_inner_J20)
        rate_pareto['J20'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_pareto['J20'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        # Conventional PGA — full depth
        _, _, F, W = model_conv_PGA.execute_PGA(
            H_test, R_ss, snr_ss, n_iter_outer)
        rate_pareto['conv'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_pareto['conv'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        # Conv PGA + GI — full depth
        _, _, F, W = model_conv_GI_notrain.execute_PGA(
            H_test, R_ss, snr_ss, n_iter_outer)
        rate_pareto['conv_gi'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_pareto['conv_gi'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        # Unfolded PGA J=1 — full depth
        _, _, F, W = model_UPGA_J1.execute_PGA(
            H_test, R_ss, snr_ss, n_iter_outer)
        rate_pareto['upga_j1'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_pareto['upga_j1'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        # Unfolded PGA J=1 + GI + CIRS — full depth
        _, _, F, W = model_J1_AGT.execute_PGA(
            H_test, R_ss, snr_ss, _actual_I_J1_AGT)
        rate_pareto['upga_j1_gi'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_pareto['upga_j1_gi'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        # Student J=10, I=60 — full 60 iters
        _, _, F, W = model_v2_c1.execute_PGA(
            H_test, R_ss, snr_ss, 60, n_iter_inner_J10)
        rate_pareto['v2_c1'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_pareto['v2_c1'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        # Student + CIRS — full 60 iters
        _, _, F, W = model_v2_c2.execute_PGA(
            H_test, R_ss, snr_ss, 60, n_iter_inner_J10)
        rate_pareto['v2_c2'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_pareto['v2_c2'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        # Student + GI — full 60 iters
        _, _, F, W = model_v2_c3.execute_PGA(
            H_test, R_ss, snr_ss, 60, n_iter_inner_J10)
        rate_pareto['v2_c3'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_pareto['v2_c3'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        # Student + GI + CIRS — full 60 iters
        _, _, F, W = model_v2_c4.execute_PGA(
            H_test, R_ss, snr_ss, 60, n_iter_inner_J10)
        rate_pareto['v2_c4'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_pareto['v2_c4'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

# Sort each curve by MSE so the Pareto frontier draws cleanly
def sort_by_mse(mse_list, rate_list):
    pairs = sorted(zip(mse_list, rate_list), key=lambda x: x[0])
    mse_sorted, rate_sorted = zip(*pairs)
    return list(mse_sorted), list(rate_sorted)

fig, ax = plt.subplots(figsize=(7, 5))

mse_s, rate_s = sort_by_mse(mse_pareto['J20'], rate_pareto['J20'])
ax.plot(mse_s, rate_s, '-', color=COLORS['teacher'], linewidth=3, marker='o', markersize=6, label=LABELS['teacher'])

mse_s, rate_s = sort_by_mse(mse_pareto['conv'], rate_pareto['conv'])
ax.plot(mse_s, rate_s, '--', color=COLORS['conv'], linewidth=3, marker='s', markersize=6, label=LABELS['conv'])

mse_s, rate_s = sort_by_mse(mse_pareto['conv_gi'], rate_pareto['conv_gi'])
ax.plot(mse_s, rate_s, ':', color=COLORS['conv_gi'], linewidth=2.5, marker='v', markersize=6, label=LABELS['conv_gi'])

mse_s, rate_s = sort_by_mse(mse_pareto['upga_j1'], rate_pareto['upga_j1'])
ax.plot(mse_s, rate_s, '-.', color=COLORS['upga_j1'], linewidth=3, marker='^', markersize=6, label=LABELS['upga_j1'])

mse_s, rate_s = sort_by_mse(mse_pareto['upga_j1_gi'], rate_pareto['upga_j1_gi'])
ax.plot(mse_s, rate_s, linestyle=(0,(3,1,1,1)), color=COLORS['upga_j1_gi'], linewidth=2.5, marker='p', markersize=6, label=LABELS['upga_j1_gi'])

mse_s, rate_s = sort_by_mse(mse_pareto['v2_c1'], rate_pareto['v2_c1'])
ax.plot(mse_s, rate_s, ':', color=COLORS['v2c1'], linewidth=3, marker='h', markersize=6, label=LABELS['v2c1'])

mse_s, rate_s = sort_by_mse(mse_pareto['v2_c2'], rate_pareto['v2_c2'])
ax.plot(mse_s, rate_s, '-.', color=COLORS['v2c2'], linewidth=2.5, marker='x', markersize=6, label=LABELS['v2c2'])

mse_s, rate_s = sort_by_mse(mse_pareto['v2_c3'], rate_pareto['v2_c3'])
ax.plot(mse_s, rate_s, linestyle=(0,(5,2)), color=COLORS['v2c3'], linewidth=2.5, marker='D', markersize=6, label=LABELS['v2c3'])

mse_s, rate_s = sort_by_mse(mse_pareto['v2_c4'], rate_pareto['v2_c4'])
ax.plot(mse_s, rate_s, '-', color=COLORS['v2c4'], linewidth=3.5, marker='*', markersize=8, label=LABELS['v2c4'])

#Annotate only 0, 6, 12 dB on teacher curve
#for i, snr_val in enumerate(snr_dB_list_np):
    #if snr_val in [0, 6, 12]:
        #ax.annotate(f'{int(snr_val)}dB',
                   # xy=(mse_pareto['J20'][i], rate_pareto['J20'][i]),
                   # xytext=(6, 4),
                   # textcoords='offset points',
                   # fontsize=8,
                   # color=COLORS['teacher'],
                   # fontweight='bold')

ax.set_xlabel(r'Average radar beampattern MSE [dB]', fontsize=14)
ax.set_ylabel(r'$R$ [bits/s/Hz]', fontsize=14)
ax.grid()
ax.legend(loc='best',
          bbox_to_anchor=(1.01, 1),
          borderaxespad=0,
          fontsize=11,
          ncol=1)
plt.tight_layout(rect=[0, 0, 0.75, 1])
plt.savefig(directory_result + f'v2_30l_rate_vs_beam_full.png',
            dpi=300, bbox_inches='tight')
plt.savefig(directory_result + f'v2_30l_rate_vs_beam_full.eps',
            bbox_inches='tight')
print('Figure 6 saved.')

# plt.show()
# print('\nDone.')


# ── Figure 6: Rate vs Beam MSE (Pareto frontier) ──────────────────────────────
# plt.figure(6)
# plt.plot(mse_vs_snr['J20'],        rate_vs_snr['J20'],        '-',              color=COLORS['teacher'],    linewidth=3,   marker='o', markersize=6, label=LABELS['teacher'])
# plt.plot(mse_vs_snr['conv'],       rate_vs_snr['conv'],       '--',             color=COLORS['conv'],       linewidth=3,   marker='s', markersize=6, label=LABELS['conv'])
# plt.plot(mse_vs_snr['conv_gi'],    rate_vs_snr['conv_gi'],    ':',              color=COLORS['conv_gi'],    linewidth=2.5, marker='v', markersize=6, label=LABELS['conv_gi'])
# plt.plot(mse_vs_snr['upga_j1'],    rate_vs_snr['upga_j1'],    '-.',             color=COLORS['upga_j1'],    linewidth=3,   marker='^', markersize=6, label=LABELS['upga_j1'])
# plt.plot(mse_vs_snr['upga_j1_gi'], rate_vs_snr['upga_j1_gi'], ls=(0,(3,1,1,1)),color=COLORS['upga_j1_gi'], linewidth=2.5, marker='p', markersize=6, label=LABELS['upga_j1_gi'])
# plt.plot(mse_vs_snr['v2_c1'],      rate_vs_snr['v2_c1'],      ':',              color=COLORS['v2c1'],       linewidth=3,   marker='h', markersize=6, label=LABELS['v2c1'])
# plt.plot(mse_vs_snr['v2_c2'],      rate_vs_snr['v2_c2'],      '-.',             color=COLORS['v2c2'],       linewidth=2.5, marker='x', markersize=6, label=LABELS['v2c2'])
# plt.plot(mse_vs_snr['v2_c3'],      rate_vs_snr['v2_c3'],      ls=(0,(5,2)),     color=COLORS['v2c3'],       linewidth=2.5, marker='D', markersize=6, label=LABELS['v2c3'])
# plt.plot(mse_vs_snr['v2_c4'],      rate_vs_snr['v2_c4'],      '-',              color=COLORS['v2c4'],       linewidth=3.5, marker='*', markersize=8, label=LABELS['v2c4'])

#for i, snr_val in enumerate(snr_dB_list_np):
 #   if snr_val % 4 == 0:
  #      plt.annotate(f'{int(snr_val)}dB',
   #                  xy=(mse_vs_snr['J20'][i], rate_vs_snr['J20'][i]),
    #                 xytext=(4, -10),
     #                textcoords='offset points',
      #               fontsize=7,
       #              color=COLORS['teacher'])

# plt.xlabel(r'Average radar beampattern MSE [dB]', fontsize=14)
# plt.ylabel(r'$R$ [bits/s/Hz]', fontsize=14)
# plt.grid()
# plt.legend(loc='best', fontsize=11)
# plt.tight_layout()
# plt.savefig(directory_result + f'v2_30l_rate_vs_beam_I{EVAL_ITERS}_neww.png')
# plt.savefig(directory_result + f'v2_30l_rate_vs_beam_I{EVAL_ITERS}_neww.eps')

# plt.show()
# print('\nDone.')
