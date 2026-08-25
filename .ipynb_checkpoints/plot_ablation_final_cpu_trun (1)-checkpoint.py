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

iter_number_UPGA_J20     = np.array(list(range(n_iter_outer + 1)))   # [0..120]
iter_number_UPGA_J10_I60 = np.array(list(range(60 + 1)))             # [0..60]

# ── Load models ───────────────────────────────────────────────────────────────
print('Loading models...')

# ── Teacher (J=20, I=120) ─────────────────────────────────────────────────────
model_UPGA_J20 = PGA_Unfold_J20(step_size_UPGA_J20)
model_UPGA_J20.load_state_dict(
    torch.load('./model/UPGA_J20_320.pth', map_location='cpu'))
model_UPGA_J20.eval()
with torch.no_grad():
    sum_rate_J20, tau_J20, _, _ = model_UPGA_J20.execute_PGA(
        H_test, R, snr, n_iter_outer, n_iter_inner_J20)
rate_iter_J20 = [r.detach().numpy() for r in (sum(sum_rate_J20) / len(H_test[0]))]
tau_iter_J20  = [e.detach().numpy() for e in (sum(tau_J20)      / len(H_test[0]))]

# ── Conventional PGA (J=1, I=120) ────────────────────────────────────────────
model_UPGA_J1 = PGA_Conv(step_size_UPGA_J1)
model_UPGA_J1.load_state_dict(
    torch.load('./model/UPGA_J1.pth', map_location='cpu'))
model_UPGA_J1.eval()
with torch.no_grad():
    sum_rate_J1, tau_J1, _, _ = model_UPGA_J1.execute_PGA(
        H_test, R, snr, n_iter_outer)
rate_iter_J1 = [r.detach().numpy() for r in (sum(sum_rate_J1) / len(H_test[0]))]
tau_iter_J1  = [e.detach().numpy() for e in (sum(tau_J1)      / len(H_test[0]))]

# ── Conventional PGA + GI ─────────────────────────────────────────────────────
_ckpt_J1_AGT     = torch.load(
    './model/UPGA_J1.pth_I120_AGT_teacher_avg_inner.pth',
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

# by creating a PGA_Unfold_J20 with J10 step sizes
model_RKD2 = PGA_Unfold_J10(step_size_UPGA_J10)
model_RKD2.load_state_dict(torch.load('./model/UPGA_J10_all.pth',map_location='cpu')) 
model_RKD2.eval()
with torch.no_grad():
    sum_rate_RKD2, tau_RKD2, _, _ = model_RKD2.execute_PGA(
        H_test, R, snr, n_iter_outer, n_iter_inner_J10)
rate_iter_RKD2 = [r.detach().numpy() for r in (sum(sum_rate_RKD2) / len(H_test[0]))][:61]
tau_iter_RKD2  = [e.detach().numpy() for e in (sum(tau_RKD2)      / len(H_test[0]))][:61]

# Full 120 iterations
rate_iter_RKD2_full = [r.detach().numpy() for r in (sum(sum_rate_RKD2) / len(H_test[0]))]
tau_iter_RKD2_full  = [e.detach().numpy() for e in (sum(tau_RKD2)      / len(H_test[0]))]

# ── Student + CI-RKD (flat init + RKD, I=60) ─────────────────────────────────
model_RKD3 = PGA_Unfold_J10_I60(step_size_UPGA_J10_I60)
model_RKD3.load_state_dict(torch.load(
    './model/64TX_4UE_4RF/UPGA_J10_all.pth_I60_noinit_CI_RKD_dist25.0_angle50.0_Kl15_win20.pth',
    map_location='cpu'))
model_RKD3.eval()
with torch.no_grad():
    sum_rate_RKD3, tau_RKD3, _, _ = model_RKD3.execute_PGA(
        H_test, R, snr, 60, n_iter_inner_J10)
rate_iter_RKD3 = [r.detach().numpy() for r in (sum(sum_rate_RKD3) / len(H_test[0]))]
tau_iter_RKD3  = [e.detach().numpy() for e in (sum(tau_RKD3)      / len(H_test[0]))]

# ── Student + GI (AGT init, no RKD, I=60) ────────────────────────────────────
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

# ── Student + GI + CI-RKD (proposed, I=60) ───────────────────────────────────
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

print('All models loaded.\n')

# ── Compute objectives ────────────────────────────────────────────────────────
obj_iter_J20    = [r - OMEGA * t for r, t in zip(rate_iter_J20,    tau_iter_J20)]
obj_iter_J1     = [r - OMEGA * t for r, t in zip(rate_iter_J1,     tau_iter_J1)]
obj_iter_J1_AGT = [r - OMEGA * t for r, t in zip(rate_iter_J1_AGT, tau_iter_J1_AGT)]
obj_iter_RKD2   = [r - OMEGA * t for r, t in zip(rate_iter_RKD2,   tau_iter_RKD2)]
obj_iter_RKD3   = [r - OMEGA * t for r, t in zip(rate_iter_RKD3,   tau_iter_RKD3)]
obj_iter_RKD1   = [r - OMEGA * t for r, t in zip(rate_iter_RKD1,   tau_iter_RKD1)]
obj_iter_RKD    = [r - OMEGA * t for r, t in zip(rate_iter_RKD,    tau_iter_RKD)]
obj_iter_RKD2_full = [r - OMEGA * t for r, t in zip(rate_iter_RKD2_full, tau_iter_RKD2_full)]

x_long  = iter_number_UPGA_J20        # [0..120] teacher and conv PGA
x_short = iter_number_UPGA_J10_I60    # [0..60]  all student models

# ── Print final values ────────────────────────────────────────────────────────
print('\n' + '='*78)
print(f'Final values at SNR = {snr_dB} dB,  omega = {OMEGA}')
print('='*78)
print(f'  {"Model":<42} {"Obj":>8} {"R [bits/s/Hz]":>15} {"Beam Error":>10}')
print(f'  {"-"*78}')
results = {
    'Teacher (J=20, I=120)'                  : (obj_iter_J20[-1],    rate_iter_J20[-1],    tau_iter_J20[-1]),
    'Conventional PGA'                        : (obj_iter_J1[-1],     rate_iter_J1[-1],     tau_iter_J1[-1]),
    'Conventional PGA+GI'                     : (obj_iter_J1_AGT[-1], rate_iter_J1_AGT[-1], tau_iter_J1_AGT[-1]),
    'Student (J=10,I=60)'   : (obj_iter_RKD2[-1],   rate_iter_RKD2[-1],   tau_iter_RKD2[-1]),
    'Student (J=10,I=120)'                    : (obj_iter_RKD2_full[-1],   rate_iter_RKD2_full[-1],   tau_iter_RKD2_full[-1]),
    'Student+CI-RKD'                          : (obj_iter_RKD3[-1],   rate_iter_RKD3[-1],   tau_iter_RKD3[-1]),
    'Student+GI'                              : (obj_iter_RKD1[-1],   rate_iter_RKD1[-1],   tau_iter_RKD1[-1]),
    'Student+GI+CI-RKD'                       : (obj_iter_RKD[-1],    rate_iter_RKD[-1],    tau_iter_RKD[-1]),
    
}
for name, (obj, rate, tau) in results.items():
    print(f'  {name:<42} {obj:>8.4f} {rate:>15.4f} {tau:>10.4f}')
print('='*78 + '\n')

# ══════════════════════════════════════════════════════════════════════════════
# CONVERGENCE PLOTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Figure 1: Objective vs Iterations ────────────────────────────────────────
plt.figure(1)
plt.plot(x_long,             obj_iter_J20,    '-',          color='#2CA02C', linewidth=3, marker='o', markevery=8, markersize=6, label=r'Teacher $(J=20, I=120)$')
plt.plot(x_long,             obj_iter_J1,     '--',         color='#FF7F0E', linewidth=3, marker='s', markevery=8, markersize=6, label=r'Conventional PGA')
plt.plot(iter_number_J1_AGT, obj_iter_J1_AGT, '-.',         color='#17BECF', linewidth=3, marker='p', markevery=8, markersize=6, label=r'Conventional PGA+GI')
plt.plot(x_short,            obj_iter_RKD2,   ':',          color='#0047FF', linewidth=3, marker='h', markevery=6, markersize=6, label=r'Student $(J=10, I=60)$')
plt.plot(x_short,            obj_iter_RKD3,   '-.',         color='#E377C2', linewidth=3, marker='v', markevery=6, markersize=6, label=r'Student+CI-RKD')
plt.plot(x_short,            obj_iter_RKD1,   ls=(0,(5,2)), color='#D62728', linewidth=3, marker='D', markevery=6, markersize=6, label=r'Student+GI')
plt.plot(x_short,            obj_iter_RKD,    '-',          color='#9467BD', linewidth=3.5,marker='*', markevery=6, markersize=8, label=r'Student+GI+CI-RKD')
# Full 120 iterations
plt.plot(x_long,  obj_iter_RKD2_full, ':',  color='#8C564B', linewidth=2, marker='h', markevery=8, markersize=5, label=r'Student $(J=10, I=120)$', alpha=0.5)
plt.xlabel(r'Number of iterations/layers $(I)$', fontsize=14)
plt.ylabel(r'$R - \lambda\bar{\tau}$', fontsize=14)
plt.xlim(0, 120)
plt.grid()
plt.legend(loc='best', fontsize=11)
plt.tight_layout()
plt.savefig(directory_result + 'ablation_obj_vs_iter.png')
plt.savefig(directory_result + 'ablation_obj_vs_iter.eps')

# ── Figure 2: Rate vs Iterations ─────────────────────────────────────────────
plt.figure(2)
plt.plot(x_long,             rate_iter_J20,    '-',          color='#2CA02C', linewidth=3, marker='o', markevery=8, markersize=6, label=r'Teacher $(J=20, I=120)$')
plt.plot(x_long,             rate_iter_J1,     '--',         color='#FF7F0E', linewidth=3, marker='s', markevery=8, markersize=6, label=r'Conventional PGA')
plt.plot(iter_number_J1_AGT, rate_iter_J1_AGT, '-.',         color='#17BECF', linewidth=3, marker='p', markevery=8, markersize=6, label=r'Conventional PGA+GI')
plt.plot(x_short,            rate_iter_RKD2,   ':',          color='#0047FF', linewidth=3, marker='h', markevery=6, markersize=6, label=r'Student $(J=10, I=60)$')
plt.plot(x_short,            rate_iter_RKD3,   '-.',         color='#E377C2', linewidth=3, marker='v', markevery=6, markersize=6, label=r'Student+CI-RKD')
plt.plot(x_short,            rate_iter_RKD1,   ls=(0,(5,2)), color='#D62728', linewidth=3, marker='D', markevery=6, markersize=6, label=r'Student+GI')
plt.plot(x_short,            rate_iter_RKD,    '-',          color='#9467BD', linewidth=3.5,marker='*', markevery=6, markersize=8, label=r'Student+GI+CI-RKD')
# Full 120 iterations
plt.plot(x_long,  rate_iter_RKD2_full, ':',  color='#8C564B', linewidth=2, marker='h', markevery=8, markersize=5, label=r'Student $(J=10, I=120)$', alpha=0.5)
plt.xlabel(r'Number of iterations/layers $(I)$', fontsize=14)
plt.ylabel(r'$R$ [bits/s/Hz]', fontsize=14)
plt.xlim(0, 120)
plt.grid()
plt.legend(loc='best', fontsize=11)
plt.tight_layout()
plt.savefig(directory_result + 'ablation_rate_vs_iter.png')
plt.savefig(directory_result + 'ablation_rate_vs_iter.eps')

# ── Figure 3: Beam Error vs Iterations ───────────────────────────────────────
plt.figure(3)
plt.plot(x_long,             tau_iter_J20,    '-',          color='#2CA02C', linewidth=3, marker='o', markevery=8, markersize=6, label=r'Teacher $(J=20, I=120)$')
plt.plot(x_long,             tau_iter_J1,     '--',         color='#FF7F0E', linewidth=3, marker='s', markevery=8, markersize=6, label=r'Conventional PGA')
plt.plot(iter_number_J1_AGT, tau_iter_J1_AGT, '-.',         color='#17BECF', linewidth=3, marker='p', markevery=8, markersize=6, label=r'Conventional PGA+GI')
plt.plot(x_short,            tau_iter_RKD2,   ':',          color='#0047FF', linewidth=3, marker='h', markevery=6, markersize=6, label=r'Student $(J=10, I=60)$')
plt.plot(x_short,            tau_iter_RKD3,   '-.',         color='#E377C2', linewidth=3, marker='v', markevery=6, markersize=6, label=r'Student+CI-RKD')
plt.plot(x_short,            tau_iter_RKD1,   ls=(0,(5,2)), color='#D62728', linewidth=3, marker='D', markevery=6, markersize=6, label=r'Student+GI')
plt.plot(x_short,            tau_iter_RKD,    '-',          color='#9467BD', linewidth=3.5,marker='*', markevery=6, markersize=8, label=r'Student+GI+CI-RKD')
# Full 120 iterations
plt.plot(x_long,  tau_iter_RKD2_full, ':',  color='#8C564B', linewidth=2, marker='h', markevery=8, markersize=5, label=r'Student $(J=10, I=120)$', alpha=0.5)
plt.xlabel(r'Number of iterations/layers $(I)$', fontsize=14)
plt.ylabel(r'$\bar{\tau}$', fontsize=14)
plt.xlim(0, 120)
plt.grid()
plt.legend(loc='best', fontsize=11)
plt.tight_layout()
plt.savefig(directory_result + 'ablation_beam_vs_iter.png')
plt.savefig(directory_result + 'ablation_beam_vs_iter.eps')

# ══════════════════════════════════════════════════════════════════════════════
# SNR SWEEP
# ══════════════════════════════════════════════════════════════════════════════
print('Running SNR sweep...')

rate_vs_snr = {'J20': [], 'J1': [], 'J1_AGT': [], 'J10_trunc': [], 'J10': [],
               'flat_rkd': [], 'agt': [], 'agt_rkd': []}
mse_vs_snr  = {'J20': [], 'J1': [], 'J1_AGT': [], 'J10_trunc': [], 'J10': [],
               'flat_rkd': [], 'agt': [], 'agt_rkd': []}

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

        _, _, F, W = model_UPGA_J1.execute_PGA(
            H_test, R_ss, snr_ss, n_iter_outer)
        rate_vs_snr['J1'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['J1'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        _, _, F, W = model_J1_AGT.execute_PGA(
            H_test, R_ss, snr_ss, _actual_I_J1_AGT)
        rate_vs_snr['J1_AGT'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['J1_AGT'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        # Truncated: run only 60 iterations to get F,W at I=60
        _, _, F, W = model_RKD2.execute_PGA(
            H_test, R_ss, snr_ss, 60, n_iter_inner_J10)
        rate_vs_snr['J10_trunc'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['J10_trunc'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        # Truncated: run only 60 iterations to get F,W at I=60
        _, _, F, W = model_RKD2.execute_PGA(
            H_test, R_ss, snr_ss, n_iter_outer, n_iter_inner_J10)
        rate_vs_snr['J10'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['J10'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        _, _, F, W = model_RKD3.execute_PGA(
            H_test, R_ss, snr_ss, 60, n_iter_inner_J10)
        rate_vs_snr['flat_rkd'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['flat_rkd'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        _, _, F, W = model_RKD1.execute_PGA(
            H_test, R_ss, snr_ss, 60, n_iter_inner_J10)
        rate_vs_snr['agt'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['agt'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

        _, _, F, W = model_RKD.execute_PGA(
            H_test, R_ss, snr_ss, 60, n_iter_inner_J10)
        rate_vs_snr['agt_rkd'].append(get_sum_rate(H_test, F, W, snr_ss).item())
        mse_vs_snr['agt_rkd'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

snr_dB_list_np = np.array(snr_dB_list)

# ── Figure 4: Rate vs SNR ─────────────────────────────────────────────────────
plt.figure(4)
plt.plot(snr_dB_list_np, rate_vs_snr['J20'],       '-',          color='#2CA02C', linewidth=3, marker='o', markersize=6, label=r'Teacher $(J=20, I=120)$')
plt.plot(snr_dB_list_np, rate_vs_snr['J1'],        '--',         color='#FF7F0E', linewidth=3, marker='s', markersize=6, label=r'Conventional PGA')
plt.plot(snr_dB_list_np, rate_vs_snr['J1_AGT'],    '-.',         color='#17BECF', linewidth=3, marker='p', markersize=6, label=r'Conventional PGA+GI')
plt.plot(snr_dB_list_np, rate_vs_snr['J10_trunc'], ':',          color='#0047FF', linewidth=3, marker='h', markersize=6, label=r'Student $(J=10, I=60)$')
plt.plot(snr_dB_list_np, rate_vs_snr['flat_rkd'],  '-.',         color='#E377C2', linewidth=3, marker='v', markersize=6, label=r'Student+CI-RKD')
plt.plot(snr_dB_list_np, rate_vs_snr['agt'],       ls=(0,(5,2)), color='#D62728', linewidth=3, marker='D', markersize=6, label=r'Student+GI')
plt.plot(snr_dB_list_np, rate_vs_snr['agt_rkd'],   '-',          color='#9467BD', linewidth=3.5,marker='*', markersize=8, label=r'Student+GI+CI-RKD')
# Full 120 iterations
plt.plot(snr_dB_list_np, rate_vs_snr['J10'], ':', color='#8C564B', linewidth=2, marker='h', markevery=8, markersize=5, label=r'Student $(J=10, I=120)$', alpha=0.5)
plt.xlabel('SNR [dB]', fontsize=14)
plt.ylabel(r'$R$ [bits/s/Hz]', fontsize=14)
plt.xticks(snr_dB_list_np)
plt.grid()
plt.legend(loc='best', fontsize=11)
plt.tight_layout()
plt.savefig(directory_result + 'ablation_rate_vs_SNR.png')
plt.savefig(directory_result + 'ablation_rate_vs_SNR.eps')

# ── Figure 5: Beam MSE vs SNR ─────────────────────────────────────────────────
plt.figure(5)
plt.plot(snr_dB_list_np, mse_vs_snr['J20'],       '-',          color='#2CA02C', linewidth=3, marker='o', markersize=6, label=r'Teacher $(J=20, I=120)$')
plt.plot(snr_dB_list_np, mse_vs_snr['J1'],        '--',         color='#FF7F0E', linewidth=3, marker='s', markersize=6, label=r'Conventional PGA')
plt.plot(snr_dB_list_np, mse_vs_snr['J1_AGT'],    '-.',         color='#17BECF', linewidth=3, marker='p', markersize=6, label=r'Conventional PGA+GI')
plt.plot(snr_dB_list_np, mse_vs_snr['J10_trunc'], ':',          color='#0047FF', linewidth=3, marker='h', markersize=6, label=r'Student $(J=10, I=60)$')
plt.plot(snr_dB_list_np, mse_vs_snr['flat_rkd'],  '-.',         color='#E377C2', linewidth=3, marker='v', markersize=6, label=r'Student+CI-RKD')
plt.plot(snr_dB_list_np, mse_vs_snr['agt'],       ls=(0,(5,2)), color='#D62728', linewidth=3, marker='D', markersize=6, label=r'Student+GI')
plt.plot(snr_dB_list_np, mse_vs_snr['agt_rkd'],   '-',          color='#9467BD', linewidth=3.5,marker='*', markersize=8, label=r'Student+GI+CI-RKD')
# Full 120 iterations
plt.plot(snr_dB_list_np,  mse_vs_snr['J10'], ':',  color='#8C564B', linewidth=2, marker='h', markevery=8, markersize=5, label=r'Student $(J=10, I=120)$', alpha=0.5)
plt.xlabel('SNR [dB]', fontsize=14)
plt.ylabel('Average radar beampattern MSE [dB]', fontsize=14)
plt.xticks(snr_dB_list_np)
plt.grid()
plt.legend(loc='best', fontsize=11)
plt.tight_layout()
plt.savefig(directory_result + 'ablation_beam_vs_SNR.png')
plt.savefig(directory_result + 'ablation_beam_vs_SNR.eps')

plt.show()
print('\nDone.')