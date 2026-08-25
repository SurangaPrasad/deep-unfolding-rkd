from PGA_models import *
import random
import numpy as np
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Load data ─────────────────────────────────────────────────────────────────
H_train, H_test0 = get_data_tensor(data_source)
H_test = H_test0[:, :test_size, :, :]
R, at0, theta, ideal_beam = get_radar_data(snr_dB, H_test)
at = at0[:, :test_size, :, :]
H_test = H_test.to(device)
R = R.to(device)
at = at.to(device)

iter_number_J20      = np.array(list(range(n_iter_outer + 1)))
iter_number_J1       = np.array(list(range(n_iter_outer + 1)))
iter_number_J10_I60  = np.array(list(range(60 + 1)))

I_T = n_iter_outer       # 120
I_S = n_iter_outer // 2  # 60
J_T = n_iter_inner_J20   # 20
J_S = n_iter_inner_J10   # 10

# ── Style — same colors/markers as v4, legend inside ─────────────────────────
STYLE = {
    'teacher'  : dict(color='#2CA02C', ls='-',       lw=2.0, marker='o', markevery=8,  markersize=6, label=r'$R_T$'),
    'J1'       : dict(color='#FF7F0E', ls='--',      lw=2.0, marker='s', markevery=8,  markersize=6, label=r'$R_{J_1}$'),
    'flat'     : dict(color='#1F77B4', ls='-.',      lw=2.0, marker='^', markevery=6,  markersize=6, label=r'$R_\mathrm{flat}$'),
    'flat_rkd' : dict(color='#E377C2', ls=':',       lw=2.0, marker='v', markevery=6,  markersize=6, label=r'$R_\mathrm{flat+RKD}$'),
    'agt'      : dict(color='#D62728', ls=(0,(5,2)), lw=2.0, marker='D', markevery=6,  markersize=6, label=r'$R_\mathrm{AGT}$'),
    'agt_rkd'  : dict(color='#9467BD', ls='-',       lw=2.5, marker='*', markevery=6,  markersize=8, label=r'$R_\mathrm{AGT+RKD}$'),
}

# SNR styles — marker at every point
STYLE_SNR = {k: {**v, 'markevery': 1} for k, v in STYLE.items()}

PLOT_KEYS = ['teacher', 'J1', 'flat', 'flat_rkd', 'agt', 'agt_rkd']

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════════════════════

print('\nLoading models...')

model_teacher = PGA_Unfold_J20(step_size_UPGA_J20).to(device)
model_teacher.load_state_dict(torch.load(model_file_name_UPGA_J20, map_location=device))
model_teacher.eval()

model_J1 = PGA_Conv(step_size_UPGA_J1).to(device)
model_J1.load_state_dict(torch.load(model_file_name_UPGA_J1, map_location=device))
model_J1.eval()

# Cell 2-2: AGT init + CI-RKD
model_agt_rkd = PGA_Unfold_J10_I60(step_size_UPGA_J10_I60).to(device)
model_agt_rkd.load_state_dict(torch.load(
    directory_model + 'UPGA_J10.pth_I60_CI_RKD_sym_inner_avg_pairs_Kl15_win20.pth', map_location=device))
model_agt_rkd.eval()

# Cell 2-1: AGT init | No RKD
model_agt = PGA_Unfold_J10_I60(step_size_UPGA_J10_I60).to(device)
model_agt.load_state_dict(torch.load(
    directory_model + 'UPGA_J10.pth_I60_init_sym_inner_avg_pairs_Kl15_win20.pth', map_location=device))
model_agt.eval()

# Cell 1-1: Flat init | No RKD
model_flat = PGA_Unfold_J10_I60(step_size_UPGA_J10_I60).to(device)
model_flat.load_state_dict(torch.load(
    directory_model + 'UPGA_J10.pth_I60_basic_sym_inner_avg_pairs_Kl15_win20.pth', map_location=device))
model_flat.eval()

# Cell 1-2: Flat init + CI-RKD
model_flat_rkd = PGA_Unfold_J10_I60(step_size_UPGA_J10_I60).to(device)
model_flat_rkd.load_state_dict(torch.load(
    directory_model + 'UPGA_J10_all.pth_I60_noinit_CI_RKD_dist25.0_angle50.0_Kl15_win20.pth', map_location=device))
model_flat_rkd.eval()

print('All models loaded.')

# ══════════════════════════════════════════════════════════════════════════════
# CONVERGENCE
# ══════════════════════════════════════════════════════════════════════════════

print('\nRunning convergence eval...')

with torch.no_grad():
    rt, tt, _, _ = model_teacher.execute_PGA(H_test, R, snr, I_T, J_T)
    rate_T   = [r.detach().cpu().numpy() for r in (sum(rt) / len(H_test[0]))]
    tau_T    = [e.detach().cpu().numpy() for e in (sum(tt) / len(H_test[0]))]

    rt, tt, _, _ = model_J1.execute_PGA(H_test, R, snr, I_T)
    rate_J1  = [r.detach().cpu().numpy() for r in (sum(rt) / len(H_test[0]))]
    tau_J1   = [e.detach().cpu().numpy() for e in (sum(tt) / len(H_test[0]))]

    rt, tt, _, _ = model_flat.execute_PGA(H_test, R, snr, I_S, J_S)
    rate_flat = [r.detach().cpu().numpy() for r in (sum(rt) / len(H_test[0]))]
    tau_flat  = [e.detach().cpu().numpy() for e in (sum(tt) / len(H_test[0]))]

    rt, tt, _, _ = model_flat_rkd.execute_PGA(H_test, R, snr, I_S, J_S)
    rate_flat_rkd = [r.detach().cpu().numpy() for r in (sum(rt) / len(H_test[0]))]
    tau_flat_rkd  = [e.detach().cpu().numpy() for e in (sum(tt) / len(H_test[0]))]

    rt, tt, _, _ = model_agt.execute_PGA(H_test, R, snr, I_S, J_S)
    rate_agt = [r.detach().cpu().numpy() for r in (sum(rt) / len(H_test[0]))]
    tau_agt  = [e.detach().cpu().numpy() for e in (sum(tt) / len(H_test[0]))]

    rt, tt, _, _ = model_agt_rkd.execute_PGA(H_test, R, snr, I_S, J_S)
    rate_agt_rkd = [r.detach().cpu().numpy() for r in (sum(rt) / len(H_test[0]))]
    tau_agt_rkd  = [e.detach().cpu().numpy() for e in (sum(tt) / len(H_test[0]))]

obj_T       = [r - OMEGA * t for r, t in zip(rate_T,       tau_T)]
obj_J1      = [r - OMEGA * t for r, t in zip(rate_J1,      tau_J1)]
obj_flat    = [r - OMEGA * t for r, t in zip(rate_flat,    tau_flat)]
obj_flat_rkd= [r - OMEGA * t for r, t in zip(rate_flat_rkd,tau_flat_rkd)]
obj_agt     = [r - OMEGA * t for r, t in zip(rate_agt,     tau_agt)]
obj_agt_rkd = [r - OMEGA * t for r, t in zip(rate_agt_rkd, tau_agt_rkd)]

x_long  = iter_number_J20
x_short = iter_number_J10_I60

# ── Figure 1: Objective vs iterations ────────────────────────────────────────
plt.figure(1)
plt.plot(x_long,  obj_T,        **STYLE['teacher'])
plt.plot(x_long,  obj_J1,       **STYLE['J1'])
plt.plot(x_short, obj_flat,     **STYLE['flat'])
plt.plot(x_short, obj_flat_rkd, **STYLE['flat_rkd'])
plt.plot(x_short, obj_agt,      **STYLE['agt'])
plt.plot(x_short, obj_agt_rkd,  **STYLE['agt_rkd'])
plt.xlabel(r'Number of iterations/layers $(I)$', fontsize=14)
plt.ylabel(r'$R - \omega\bar{\tau}$ [bits/s/Hz]', fontsize=14)
plt.xlim(0, 120)
plt.grid()
plt.legend(loc='best', fontsize=12)
plt.savefig(directory_result + 'ablation_obj_vs_iter.png')
plt.savefig(directory_result + 'ablation_obj_vs_iter.eps')
print('Saved ablation_obj_vs_iter')

# ── Figure 2: Rate vs iterations ──────────────────────────────────────────────
plt.figure(2)
plt.plot(x_long,  rate_T,        **STYLE['teacher'])
plt.plot(x_long,  rate_J1,       **STYLE['J1'])
plt.plot(x_short, rate_flat,     **STYLE['flat'])
plt.plot(x_short, rate_flat_rkd, **STYLE['flat_rkd'])
plt.plot(x_short, rate_agt,      **STYLE['agt'])
plt.plot(x_short, rate_agt_rkd,  **STYLE['agt_rkd'])
plt.xlabel(r'Number of iterations/layers $(I)$', fontsize=14)
plt.ylabel(r'$R$ [bits/s/Hz]', fontsize=14)
plt.xlim(0, 120)
plt.grid()
plt.legend(loc='best', fontsize=12)
plt.savefig(directory_result + 'ablation_rate_vs_iter.png')
plt.savefig(directory_result + 'ablation_rate_vs_iter.eps')
print('Saved ablation_rate_vs_iter')

# ── Figure 3: Beam error vs iterations ───────────────────────────────────────
plt.figure(3)
plt.plot(x_long,  tau_T,        **STYLE['teacher'])
plt.plot(x_long,  tau_J1,       **STYLE['J1'])
plt.plot(x_short, tau_flat,     **STYLE['flat'])
plt.plot(x_short, tau_flat_rkd, **STYLE['flat_rkd'])
plt.plot(x_short, tau_agt,      **STYLE['agt'])
plt.plot(x_short, tau_agt_rkd,  **STYLE['agt_rkd'])
plt.xlabel(r'Number of iterations/layers $(I)$', fontsize=14)
plt.ylabel(r'$\bar{\tau}$', fontsize=14)
plt.xlim(0, 120)
plt.grid()
plt.legend(loc='best', fontsize=12)
plt.savefig(directory_result + 'ablation_beam_vs_iter.png')
plt.savefig(directory_result + 'ablation_beam_vs_iter.eps')
print('Saved ablation_beam_vs_iter')

# ══════════════════════════════════════════════════════════════════════════════
# SNR SWEEP
# ══════════════════════════════════════════════════════════════════════════════

print('\nRunning SNR sweep...')

rate_vs_snr = {k: np.zeros(len(snr_dB_list)) for k in PLOT_KEYS}
mse_vs_snr  = {k: np.zeros(len(snr_dB_list)) for k in PLOT_KEYS}

for ss in range(len(snr_dB_list)):
    snr_dB_ss = snr_dB_list[ss]
    snr_ss    = 10 ** (snr_dB_ss / 10)
    print(f'  SNR = {snr_dB_ss} dB')

    R_ss, at_ss, _, _ = get_radar_data(snr_dB_ss, H_test.cpu())
    R_ss  = R_ss.to(device)
    at_ss = at_ss[:, :test_size, :, :].to(device)

    with torch.no_grad():
        # Teacher
        _, _, F, W = model_teacher.execute_PGA(H_test, R_ss, snr_ss, I_T, J_T)
        rate_vs_snr['teacher'][ss] = get_sum_rate(H_test, F, W, snr_ss).item()
        mse_vs_snr['teacher'][ss]  = get_MSE(F, W, at_ss, R_ss, snr_ss).item()

        # J1
        _, _, F, W = model_J1.execute_PGA(H_test, R_ss, snr_ss, I_T)
        rate_vs_snr['J1'][ss] = get_sum_rate(H_test, F, W, snr_ss).item()
        mse_vs_snr['J1'][ss]  = get_MSE(F, W, at_ss, R_ss, snr_ss).item()

        # Flat | No RKD
        _, _, F, W = model_flat.execute_PGA(H_test, R_ss, snr_ss, I_S, J_S)
        rate_vs_snr['flat'][ss] = get_sum_rate(H_test, F, W, snr_ss).item()
        mse_vs_snr['flat'][ss]  = get_MSE(F, W, at_ss, R_ss, snr_ss).item()

        # Flat + CI-RKD
        _, _, F, W = model_flat_rkd.execute_PGA(H_test, R_ss, snr_ss, I_S, J_S)
        rate_vs_snr['flat_rkd'][ss] = get_sum_rate(H_test, F, W, snr_ss).item()
        mse_vs_snr['flat_rkd'][ss]  = get_MSE(F, W, at_ss, R_ss, snr_ss).item()

        # AGT | No RKD
        _, _, F, W = model_agt.execute_PGA(H_test, R_ss, snr_ss, I_S, J_S)
        rate_vs_snr['agt'][ss] = get_sum_rate(H_test, F, W, snr_ss).item()
        mse_vs_snr['agt'][ss]  = get_MSE(F, W, at_ss, R_ss, snr_ss).item()

        # AGT + CI-RKD
        _, _, F, W = model_agt_rkd.execute_PGA(H_test, R_ss, snr_ss, I_S, J_S)
        rate_vs_snr['agt_rkd'][ss] = get_sum_rate(H_test, F, W, snr_ss).item()
        mse_vs_snr['agt_rkd'][ss]  = get_MSE(F, W, at_ss, R_ss, snr_ss).item()

# ── Figure 4: Rate vs SNR ─────────────────────────────────────────────────────
plt.figure(4)
for k in PLOT_KEYS:
    plt.plot(snr_dB_list, rate_vs_snr[k], **STYLE_SNR[k])
plt.xlabel('SNR [dB]', fontsize=14)
plt.ylabel(r'$R$ [bits/s/Hz]', fontsize=14)
plt.xticks(snr_dB_list)
plt.grid()
plt.legend(loc='best', fontsize=12)
plt.savefig(directory_result + 'ablation_rate_vs_SNR.png')
plt.savefig(directory_result + 'ablation_rate_vs_SNR.eps')
print('Saved ablation_rate_vs_SNR')

# ── Figure 5: Beam MSE vs SNR ─────────────────────────────────────────────────
plt.figure(5)
for k in PLOT_KEYS:
    plt.plot(snr_dB_list, mse_vs_snr[k], **STYLE_SNR[k])
plt.xlabel('SNR [dB]', fontsize=14)
plt.ylabel('Average radar beampattern MSE [dB]', fontsize=14)
plt.xticks(snr_dB_list)
plt.grid()
plt.legend(loc='best', fontsize=12)
plt.savefig(directory_result + 'ablation_beam_vs_SNR.png')
plt.savefig(directory_result + 'ablation_beam_vs_SNR.eps')
print('Saved ablation_beam_vs_SNR')

plt.show()
print('\nDone.')
