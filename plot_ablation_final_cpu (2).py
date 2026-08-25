from PGA_models import *
import random
import numpy as np
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

run_program = 1
plot_figure = 1
save_result = 0
seed = 3407

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ///////////////////////////////////////// LOAD DATA ///////////////////////////////////
H_train, H_test0 = get_data_tensor(data_source)
H_test = H_test0[:, :test_size, :, :]

# Load radar data at the default SNR for convergence plots
R, at0, theta, ideal_beam = get_radar_data(snr_dB, H_test)
at = at0[:, :test_size, :, :]

iter_number_UPGA_J20     = np.array(list(range(n_iter_outer + 1)))
iter_number_UPGA_J1      = np.array(list(range(n_iter_outer + 1)))
iter_number_UPGA_J10_I60 = np.array(list(range(60 + 1)))

run_RKD_Distillation  = 1
run_RKD_Distillation1 = 1
run_RKD_Distillation2 = 1
run_RKD_Distillation3 = 1

if run_program == 1:

    # ── Teacher (J=20, I=120) ─────────────────────────────────────────────────
    print('Running Teacher (J=20)...')
    model_UPGA_J20 = PGA_Unfold_J20(step_size_UPGA_J20)
    model_UPGA_J20.load_state_dict(torch.load(model_file_name_UPGA_J20, map_location='cpu'))
    model_UPGA_J20.eval()
    with torch.no_grad():
        sum_rate_J20, tau_J20, _, _ = model_UPGA_J20.execute_PGA(H_test, R, snr, n_iter_outer, n_iter_inner_J20)
    rate_iter_J20 = [r.detach().numpy() for r in (sum(sum_rate_J20) / len(H_test[0]))]
    tau_iter_J20  = [e.detach().numpy() for e in (sum(tau_J20)      / len(H_test[0]))]

    # ── J1 (I=120) ───────────────────────────────────────────────────────────
    print('Running Conventional PGA (J=1)...')
    model_UPGA_J1 = PGA_Conv(step_size_UPGA_J1)
    model_UPGA_J1.load_state_dict(torch.load('./model/UPGA_J1.pth', map_location='cpu'))
    model_UPGA_J1.eval()
    with torch.no_grad():
        sum_rate_J1, tau_J1, _, _ = model_UPGA_J1.execute_PGA(H_test, R, snr, n_iter_outer)
    rate_iter_J1 = [r.detach().numpy() for r in (sum(sum_rate_J1) / len(H_test[0]))]
    tau_iter_J1  = [e.detach().numpy() for e in (sum(tau_J1)      / len(H_test[0]))]

    # ── J1 AGT init ───────────────────────────────────────────────────────────
    print('Running Conventional PGA + GI...')
    _ckpt_J1_AGT = torch.load('./model/UPGA_J1.pth_I120_AGT_teacher_avg_inner.pth', map_location='cpu')
    _actual_I_J1_AGT = _ckpt_J1_AGT['step_size'].shape[0]
    print(f'  J1 AGT step_size shape: {list(_ckpt_J1_AGT["step_size"].shape)} -> I={_actual_I_J1_AGT}')
    _ss_J1_AGT = torch.zeros(_actual_I_J1_AGT, K + 1)
    model_J1_AGT = PGA_Conv(_ss_J1_AGT)
    model_J1_AGT.load_state_dict(_ckpt_J1_AGT)
    model_J1_AGT.eval()
    with torch.no_grad():
        sum_rate_J1_AGT, tau_J1_AGT, _, _ = model_J1_AGT.execute_PGA(H_test, R, snr, _actual_I_J1_AGT)
    rate_iter_J1_AGT = [r.detach().numpy() for r in (sum(sum_rate_J1_AGT) / len(H_test[0]))]
    tau_iter_J1_AGT  = [e.detach().numpy() for e in (sum(tau_J1_AGT)      / len(H_test[0]))]
    iter_number_J1_AGT = np.array(list(range(_actual_I_J1_AGT + 1)))

    # ── Cell 2-2: AGT init + CI-RKD ───────────────────────────────────────────
    if run_RKD_Distillation == 1:
        print('Running Student+GI+CI-RKD...')
        model_RKD = PGA_Unfold_J10_I60(step_size_UPGA_J10_I60)
        model_RKD.load_state_dict(torch.load('./model/UPGA_J10.pth_I60_CI_RKD_sym_inner_avg_pairs_Kl15_win20.pth', map_location='cpu'))
        model_RKD.eval()
        with torch.no_grad():
            sum_rate_RKD, tau_RKD, _, _ = model_RKD.execute_PGA(H_test, R, snr, 60, n_iter_inner_J10)
        rate_iter_RKD = [r.detach().numpy() for r in (sum(sum_rate_RKD) / len(H_test[0]))]
        tau_iter_RKD  = [e.detach().numpy() for e in (sum(tau_RKD)      / len(H_test[0]))]

    # ── Cell 2-1: AGT init | No RKD ───────────────────────────────────────────
    if run_RKD_Distillation1 == 1:
        print('Running Student+GI...')
        model_RKD1 = PGA_Unfold_J10_I60(step_size_UPGA_J10_I60)
        model_RKD1.load_state_dict(torch.load('./model/UPGA_J10.pth_I60_init_sym_inner_avg_pairs_Kl15_win20.pth', map_location='cpu'))
        model_RKD1.eval()
        with torch.no_grad():
            sum_rate_RKD1, tau_RKD1, _, _ = model_RKD1.execute_PGA(H_test, R, snr, 60, n_iter_inner_J10)
        rate_iter_RKD1 = [r.detach().numpy() for r in (sum(sum_rate_RKD1) / len(H_test[0]))]
        tau_iter_RKD1  = [e.detach().numpy() for e in (sum(tau_RKD1)      / len(H_test[0]))]

    # ── Cell 1-1: Flat init | No RKD ──────────────────────────────────────────
    if run_RKD_Distillation2 == 1:
        print('Running Student (flat init, no RKD)...')
        model_RKD2 = PGA_Unfold_J10_I60(step_size_UPGA_J10_I60)
        model_RKD2.load_state_dict(torch.load('./model/UPGA_J10.pth_I60_basic_sym_inner_avg_pairs_Kl15_win20.pth', map_location='cpu'))
        model_RKD2.eval()
        with torch.no_grad():
            sum_rate_RKD2, tau_RKD2, _, _ = model_RKD2.execute_PGA(H_test, R, snr, 60, n_iter_inner_J10)
        rate_iter_RKD2 = [r.detach().numpy() for r in (sum(sum_rate_RKD2) / len(H_test[0]))]
        tau_iter_RKD2  = [e.detach().numpy() for e in (sum(tau_RKD2)      / len(H_test[0]))]

    # ── Cell 1-2: Flat init + CI-RKD ──────────────────────────────────────────
    if run_RKD_Distillation3 == 1:
        print('Running Student+CI-RKD...')
        model_RKD3 = PGA_Unfold_J10_I60(step_size_UPGA_J10_I60)
        model_RKD3.load_state_dict(torch.load('./model/UPGA_J10_all.pth_I60_noinit_CI_RKD_dist25.0_angle50.0_Kl15_win20.pth', map_location='cpu'))
        model_RKD3.eval()
        with torch.no_grad():
            sum_rate_RKD3, tau_RKD3, _, _ = model_RKD3.execute_PGA(H_test, R, snr, 60, n_iter_inner_J10)
        rate_iter_RKD3 = [r.detach().numpy() for r in (sum(sum_rate_RKD3) / len(H_test[0]))]
        tau_iter_RKD3  = [e.detach().numpy() for e in (sum(tau_RKD3)      / len(H_test[0]))]


if plot_figure == 1:

    # ── Compute objectives ────────────────────────────────────────────────────
    obj_iter_J20    = [r - OMEGA * t for r, t in zip(rate_iter_J20,    tau_iter_J20)]
    obj_iter_J1     = [r - OMEGA * t for r, t in zip(rate_iter_J1,     tau_iter_J1)]
    obj_iter_J1_AGT = [r - OMEGA * t for r, t in zip(rate_iter_J1_AGT, tau_iter_J1_AGT)]
    obj_iter_RKD    = [r - OMEGA * t for r, t in zip(rate_iter_RKD,    tau_iter_RKD)]
    obj_iter_RKD1   = [r - OMEGA * t for r, t in zip(rate_iter_RKD1,   tau_iter_RKD1)]
    obj_iter_RKD2   = [r - OMEGA * t for r, t in zip(rate_iter_RKD2,   tau_iter_RKD2)]
    obj_iter_RKD3   = [r - OMEGA * t for r, t in zip(rate_iter_RKD3,   tau_iter_RKD3)]

    x_long  = iter_number_UPGA_J20
    x_short = iter_number_UPGA_J10_I60

    # ══════════════════════════════════════════════════════════════════════════
    # Figure 1 — Objective vs Iterations
    # ══════════════════════════════════════════════════════════════════════════
    plt.figure(1)
    plt.plot(x_long,             obj_iter_J20,    '-',          color='#2CA02C', linewidth=3, marker='o', markevery=8, markersize=6, label=r'Teacher $(J=20, I=120)$')
    plt.plot(x_long,             obj_iter_J1,     '--',         color='#FF7F0E', linewidth=3, marker='s', markevery=8, markersize=6, label=r'Conventional PGA')
    plt.plot(iter_number_J1_AGT, obj_iter_J1_AGT, '-.',         color='#17BECF', linewidth=3, marker='p', markevery=8, markersize=6, label=r'Conventional PGA+GI')
    plt.plot(x_short,            obj_iter_RKD2,   '-.',         color='#1F77B4', linewidth=3, marker='^', markevery=6, markersize=6, label=r'Student $(J=10, I=60)$')
    plt.plot(x_short,            obj_iter_RKD3,   ':',          color='#E377C2', linewidth=3, marker='v', markevery=6, markersize=6, label=r'Student+CI-RKD')
    plt.plot(x_short,            obj_iter_RKD1,   ls=(0,(5,2)), color='#D62728', linewidth=3, marker='D', markevery=6, markersize=6, label=r'Student+GI')
    plt.plot(x_short,            obj_iter_RKD,    '-',          color='#9467BD', linewidth=3.5, marker='*', markevery=6, markersize=8, label=r'Student+GI+CI-RKD')
    plt.xlabel(r'Number of iterations/layers $(I)$', fontsize=14)
    plt.ylabel(r'$R - \lambda\bar{\tau}$', fontsize=14)
    plt.xlim(0, 120)
    plt.grid()
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig(directory_result + 'ablation_obj_vs_iter.png')
    plt.savefig(directory_result + 'ablation_obj_vs_iter.eps')

    # ══════════════════════════════════════════════════════════════════════════
    # Figure 2 — Rate vs Iterations
    # ══════════════════════════════════════════════════════════════════════════
    plt.figure(2)
    plt.plot(x_long,             rate_iter_J20,    '-',          color='#2CA02C', linewidth=3, marker='o', markevery=8, markersize=6, label=r'Teacher $(J=20, I=120)$')
    plt.plot(x_long,             rate_iter_J1,     '--',         color='#FF7F0E', linewidth=3, marker='s', markevery=8, markersize=6, label=r'Conventional PGA')
    plt.plot(iter_number_J1_AGT, rate_iter_J1_AGT, '-.',         color='#17BECF', linewidth=3, marker='p', markevery=8, markersize=6, label=r'Conventional PGA+GI')
    plt.plot(x_short,            rate_iter_RKD2,   '-.',         color='#1F77B4', linewidth=3, marker='^', markevery=6, markersize=6, label=r'Student $(J=10, I=60)$')
    plt.plot(x_short,            rate_iter_RKD3,   ':',          color='#E377C2', linewidth=3, marker='v', markevery=6, markersize=6, label=r'Student+CI-RKD')
    plt.plot(x_short,            rate_iter_RKD1,   ls=(0,(5,2)), color='#D62728', linewidth=3, marker='D', markevery=6, markersize=6, label=r'Student+GI')
    plt.plot(x_short,            rate_iter_RKD,    '-',          color='#9467BD', linewidth=3.5, marker='*', markevery=6, markersize=8, label=r'Student+GI+CI-RKD')
    plt.xlabel(r'Number of iterations/layers $(I)$', fontsize=14)
    plt.ylabel(r'$R$ [bits/s/Hz]', fontsize=14)
    plt.xlim(0, 120)
    plt.grid()
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig(directory_result + 'ablation_rate_vs_iter.png')
    plt.savefig(directory_result + 'ablation_rate_vs_iter.eps')

    # ══════════════════════════════════════════════════════════════════════════
    # Figure 3 — Beam Error vs Iterations
    # ══════════════════════════════════════════════════════════════════════════
    plt.figure(3)
    plt.plot(x_long,             tau_iter_J20,    '-',          color='#2CA02C', linewidth=3, marker='o', markevery=8, markersize=6, label=r'Teacher $(J=20, I=120)$')
    plt.plot(x_long,             tau_iter_J1,     '--',         color='#FF7F0E', linewidth=3, marker='s', markevery=8, markersize=6, label=r'Conventional PGA')
    plt.plot(iter_number_J1_AGT, tau_iter_J1_AGT, '-.',         color='#17BECF', linewidth=3, marker='p', markevery=8, markersize=6, label=r'Conventional PGA+GI')
    plt.plot(x_short,            tau_iter_RKD2,   '-.',         color='#1F77B4', linewidth=3, marker='^', markevery=6, markersize=6, label=r'Student $(J=10, I=60)$')
    plt.plot(x_short,            tau_iter_RKD3,   ':',          color='#E377C2', linewidth=3, marker='v', markevery=6, markersize=6, label=r'Student+CI-RKD')
    plt.plot(x_short,            tau_iter_RKD1,   ls=(0,(5,2)), color='#D62728', linewidth=3, marker='D', markevery=6, markersize=6, label=r'Student+GI')
    plt.plot(x_short,            tau_iter_RKD,    '-',          color='#9467BD', linewidth=3.5, marker='*', markevery=6, markersize=8, label=r'Student+GI+CI-RKD')
    plt.xlabel(r'Number of iterations/layers $(I)$', fontsize=14)
    plt.ylabel(r'$\bar{\tau}$', fontsize=14)
    plt.xlim(0, 120)
    plt.grid()
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig(directory_result + 'ablation_beam_vs_iter.png')
    plt.savefig(directory_result + 'ablation_beam_vs_iter.eps')

    # ══════════════════════════════════════════════════════════════════════════
    # SNR SWEEP
    # ══════════════════════════════════════════════════════════════════════════
    print('Running SNR sweep...')

    rate_vs_snr = {'J20': [], 'J1': [], 'J1_AGT': [], 'flat': [], 'flat_rkd': [], 'agt': [], 'agt_rkd': []}
    mse_vs_snr  = {'J20': [], 'J1': [], 'J1_AGT': [], 'flat': [], 'flat_rkd': [], 'agt': [], 'agt_rkd': []}

    for ss in range(len(snr_dB_list)):
        snr_dB_ss = snr_dB_list[ss]
        snr_ss    = 10 ** (snr_dB_ss / 10)
        print(f'  SNR = {snr_dB_ss} dB')

        # Load radar data for this specific SNR point
        R_ss, at_ss, _, _ = get_radar_data(snr_dB_ss, H_test)
        at_ss = at_ss[:, :test_size, :, :]

        with torch.no_grad():
            _, _, F, W = model_UPGA_J20.execute_PGA(H_test, R_ss, snr_ss, n_iter_outer, n_iter_inner_J20)
            rate_vs_snr['J20'].append(get_sum_rate(H_test, F, W, snr_ss).item())
            mse_vs_snr['J20'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

            _, _, F, W = model_UPGA_J1.execute_PGA(H_test, R_ss, snr_ss, n_iter_outer)
            rate_vs_snr['J1'].append(get_sum_rate(H_test, F, W, snr_ss).item())
            mse_vs_snr['J1'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

            _, _, F, W = model_J1_AGT.execute_PGA(H_test, R_ss, snr_ss, _actual_I_J1_AGT)
            rate_vs_snr['J1_AGT'].append(get_sum_rate(H_test, F, W, snr_ss).item())
            mse_vs_snr['J1_AGT'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

            _, _, F, W = model_RKD2.execute_PGA(H_test, R_ss, snr_ss, 60, n_iter_inner_J10)
            rate_vs_snr['flat'].append(get_sum_rate(H_test, F, W, snr_ss).item())
            mse_vs_snr['flat'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

            _, _, F, W = model_RKD3.execute_PGA(H_test, R_ss, snr_ss, 60, n_iter_inner_J10)
            rate_vs_snr['flat_rkd'].append(get_sum_rate(H_test, F, W, snr_ss).item())
            mse_vs_snr['flat_rkd'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

            _, _, F, W = model_RKD1.execute_PGA(H_test, R_ss, snr_ss, 60, n_iter_inner_J10)
            rate_vs_snr['agt'].append(get_sum_rate(H_test, F, W, snr_ss).item())
            mse_vs_snr['agt'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

            _, _, F, W = model_RKD.execute_PGA(H_test, R_ss, snr_ss, 60, n_iter_inner_J10)
            rate_vs_snr['agt_rkd'].append(get_sum_rate(H_test, F, W, snr_ss).item())
            mse_vs_snr['agt_rkd'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

    snr_dB_list_np = np.array(snr_dB_list)

    # ── Figure 4: Rate vs SNR ─────────────────────────────────────────────────
    plt.figure(4)
    plt.plot(snr_dB_list_np, rate_vs_snr['J20'],      '-',          color='#2CA02C', linewidth=3, marker='o', markersize=6, label=r'Teacher $(J=20, I=120)$')
    plt.plot(snr_dB_list_np, rate_vs_snr['J1'],       '--',         color='#FF7F0E', linewidth=3, marker='s', markersize=6, label=r'Conventional PGA')
    plt.plot(snr_dB_list_np, rate_vs_snr['J1_AGT'],   '-.',         color='#17BECF', linewidth=3, marker='p', markersize=6, label=r'Conventional PGA+GI')
    plt.plot(snr_dB_list_np, rate_vs_snr['flat'],     '-.',         color='#1F77B4', linewidth=3, marker='^', markersize=6, label=r'Student $(J=10, I=60)$')
    plt.plot(snr_dB_list_np, rate_vs_snr['flat_rkd'], ':',          color='#E377C2', linewidth=3, marker='v', markersize=6, label=r'Student+CI-RKD')
    plt.plot(snr_dB_list_np, rate_vs_snr['agt'],      ls=(0,(5,2)), color='#D62728', linewidth=3, marker='D', markersize=6, label=r'Student+GI')
    plt.plot(snr_dB_list_np, rate_vs_snr['agt_rkd'],  '-',          color='#9467BD', linewidth=3.5, marker='*', markersize=8, label=r'Student+GI+CI-RKD')
    plt.xlabel('SNR [dB]', fontsize=14)
    plt.ylabel(r'$R$ [bits/s/Hz]', fontsize=14)
    plt.xticks(snr_dB_list_np)
    plt.grid()
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig(directory_result + 'ablation_rate_vs_SNR.png')
    plt.savefig(directory_result + 'ablation_rate_vs_SNR.eps')

    # ── Figure 5: Beam MSE vs SNR ─────────────────────────────────────────────
    plt.figure(5)
    plt.plot(snr_dB_list_np, mse_vs_snr['J20'],       '-',          color='#2CA02C', linewidth=3, marker='o', markersize=6, label=r'Teacher $(J=20, I=120)$')
    plt.plot(snr_dB_list_np, mse_vs_snr['J1'],        '--',         color='#FF7F0E', linewidth=3, marker='s', markersize=6, label=r'Conventional PGA')
    plt.plot(snr_dB_list_np, mse_vs_snr['J1_AGT'],    '-.',         color='#17BECF', linewidth=3, marker='p', markersize=6, label=r'Conventional PGA+GI')
    plt.plot(snr_dB_list_np, mse_vs_snr['flat'],      '-.',         color='#1F77B4', linewidth=3, marker='^', markersize=6, label=r'Student $(J=10, I=60)$')
    plt.plot(snr_dB_list_np, mse_vs_snr['flat_rkd'],  ':',          color='#E377C2', linewidth=3, marker='v', markersize=6, label=r'Student+CI-RKD')
    plt.plot(snr_dB_list_np, mse_vs_snr['agt'],       ls=(0,(5,2)), color='#D62728', linewidth=3, marker='D', markersize=6, label=r'Student+GI')
    plt.plot(snr_dB_list_np, mse_vs_snr['agt_rkd'],   '-',          color='#9467BD', linewidth=3.5, marker='*', markersize=8, label=r'Student+GI+CI-RKD')
    plt.xlabel('SNR [dB]', fontsize=14)
    plt.ylabel('Average radar beampattern MSE [dB]', fontsize=14)
    plt.xticks(snr_dB_list_np)
    plt.grid()
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig(directory_result + 'ablation_beam_vs_SNR.png')
    plt.savefig(directory_result + 'ablation_beam_vs_SNR.eps')

    

    plt.show()
    print('Done.')