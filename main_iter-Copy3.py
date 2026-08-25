from PGA_models import *
import random
import numpy as np
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

run_program = 1
plot_figure = 1
save_result = 0
seed = 3407

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ///////////////////////////////////////// SHOW OBJECTIVE VALUES OVER ITERATIONS ///////////////////////////////////
# Load training data
H_train, H_test0 = get_data_tensor(data_source)
H_test = H_test0[:, :test_size, :, :]

R, at0, theta, ideal_beam = get_radar_data(snr_dB, H_test)
at = at0[:, : test_size, :, :]
H_train = H_train.to(device)
H_test = H_test.to(device)
R = R.to(device)
at = at.to(device)

iter_number_conv_PGA = np.array(list(range(n_iter_outer + 1)))
iter_number_UPGA_J1 = np.array(list(range(n_iter_outer + 1)))
iter_number_UPGA_J10 = np.array(list(range(n_iter_outer + 1)))
iter_number_UPGA_rkd= np.array(list(range(n_iter_outer + 1)))
iter_number_UPGA_J20 = np.array(list(range(n_iter_outer + 1)))
iter_number_UPGA_J20_I60 = np.array(list(range(60 + 1)))
iter_number_UPGA_J10_I60 = np.array(list(range(60 + 1)))
iter_number_student_J10_80 = np.array(list(range(80 + 1)))

run_RKD_Distillation = 1
run_RKD_Distillation1 =1
run_RKD_Distillation2 =1
run_RKD_Distillation3 =1
run_UPGA_J10=0
run_UPGA_J20=0

if run_program == 1:
    # ====================================================== Conv. PGA ====================================
    if run_conv_PGA == 1:
        print('Running conventional PGA...')
        model_conv_PGA = PGA_Conv(step_size_conv_PGA)
        rate_conv, tau_conv, F_conv, W_conv = model_conv_PGA.execute_PGA(H_test, R, snr, n_iter_outer)
        rate_iter_conv = [r.detach().numpy() for r in (sum(rate_conv) / len(H_test[0]))]
        tau_iter_conv = [e.detach().numpy() for e in (sum(tau_conv) / (len(H_test[0])))]
    # ====================================================== Conv. PGA with J = 10 ====================================
    if run_conv_PGA_J10 == 1:
        print('Running conventional PGA with J = 10...')
        model_conv_PGA_J10 = PGA_Unfold_J10(step_size_UPGA_J10).to(device)
        rate_conv_PGA_J10, tau_conv_PGA_J10, F_conv_PGA_J10, W_conv_PGA_J10 = model_conv_PGA_J10.execute_PGA(H_test, R, snr,
                                                                                           n_iter_outer,
                                                                                           n_iter_inner_J10)
        rate_iter_conv_PGA_J10 = [r.detach().numpy() for r in (sum(rate_conv_PGA_J10) / len(H_test[0]))]
        tau_iter_conv_PGA_J10 = [e.detach().numpy() for e in (sum(tau_conv_PGA_J10) / (len(H_test[0])))]

    # ====================================================== Unfolded PGA with J = 1====================================
    if run_RKD_Distillation == 1:
        print('Running unfolded PGA with RKD distillation...')
        # Create new model and load states
        model_RKD = PGA_Unfold_J10_I60(step_size_UPGA_J10_I60).to(device)
        model_RKD.load_state_dict(torch.load(directory_model  + 'UPGA_J10_320.pth_I60_CI_RKD_sym_inner_avg_pairs_Kl15_win20_basic.pth',map_location=device))

        sum_rate_RKD, tau_RKD, F_RKD, W_RKD = model_RKD.execute_PGA(H_test, R, snr,60,n_iter_inner_J10)
        rate_iter_RKD = [r.detach().cpu().numpy() for r in (sum(sum_rate_RKD) / len(H_test[0]))]
        tau_iter_RKD = [e.detach().cpu().numpy() for e in (sum(tau_RKD) / (len(H_test[0])))]

    if run_RKD_Distillation1 == 1:
        print('Running unfolded PGA with RKD distillation...')
        # Create new model and load states
        model_RKD1 = PGA_Unfold_J10_I60(step_size_UPGA_J10_I60).to(device)
        model_RKD1.load_state_dict(torch.load(directory_model  + 'UPGA_J10_320.pth_I60_no_init_sym_inner_avg_pairs_Kl15_win20agt_new15epoch.pth',map_location=device))

        sum_rate_RKD1, tau_RKD1, F_RKD1, W_RKD1 = model_RKD1.execute_PGA(H_test, R, snr,60,n_iter_inner_J10)
        rate_iter_RKD1 = [r.detach().cpu().numpy() for r in (sum(sum_rate_RKD1) / len(H_test[0]))]
        tau_iter_RKD1 = [e.detach().cpu().numpy() for e in (sum(tau_RKD1) / (len(H_test[0])))]    
    

    if run_RKD_Distillation2 == 1:
        print('Running unfolded PGA with RKD distillation...')
        # Create new model and load states
        model_RKD2 = PGA_Unfold_J10_I60(step_size_UPGA_J10_I60).to(device)
        model_RKD2.load_state_dict(torch.load(directory_model  + 'UPGA_J10_320.pth_I60_sym_inner_avg_pairs_Kl15_win20agt_init_new.pth',map_location=device))
#UPGA_J10.pth_I60_basic_sym_inner_avg_pairs_Kl15_win20.pth
        sum_rate_RKD2, tau_RKD2, F_RKD2, W_RKD2 = model_RKD2.execute_PGA(H_test, R, snr,60,n_iter_inner_J10)
        rate_iter_RKD2 = [r.detach().cpu().numpy() for r in (sum(sum_rate_RKD2) / len(H_test[0]))]
        tau_iter_RKD2 = [e.detach().cpu().numpy() for e in (sum(tau_RKD2) / (len(H_test[0])))]    

    if run_RKD_Distillation3 == 1:
        print('Running unfolded PGA with RKD distillation...')
        # Create new model and load states
        model_RKD3 = PGA_Unfold_J10_I60(step_size_UPGA_J10_I60).to(device)
        model_RKD3.load_state_dict(torch.load(directory_model  + 'UPGA_J10_320.pth_I60_noinit_CI_RKD_dist25.0_angle50.0_Kl15_win20new.pth',map_location=device))

        sum_rate_RKD3, tau_RKD3, F_RKD3, W_RKD3= model_RKD3.execute_PGA(H_test, R, snr, 60, n_iter_inner_J10 )
        rate_iter_RKD3 = [r.detach().cpu().numpy() for r in (sum(sum_rate_RKD3) / len(H_test[0]))]
        tau_iter_RKD3 = [e.detach().cpu().numpy() for e in (sum(tau_RKD3) / (len(H_test[0])))]   

    
    # ====================================================== Proposed Unfolded PGA light ====================================
    if run_UPGA_J10 == 1:
        print('Running unfolded PGA with J = 10...')
        # Create new model and load states
        model_UPGA_J10 = PGA_Unfold_J10(step_size_UPGA_J10).to(device)
        model_UPGA_J10.load_state_dict(torch.load(directory_model  + 'UPGA_J10_all.pth',map_location=device))

        sum_rate_UPGA_J10, tau_UPGA_J10, F_UPGA_J10, W_UPGA_J10 = model_UPGA_J10.execute_PGA(H_test, R,
                                                                                             snr,
                                                                                             n_iter_outer,
                                                                                             n_iter_inner_J10)
        rate_iter_UPGA_J10 = [r.detach().cpu().numpy() for r in (sum(sum_rate_UPGA_J10) / len(H_test[0]))]
        tau_iter_UPGA_J10 = [e.detach().cpu().numpy() for e in (sum(tau_UPGA_J10) / (len(H_test[0])))]

    # ====================================================== Proposed Unfolded PGA ====================================
    if run_UPGA_J20 == 1:
        print('Running unfolded PGA with J = 20...')
        # Create new model and load states
        model_UPGA_J20 = PGA_Unfold_J20(step_size_UPGA_J20).to(device)
        model_UPGA_J20.load_state_dict(torch.load(directory_model  + 'UPGA_J20_all.pth',map_location=device))

        sum_rate_UPGA_J20, tau_UPGA_J20, F_UPGA_J20, W_UPGA_J20 = model_UPGA_J20.execute_PGA(H_test, R, snr, n_iter_outer,n_iter_inner_J20)
        rate_iter_UPGA_J20 = [r.detach().cpu().numpy() for r in (sum(sum_rate_UPGA_J20) / len(H_test[0]))]
        tau_iter_UPGA_J20 = [e.detach().cpu().numpy() for e in (sum(tau_UPGA_J20) / (len(H_test[0])))]

if plot_figure == 1:
    fig_obj=plt.figure(4)
    print('Plotting')
    # Teacher (J=20)
    if run_UPGA_J20 == 1:
        obj_iter_UPGA_J20 = [
            rate - OMEGA * tau
            for rate, tau in zip(rate_iter_UPGA_J20, tau_iter_UPGA_J20)
        ]
        plt.plot(
            iter_number_UPGA_J20,
            obj_iter_UPGA_J20,
            'g-',
            linewidth=2,
            markersize=3,
            label='Teacher (J=20)'
        )

    # Student without RKD (J=10)
    if run_UPGA_J10 == 1:
        obj_iter_UPGA_J10 = [
            rate - OMEGA * tau
            for rate, tau in zip(rate_iter_UPGA_J10, tau_iter_UPGA_J10)
        ]
        plt.plot(
            iter_number_UPGA_J10,
            obj_iter_UPGA_J10,
            'k-',
            linewidth=2,
            markersize=3,
            label='Student w/o RKD (J=10)'
        )

    # RKD student (J=10)
    if run_RKD_Distillation == 1:
        obj_iter_UPGA_J10_rkd = [
            rate - OMEGA * tau
            for rate, tau in zip(rate_iter_RKD, tau_iter_RKD)
        ]
        plt.plot(
            iter_number_UPGA_J10_I60,
            obj_iter_UPGA_J10_rkd,
            'y--',
            linewidth=2,
            markersize=3,
            label='Rkd +init'
        )
    if run_RKD_Distillation1 == 1:
        obj_iter_UPGA_J10_rkd1 = [
            rate - OMEGA * tau
            for rate, tau in zip(rate_iter_RKD1, tau_iter_RKD1)
        ]
        plt.plot(
            iter_number_UPGA_J10_I60,
            obj_iter_UPGA_J10_rkd1,
            'r--',
            linewidth=2,
            markersize=3,
            label='Init only'
        )

     # RKD student (J=10)
    if run_RKD_Distillation3 == 1:
        obj_iter_UPGA_J10_rkd3 = [
            rate - OMEGA * tau
            for rate, tau in zip(rate_iter_RKD3, tau_iter_RKD3)
        ]
        plt.plot(
            iter_number_UPGA_J10_I60,
            obj_iter_UPGA_J10_rkd3,
            'm--',
            linewidth=2,
            markersize=3,
            label='rkd only'
        )    

    if run_RKD_Distillation2 == 1:
        obj_iter_UPGA_J10_rkd2 = [
            rate - OMEGA * tau
            for rate, tau in zip(rate_iter_RKD2, tau_iter_RKD2)
        ]
        plt.plot(
            iter_number_UPGA_J10_I60,
            obj_iter_UPGA_J10_rkd2,
            'b-.',
            linewidth=2,
            markersize=3,
            label='Base'
        )    

    plt.xlabel('Iterations / Layers (I)', fontsize=14)
    plt.ylabel(r'$R - \omega \bar{\tau}$', fontsize=14)
    plt.xlim(0,120)
    plt.grid()
    plt.legend(fontsize=12)

    plt.savefig(directory_result + 'singlepass_j10_I80_fw.png')

    # ///////////////////////////////////////// SHOW OBJECTIVE VALUES OVER ITERATIONS ///////////////////////////////////
    benchmark = 0
    iter_number_conv_PGA = np.array(list(range(n_iter_outer + 1)))
    iter_number_UPGA_J1 = np.array(list(range(n_iter_outer + 1)))
    iter_number_UPGA_J10 = np.array(list(range(n_iter_outer + 1)))
    iter_number_UPGA_rkd= np.array(list(range(n_iter_outer + 1)))
    iter_number_UPGA_J20 = np.array(list(range(n_iter_outer + 1)))

    # //////////////////////////////// LOADING RESULTS //////////////////////////////////////////
    if save_result == 1:
        if run_conv_PGA == 1:
            result_file_name = directory_result + 'result_vs_iter_conv.npz'
            result = np.load(result_file_name)
            rate_iter_conv, tau_iter_conv, beam_conv_PGA = result['name1'], result['name2'], result['name3']
        if run_UPGA_J1 == 1:
            result_file_name = directory_result + 'result_vs_iter_UPGA_J1.npz'
            result = np.load(result_file_name)
            rate_iter_conv, tau_iter_conv, beam_conv_PGA = result['name1'], result['name2'], result['name3']
        if run_UPGA_J10 == 1:
            result_file_name = directory_result + 'result_vs_iter_UPGA_J10.npz'
            result = np.load(result_file_name)
            rate_iter_conv, tau_iter_conv, beam_conv_PGA = result['name1'], result['name2'], result['name3']
        if run_UPGA_J20 == 1:
            result_file_name = directory_result + 'result_vs_iter_UPGA_J20.npz'
            result = np.load(result_file_name)
            rate_iter_conv, tau_iter_conv, beam_conv_PGA = result['name1'], result['name2'], result['name3']

    #  /////////////////////////////////////////////////////////////////////////////////////////
    #                               PLOT FIGURES
    # //////////////////////////////////////////////////////////////////////////////////////////
    print('Plotting figures...')
    system_params = (
        rf'$N={Nt}, M={M}, N_{{\mathrm{{RF}}}}={Nrf}, '
        rf'\mathrm{{SNR}}={snr_dB} \mathrm{{dB}}, '
        rf'\omega={OMEGA}$'
    )

    # load benchmark results
    if benchmark == 1:
        benchmark_results = scipy.io.loadmat(directory_benchmark + 'result_benchmark')
        rate_ZF = np.squeeze(benchmark_results['rate_ZF_mean'])
        rate_SCA = np.squeeze(benchmark_results['rate_SCA_mean'])
        tau_ZF = np.squeeze(benchmark_results['tau_ZF_mean'])
        tau_SCA = np.squeeze(benchmark_results['tau_SCA_mean'])

        idx_snr = np.where(snr_dB_list == snr_dB)
        rate_ZF = rate_ZF[idx_snr] * np.ones(n_iter_outer + 1)
        rate_SCA = rate_SCA[idx_snr] * np.ones(n_iter_outer + 1)
        tau_ZF = tau_ZF[idx_snr] * np.ones(n_iter_outer + 1)
        tau_SCA = tau_SCA[idx_snr] * np.ones(n_iter_outer + 1)

        beam_ZF = np.squeeze(benchmark_results['beam_ZF_mean'][:, idx_snr])
        beam_SCA = np.squeeze(benchmark_results['beam_SCA_mean'][:, idx_snr])



    # ==================================== RATES ================================================
    plt.figure(1)
    if run_UPGA_J20 == 1:
        plt.plot(iter_number_UPGA_J20, rate_iter_UPGA_J20, '-', markevery=3, color='green', linewidth=2, markersize=7,
                 label='Teacher (J=20)')
    if run_UPGA_J10 == 1:
        plt.plot(iter_number_UPGA_J10, rate_iter_UPGA_J10, '--', markevery=3, color='blue', linewidth=2, markersize=7,
                 label='Student w/o RKD (J=10)')
    if run_RKD_Distillation3 == 1:
        plt.plot(iter_number_UPGA_J10_I60, rate_iter_RKD3, '-.', markevery=3, color='purple', linewidth=2, markersize=7,
                 label='RKD new')
    if run_RKD_Distillation == 1:
        plt.plot(iter_number_UPGA_J10_I60, rate_iter_RKD, '-.', markevery=3, color='purple', linewidth=2, markersize=7,
                 label='RKD outer+inner')    
    if run_RKD_Distillation == 1:
        plt.plot(iter_number_UPGA_J10_I60, rate_iter_RKD, ':*', markevery=3, color='black', linewidth=2, markersize=7,
                 label='Proposed init student model')
    plt.xlabel(r'Number of iterations/layers $(I)$', fontsize="14")
    plt.ylabel('$R$ [bits/s/Hz]', fontsize="14")
    plt.grid()
    plt.legend(loc='best', fontsize="14", labelspacing=0.15)
    plt.savefig(directory_result + 'rate_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.png')
    plt.savefig(directory_result + 'rate_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.eps')

    # plot beam errors ////////////////////////////////////////////////////////////////////
    fig_tau = plt.figure(2)
    if run_UPGA_J20 == 1:
        plt.plot(iter_number_UPGA_J20, tau_iter_UPGA_J20, '-', markevery=3, color='green', linewidth=2, markersize=7,
                 label='Teacher (J=20)')
    if run_UPGA_J10 == 1:
        plt.plot(iter_number_UPGA_J10, tau_iter_UPGA_J10, '--', markevery=3, color='blue', linewidth=2, markersize=7,
                 label='Student w/o RKD (J=10)')
    if run_RKD_Distillation3 == 1:
        plt.plot(iter_number_UPGA_J10_I60, tau_iter_RKD3, '-.', markevery=3, color='purple', linewidth=2, markersize=7,
                 label='rkd inner only')
    if run_RKD_Distillation == 1:
        plt.plot(iter_number_UPGA_J10_I60, tau_iter_RKD, '-.', markevery=3, color='purple', linewidth=2, markersize=7,
                 label='RKD outer+inner')  
    if run_RKD_Distillation == 1:
        plt.plot(iter_number_UPGA_J10_I60, tau_iter_RKD, ':*', markevery=3, color='black', linewidth=2, markersize=7,
                 label='Proposed init student model')
    plt.xlabel(r'Number of iterations/layers $(I)$', fontsize="14")
    plt.ylabel(r'$\bar{\tau}$', fontsize="14")
    plt.grid()
    plt.legend(loc='best', fontsize="14", labelspacing=0.15)
    plt.savefig(directory_result + 'beampattern_error_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.png')
    plt.savefig(directory_result + 'beampattern_error_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.eps')

    # plot rate - beam errors tradeoff ////////////////////////////////////////////////////////////////////
    fig_tradeoff = plt.figure(3)
    if run_UPGA_J20 == 1:
        plt.plot(tau_iter_UPGA_J20, rate_iter_UPGA_J20, '-', markevery=3, color='green', linewidth=2, markersize=7,
                 label='Teacher (J=20)')
    if run_UPGA_J10 == 1:
        plt.plot(tau_iter_UPGA_J10, rate_iter_UPGA_J10, '--', markevery=3, color='blue', linewidth=2, markersize=7,
                 label='Student w/o RKD (J=10)')
    if run_RKD_Distillation3 == 1:
        plt.plot(tau_iter_RKD3, rate_iter_RKD3, '-.', markevery=3, color='purple', linewidth=2, markersize=7,
                 label='inner only')
    if run_RKD_Distillation == 1:
        plt.plot(tau_iter_RKD, rate_iter_RKD, '-.', markevery=3, color='purple', linewidth=2, markersize=7,
                 label='RKD outer+inner')  
    if run_RKD_Distillation == 1:
        plt.plot(tau_iter_RKD, rate_iter_RKD, ':*', markevery=3, color='black', linewidth=2, markersize=7,
                 label='Proposed init student model')
    plt.xlabel(r'$\bar{\tau}$', fontsize="14")
    plt.ylabel(r'$R$ [bits/s/Hz]', fontsize="14")
    plt.grid()
    plt.legend(loc='best', fontsize="14", labelspacing=0.15)
    plt.savefig(directory_result + 'tradeoff_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.png')
    plt.savefig(directory_result + 'tradeoff_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.eps')

    plt.show()

