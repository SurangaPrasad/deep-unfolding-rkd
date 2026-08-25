from PGA_models import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

run_program = 1
plot_figure = 1
save_result = 1
# torch.manual_seed(3407)
# ///////////////////////////////////////// SHOW OBJECTIVE VALUES OVER ITERATIONS ///////////////////////////////////
# Load training data
H_train, H_test0 = get_data_tensor(data_source)
H_test = H_test0[:, :test_size, :, :]

R, at0, theta, ideal_beam = get_radar_data(snr_dB, H_test)
at = at0[:, : test_size, :, :]

OMEGA = [0.1, 0.3]  # weight for beampattern error in the objective function

if run_program == 1:
    rate_iter = {}
    tau_iter = {}


    # ====================================================== Conv. PGA with J = 10 ====================================
    if run_conv_PGA_J10 == 1:
        print('Running conventional PGA with J = 10...')
        model_conv_PGA_J10 = PGA_Unfold_J10(step_size_UPGA_J10)
        rate_conv_PGA_J10, tau_conv_PGA_J10, F_conv_PGA_J10, W_conv_PGA_J10 = model_conv_PGA_J10.execute_PGA(H_test, R, snr,
                                                                                           n_iter_outer,
                                                                                           n_iter_inner_J10)
        rate_iter_conv_PGA_J10 = [r.detach().numpy() for r in (sum(rate_conv_PGA_J10) / len(H_test[0]))]
        tau_iter_conv_PGA_J10 = [e.detach().numpy() for e in (sum(tau_conv_PGA_J10) / (len(H_test[0])))]

    # ====================================================== Proposed Unfolded PGA light ====================================
    if run_UPGA_J10 == 1:
        print('Running unfolded PGA with J = 10...')
        # Create new model and load states
        model_UPGA_J10 = PGA_Unfold_J10(step_size_UPGA_J10)
        model_UPGA_J10.load_state_dict(torch.load(model_file_name_UPGA_J10))

        sum_rate_UPGA_J10, tau_UPGA_J10, F_UPGA_J10, W_UPGA_J10 = model_UPGA_J10.execute_PGA(H_test, R,
                                                                                             snr,
                                                                                             n_iter_outer,
                                                                                             n_iter_inner_J10)
        rate_iter_UPGA_J10 = [r.detach().numpy() for r in (sum(sum_rate_UPGA_J10) / len(H_test[0]))]
        tau_iter_UPGA_J10 = [e.detach().numpy() for e in (sum(tau_UPGA_J10) / (len(H_test[0])))]

print('Finished running algorithms for all omega values.')


# Plot objective function (rate - OMEGA*tau) for all OMEGA values
if plot_figure == 1:
    iter_number_UPGA_J10 = np.array(list(range(n_iter_outer + 1)))
    
    fig_obj = plt.figure(5)
    for omega_val in OMEGA:
        if run_conv_PGA_J10 == 1:
            obj_iter_conv_PGA_J10 = [rate - omega_val * tau for rate, tau in zip(rate_iter_conv_PGA_J10, tau_iter_conv_PGA_J10)]
            plt.plot(iter_number_UPGA_J10, obj_iter_conv_PGA_J10, ':*', markevery=5, color='orange', linewidth=3, markersize=7, label=f'PGA (J=10, ω={omega_val})')
        if run_UPGA_J10 == 1:
            obj_iter_UPGA_J10 = [rate - omega_val * tau for rate, tau in zip(rate_iter_UPGA_J10, tau_iter_UPGA_J10)]
            plt.plot(iter_number_UPGA_J10, obj_iter_UPGA_J10, ':*', markevery=5, color='blue', linewidth=3, markersize=7, label=f'{label_UPGA_J10} (ω={omega_val})')
        # plt.title(system_params)
    plt.xlabel(r'Number of iterations/layers $(I)$', fontsize="14")
    plt.ylabel(r'$R - \omega \bar{\tau}$', fontsize="14")
    # plt.ylim(-70, 20)
    plt.grid()
    plt.legend(loc='best', fontsize="14", labelspacing  = 0.15)

    # save figure
    plt.savefig(directory_result + 'objective_vs_iter_' + str(Nt) + '_' + str(omega_val) + '.png')
    plt.savefig(directory_result + 'objective_vs_iter_' + str(Nt) + '_' + str(omega_val) + '.eps')