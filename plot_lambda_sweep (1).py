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

# ── Lambda values and evaluation depth ────────────────────────────────────────
lambda_list = [0.15, 0.30, 0.45]
lambda_tags = ['015', '030', '045']   # used in file names
EVAL_ITERS  =  40                      # evaluate all models at I=40

# ── Load data ─────────────────────────────────────────────────────────────────
H_train, H_test0 = get_data_tensor(data_source)
H_test = H_test0[:, :test_size, :, :]

# ── Define v2 student model class ─────────────────────────────────────────────
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
                F, W)

# ── Storage ───────────────────────────────────────────────────────────────────
# rate_vs_snr[lam_tag][model_key] = list of rates over snr_dB_list
rate_vs_snr = {tag: {'J20': [], 'c1': [], 'c2': [], 'c3': [], 'c4': []}
               for tag in lambda_tags}
mse_vs_snr  = {tag: {'J20': [], 'c1': [], 'c2': [], 'c3': [], 'c4': []}
               for tag in lambda_tags}

# ── Loop over lambda values ───────────────────────────────────────────────────
for lam, tag in zip(lambda_list, lambda_tags):
    print(f'\n{"="*60}')
    print(f'  lambda_r = {lam}  (tag={tag})')
    print(f'{"="*60}')

    snr_val = 10 ** (snr_dB / 10)

    # Load radar data at this lambda_r
    R_lam, at_lam, _, _ = get_radar_data(snr_dB, H_test)
    at_lam = at_lam[:, :test_size, :, :]

    # ── Load teacher ──────────────────────────────────────────────────────────
    print(f'  Loading teacher...')
    model_teacher = PGA_Unfold_J20(step_size_UPGA_J20)
    model_teacher.load_state_dict(torch.load(
        f'UPGA_J20_I120_w_{tag}.pth',
        map_location='cpu'))
    model_teacher.eval()

    # ── Load student cells ────────────────────────────────────────────────────
    print(f'  Loading student cells...')

    model_c1 = PGA_Unfold_J10_I60_v2(step_size_UPGA_J10_I60)
    model_c1.load_state_dict(torch.load(
        f'./model/UPGA_J10_320.pth_I60_cell_1_flat0.01_noRKD_Kl15_win20_030.pth',
        map_location='cpu'))
    model_c1.eval()

    model_c2 = PGA_Unfold_J10_I60_v2(step_size_UPGA_J10_I60)
    model_c2.load_state_dict(torch.load(
        f'./model/UPGA_J10_320.pth_I60_cell_2_flat0.01_RKDlog_30layers_Kl15_win20_{tag}.pth',
        map_location='cpu'))
    model_c2.eval()

    model_c3 = PGA_Unfold_J10_I60_v2(step_size_UPGA_J10_I60)
    model_c3.load_state_dict(torch.load(
        f'./model/UPGA_J10_320.pth_I60_cell_3_AGT_avg_pairs_noRKD_Kl15_win20_{tag}.pth',
        map_location='cpu'))
    model_c3.eval()

    model_c4 = PGA_Unfold_J10_I60_v2(step_size_UPGA_J10_I60)
    model_c4.load_state_dict(torch.load(
        f'./model/UPGA_J10_320.pth_I60_cell_4_AGT_avg_pairs_RKDlog_30layers_Kl15_win20_{tag}.pth',
        map_location='cpu'))
    model_c4.eval()

    # ── SNR sweep ─────────────────────────────────────────────────────────────
    #print(f'  Running SNR sweep at I={EVAL_ITERS}...')
    for ss in range(len(snr_dB_list)):
        snr_dB_ss = snr_dB_list[ss]
        snr_ss    = 10 ** (snr_dB_ss / 10)
        print(f'    SNR = {snr_dB_ss} dB')

        R_ss, at_ss, _, _ = get_radar_data(snr_dB_ss, H_test)
        at_ss = at_ss[:, :test_size, :, :]

        with torch.no_grad():
            # Teacher — full depth
            _, _, F, W = model_teacher.execute_PGA(
                H_test, R_ss, snr_ss, n_iter_outer, n_iter_inner_J20)
            rate_vs_snr[tag]['J20'].append(get_sum_rate(H_test, F, W, snr_ss).item())
            mse_vs_snr[tag]['J20'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

            # Student cells — evaluated at EVAL_ITERS
            _, _, F, W = model_c1.execute_PGA(
                H_test, R_ss, snr_ss, 60, n_iter_inner_J10)
            rate_vs_snr[tag]['c1'].append(get_sum_rate(H_test, F, W, snr_ss).item())
            mse_vs_snr[tag]['c1'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

            _, _, F, W = model_c2.execute_PGA(
                H_test, R_ss, snr_ss, 60, n_iter_inner_J10)
            rate_vs_snr[tag]['c2'].append(get_sum_rate(H_test, F, W, snr_ss).item())
            mse_vs_snr[tag]['c2'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

            _, _, F, W = model_c3.execute_PGA(
                H_test, R_ss, snr_ss, 60, n_iter_inner_J10)
            rate_vs_snr[tag]['c3'].append(get_sum_rate(H_test, F, W, snr_ss).item())
            mse_vs_snr[tag]['c3'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

            _, _, F, W = model_c4.execute_PGA(
                H_test, R_ss, snr_ss, 60, n_iter_inner_J10)
            rate_vs_snr[tag]['c4'].append(get_sum_rate(H_test, F, W, snr_ss).item())
            mse_vs_snr[tag]['c4'].append(get_MSE(F, W, at_ss, R_ss, snr_ss).item())

    # ── Print numerical results ───────────────────────────────────────────────
    print(f'\n  Results at lambda_r={lam}, SNR={snr_dB_list[-1]} dB, I={60}:')
    print(f'  {"Model":<25} {"R [bits/s/Hz]":>15} {"MSE [dB]":>12}')
    print(f'  {"-"*55}')
    for key, name in [('J20','Teacher (J=20,I=120)'),
                      ('c1', 'cell_1: flat, no KD'),
                      ('c2', 'cell_2: flat+RKD+log'),
                      ('c3', 'cell_3: GI, no RKD'),
                      ('c4', 'cell_4: GI+RKD+log')]:
        print(f'  {name:<25} {rate_vs_snr[tag][key][-1]:>15.4f}'
              f' {mse_vs_snr[tag][key][-1]:>12.4f}')



# ── Extract values at fixed SNR (last SNR point) for lambda sweep plots ────────
# Each point = final rate or MSE at that lambda_r value, evaluated at EVAL_ITERS
# Change snr_idx to use a different SNR point (e.g. 0 = lowest, -1 = highest)
snr_idx = -1   # use highest SNR point

rate_vs_lam = {key: [] for key in ['J20', 'c1', 'c2', 'c3', 'c4']}
mse_vs_lam  = {key: [] for key in ['J20', 'c1', 'c2', 'c3', 'c4']}

for tag in lambda_tags:
    for key in ['J20', 'c1', 'c2', 'c3', 'c4']:
        rate_vs_lam[key].append(rate_vs_snr[tag][key][snr_idx])
        mse_vs_lam[key].append(mse_vs_snr[tag][key][snr_idx])

lambda_np = np.array(lambda_list)

# ── Print numerical results ───────────────────────────────────────────────────
print('\n' + '='*70)
print(f'Rate and MSE vs lambda_r at SNR={snr_dB_list[snr_idx]} dB')
print('='*70)
print(f'  {"Model":<25} ', end='')
for lam in lambda_list:
    print(f'  lam={lam}', end='')
print()
print(f'  {"-"*70}')
for key, name in [('J20','Teacher'),('c1','Cell 1'),
                   ('c2','Cell 2'),('c3','Cell 3'),('c4','Cell 4')]:
    print(f'  {name:<25} ', end='')
    for r in rate_vs_lam[key]:
        print(f'  {r:>8.4f}', end='')
    print()
print('='*70)


# ── Figure 1: Rate vs Lambda ──────────────────────────────────────────────────
plt.figure(1, figsize=(7, 5))
plt.plot(lambda_np, rate_vs_lam['J20'], '-',          color='#2CA02C', linewidth=3,   marker='o', markersize=8, label=r'Teacher $(J=20, I=120)$')
plt.plot(lambda_np, rate_vs_lam['c1'],  ':',          color='#E377C2', linewidth=2.5, marker='^', markersize=7, label=r'Cell 1: flat, no KD')
plt.plot(lambda_np, rate_vs_lam['c2'],  '--',         color='#0047FF', linewidth=2.5, marker='s', markersize=7, label=r'Cell 2: flat+RKD+log')
plt.plot(lambda_np, rate_vs_lam['c3'],  ls=(0,(5,2)), color='#FF9500', linewidth=2.5, marker='D', markersize=7, label=r'Cell 3: GI, no RKD')
plt.plot(lambda_np, rate_vs_lam['c4'],  '-',          color='#9467BD', linewidth=3.5, marker='*', markersize=9, label=r'Cell 4: GI+RKD+log')
plt.xlabel(r'$\lambda_r$', fontsize=14)
plt.ylabel(r'$R$ [bits/s/Hz]', fontsize=14)
plt.xticks(lambda_np, [str(l) for l in lambda_list])
plt.grid()
plt.legend(loc='best', fontsize=11)
plt.tight_layout()
plt.savefig(directory_result + f'lambda_sweep_rate_final.png')
plt.savefig(directory_result + f'lambda_sweep_rate_final.eps')

# ── Figure 2: Beam MSE vs Lambda ─────────────────────────────────────────────
plt.figure(2, figsize=(7, 5))
plt.plot(lambda_np, mse_vs_lam['J20'], '-',          color='#2CA02C', linewidth=3,   marker='o', markersize=8, label=r'Teacher $(J=20, I=120)$')
plt.plot(lambda_np, mse_vs_lam['c1'],  ':',          color='#E377C2', linewidth=2.5, marker='^', markersize=7, label=r'Cell 1: flat, no KD')
plt.plot(lambda_np, mse_vs_lam['c2'],  '--',         color='#0047FF', linewidth=2.5, marker='s', markersize=7, label=r'Cell 2: flat+RKD+log')
plt.plot(lambda_np, mse_vs_lam['c3'],  ls=(0,(5,2)), color='#FF9500', linewidth=2.5, marker='D', markersize=7, label=r'Cell 3: GI, no RKD')
plt.plot(lambda_np, mse_vs_lam['c4'],  '-',          color='#9467BD', linewidth=3.5, marker='*', markersize=9, label=r'Cell 4: GI+RKD+log')
plt.xlabel(r'$\lambda_r$', fontsize=14)
plt.ylabel('Average radar beampattern MSE [dB]', fontsize=14)
plt.xticks(lambda_np, [str(l) for l in lambda_list])
plt.grid()
plt.legend(loc='best', fontsize=11)
plt.tight_layout()
plt.savefig(directory_result + f'lambda_sweep_beam_I{EVAL_ITERS}.png')
plt.savefig(directory_result + f'lambda_sweep_beam_I{EVAL_ITERS}.eps')

plt.show()
print('\nDone.')