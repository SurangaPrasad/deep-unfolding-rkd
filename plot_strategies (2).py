from PGA_models import *
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

run_program  = 1
plot_figure  = 1
seed = 3407

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ══════════════════════════════════════════════════════════════════════════════
# DATA AND RADAR
# ══════════════════════════════════════════════════════════════════════════════

H_train, H_test0 = get_data_tensor(data_source)
H_test = H_test0[:, :test_size, :, :]

R, at0, theta, ideal_beam = get_radar_data(snr_dB, H_test)
at = at0[:, :test_size, :, :]

H_test = H_test.to(device)
R      = R.to(device)
at     = at.to(device)

# ══════════════════════════════════════════════════════════════════════════════
# WHICH STUDENT ARCHITECTURE TO COMPARE
# Set I_S and n_inner to match what you trained
# ══════════════════════════════════════════════════════════════════════════════

I_S    = 60              # student outer iterations — change to 80 if needed
n_inner_student = n_iter_inner_J10   # change to n_iter_inner_J20 if J20 student

# Iteration axes
iter_number_J20_120 = np.array(list(range(n_iter_outer + 1)))  # 0..120
iter_number_J10_120 = np.array(list(range(n_iter_outer + 1)))  # 0..120
iter_number_student = np.array(list(range(I_S + 1)))           # 0..I_S

# ══════════════════════════════════════════════════════════════════════════════
# STUDENT MODEL CLASSES
# Match whichever class was used during training
# ══════════════════════════════════════════════════════════════════════════════

class PGA_Student_J10(PGA_Unfold_J10):
    """J=10 inner, configurable outer. step_size: [J10, I_S, K+1]"""
    def execute_PGA_eval(self, H, R, Pt, n_iter_outer, n_iter_inner):
        rate_init, tau_init, F, W = initialize(H, R, Pt, initial_normalization)
        rate_over = torch.zeros(n_iter_outer, H.shape[1], device=H.device)
        tau_over  = torch.zeros(n_iter_outer, H.shape[1], device=H.device)
        for ii in range(n_iter_outer):
            for jj in range(n_iter_inner):
                gF_c = get_grad_F_com(H, F, W)
                gF_r = get_grad_F_rad(F, W, R)
                step = self.step_size[jj][ii][0]
                F = F + step * gF_c * WEIGHT_F_COM \
                      - step * gF_r * WEIGHT_F_RAD
                if sum(torch.abs(F[0, :, 0, 0])) > 1e3:
                    F = normalize_power(F, W, H, Pt)
            F = F / torch.abs(F)
            W_new = W.clone().detach()
            gW_c  = get_grad_W_com(H, F, W)
            gW_r  = get_grad_W_rad(F, W, R)
            for k in range(K):
                step     = self.step_size[0][ii][k+1]
                W_new[k] = (W[k].clone().detach()
                            + step * gW_c[k] * WEIGHT_W_COM
                            - step * gW_r[k] * WEIGHT_W_RAD)
            F, W = normalize(F, W_new, H, Pt)
            rate_over[ii] = get_sum_rate(H, F, W, Pt)
            tau_over[ii]  = get_beam_error(H, F, W, R, Pt)
        rates = torch.cat([rate_init, rate_over], dim=0)
        taus  = torch.cat([tau_init,  tau_over],  dim=0)
        return torch.transpose(rates, 0, 1), torch.transpose(taus, 0, 1), F, W


class PGA_Student_J20(PGA_Unfold_J20):
    """J=20 inner, configurable outer. step_size: [J20, I_S, K+1]"""
    def execute_PGA_eval(self, H, R, Pt, n_iter_outer, n_iter_inner):
        rate_init, tau_init, F, W = initialize(H, R, Pt, initial_normalization)
        rate_over = torch.zeros(n_iter_outer, H.shape[1], device=H.device)
        tau_over  = torch.zeros(n_iter_outer, H.shape[1], device=H.device)
        for ii in range(n_iter_outer):
            for jj in range(n_iter_inner):
                gF_c = get_grad_F_com(H, F, W)
                gF_r = get_grad_F_rad(F, W, R)
                step = self.step_size[jj][ii][0]
                F = F + step * gF_c * WEIGHT_F_COM \
                      - step * gF_r * WEIGHT_F_RAD
                if sum(torch.abs(F[0, :, 0, 0])) > 1e3:
                    F = normalize_power(F, W, H, Pt)
            F = F / torch.abs(F)
            W_new = W.clone().detach()
            gW_c  = get_grad_W_com(H, F, W)
            gW_r  = get_grad_W_rad(F, W, R)
            for k in range(K):
                step     = self.step_size[0][ii][k+1]
                W_new[k] = (W[k].clone().detach()
                            + step * gW_c[k] * WEIGHT_W_COM
                            - step * gW_r[k] * WEIGHT_W_RAD)
            F, W = normalize(F, W_new, H, Pt)
            rate_over[ii] = get_sum_rate(H, F, W, Pt)
            tau_over[ii]  = get_beam_error(H, F, W, R, Pt)
        rates = torch.cat([rate_init, rate_over], dim=0)
        taus  = torch.cat([tau_init,  tau_over],  dim=0)
        return torch.transpose(rates, 0, 1), torch.transpose(taus, 0, 1), F, W


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_and_run(model_class, ss_shape, path, n_outer, n_inner):
    ss = torch.zeros(ss_shape)
    m  = model_class(ss).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    with torch.no_grad():
        rate_iter, tau_iter, _, _ = m.execute_PGA_eval(
            H_test, R, snr, n_outer, n_inner)
    rate = [r.detach().cpu().numpy() for r in (sum(rate_iter) / len(H_test[0]))]
    tau  = [e.detach().cpu().numpy() for e in (sum(tau_iter)  / len(H_test[0]))]
    return rate, tau

def safe_load(model_class, ss_shape, path, n_outer, n_inner, label):
    try:
        return load_and_run(model_class, ss_shape, path, n_outer, n_inner)
    except FileNotFoundError:
        print(f'  [SKIP] {label} — file not found: {path}')
        return None, None

# Shape helpers
def ss_J10(I): return (n_iter_inner_J10, I, K + 1)
def ss_J20(I): return (n_iter_inner_J20, I, K + 1)
def spJ10(tag): return model_file_name_UPGA_J10.replace('J10', tag) + '.pth'
def spJ20(tag): return model_file_name_UPGA_J20.replace('J20', tag) + '.pth'


# ══════════════════════════════════════════════════════════════════════════════
# RUN MODELS
# ══════════════════════════════════════════════════════════════════════════════

if run_program == 1:

    # ── Teacher J20/120 ───────────────────────────────────────────────────────
    print('Loading Teacher (J=20, I=120)...')
    model_J20 = PGA_Unfold_J20(step_size_UPGA_J20).to(device)
    model_J20.load_state_dict(torch.load(model_file_name_UPGA_J20, map_location=device))
    model_J20.eval()
    with torch.no_grad():
        rate_J20_r, tau_J20_r, _, _ = model_J20.execute_PGA(
            H_test, R, snr, n_iter_outer, n_iter_inner_J20)
    rate_iter_J20 = [r.detach().cpu().numpy() for r in (sum(rate_J20_r) / len(H_test[0]))]
    tau_iter_J20  = [e.detach().cpu().numpy() for e in (sum(tau_J20_r)  / len(H_test[0]))]

    # ── Student J10/120 (no distillation, original) ───────────────────────────
    print('Loading Student J10/120 (no distillation)...')
    model_J10 = PGA_Unfold_J10(step_size_UPGA_J10).to(device)
    model_J10.load_state_dict(torch.load(model_file_name_UPGA_J10, map_location=device))
    model_J10.eval()
    with torch.no_grad():
        rate_J10_r, tau_J10_r, _, _ = model_J10.execute_PGA(
            H_test, R, snr, n_iter_outer, n_iter_inner_J10)
    rate_iter_J10 = [r.detach().cpu().numpy() for r in (sum(rate_J10_r) / len(H_test[0]))]
    tau_iter_J10  = [e.detach().cpu().numpy() for e in (sum(tau_J10_r)  / len(H_test[0]))]

    # ── Strategy 1: Trajectory-shape RKD ─────────────────────────────────────
    print('Loading Strategy 1 (Trajectory-shape RKD)...')
    rate_S1_J10_60, tau_S1_J10_60 = safe_load(
        PGA_Student_J10, ss_J10(60),
        spJ10('J10_60_traj'), 60, n_iter_inner_J10, 'S1 J10/I60')

    rate_S1_J10_80, tau_S1_J10_80 = safe_load(
        PGA_Student_J10, ss_J10(80),
        spJ10('J10_80_traj'), 80, n_iter_inner_J10, 'S1 J10/I80')

    rate_S1_J20_60, tau_S1_J20_60 = safe_load(
        PGA_Student_J20, ss_J20(60),
        spJ20('J20_60_traj'), 60, n_iter_inner_J20, 'S1 J20/I60')

    rate_S1_J20_80, tau_S1_J20_80 = safe_load(
        PGA_Student_J20, ss_J20(80),
        spJ20('J20_80_traj'), 80, n_iter_inner_J20, 'S1 J20/I80')

    # ── Strategy 3: Gradient direction distillation ───────────────────────────
    print('Loading Strategy 3 (Gradient direction)...')
    rate_S3_J10_60, tau_S3_J10_60 = safe_load(
        PGA_Student_J10, ss_J10(60),
        spJ10('J10_60_grad'), 60, n_iter_inner_J10, 'S3 J10/I60')

    rate_S3_J10_80, tau_S3_J10_80 = safe_load(
        PGA_Student_J10, ss_J10(80),
        spJ10('J10_80_grad'), 80, n_iter_inner_J10, 'S3 J10/I80')

    rate_S3_J20_60, tau_S3_J20_60 = safe_load(
        PGA_Student_J20, ss_J20(60),
        spJ20('J20_60_grad'), 60, n_iter_inner_J20, 'S3 J20/I60')

    rate_S3_J20_80, tau_S3_J20_80 = safe_load(
        PGA_Student_J20, ss_J20(80),
        spJ20('J20_80_grad'), 80, n_iter_inner_J20, 'S3 J20/I80')

    # ── Strategy 4: Continuous schedule distillation ──────────────────────────
    print('Loading Strategy 4 (Continuous schedule)...')
    rate_S4_J10_60, tau_S4_J10_60 = safe_load(
        PGA_Student_J10, ss_J10(60),
        spJ10('J10_60_css'), 60, n_iter_inner_J10, 'S4 J10/I60')

    rate_S4_J10_80, tau_S4_J10_80 = safe_load(
        PGA_Student_J10, ss_J10(80),
        spJ10('J10_80_css'), 80, n_iter_inner_J10, 'S4 J10/I80')

    rate_S4_J20_60, tau_S4_J20_60 = safe_load(
        PGA_Student_J20, ss_J20(60),
        spJ20('J20_60_css'), 60, n_iter_inner_J20, 'S4 J20/I60')

    rate_S4_J20_80, tau_S4_J20_80 = safe_load(
        PGA_Student_J20, ss_J20(80),
        spJ20('J20_80_css'), 80, n_iter_inner_J20, 'S4 J20/I80')


# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def obj(rate, tau):
    if rate is None: return None
    return [r - OMEGA * t for r, t in zip(rate, tau)]


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING — one set of 4 figures per student variant
# Adjust I_S and the curves you want below
# ══════════════════════════════════════════════════════════════════════════════

def plot_variant(I_S_plot,
                 rate_S1, tau_S1,
                 rate_S3, tau_S3,
                 rate_S4, tau_S4,
                 suffix):
    """
    Plot 4 figures (obj, rate, tau, tradeoff) for one student variant.
    suffix : string appended to saved filenames, e.g. 'J10_I60'
    """
    iter_student = np.array(list(range(I_S_plot + 1)))

    obj_J20 = obj(rate_iter_J20, tau_iter_J20)
    obj_J10 = obj(rate_iter_J10, tau_iter_J10)
    obj_S1  = obj(rate_S1, tau_S1)
    obj_S3  = obj(rate_S3, tau_S3)
    obj_S4  = obj(rate_S4, tau_S4)

    # ── Figure 4: R - ω·τ vs iterations ──────────────────────────────────────
    plt.figure(4); plt.clf()

    plt.plot(iter_number_J20_120, obj_J20,
             'g-', linewidth=2, markersize=3, label='Teacher (J=20, I=120)')
    plt.plot(iter_number_J10_120, obj_J10,
             'b-', linewidth=2, markersize=3, label='Student w/o distillation (J=10, I=120)')

    if obj_S1 is not None:
        plt.plot(iter_student, obj_S1,
                 'r-.', linewidth=2, markersize=3,
                 label=f'S1: Traj-RKD ({suffix})')
    if obj_S3 is not None:
        plt.plot(iter_student, obj_S3,
                 'm--', linewidth=2, markersize=3,
                 label=f'S3: Grad-Dir ({suffix})')
    if obj_S4 is not None:
        plt.plot(iter_student, obj_S4,
                 'c:', linewidth=2, markersize=3,
                 label=f'S4: Cont-Sched ({suffix})')

    plt.xlabel('Iterations / Layers (I)', fontsize=14)
    plt.ylabel(r'$R - \omega \bar{\tau}$', fontsize=14)
    plt.xlim(0, 120)
    plt.grid(); plt.legend(fontsize=11); plt.tight_layout()
    plt.savefig(directory_result + f'obj_vs_iter_{suffix}.png')
    plt.savefig(directory_result + f'obj_vs_iter_{suffix}.eps')

    # ── Figure 1: Rate vs iterations ──────────────────────────────────────────
    plt.figure(1); plt.clf()

    plt.plot(iter_number_J20_120, rate_iter_J20,
             '-', markevery=3, color='green', linewidth=2, markersize=7,
             label='Teacher (J=20, I=120)')
    plt.plot(iter_number_J10_120, rate_iter_J10,
             '--', markevery=3, color='blue', linewidth=2, markersize=7,
             label='Student w/o distillation (J=10, I=120)')
    if rate_S1 is not None:
        plt.plot(iter_student, rate_S1,
                 '-.', markevery=3, color='red', linewidth=2, markersize=7,
                 label=f'S1: Traj-RKD ({suffix})')
    if rate_S3 is not None:
        plt.plot(iter_student, rate_S3,
                 '--', markevery=3, color='magenta', linewidth=2, markersize=7,
                 label=f'S3: Grad-Dir ({suffix})')
    if rate_S4 is not None:
        plt.plot(iter_student, rate_S4,
                 ':', markevery=3, color='cyan', linewidth=2, markersize=7,
                 label=f'S4: Cont-Sched ({suffix})')

    plt.xlabel(r'Number of iterations/layers $(I)$', fontsize=14)
    plt.ylabel('$R$ [bits/s/Hz]', fontsize=14)
    plt.grid(); plt.legend(loc='best', fontsize=11, labelspacing=0.15)
    plt.tight_layout()
    plt.savefig(directory_result + f'rate_vs_iter_{suffix}_{Nt}_{OMEGA}.png')
    plt.savefig(directory_result + f'rate_vs_iter_{suffix}_{Nt}_{OMEGA}.eps')

    # ── Figure 2: Beam error vs iterations ───────────────────────────────────
    plt.figure(2); plt.clf()

    plt.plot(iter_number_J20_120, tau_iter_J20,
             '-', markevery=3, color='green', linewidth=2, markersize=7,
             label='Teacher (J=20, I=120)')
    plt.plot(iter_number_J10_120, tau_iter_J10,
             '--', markevery=3, color='blue', linewidth=2, markersize=7,
             label='Student w/o distillation (J=10, I=120)')
    if tau_S1 is not None:
        plt.plot(iter_student, tau_S1,
                 '-.', markevery=3, color='red', linewidth=2, markersize=7,
                 label=f'S1: Traj-RKD ({suffix})')
    if tau_S3 is not None:
        plt.plot(iter_student, tau_S3,
                 '--', markevery=3, color='magenta', linewidth=2, markersize=7,
                 label=f'S3: Grad-Dir ({suffix})')
    if tau_S4 is not None:
        plt.plot(iter_student, tau_S4,
                 ':', markevery=3, color='cyan', linewidth=2, markersize=7,
                 label=f'S4: Cont-Sched ({suffix})')

    plt.xlabel(r'Number of iterations/layers $(I)$', fontsize=14)
    plt.ylabel(r'$\bar{\tau}$', fontsize=14)
    plt.grid(); plt.legend(loc='best', fontsize=11, labelspacing=0.15)
    plt.tight_layout()
    plt.savefig(directory_result + f'beampattern_error_vs_iter_{suffix}_{Nt}_{OMEGA}.png')
    plt.savefig(directory_result + f'beampattern_error_vs_iter_{suffix}_{Nt}_{OMEGA}.eps')

    # ── Figure 3: Rate-beampattern tradeoff ──────────────────────────────────
    plt.figure(3); plt.clf()

    plt.plot(tau_iter_J20, rate_iter_J20,
             '-', markevery=3, color='green', linewidth=2, markersize=7,
             label='Teacher (J=20, I=120)')
    plt.plot(tau_iter_J10, rate_iter_J10,
             '--', markevery=3, color='blue', linewidth=2, markersize=7,
             label='Student w/o distillation (J=10, I=120)')
    if rate_S1 is not None:
        plt.plot(tau_S1, rate_S1,
                 '-.', markevery=3, color='red', linewidth=2, markersize=7,
                 label=f'S1: Traj-RKD ({suffix})')
    if rate_S3 is not None:
        plt.plot(tau_S3, rate_S3,
                 '--', markevery=3, color='magenta', linewidth=2, markersize=7,
                 label=f'S3: Grad-Dir ({suffix})')
    if rate_S4 is not None:
        plt.plot(tau_S4, rate_S4,
                 ':', markevery=3, color='cyan', linewidth=2, markersize=7,
                 label=f'S4: Cont-Sched ({suffix})')

    plt.xlabel(r'$\bar{\tau}$', fontsize=14)
    plt.ylabel(r'$R$ [bits/s/Hz]', fontsize=14)
    plt.grid(); plt.legend(loc='best', fontsize=11, labelspacing=0.15)
    plt.tight_layout()
    plt.savefig(directory_result + f'tradeoff_vs_iter_{suffix}_{Nt}_{OMEGA}.png')
    plt.savefig(directory_result + f'tradeoff_vs_iter_{suffix}_{Nt}_{OMEGA}.eps')

    # ── Print summary table ───────────────────────────────────────────────────
    print(f'\n── {suffix} results ────────────────────────────────────────')
    print(f'{"Model":<35} {"R-ωτ":>8} {"Rate":>8} {"τ":>8}')
    print('─' * 62)
    for label, r, t in [
        ('Teacher (J=20, I=120)',           rate_iter_J20, tau_iter_J20),
        ('Student J10/I120 (no distill.)',  rate_iter_J10, tau_iter_J10),
        (f'S1 Traj-RKD ({suffix})',         rate_S1, tau_S1),
        (f'S3 Grad-Dir ({suffix})',         rate_S3, tau_S3),
        (f'S4 Cont-Sched ({suffix})',       rate_S4, tau_S4),
    ]:
        if r is None:
            print(f'{label:<35}  not trained')
        else:
            o = r[-1] - OMEGA * t[-1]
            print(f'{label:<35} {o:>8.4f} {r[-1]:>8.4f} {t[-1]:>8.4f}')
    print('─' * 62)


if plot_figure == 1:

    # ── Plot J10/I60 variant ──────────────────────────────────────────────────
    plot_variant(
        I_S_plot=60,
        rate_S1=rate_S1_J10_60, tau_S1=tau_S1_J10_60,
        rate_S3=rate_S3_J10_60, tau_S3=tau_S3_J10_60,
        rate_S4=rate_S4_J10_60, tau_S4=tau_S4_J10_60,
        suffix='J10_I60')

    # ── Plot J10/I80 variant ──────────────────────────────────────────────────
    plot_variant(
        I_S_plot=80,
        rate_S1=rate_S1_J10_80, tau_S1=tau_S1_J10_80,
        rate_S3=rate_S3_J10_80, tau_S3=tau_S3_J10_80,
        rate_S4=rate_S4_J10_80, tau_S4=tau_S4_J10_80,
        suffix='J10_I80')

    # ── Plot J20/I60 variant ──────────────────────────────────────────────────
    plot_variant(
        I_S_plot=60,
        rate_S1=rate_S1_J20_60, tau_S1=tau_S1_J20_60,
        rate_S3=rate_S3_J20_60, tau_S3=tau_S3_J20_60,
        rate_S4=rate_S4_J20_60, tau_S4=tau_S4_J20_60,
        suffix='J20_I60')

    # ── Plot J20/I80 variant ──────────────────────────────────────────────────
    plot_variant(
        I_S_plot=80,
        rate_S1=rate_S1_J20_80, tau_S1=tau_S1_J20_80,
        rate_S3=rate_S3_J20_80, tau_S3=tau_S3_J20_80,
        rate_S4=rate_S4_J20_80, tau_S4=tau_S4_J20_80,
        suffix='J20_I80')

    plt.show()
