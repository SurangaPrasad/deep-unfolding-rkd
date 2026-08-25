from PGA_models import *
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

run_program = 1
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

iter_J20_120 = np.array(list(range(n_iter_outer + 1)))  # reference axis

# ══════════════════════════════════════════════════════════════════════════════
# STUDENT MODEL CLASSES
# These must match exactly what was used in the strategy training files
# ══════════════════════════════════════════════════════════════════════════════

class PGA_Student_J10(PGA_Unfold_J10):
    """J=10 inner, configurable I_S outer. step_size: [J10, I_S, K+1]"""
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
    """J=20 inner, configurable I_S outer. step_size: [J20, I_S, K+1]"""
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
        print(f'  [SKIP] {label} — file not found')
        return None, None

def obj(rate, tau):
    if rate is None: return None
    return [r - OMEGA * t for r, t in zip(rate, tau)]

# Path helpers — must match filenames saved by strategy training files
def ss_J10(I): return (n_iter_inner_J10, I, K + 1)
def ss_J20(I): return (n_iter_inner_J20, I, K + 1)
def spJ10(tag):
    return model_file_name_UPGA_J10.replace("J10", tag)
def spJ20(tag):
    return model_file_name_UPGA_J20.replace("J20", tag)


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL MODELS
# ══════════════════════════════════════════════════════════════════════════════

if run_program == 1:

    # ── Teacher J20/120 (upper bound reference) ───────────────────────────────
    print('Loading Teacher J20/I120...')
    model_teacher = PGA_Unfold_J20(step_size_UPGA_J20).to(device)
    model_teacher.load_state_dict(
        torch.load(model_file_name_UPGA_J20, map_location=device))
    model_teacher.eval()
    with torch.no_grad():
        r_t, tau_t, _, _ = model_teacher.execute_PGA(
            H_test, R, snr, n_iter_outer, n_iter_inner_J20)
    rate_iter_J20 = [r.detach().cpu().numpy()
                     for r in (sum(r_t) / len(H_test[0]))]
    tau_iter_J20  = [e.detach().cpu().numpy()
                     for e in (sum(tau_t) / len(H_test[0]))]

    # ──────────────────────────────────────────────────────────────────────────
    # J10 / I=60 variant
    # Pure baseline: SAME J=10, SAME I=60, flat init, task loss only
    # Distilled:     SAME J=10, SAME I=60, flat init, + distillation loss
    # This is the only fair comparison.
    # ──────────────────────────────────────────────────────────────────────────

    print('\nLoading J10/I60 models...')
    rate_J10_60_pure, tau_J10_60_pure = safe_load(
        PGA_Student_J10, ss_J10(60), spJ10('J10_60_pure'),
        60, n_iter_inner_J10, 'J10/I60 pure')
    rate_J10_60_traj, tau_J10_60_traj = safe_load(
        PGA_Student_J10, ss_J10(60), spJ10('J10_60_traj'),
        60, n_iter_inner_J10, 'J10/I60 traj-RKD')
    rate_J10_60_grad, tau_J10_60_grad = safe_load(
        PGA_Student_J10, ss_J10(60), spJ10('J10_60_grad'),
        60, n_iter_inner_J10, 'J10/I60 grad-dir')
    rate_J10_60_css, tau_J10_60_css = safe_load(
        PGA_Student_J10, ss_J10(60), spJ10('J10_60_css'),
        60, n_iter_inner_J10, 'J10/I60 cont-sched')

    # ── J10 / I=80 variant ────────────────────────────────────────────────────
    print('Loading J10/I80 models...')
    rate_J10_80_pure, tau_J10_80_pure = safe_load(
        PGA_Student_J10, ss_J10(80), spJ10('J10_80_pure'),
        80, n_iter_inner_J10, 'J10/I80 pure')
    rate_J10_80_traj, tau_J10_80_traj = safe_load(
        PGA_Student_J10, ss_J10(80), spJ10('J10_80_traj'),
        80, n_iter_inner_J10, 'J10/I80 traj-RKD')
    rate_J10_80_grad, tau_J10_80_grad = safe_load(
        PGA_Student_J10, ss_J10(80), spJ10('J10_80_grad'),
        80, n_iter_inner_J10, 'J10/I80 grad-dir')
    rate_J10_80_css, tau_J10_80_css = safe_load(
        PGA_Student_J10, ss_J10(80), spJ10('J10_80_css'),
        80, n_iter_inner_J10, 'J10/I80 cont-sched')

    # ── J20 / I=60 variant ────────────────────────────────────────────────────
    print('Loading J20/I60 models...')
    rate_J20_60_pure, tau_J20_60_pure = safe_load(
        PGA_Student_J20, ss_J20(60), spJ20('J20_60_pure'),
        60, n_iter_inner_J20, 'J20/I60 pure')
    rate_J20_60_traj, tau_J20_60_traj = safe_load(
        PGA_Student_J20, ss_J20(60), spJ20('J20_60_traj'),
        60, n_iter_inner_J20, 'J20/I60 traj-RKD')
    rate_J20_60_grad, tau_J20_60_grad = safe_load(
        PGA_Student_J20, ss_J20(60), spJ20('J20_60_grad'),
        60, n_iter_inner_J20, 'J20/I60 grad-dir')
    rate_J20_60_css, tau_J20_60_css = safe_load(
        PGA_Student_J20, ss_J20(60), spJ20('J20_60_css'),
        60, n_iter_inner_J20, 'J20/I60 cont-sched')

    # ── J20 / I=80 variant ────────────────────────────────────────────────────
    print('Loading J20/I80 models...')
    rate_J20_80_pure, tau_J20_80_pure = safe_load(
        PGA_Student_J20, ss_J20(80), spJ20('J20_80_pure'),
        80, n_iter_inner_J20, 'J20/I80 pure')
    rate_J20_80_traj, tau_J20_80_traj = safe_load(
        PGA_Student_J20, ss_J20(80), spJ20('J20_80_traj'),
        80, n_iter_inner_J20, 'J20/I80 traj-RKD')
    rate_J20_80_grad, tau_J20_80_grad = safe_load(
        PGA_Student_J20, ss_J20(80), spJ20('J20_80_grad'),
        80, n_iter_inner_J20, 'J20/I80 grad-dir')
    rate_J20_80_css, tau_J20_80_css = safe_load(
        PGA_Student_J20, ss_J20(80), spJ20('J20_80_css'),
        80, n_iter_inner_J20, 'J20/I80 cont-sched')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT FUNCTION — one set of 4 figures per variant
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# PLOT FUNCTION — saves 4 figures per variant (NO plt.show())
# Each figure has 3 curves:
#   Teacher
#   Pure Student (No KD)
#   Student + RKD
# Saves BOTH .png and .eps
# ══════════════════════════════════════════════════════════════════════════════

def plot_variant(I_S,
                 rate_pure, tau_pure,
                 rate_traj, tau_traj,
                 suffix):

    iter_s = np.arange(I_S + 1)

    obj_teacher = obj(rate_iter_J20, tau_iter_J20)
    obj_pure    = obj(rate_pure, tau_pure)
    obj_traj    = obj(rate_traj, tau_traj)

    # =====================================================
    # FIGURE 1: Objective
    # =====================================================
    plt.figure(figsize=(8,6))

    plt.plot(iter_J20_120, obj_teacher,
             'g-', linewidth=3, label='Teacher J20/I120')

    if obj_pure is not None:
        plt.plot(iter_s, obj_pure,
                 'b--', linewidth=3, label='Student No KD')

    if obj_traj is not None:
        plt.plot(iter_s, obj_traj,
                 'r-.', linewidth=3, label='Student + RKD')

    plt.xlabel('Iterations')
    plt.ylabel(r'$R-\omega\tau$')
    plt.title(f'Objective ({suffix})')
    plt.xlim(0,120)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(directory_result + f'obj_{suffix}.png')
    plt.savefig(directory_result + f'obj_{suffix}.eps')
    plt.close()


    # =====================================================
    # FIGURE 2: Rate
    # =====================================================
    plt.figure(figsize=(8,6))

    plt.plot(iter_J20_120, rate_iter_J20,
             'g-', linewidth=3, label='Teacher J20/I120')

    if rate_pure is not None:
        plt.plot(iter_s, rate_pure,
                 'b--', linewidth=3, label='Student No KD')

    if rate_traj is not None:
        plt.plot(iter_s, rate_traj,
                 'r-.', linewidth=3, label='Student + RKD')

    plt.xlabel('Iterations')
    plt.ylabel('Rate')
    plt.title(f'Rate ({suffix})')
    plt.xlim(0,120)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(directory_result + f'rate_{suffix}.png')
    plt.savefig(directory_result + f'rate_{suffix}.eps')
    plt.close()


    # =====================================================
    # FIGURE 3: Beam Error
    # =====================================================
    plt.figure(figsize=(8,6))

    plt.plot(iter_J20_120, tau_iter_J20,
             'g-', linewidth=3, label='Teacher J20/I120')

    if tau_pure is not None:
        plt.plot(iter_s, tau_pure,
                 'b--', linewidth=3, label='Student No KD')

    if tau_traj is not None:
        plt.plot(iter_s, tau_traj,
                 'r-.', linewidth=3, label='Student + RKD')

    plt.xlabel('Iterations')
    plt.ylabel(r'$\tau$')
    plt.title(f'Beam Error ({suffix})')
    plt.xlim(0,120)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(directory_result + f'tau_{suffix}.png')
    plt.savefig(directory_result + f'tau_{suffix}.eps')
    plt.close()


    # =====================================================
    # FIGURE 4: Tradeoff
    # =====================================================
    plt.figure(figsize=(8,6))

    plt.plot(tau_iter_J20, rate_iter_J20,
             'g-', linewidth=3, label='Teacher J20/I120')

    if rate_pure is not None:
        plt.plot(tau_pure, rate_pure,
                 'b--', linewidth=3, label='Student No KD')

    if rate_traj is not None:
        plt.plot(tau_traj, rate_traj,
                 'r-.', linewidth=3, label='Student + RKD')

    plt.xlabel(r'$\tau$')
    plt.ylabel('Rate')
    plt.title(f'Rate-Beam Tradeoff ({suffix})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(directory_result + f'tradeoff_{suffix}.png')
    plt.savefig(directory_result + f'tradeoff_{suffix}.eps')
    plt.close()
# ══════════════════════════════════════════════════════════════════════════════
# PRODUCE ALL FOUR VARIANT PLOTS
# ══════════════════════════════════════════════════════════════════════════════

if plot_figure == 0:

    plot_variant(
        60,
        rate_J10_60_pure, tau_J10_60_pure,
        rate_J10_60_traj, tau_J10_60_traj,
        'J10_I60'
    )

    plot_variant(
        80,
        rate_J10_80_pure, tau_J10_80_pure,
        rate_J10_80_traj, tau_J10_80_traj,
        'J10_I80'
    )

    plot_variant(
        60,
        rate_J20_60_pure, tau_J20_60_pure,
        rate_J20_60_traj, tau_J20_60_traj,
        'J20_I60'
    )

    plot_variant(
        80,
        rate_J20_80_pure, tau_J20_80_pure,
        rate_J20_80_traj, tau_J20_80_traj,
        'J20_I80'
    )

def plot_grad_vs_pure_variant(I_S,
                              rate_pure, tau_pure,
                              rate_grad, tau_grad,
                              suffix):

    iter_s = np.arange(I_S + 1)

    obj_teacher = obj(rate_iter_J20, tau_iter_J20)
    obj_pure    = obj(rate_pure, tau_pure)
    obj_grad    = obj(rate_grad, tau_grad)

    # Objective
    plt.figure()
    plt.plot(iter_J20_120, obj_teacher, 'g-', linewidth=2, label='Teacher J20/I120')

    if obj_pure is not None:
        plt.plot(iter_s, obj_pure, 'b--', linewidth=2, label='Student No KD')

    if obj_grad is not None:
        plt.plot(iter_s, obj_grad, 'm-.', linewidth=2, label='Student + Grad-RKD')

    plt.xlabel('Iterations')
    plt.ylabel(r'$R-\omega\tau$')
    plt.title(f'Objective Grad-RKD vs No-KD ({suffix})')
    plt.xlim(0,120)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory_result + f'obj_grad_vs_pure_{suffix}.png')
    plt.savefig(directory_result + f'obj_grad_vs_pure_{suffix}.eps')
    plt.close()

    # Rate
    plt.figure()
    plt.plot(iter_J20_120, rate_iter_J20, 'g-', linewidth=2, label='Teacher J20/I120')

    if rate_pure is not None:
        plt.plot(iter_s, rate_pure, 'b--', linewidth=2, label='Student No KD')

    if rate_grad is not None:
        plt.plot(iter_s, rate_grad, 'm-.', linewidth=2, label='Student + Grad-RKD')

    plt.xlabel('Iterations')
    plt.ylabel('Rate')
    plt.title(f'Rate Grad-RKD vs No-KD ({suffix})')
    plt.xlim(0,120)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory_result + f'rate_grad_vs_pure_{suffix}.png')
    plt.savefig(directory_result + f'rate_grad_vs_pure_{suffix}.eps')
    plt.close()

    # Beam error
    plt.figure()
    plt.plot(iter_J20_120, tau_iter_J20, 'g-', linewidth=2, label='Teacher J20/I120')

    if tau_pure is not None:
        plt.plot(iter_s, tau_pure, 'b--', linewidth=2, label='Student No KD')

    if tau_grad is not None:
        plt.plot(iter_s, tau_grad, 'm-.', linewidth=2, label='Student + Grad-RKD')

    plt.xlabel('Iterations')
    plt.ylabel(r'$\tau$')
    plt.title(f'Beam Error Grad-RKD vs No-KD ({suffix})')
    plt.xlim(0,120)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory_result + f'tau_grad_vs_pure_{suffix}.png')
    plt.savefig(directory_result + f'tau_grad_vs_pure_{suffix}.eps')
    plt.close()

    # Tradeoff
    plt.figure()
    plt.plot(tau_iter_J20, rate_iter_J20, 'g-', linewidth=2, label='Teacher J20/I120')

    if rate_pure is not None:
        plt.plot(tau_pure, rate_pure, 'b--', linewidth=2, label='Student No KD')

    if rate_grad is not None:
        plt.plot(tau_grad, rate_grad, 'm-.', linewidth=2, label='Student + Grad-RKD')

    plt.xlabel(r'$\tau$')
    plt.ylabel('Rate')
    plt.title(f'Rate-Beam Tradeoff Grad-RKD vs No-KD ({suffix})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory_result + f'tradeoff_grad_vs_pure_{suffix}.png')
    plt.savefig(directory_result + f'tradeoff_grad_vs_pure_{suffix}.eps')
    plt.close()

if plot_figure == 1:

    plot_grad_vs_pure_variant(
        60,
        rate_J10_60_pure, tau_J10_60_pure,
        rate_J10_60_grad, tau_J10_60_grad,
        'J10_I60'
    )

    plot_grad_vs_pure_variant(
        80,
        rate_J10_80_pure, tau_J10_80_pure,
        rate_J10_80_grad, tau_J10_80_grad,
        'J10_I80'
    )

    plot_grad_vs_pure_variant(
        60,
        rate_J20_60_pure, tau_J20_60_pure,
        rate_J20_60_grad, tau_J20_60_grad,
        'J20_I60'
    )

    plot_grad_vs_pure_variant(
        80,
        rate_J20_80_pure, tau_J20_80_pure,
        rate_J20_80_grad, tau_J20_80_grad,
        'J20_I80'
    )