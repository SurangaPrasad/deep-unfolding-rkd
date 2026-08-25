import math
import torch
import torch.nn as nn
from utility import *
from torch.utils.checkpoint import checkpoint

def analog_block_pga(F, W, H, R, step_size, ii, Pt, n_iter_inner):

    for jj in range(n_iter_inner):

        grad_F_com = get_grad_F_com(H, F, W)
        grad_F_rad = get_grad_F_rad(F, W, R)

        if grad_F_com.isnan().any() or grad_F_rad.isnan().any():
            print('Error NaN gradients!!!!!!!!!!!!!!!')

        mu = step_size[jj][ii][0]

        F = F + mu * (
            grad_F_com * WEIGHT_F_COM
            - grad_F_rad * WEIGHT_F_RAD
        )

        # safer + faster check (avoid Python sum)
        if torch.abs(F[0, :, 0, 0]).sum() > 1e3:
            F = normalize_power(F, W, H, Pt)

    return F

def clamp_complex_magnitude(delta, max_magnitude):
    """Scale complex updates so their magnitude never exceeds `max_magnitude`."""
    magnitude = torch.abs(delta)
    scale = torch.clamp(max_magnitude / (magnitude + 1e-12), max=1.0)
    return delta * scale


def sanitize_complex_tensor(tensor):
    """Replace NaN/Inf entries in a complex tensor with safe finite values."""
    real = torch.where(torch.isfinite(tensor.real), tensor.real, torch.zeros_like(tensor.real))
    imag = torch.where(torch.isfinite(tensor.imag), tensor.imag, torch.zeros_like(tensor.imag))
    return torch.complex(real, imag)


def project_unit_modulus(F, eps=1e-12, active_mask=None):
    """Project complex entries to unit modulus without introducing NaNs."""
    magnitude = torch.abs(F)
    safe_magnitude = torch.where(magnitude > eps, magnitude, torch.ones_like(magnitude))
    projected = F / safe_magnitude
    projected = torch.where(magnitude > eps, projected, torch.zeros_like(F))

    if active_mask is not None:
        mask = active_mask
        while mask.dim() < F.dim():
            mask = mask.unsqueeze(0)
        mask = mask.to(dtype=F.dtype, device=F.device)
        mask_bool = mask.abs() > 0

        # Rebuild the phase for entries that were driven below eps inside the active region.
        phase = torch.polar(torch.ones_like(F.real), torch.angle(F))
        projected = torch.where(mask_bool & (magnitude <= eps), phase, projected)
        projected = torch.where(mask_bool, projected, torch.zeros_like(projected))

    return projected

# /////////////////////////////////////////////////////////////////////////////////////////
#                             PGA MODEL CLASSES
# /////////////////////////////////////////////////////////////////////////////////////////
class PGA_Unfold_J3_I60_CI(nn.Module):
    """
    Student — 60 outer × 3 inner iterations.
    step_size shape: [3, 60, K+1]
    ~13× cheaper than teacher at inference.

    STABILITY NOTE: With only 3 inner steps, each step size has an
    outsized effect on F. max_step_size is set conservatively to 0.05.
    Monitor step_size max during training — if it approaches max_step_size
    consistently, reduce the learning rate rather than raising the clamp.
    """

    def __init__(self, step_size_init):
        super().__init__()
        if isinstance(step_size_init, torch.Tensor) and step_size_init.dim() == 3:
            self.step_size = nn.Parameter(step_size_init.float().clone())
        else:
            self.step_size = nn.Parameter(
                step_size_init * torch.ones(N_INNER_S, I_S, K + 1))

    def execute_PGA_with_both_windows(self, H, R, Pt,
                                       n_iter_outer, n_iter_inner, K_layers):
        rate_init, tau_init, F, W = initialize(H, R, Pt, initial_normalization)
        rate_over_iters = torch.zeros(n_iter_outer, H.shape[1], device=H.device)
        tau_over_iters  = torch.zeros(n_iter_outer, H.shape[1], device=H.device)
        F_traj_first, F_traj_last = [], []

        for ii in range(n_iter_outer):
            for jj in range(n_iter_inner):
                grad_F_com  = get_grad_F_com(H, F, W)
                grad_F_rad  = get_grad_F_rad(F, W, R)
                delta_F_com = self.step_size[jj][ii][0] * grad_F_com
                delta_F_rad = self.step_size[jj][ii][0] * grad_F_rad
                F = F + delta_F_com * WEIGHT_F_COM - delta_F_rad * WEIGHT_F_RAD
                F = normalize_power(F, W, H, Pt)
            F = F / torch.abs(F)

            W_new      = W.clone().detach()
            grad_W_com = get_grad_W_com(H, F, W)
            grad_W_rad = get_grad_W_rad(F, W, R)
            for k in range(K):
                delta_W_com = self.step_size[0][ii][k+1] * grad_W_com[k]
                delta_W_rad = self.step_size[0][ii][k+1] * grad_W_rad[k]
                W_new[k] = (W[k].clone().detach()
                            + delta_W_com * WEIGHT_W_COM
                            - delta_W_rad * WEIGHT_W_RAD)
            F, W = normalize(F, W_new, H, Pt)

            if ii < K_layers:
                F_traj_first.append(F.clone())
            if ii >= n_iter_outer - K_layers:
                F_traj_last.append(F.clone())

            rate_over_iters[ii] = get_sum_rate(H, F, W, Pt)
            tau_over_iters[ii]  = get_beam_error(H, F, W, R, Pt)

        rates = torch.cat([rate_init, rate_over_iters], dim=0)
        taus  = torch.cat([tau_init,  tau_over_iters],  dim=0)
        return (torch.transpose(rates, 0, 1),
                torch.transpose(taus,  0, 1),
                F, W, F_traj_first, F_traj_last)
#  ================================ PGA conventional with different inner iterations ===========================================
class PGA_Conv_comp_grad(nn.Module):

    def __init__(self, step_size):
        super().__init__()
        self.step_size = nn.Parameter(step_size)  # parameters = (mu, lambda)

    # =========== Projection Gradient Ascent execution ===================
    def execute_PGA(self, H, R, Pt, n_iter_outer, n_iter_inner, weight_grad_F_rad, init_method):
        rate_init, tau_init, F, W = initialize_schemes(H, R, Pt, init_method)
        rate_over_iters = torch.zeros(n_iter_outer, len(H[0]), device=H.device)  # save rates over iterations
        tau_over_iters = torch.zeros(n_iter_outer, len(H[0]), device=H.device)  # save beampattern errors over iterations
        # update F and W over iterations
        for ii in range(n_iter_outer):
            for jj in range(n_iter_inner):
                # update F
                grad_F_com = get_grad_F_com(H, F, W)
                grad_F_rad = get_grad_F_rad(F, W, R)
                # self.step_size[ii][0]
                delta_F_com = self.step_size[ii][0] * grad_F_com
                delta_F_rad = self.step_size[ii][0] * grad_F_rad
                F = F + delta_F_com - delta_F_rad * OMEGA
                # normalize by power to ensure non-NaN gradients if F becomes too large
                if sum(torch.abs(F[0, :, 0, 0])) > 1e1:
                    F = normalize_power(F, W, H, Pt)
                # Projection
                F = project_unit_modulus(F)

                # update W
                W_new = W.clone().detach()
                # compute gradients
                grad_W_k_com = get_grad_W_com(H, F, W)
                grad_W_k_rad = get_grad_W_rad(F, W, R)
                for k in range(K):
                    delta_W_com = self.step_size[ii][k + 1] * grad_W_k_com[k]
                    delta_W_rad = self.step_size[ii][k + 1] * grad_W_k_rad[k]
                    W_new[k] = W[k].clone().detach() + delta_W_com * WEIGHT_W_COM - delta_W_rad * WEIGHT_W_RAD

            # projection
            F, W = normalize(F, W_new, H, Pt)

            # get the rate in this iteration
            rate_over_iters[ii] = get_sum_rate(H, F, W, Pt)
            rates = torch.cat([rate_init, rate_over_iters], dim=0)
            tau_over_iters[ii] = get_beam_error(H, F, W, R, Pt)
            taus = torch.cat([tau_init, tau_over_iters], dim=0)
            # print(torch.linalg.matrix_norm(F @ W, ord='fro') ** 2)
        return torch.transpose(rates,  0, 1), torch.transpose(taus,  0, 1), F, W

#  ================================ UPGA with J = 1 and conventional ===========================================
class PGA_Conv(nn.Module):

    def __init__(self, step_size):
        super().__init__()
        self.step_size = nn.Parameter(step_size)  # parameters = (mu, lambda)

    # =========== Projection Gradient Ascent execution ===================
    def execute_PGA(self, H, R, Pt, n_iter_outer):
        rate_init, tau_init, F, W = initialize(H, R, Pt, initial_normalization)
        rate_over_iters = torch.zeros(n_iter_outer, len(H[0]), device=H.device)  # save rates over iterations
        tau_over_iters = torch.zeros(n_iter_outer, len(H[0]), device=H.device)  # save beampattern errors over iterations
        # update F and W over iterations
        for ii in range(n_iter_outer):
            # update F
            grad_F_com = get_grad_F_com(H, F, W)
            grad_F_rad = get_grad_F_rad(F, W, R)
            # self.step_size[ii][0]
            delta_F_com = self.step_size[ii][0] * grad_F_com
            delta_F_rad = self.step_size[ii][0] * grad_F_rad
            F = F + delta_F_com * WEIGHT_F_COM - delta_F_rad * WEIGHT_F_RAD

            # Projection
            F = project_unit_modulus(F)

            # update W
            W_new = W.clone().detach()
            # compute gradients
            grad_W_k_com = get_grad_W_com(H, F, W)
            grad_W_k_rad = get_grad_W_rad(F, W, R)
            for k in range(K):
                delta_W_com = self.step_size[ii][k + 1] * grad_W_k_com[k]
                delta_W_rad = self.step_size[ii][k + 1] * grad_W_k_rad[k]
                W_new[k] = W[k].clone().detach() + delta_W_com * WEIGHT_W_COM - delta_W_rad * WEIGHT_W_RAD

            # projection
            F, W = normalize(F, W_new, H, Pt)

            # get the rate in this iteration
            rate_over_iters[ii] = get_sum_rate(H, F, W, Pt)
            rates = torch.cat([rate_init, rate_over_iters], dim=0)
            tau_over_iters[ii] = get_beam_error(H, F, W, R, Pt)
            taus = torch.cat([tau_init, tau_over_iters], dim=0)
            # print(torch.linalg.matrix_norm(F @ W, ord='fro') ** 2)
        return torch.transpose(rates,  0, 1), torch.transpose(taus,  0, 1), F, W

# ============================================== Proposed PGA model light=============================
class PGA_Unfold_J10(nn.Module):

    def __init__(self, step_size):
        super().__init__()
        self.step_size = nn.Parameter(step_size)  # parameters = (mu, lambda)

    # =========== Projection Gradient Ascent execution ===================
    def execute_PGA(self, H, R, Pt, n_iter_outer, n_iter_inner):
        rate_init, tau_init, F, W = initialize(H, R, Pt, initial_normalization)
        rate_over_iters = torch.zeros(n_iter_outer, len(H[0]), device=H.device)  # save rates over iterations
        tau_over_iters = torch.zeros(n_iter_outer, len(H[0]), device=H.device)  # save beam errors over iterations

        for ii in range(n_iter_outer):
            F = checkpoint(
                analog_block_pga,
                F, W, H, R, self.step_size, ii, Pt, n_iter_inner,use_reentrant=False
            )
            # Projection
            F = project_unit_modulus(F)

            # update W
            W_new = W.clone().detach()
            # compute gradients
            grad_W_k_com = get_grad_W_com(H, F, W)
            grad_W_k_rad = get_grad_W_rad(F, W, R)
            for k in range(K):
                delta_W_com = self.step_size[0][ii][k + 1] * grad_W_k_com[k]
                delta_W_rad = self.step_size[0][ii][k + 1] * grad_W_k_rad[k]
                W_new[k] = W[k].clone().detach() + delta_W_com * WEIGHT_W_COM - delta_W_rad * WEIGHT_W_RAD
            # Projection
            F, W = normalize(F, W_new, H, Pt)

            # get the rate in this iteration
            rate_over_iters[ii] = get_sum_rate(H, F, W, Pt)
            # print(rate_over_iters[ii])
            rates = torch.cat([rate_init, rate_over_iters], dim=0)
            tau_over_iters[ii] = get_beam_error(H, F, W, R, Pt)
            taus = torch.cat([tau_init, tau_over_iters], dim=0)

        return torch.transpose(rates, 0, 1), torch.transpose(taus, 0, 1), F, W

# ============================================== Proposed PGA model=============================
class PGA_Unfold_J20(nn.Module):

    def __init__(self, step_size):
        super().__init__()
        self.step_size = nn.Parameter(step_size)  # parameters = (mu, lambda)

    # =========== Projection Gradient Ascent execution ===================
    def execute_PGA(self, H, R, Pt, n_iter_outer, n_iter_inner):
        rate_init, tau_init, F, W = initialize(H, R, Pt, initial_normalization)
        rate_over_iters = torch.zeros(n_iter_outer, len(H[0]), device=H.device)  # save rates over iterations
        tau_over_iters = torch.zeros(n_iter_outer, len(H[0]), device=H.device)  # save beam errors over iterations

        for ii in range(n_iter_outer):
            F = checkpoint(
                analog_block_pga,
                F, W, H, R, self.step_size, ii, Pt, n_iter_inner,use_reentrant=False
            )
            # Projection
            F = project_unit_modulus(F)

            # update W
            W_new = W.clone().detach()
            # compute gradients
            grad_W_k_com = get_grad_W_com(H, F, W)
            grad_W_k_rad = get_grad_W_rad(F, W, R)
            for k in range(K):
                delta_W_com = self.step_size[0][ii][k + 1] * grad_W_k_com[k]
                delta_W_rad = self.step_size[0][ii][k + 1] * grad_W_k_rad[k]
                W_new[k] = W[k].clone().detach() + delta_W_com * WEIGHT_W_COM - delta_W_rad * WEIGHT_W_RAD

            # Projection
            F, W = normalize(F, W_new, H, Pt)

            # get the rate in this iteration
            rate_over_iters[ii] = get_sum_rate(H, F, W, Pt)
            # print(rate_over_iters[ii])
            rates = torch.cat([rate_init, rate_over_iters], dim=0)
            tau_over_iters[ii] = get_beam_error(H, F, W, R, Pt)
            taus = torch.cat([tau_init, tau_over_iters], dim=0)

        return torch.transpose(rates, 0, 1), torch.transpose(taus, 0, 1), F, W



class PGA_Unfold_J5_I120_CI(nn.Module):
    """
    Student — 120 outer × 5 inner iterations.
    step_size shape: [5, 120, K+1]
    4× cheaper than teacher (600 vs 2400 steps).

    Same outer depth as teacher — no outer schedule mismatch.
    Inner RKD mapping: student inner j ←→ teacher inner (15+j).
    """

    def __init__(self, step_size_init):
        super().__init__()
        if isinstance(step_size_init, torch.Tensor) and step_size_init.dim() == 3:
            self.step_size = nn.Parameter(step_size_init.float().clone())
        else:
            self.step_size = nn.Parameter(
                step_size_init * torch.ones(N_INNER_S, I_S, K + 1))

    def execute_PGA_inner_windows(self, H, R, Pt,
                                   n_iter_outer, n_iter_inner, K_outer):
        rate_init, tau_init, F, W = initialize(H, R, Pt, initial_normalization)
        rate_over_iters = torch.zeros(n_iter_outer, H.shape[1], device=H.device)
        tau_over_iters  = torch.zeros(n_iter_outer, H.shape[1], device=H.device)
        F_inner_S       = []

        for ii in range(n_iter_outer):
            collect_this_outer = (ii < K_outer)
            for jj in range(n_iter_inner):
                grad_F_com  = get_grad_F_com(H, F, W)
                grad_F_rad  = get_grad_F_rad(F, W, R)
                delta_F_com = self.step_size[jj][ii][0] * grad_F_com
                delta_F_rad = self.step_size[jj][ii][0] * grad_F_rad
                F = F + delta_F_com * WEIGHT_F_COM - delta_F_rad * WEIGHT_F_RAD
                F = normalize_power(F, W, H, Pt)
                if collect_this_outer:
                    F_inner_S.append(F.clone())
            F = F / torch.abs(F)

            W_new      = W.clone().detach()
            grad_W_com = get_grad_W_com(H, F, W)
            grad_W_rad = get_grad_W_rad(F, W, R)
            for k in range(K):
                delta_W_com = self.step_size[0][ii][k+1] * grad_W_com[k]
                delta_W_rad = self.step_size[0][ii][k+1] * grad_W_rad[k]
                W_new[k] = (W[k].clone().detach()
                            + delta_W_com * WEIGHT_W_COM
                            - delta_W_rad * WEIGHT_W_RAD)
            F, W = normalize(F, W_new, H, Pt)

            rate_over_iters[ii] = get_sum_rate(H, F, W, Pt)
            tau_over_iters[ii]  = get_beam_error(H, F, W, R, Pt)

        rates = torch.cat([rate_init, rate_over_iters], dim=0)
        taus  = torch.cat([tau_init,  tau_over_iters],  dim=0)
        return (torch.transpose(rates, 0, 1),
                torch.transpose(taus,  0, 1),
                F, W, F_inner_S)
class PGA_Unfold_J20_I60(nn.Module):

    def __init__(self, step_size):
        super().__init__()
        self.step_size = nn.Parameter(step_size)  # parameters = (mu, lambda)

    # =========== Projection Gradient Ascent execution ===================
    def execute_PGA(self, H, R, Pt, n_iter_outer, n_iter_inner):
        rate_init, tau_init, F, W = initialize(H, R, Pt, initial_normalization)
        rate_over_iters = torch.zeros(60, len(H[0]), device=H.device)  # save rates over iterations
        tau_over_iters = torch.zeros(60, len(H[0]), device=H.device)  # save beam errors over iterations

        for ii in range(n_iter_outer):
            F = checkpoint(
                analog_block_pga,
                F, W, H, R, self.step_size, ii, Pt, n_iter_inner,use_reentrant=False
            )
            # Projection
            F = project_unit_modulus(F)

            # update W
            W_new = W.clone().detach()
            # compute gradients
            grad_W_k_com = get_grad_W_com(H, F, W)
            grad_W_k_rad = get_grad_W_rad(F, W, R)
            for k in range(K):
                delta_W_com = self.step_size[0][ii][k + 1] * grad_W_k_com[k]
                delta_W_rad = self.step_size[0][ii][k + 1] * grad_W_k_rad[k]
                W_new[k] = W[k].clone().detach() + delta_W_com * WEIGHT_W_COM - delta_W_rad * WEIGHT_W_RAD

            # Projection
            F, W = normalize(F, W_new, H, Pt)

            # get the rate in this iteration
            rate_over_iters[ii] = get_sum_rate(H, F, W, Pt)
            # print(rate_over_iters[ii])
            rates = torch.cat([rate_init, rate_over_iters], dim=0)
            tau_over_iters[ii] = get_beam_error(H, F, W, R, Pt)
            taus = torch.cat([tau_init, tau_over_iters], dim=0)

        return torch.transpose(rates, 0, 1), torch.transpose(taus, 0, 1), F, W

class PGA_Unfold_J10_I60(PGA_Unfold_J10):
    """
    Student — 60 outer × 10 inner.
    step_size shape: [10, 60, K+1]

    4× cheaper than teacher at inference (600 vs 2400 steps).
    Single forward pass collects first-K F for CI-RKD.
    """

    def execute_PGA_single_pass(self, H, R, Pt,
                                 n_iter_outer, n_iter_inner, K_layers):
        """
        Single 60-iteration pass.
        Collects F for iters 0..K_layers-1 (for CI-RKD early window).
        K_layers=0: no F stored, pure task loss.
        """
        rate_init, tau_init, F, W = initialize(H, R, Pt, initial_normalization)
        rate_over_iters = torch.zeros(n_iter_outer, H.shape[1], device=H.device)
        tau_over_iters  = torch.zeros(n_iter_outer, H.shape[1], device=H.device)
        F_traj = []

        for ii in range(n_iter_outer):
            for jj in range(n_iter_inner):
                grad_F_com  = get_grad_F_com(H, F, W)
                grad_F_rad  = get_grad_F_rad(F, W, R)
                delta_F_com = self.step_size[jj][ii][0] * grad_F_com
                delta_F_rad = self.step_size[jj][ii][0] * grad_F_rad
                F = F + delta_F_com * WEIGHT_F_COM - delta_F_rad * WEIGHT_F_RAD
                if sum(torch.abs(F[0, :, 0, 0])) > 1e3:
                    F = normalize_power(F, W, H, Pt)
            F = F / torch.abs(F)

            W_new      = W.clone().detach()
            grad_W_com = get_grad_W_com(H, F, W)
            grad_W_rad = get_grad_W_rad(F, W, R)
            for k in range(K):
                delta_W_com = self.step_size[0][ii][k+1] * grad_W_com[k]
                delta_W_rad = self.step_size[0][ii][k+1] * grad_W_rad[k]
                W_new[k] = (W[k].clone().detach()
                            + delta_W_com * WEIGHT_W_COM
                            - delta_W_rad * WEIGHT_W_RAD)
            F, W = normalize(F, W_new, H, Pt)

            if ii < K_layers:
                F_traj.append(F.clone())   # in-graph ✓

            rate_over_iters[ii] = get_sum_rate(H, F, W, Pt)
            tau_over_iters[ii]  = get_beam_error(H, F, W, R, Pt)

        rates = torch.cat([rate_init, rate_over_iters], dim=0)
        taus  = torch.cat([tau_init,  tau_over_iters],  dim=0)
        return (torch.transpose(rates, 0, 1),
                torch.transpose(taus,  0, 1),
                F, W, F_traj)

class PGA_Unfold_J10_60(PGA_Unfold_J10):
    """
    Student — 80 outer × 10 inner.
    step_size shape: [10, 80, K+1]

    Extends PGA_Unfold_J10 — base class __init__ registers step_size
    correctly for J=10 inner steps.

    Single forward pass collects:
      FW_early : (F, W) at outer iters 0 .. K_layers-1
      FW_late  : (F, W) at outer iters I_S-K_layers .. I_S-1
      F_final  : F at outer iter I_S-1  (for response distillation)
      W_final  : W at outer iter I_S-1  (for response distillation)

    W inside the loop is always detached (W.clone().detach() + grad_step).
    F has grad_fn through inner loop → step_size.
    matmul(F, W) has grad_fn through F ✓
    """

    def execute_PGA_single_pass(self, H, R, Pt,
                                 n_iter_outer, n_iter_inner,
                                 K_layers):
        """
        Single 80-iteration forward pass — no repeated iterations.

        Returns:
          F_final, W_final    : final beamformers for task + response loss
          FW_early            : list of K_layers (F,W) tuples, iters 0..K-1
          FW_late             : list of K_layers (F,W) tuples, iters I-K..I-1
          rate_iter, tau_iter : convergence metrics for evaluation
        """
        rate_init, tau_init, F, W = initialize(H, R, Pt, initial_normalization)
        rate_over = torch.zeros(n_iter_outer, H.shape[1], device=H.device)
        tau_over  = torch.zeros(n_iter_outer, H.shape[1], device=H.device)

        FW_early = []
        FW_late  = []
        I = n_iter_outer

        for ii in range(I):

            # ── J=10 inner F-gradient steps ───────────────────────────────────
            for jj in range(n_iter_inner):
                gF_c = get_grad_F_com(H, F, W)
                gF_r = get_grad_F_rad(F, W, R)
                step = self.step_size[jj][ii][0]
                F = F + step * gF_c * WEIGHT_F_COM \
                      - step * gF_r * WEIGHT_F_RAD
            F = F / torch.abs(F)

            # ── Single W-gradient step ─────────────────────────────────────────
            # W_new is always detached — only F carries the gradient
            W_new = W.clone().detach()
            gW_c  = get_grad_W_com(H, F, W)
            gW_r  = get_grad_W_rad(F, W, R)
            for k in range(K):
                step    = self.step_size[0][ii][k+1]
                W_new[k] = (W[k].clone().detach()
                            + step * gW_c[k] * WEIGHT_W_COM
                            - step * gW_r[k] * WEIGHT_W_RAD)
            F, W = normalize(F, W_new, H, Pt)

            # ── Store (F, W) tuples for distillation windows ──────────────────
            # F is in-graph, W is detached
            # matmul(F, W) computed in loss has grad_fn through F ✓
            if ii < K_layers:
                FW_early.append((F.clone(), W.detach().clone()))

            if ii >= I - K_layers:
                FW_late.append((F.clone(), W.detach().clone()))

            rate_over[ii] = get_sum_rate(H, F, W, Pt)
            tau_over[ii]  = get_beam_error(H, F, W, R, Pt)

        rates = torch.cat([rate_init, rate_over], dim=0)
        taus  = torch.cat([tau_init,  tau_over],  dim=0)

        return (torch.transpose(rates, 0, 1),
                torch.transpose(taus,  0, 1),
                F, W,           # final beamformers — in-graph for loss
                FW_early,       # early window for CI-RKD
                FW_late)        # late window        

# /////////////////////////////////////////////////////////////////////////////////////////
#                             COMM GRADIENTS
# /////////////////////////////////////////////////////////////////////////////////////////


# ==================================== gradient of R_mk w.r.t. F ===========================
def get_grad_F_com(H, F, W):
    """Vectorised gradient of sum-rate w.r.t. F (no Python loop over users)."""
    F_H = F.conj().transpose(-2, -1)          # (K, B, Nrf, Nt)
    W_H = W.conj().transpose(-2, -1)           # (K, B, M, Nrf)
    V   = W @ W_H                               # (K, B, Nrf, Nrf)
    K_d = W.shape[0]

    # Per-user outer products w_m w_m^H -> V_mk = V - outer_m
    # w_cols: (K, B, M, Nrf, 1)
    w_cols = W.permute(0, 1, 3, 2).unsqueeze(-1)
    V_m    = w_cols @ w_cols.conj().transpose(-2, -1)   # (K, B, M, Nrf, Nrf)
    V_mk   = V.unsqueeze(2) - V_m                        # (K, B, M, Nrf, Nrf)

    # Channel outer products H_tilde_m = h_m h_m^H  (K, B, M, Nt, Nt)
    h       = H.unsqueeze(-1)                             # (K, B, M, Nt, 1)
    Htilde  = h @ h.conj().transpose(-2, -1)              # (K, B, M, Nt, Nt)

    # Shared: F @ V @ F_H                                 (K, B, Nt, Nt)
    FVF_H = F @ V @ F_H

    # Quadratic forms via h^H A h  (cheap compared to full Nt×Nt trace)
    qf1 = (h.conj().transpose(-2, -1) @ FVF_H.unsqueeze(2) @ h).squeeze(-1).squeeze(-1)  # (K,B,M)
    denom1 = np.log(2) * (qf1 + sigma2)

    FVmk    = F.unsqueeze(2) @ V_mk                       # (K, B, M, Nt, Nrf)
    FVmkF_H = FVmk @ F_H.unsqueeze(2)                     # (K, B, M, Nt, Nt)
    qf2 = (h.conj().transpose(-2, -1) @ FVmkF_H @ h).squeeze(-1).squeeze(-1)  # (K,B,M)
    denom2 = np.log(2) * (qf2 + sigma2)

    HtF   = Htilde @ F.unsqueeze(2)                        # (K, B, M, Nt, Nrf)
    grad1 = HtF @ V.unsqueeze(2)  / (denom1.unsqueeze(-1).unsqueeze(-1) + 1e-4)  # (K,B,M,Nt,Nrf)
    grad2 = HtF @ V_mk            / (denom2.unsqueeze(-1).unsqueeze(-1) + 1e-4)  # (K,B,M,Nt,Nrf)

    # Sum over M users, average over K frequencies
    grad_F = (grad1 - grad2).sum(dim=2) / K_d             # (K, B, Nt, Nrf)
    return grad_F

def get_grad_W_com(H, F, W):
    F_H = torch.transpose(F, 2, 3).conj()
    W_H = torch.transpose(W, 2, 3).conj()
    V = W @ W_H  # K x train_size x Nrf x Nrf
    grad_W = torch.zeros(len(H), len(H[0]), Nrf, M, dtype=H.dtype, device=H.device)

    for m in range(M):
        W_m = W
        # print(W)
        W_m_H = torch.transpose(W_m, 2, 3).conj()

        h_mk0 = torch.unsqueeze(H[:, :, m, :], dim=2)
        h_mk = torch.transpose(h_mk0, 2, 3)
        h_mk_H = torch.transpose(h_mk, 2, 3).conj()
        Htilde_mk = h_mk @ h_mk_H
        Hbar_mk = F_H @ Htilde_mk @ F

        denom_1 = np.log(2) * (get_trace(W @ W_H @ Hbar_mk) + sigma2)
        grad_W_1 = Hbar_mk @ W / denom_1[:, :, None, None]  # expand dimension

        denom_2 = np.log(2) * (get_trace(W_m @ W_m_H @ Hbar_mk) + sigma2)
        grad_W_2 = Hbar_mk @ W_m / denom_2[:, :, None, None]  # expand dimension
        mask_m = torch.ones(len(H), len(H[0]), Nrf, M, device=H.device)
        mask_m[:, :, :, m] = 0.0
        grad_W_2_masked = grad_W_2 * mask_m  # need element-wise multiplication for masking
        grad_W = grad_W + (grad_W_1 - grad_W_2_masked)

    grad_W = grad_W / K
    return grad_W



def generate_pi_matrix(B_matrix, H_tilde, Nt, Nrf):
    """
    Batched Pi matrix generation for partially connected architecture.
    
    Parameters
    ----------
    B_matrix : torch.Tensor
        Digital covariance matrix, shape (Bsz, Nrf, Nrf)
    H_tilde : torch.Tensor
        Channel outer product, shape (Bsz, Nt, Nt)
    Nt : int
        Total number of antennas
    Nrf : int
        Number of RF chains (subarrays)
    
    Returns
    -------
    Pi : torch.Tensor
        Condensed sensing/channel matrix, shape (Bsz, Nt, Nt)
    """
    Bsz = B_matrix.shape[0]
    antennas_per_rf = Nt // Nrf
    
    Pi = torch.zeros(
        (Bsz, Nt, Nt),
        dtype=H_tilde.dtype,
        device=H_tilde.device
    )
    
    Bt = B_matrix.transpose(-1, -2)  # (Bsz, Nrf, Nrf)
    
    for m in range(Nrf):
        for n in range(Nrf):
            row_start = m * antennas_per_rf
            row_end = (m + 1) * antennas_per_rf
            col_start = n * antennas_per_rf
            col_end = (n + 1) * antennas_per_rf
            
            H_sub = H_tilde[:, row_start:row_end, col_start:col_end]
            
            Pi[:, row_start:row_end, col_start:col_end] = (
                Bt[:, m, n].view(Bsz, 1, 1) * H_sub
            )
    
    return Pi


def quad_form(Pi, f):
    """
    Computes f^H Pi f for batch.
    
    Parameters
    ----------
    Pi : torch.Tensor
        Matrix, shape (B, Nt, Nt)
    f : torch.Tensor
        Vector, shape (B, Nt, 1)
    
    Returns
    -------
    result : torch.Tensor
        Quadratic form result, shape (B, 1, 1)
    """
    return torch.bmm(f.conj().transpose(-1, -2), torch.bmm(Pi, f))


def extract_active_elements_pc(F, Nt, Nrf):
    """
    Extract active (non-zero) elements from a block-diagonal analog precoder.
    
    Parameters
    ----------
    F : torch.Tensor
        Analog precoder of shape (B, Nt, Nrf), complex
    Nt : int
        Total number of antennas
    Nrf : int
        Number of RF chains (subarrays)
    
    Returns
    -------
    f_active : torch.Tensor
        Active elements of shape (B, Nt, 1)
    """
    B = F.shape[0]
    antennas_per_rf = Nt // Nrf
    
    # Initialize output
    f_active = torch.zeros((B, Nt, 1), dtype=F.dtype, device=F.device)
    
    for i in range(Nrf):
        start_idx = i * antennas_per_rf
        end_idx = (i + 1) * antennas_per_rf
        
        # Copy the active block from column i
        f_active[:, start_idx:end_idx, 0] = F[:, start_idx:end_idx, i]
    
    return f_active


def compute_digital_covariance_pc(W, Pt, Nrf, eps=1e-12):
    """
    Computes the digital precoder covariance matrix V = W W^H
    with PC-specific power normalization.
    
    Parameters
    ----------
    W : torch.Tensor
        Digital precoder, shape (B, Nrf, M)
    Pt : float or torch.Tensor
        Total transmit power (scalar or shape (B,))
    Nrf : int
        Number of RF chains (used for PC normalization)
    eps : float
        Numerical stability constant
    
    Returns
    -------
    V : torch.Tensor
        Digital covariance matrix, shape (B, Nrf, Nrf)
    """
    B = W.shape[0]
    
    # Frobenius norm ||W||_F per batch
    fro_norm = torch.linalg.norm(W, ord='fro', dim=(1, 2), keepdim=True)  # (B, 1, 1)
    
    # Handle scalar or batched Pt
    if not torch.is_tensor(Pt):
        P_vec = torch.tensor(float(Pt), device=W.device).view(1).repeat(B)
    else:
        P_vec = Pt.to(W.device).view(B) if Pt.dim() > 0 else Pt.repeat(B)
    
    # sqrt(Pt / Nrf) for PC normalization
    scale = torch.sqrt(P_vec / Nrf).view(B, 1, 1)
    
    # Normalized W (PC case)
    W_normalized = scale * W / (fro_norm + eps)
    
    # Digital covariance V = W W^H
    V = torch.bmm(W_normalized, W_normalized.conj().transpose(-1, -2))  # (B, Nrf, Nrf)
    
    return V


def get_grad_F_rad_AP(F, W, R, Pt, Nt, Nrf):
    """
    Computes the Euclidean gradient of the sensing objective
    for partially connected architecture.
    
    This is the sensing/radar gradient: ∇_f τ where τ = ||f^H Pi f - target||²
    
    Parameters
    ----------
    F : torch.Tensor
        Analog precoder, shape (K, B, Nt, Nrf) or (B, Nt, Nrf)
    W : torch.Tensor
        Digital precoder, shape (K, B, Nrf, M) or (B, Nrf, M)
    R : torch.Tensor
        Target sensing covariance matrix, shape (K, B, Nt, Nt) or (B, Nt, Nt)
    Pt : float or torch.Tensor
        Total transmit power
    Nt : int
        Total number of antennas
    Nrf : int
        Number of RF chains
    
    Returns
    -------
    grad_F : torch.Tensor
        Gradient w.r.t. F in active element space, same shape as F
    """
    # Handle both 4D (K, B, Nt, Nrf) and 3D (B, Nt, Nrf) inputs
    if F.dim() == 4:
        K_freq, Bsz, _, _ = F.shape
        F_single = F[0]  # Work with first frequency
        W_single = W[0] if W.dim() == 4 else W
        R_single = R[0] if R.dim() == 4 else R
    else:
        Bsz = F.shape[0]
        F_single = F
        W_single = W
        R_single = R
        K_freq = None
    
    # ---------------------------------------------------
    # 1. Extract active elements
    # ---------------------------------------------------
    f_active = extract_active_elements_pc(F_single, Nt, Nrf)  # (B, Nt, 1)
    
    # ---------------------------------------------------
    # 2. Compute digital covariance V = W W^H (normalized)
    # ---------------------------------------------------
    V = compute_digital_covariance_pc(W_single, Pt, Nrf)  # (B, Nrf, Nrf)
    
    # ---------------------------------------------------
    # 3. Generate Pi matrix: Pi = Σ_rf V_rf * R_block
    # ---------------------------------------------------
    Pi = generate_pi_matrix(V, R_single, Nt, Nrf)  # (B, Nt, Nt)
    
    # ---------------------------------------------------
    # 4. Compute gradient: ∇_f τ = 2 * Pi * f
    #    This is the gradient of f^H Pi f w.r.t. f
    # ---------------------------------------------------
    grad_f_active = 2.0 * torch.bmm(Pi, f_active)  # (B, Nt, 1)
    
    # ---------------------------------------------------
    # 5. Lift gradient from active vector back to F matrix
    # ---------------------------------------------------
    grad_F = torch.zeros_like(F_single)  # (B, Nt, Nrf)
    antennas_per_rf = Nt // Nrf
    for rf in range(Nrf):
        start = rf * antennas_per_rf
        end = (rf + 1) * antennas_per_rf
        grad_F[:, start:end, rf] = grad_f_active[:, start:end, 0]
    
    # ---------------------------------------------------
    # 6. Replicate for all frequencies if needed
    # ---------------------------------------------------
    if K_freq is not None:
        grad_F_all_freq = torch.cat([grad_F.unsqueeze(0)] * K_freq, dim=0)
        return grad_F_all_freq
    else:
        return grad_F


def get_grad_F_com_AP(f_active, H, F, W):
    """
    Computes ∇_a R for partially connected architecture.
    
    Parameters
    ----------
    H : (K, B, M, Nt) channel
    F : (K, B, Nt, Nrf) analog precoder
    W : (K, B, Nrf, M) digital precoder
    
    Returns
    -------
    grad_F : (K, B, Nt, Nrf) gradient of rate w.r.t. F
    """
    
    # Get dimensions from F
    K_freq, Bsz, Nt, Nrf = F.shape
    _, _, M, _ = H.shape
    
    # ---------------------------------------------------
    # 1. Extract active elements: f = N[vec(F)]
    # ---------------------------------------------------
    # f_active shape: (B, Nt, 1)
    
    # ---------------------------------------------------
    # 2. Channel outer products: H̃_m = h_m h_m^H
    # ---------------------------------------------------
    # H shape: (K, B, M, Nt), we need (B, M, Nt, Nt)
    H_single_freq = H[0]  # (B, M, Nt) - work with first frequency for gradient
    H_tilde = (
        H_single_freq.unsqueeze(-1) @ H_single_freq.conj().unsqueeze(-2)
    )  # (B, M, Nt, Nt)
    
    # ---------------------------------------------------
    # 3. Digital covariance terms
    # ---------------------------------------------------
    W_single_freq = W[0]  # (B, Nrf, M)
    w = W_single_freq.permute(0, 2, 1).unsqueeze(-1)   # (B, M, Nrf, 1)
    BmT = w @ w.conj().transpose(-1, -2)  # (B, M, Nrf, Nrf)
    B_all = BmT.sum(dim=1, keepdim=True)  # (B, 1, Nrf, Nrf)
    BmT_tilde = B_all - BmT               # (B, M, Nrf, Nrf)
    
    ln2 = torch.log(torch.tensor(2.0, device=F.device))
    grad_f_active = torch.zeros_like(f_active)
    
    # ---------------------------------------------------
    # 4. Sum over m (users)
    # ---------------------------------------------------
    for m in range(M):
        # Generate Pi matrices using digital covariance and channel outer products
        Pi_mm = generate_pi_matrix(
            BmT[:, m], H_tilde[:, m], Nt, Nrf
        )  # (B, Nt, Nt)
        
        Pi_m_tilde = generate_pi_matrix(
            BmT_tilde[:, m], H_tilde[:, m], Nt, Nrf
        )  # (B, Nt, Nt)
        
        # Quadratic forms
        fH_Pi_mm_f = quad_form(Pi_mm, f_active)  # (B, 1, 1)
        fH_Pi_m_tilde_f = quad_form(Pi_m_tilde, f_active)  # (B, 1, 1)
        
        # Denominators with noise
        denom1 = (
            fH_Pi_mm_f + fH_Pi_m_tilde_f + sigma2
        )
        denom2 = fH_Pi_m_tilde_f + sigma2
        
        # Gradient terms
        term1 = (
            fH_Pi_mm_f / (ln2 * denom1)
        )
        
        term2 = (
            2 * fH_Pi_mm_f
            * torch.bmm(Pi_m_tilde, f_active)
            / (ln2 * denom1 * denom2)
        )
        
        grad_f_active += term1 * f_active - term2
    
    # ---------------------------------------------------
    # 5. Lift gradient from active vector back to F matrix
    # ---------------------------------------------------
    grad_F = torch.zeros_like(F[0])  # (B, Nt, Nrf)
    antennas_per_rf = Nt // Nrf
    for rf in range(Nrf):
        start = rf * antennas_per_rf
        end = (rf + 1) * antennas_per_rf
        grad_F[:, start:end, rf] = grad_f_active[:, start:end, 0]
    
    # Replicate for all frequencies
    grad_F_all_freq = torch.cat([grad_F.unsqueeze(0)] * K_freq, dim=0)
    
    return grad_F_all_freq


# /////////////////////////////////////////////////////////////////////////////////////////
#                             RADAR GRADIENTS
# /////////////////////////////////////////////////////////////////////////////////////////

# ==================================== gradient of tau w.r.t. F ===========================
def get_grad_F_rad(F, W, R):
    F_H = torch.transpose(F, 2, 3).conj()
    W_H = torch.transpose(W, 2, 3).conj()
    if normalize_tau == 1:
        grad_F_K = 2 * (F @ W @ W_H @ F_H - R) @ F @ W @ W_H / torch.linalg.matrix_norm(R[:, 0, :, :], ord='fro') ** 2
    else:
        grad_F_K = 2 * (F @ W @ W_H @ F_H - R) @ F @ W @ W_H
    grad_F_sum = sum(grad_F_K)
    grad_F = grad_F_sum / K
    return grad_F

# ==================================== gradient of tau w.r.t. W ===========================
def get_grad_W_rad(F, W, R):
    F_H = torch.transpose(F, 2, 3).conj()
    W_H = torch.transpose(W, 2, 3).conj()
    if normalize_tau == 1:
        grad_W = 2 * F_H @ (F @ W @ W_H @ F_H - R) @ F @ W / torch.linalg.matrix_norm(R[:, 0, :, :], ord='fro') ** 2
    else:
        grad_W = 2 * F_H @ (F @ W @ W_H @ F_H - R) @ F @ W
    grad_W = grad_W / K
    return grad_W

# ==================compute los based on (15)
def get_sum_loss(F, W, H, R, Pt, batch_size):
    sum_rate = get_sum_rate(H, F, W, Pt)
    # sum_rate = sum(rate)
    X = F @ W
    X_H = torch.transpose(X, 2, 3).conj()
    if normalize_tau == 1:
        error = torch.linalg.matrix_norm(X @ X_H - R, ord='fro') ** 2 / torch.linalg.matrix_norm(R[:, 0, :, :] , ord='fro') ** 2
    else:
        error = torch.linalg.matrix_norm(X @ X_H - R, ord='fro') ** 2
    # sum_error = sum(sum(error))
    sum_error = torch.mean(error)
    loss = -(sum_rate - OMEGA * sum_error)# / (K * batch_size)
    return loss