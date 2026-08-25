"""
utility_gpu.py
==============
Self-contained GPU-safe utility file.
Replaces utility.py entirely — no imports from utility.py.

Usage:
    from utility_gpu import *
"""

import torch
import torch.nn as nn
import sys
import h5py
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from system_config import *


# ══════════════════════════════════════════════════════════════════════════════
# DTYPE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_real_dtype_like(tensor):
    if torch.is_complex(tensor):
        return torch.float64 if tensor.dtype == torch.complex128 else torch.float32
    return tensor.dtype


def _randn_complex(shape, device, real_dtype):
    real = torch.randn(shape, device=device, dtype=real_dtype)
    imag = torch.randn(shape, device=device, dtype=real_dtype)
    return real + 1j * imag


# ══════════════════════════════════════════════════════════════════════════════
# NORMALIZE
# ══════════════════════════════════════════════════════════════════════════════

def normalize(F, W, H, Pt):
    F = F / (torch.abs(F) + 1e-12)
    sum_norm_BB = sum(torch.linalg.matrix_norm(F @ W, ord='fro') ** 2)
    sum_norm_BB = torch.clamp(sum_norm_BB, min=1e-6)
    normalize_factor = torch.sqrt(K * Pt / sum_norm_BB).reshape(len(H[0]), 1, 1)
    W = normalize_factor * W
    return F, W


def normalize_power(F, W, H, Pt):
    sum_norm_power = sum(torch.linalg.matrix_norm(F @ W, ord='fro') ** 2)
    sum_norm_power = torch.clamp(sum_norm_power, min=1e-6)
    normalize_factor = torch.sqrt(Pt / sum_norm_power).reshape(len(H[0]), 1, 1)
    F = normalize_factor * F
    return F


# ══════════════════════════════════════════════════════════════════════════════
# TRACE
# ══════════════════════════════════════════════════════════════════════════════

def get_trace(A):
    trace_A = torch.diagonal(A, offset=0, dim1=-1, dim2=-2).sum(-1)
    return trace_A


# ══════════════════════════════════════════════════════════════════════════════
# SUM RATE — GPU-SAFE
# ══════════════════════════════════════════════════════════════════════════════

def get_sum_rate(H, F, W, Pt):
    """
    GPU-safe get_sum_rate.
    Original utility.py uses:
        rate = torch.zeros(len(H[0]), )    ← CPU tensor
        rate + sigma2                       ← Python float
    Both cause device mismatch on GPU. Fixed here.
    """
    device     = H.device
    real_dtype = _get_real_dtype_like(H)

    # sigma2 moved to device — root cause of original error
    sigma2_t = torch.as_tensor(sigma2, device=device, dtype=real_dtype)

    F_n, W_n = normalize(F, W, H, Pt)

    power_high_threshold = Pt + 1e-3
    sum_power = sum(torch.linalg.matrix_norm(F_n @ W_n, ord='fro') ** 2) / K
    if torch.any(sum_power > power_high_threshold):
        sys.stderr.write('Error: power constraint violated\n')

    F_H = torch.transpose(F_n, 2, 3).conj()
    W_H = torch.transpose(W_n, 2, 3).conj()
    V   = W_n @ W_H

    # Created on device directly
    rate = torch.zeros(len(H[0]), device=device, dtype=real_dtype)

    for m in range(M):
        W_m             = W_n.clone()
        W_m[:, :, :, m] = 0.0
        V_m             = W_m @ torch.transpose(W_m, 2, 3).conj()

        h_mk0  = torch.unsqueeze(H[:, :, m, :], dim=2)
        h_mk   = torch.transpose(h_mk0, 2, 3)
        h_mk_H = torch.transpose(h_mk, 2, 3).conj()
        Htilde = h_mk @ h_mk_H

        trace_1 = get_trace(F_n @ V   @ F_H @ Htilde)
        trace_2 = get_trace(F_n @ V_m @ F_H @ Htilde)

        rate = rate + (
            torch.log2(trace_1 + sigma2_t)
            - torch.log2(trace_2 + sigma2_t)
        ).real

    return torch.mean(rate)


# ══════════════════════════════════════════════════════════════════════════════
# BEAM ERROR
# ══════════════════════════════════════════════════════════════════════════════

def get_beam_error(H, F, W, R, Pt):
    F, W = normalize(F, W, H, Pt)
    X   = F @ W
    X_H = torch.transpose(X, 2, 3).conj()
    if normalize_tau == 1:
        error = (torch.linalg.matrix_norm(X @ X_H - R, ord='fro') ** 2
                 / torch.linalg.matrix_norm(R, ord='fro') ** 2)
    else:
        error = torch.linalg.matrix_norm(X @ X_H - R, ord='fro') ** 2
    return torch.mean(error)


# ══════════════════════════════════════════════════════════════════════════════
# TASK LOSS
# ══════════════════════════════════════════════════════════════════════════════

def get_sum_loss(F, W, H, R, Pt, B):
    """
    GPU-safe combined task loss:
        L = -(sum_rate - OMEGA * beam_error)
    """
    device  = F.device
    H       = H.to(device)
    R       = R.to(device)
    W       = W.to(device)
    omega_t = torch.as_tensor(OMEGA, device=device, dtype=torch.float32)

    sum_rate  = get_sum_rate(H, F, W, Pt)
    sum_error = get_beam_error(H, F, W, R, Pt)

    return -(sum_rate - omega_t * sum_error)


# ══════════════════════════════════════════════════════════════════════════════
# MSE
# ══════════════════════════════════════════════════════════════════════════════

def get_MSE(F, W, at, R, Pt):
    X   = F @ W
    X_H = torch.transpose(X, 2, 3).conj()
    at_H = torch.transpose(at, 2, 3).conj()
    beampattern  = torch.real(
        torch.diagonal(at_H @ X @ X_H @ at, offset=0, dim1=-1, dim2=-2)) / Pt
    beam_mean    = torch.mean(beampattern, 0)
    beam_bm      = torch.real(
        torch.diagonal(at_H @ R @ at, offset=0, dim1=-1, dim2=-2)) / Pt
    beam_bm_mean = torch.mean(beam_bm, 0)
    MSE          = (torch.abs(beam_bm_mean - beam_mean)) ** 2
    MSE_mean     = 10 * torch.log10(torch.mean(torch.mean(MSE, 1)))
    return MSE_mean


# ══════════════════════════════════════════════════════════════════════════════
# BEAMPATTERN
# ══════════════════════════════════════════════════════════════════════════════

def get_beampattern(F, W, at, Pt):
    Q    = F @ W
    at_H = torch.transpose(at, 2, 3).conj()
    Q_H  = torch.transpose(Q, 2, 3).conj()
    B    = at_H @ Q @ Q_H @ at
    Bdiag = torch.diagonal(B, offset=0, dim1=-1, dim2=-2) / Pt
    Bmean = torch.real(torch.mean(torch.mean(Bdiag, 1), 0))
    return Bmean.detach().cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# PARTIAL CONNECTION MASK
# ══════════════════════════════════════════════════════════════════════════════

def generage_partial_connection_mask(N, M, device=None, dtype=torch.cfloat):
    mask = torch.zeros((N, M), dtype=dtype, device=device)
    antennas_per_rf = N // M
    for rf in range(M):
        start_idx = rf * antennas_per_rf
        end_idx   = start_idx + antennas_per_rf
        mask[start_idx:end_idx, rf] = 1.0 + 0j
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# G MATRIX HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_mat_G(H, fre_indx, snr_dB):
    device     = H.device
    real_dtype = _get_real_dtype_like(H)
    G          = _randn_complex((len(H[0]), Nt, Nrf), device, real_dtype)
    Htmp       = torch.transpose(H[fre_indx, :, :, :], 1, 2)
    G[:, :, :M] = Htmp

    R, at0, theta, ideal_beam = get_radar_data(snr_dB, H)
    at_batch     = at0[:, :batch_size, :, :].to(device=device)
    theta_degree = np.around(theta[0, :] * 180 / np.pi)
    for t in range(Nrf - M):
        angle_index = np.where(theta_degree == theta_desire[t])
        at_tmp      = at_batch[0, :, :, angle_index]
        at          = at_tmp[:, :, 0, 0]
        G[:, :, M + t] = at

    G = torch.transpose(G, 1, 2)
    return G


def get_mat_G_SVD(H, fre_indx, snr_dB):
    device     = H.device
    real_dtype = _get_real_dtype_like(H)
    G          = _randn_complex((len(H[0]), Nt, Nrf), device, real_dtype)
    U, S, V_H  = torch.linalg.svd(H)
    V          = V_H
    G[:, :, :M] = V[:, :, :, :M]

    R, at0, theta, ideal_beam = get_radar_data(snr_dB, H)
    at_batch     = at0[:, :batch_size, :, :].to(device=device)
    theta_degree = np.around(theta[0, :] * 180 / np.pi)
    for t in range(Nrf - M):
        angle_index = np.where(theta_degree == theta_desire[t])
        at_tmp      = at_batch[0, :, :, angle_index]
        at          = at_tmp[:, :, 0, 0]
        G[:, :, M + t] = at

    G = torch.transpose(G, 1, 2)
    return G


# ══════════════════════════════════════════════════════════════════════════════
# INITIALIZE — GPU-SAFE
# ══════════════════════════════════════════════════════════════════════════════

def initialize(H, R, Pt, normalization, pc=False):
    """
    GPU-safe initialize. All tensors created on H.device.
    Original utility.py created F, W, rate_init, beam_error_init on CPU.
    """
    device       = H.device
    complex_dtype = H.dtype
    real_dtype   = _get_real_dtype_like(H)

    if init_scheme == 'conv':
        F = _randn_complex((len(H[0]), Nt, Nrf), device, real_dtype)
        F = F / torch.abs(F)
        F = torch.cat(((F[None, :, :, :],) * K), 0)
        W = torch.linalg.pinv(H @ F)

    elif init_scheme == 'prop':
        if Nrf == M:
            F = H[K // 2, :, :, :] / torch.abs(H[K // 2, :, :, :])
            F = torch.transpose(F, 1, 2)
            W = _randn_complex((K, len(H[0]), Nrf, M), device, real_dtype)
            for k in range(K):
                Hk       = H[k]
                Hp       = Hk.conj()
                Xzf      = torch.linalg.pinv(Hp)
                Wtmp     = torch.linalg.pinv(F) @ Xzf
                Wtmp_norm = torch.linalg.matrix_norm(
                    Wtmp, ord='fro').reshape(len(H[0]), 1, 1)
                W[k] = Wtmp / Wtmp_norm
            F = torch.cat(((F[None, :, :, :],) * K), 0)
        elif Nrf > M:
            G = get_mat_G(H, K // 2, snr_dB)
            F = G / torch.abs(G)
            F = torch.transpose(F, 1, 2)
            W = _randn_complex((K, len(H[0]), Nrf, M), device, real_dtype)
            for k in range(K):
                Hk        = H[k]
                Hp        = Hk.conj()
                Xzf       = torch.linalg.pinv(Hp)
                Fpinv     = torch.linalg.pinv(F)
                Wtmp      = torch.bmm(Fpinv, Xzf)
                Wtmp_norm = torch.linalg.matrix_norm(
                    Wtmp, ord='fro').reshape(len(H[0]), 1, 1)
                W[k, :, :, :] = Wtmp / Wtmp_norm
            F = torch.cat(((F[None, :, :, :],) * K), 0)
        else:
            sys.stderr.write('Error: Wrong RF chain configuration....\n')
        mask = generage_partial_connection_mask(
            Nt, Nrf, device=device, dtype=complex_dtype)
        F = F * mask if pc else F

    elif init_scheme == 'svd':
        U, S, V_H = torch.linalg.svd(H)
        V  = V_H
        F  = V[:, :, :, :Nrf]
        F  = F / torch.abs(F)
        Hp = H.conj()
        Q  = torch.linalg.pinv(Hp)
        FQ = torch.linalg.pinv(F) @ Q
        W  = FQ / (torch.linalg.matrix_norm(
            FQ, ord='fro').reshape(len(H[0]), 1, 1))

    else:
        R2, at0, theta, ideal_beam = get_radar_data(snr_dB, H)
        at           = at0[:, :batch_size, :, :]
        angles_theta = np.around(theta[0, :] * 180 / np.pi)
        idx_snr      = np.where(angles_theta == 0)
        at_tmp       = at[0, 0, :, idx_snr]
        at1          = at_tmp[:, 0, 0]
        F  = H / torch.abs(H)
        F  = torch.transpose(F, 2, 3)
        F[:, :, :, 0] = at1
        Hp = H.conj()
        Q  = torch.linalg.pinv(Hp)
        FQ = torch.linalg.pinv(F) @ Q
        W  = FQ / (torch.linalg.matrix_norm(
            FQ, ord='fro').reshape(len(H[0]), 1, 1))

    if normalization == 1:
        F, W = normalize(F, W, H, Pt)
    else:
        norm2_FW = sum(torch.linalg.matrix_norm(F @ W, ord='fro') ** 2)
        W = (torch.sqrt(Pt / norm2_FW.reshape(len(H[0]), 1, 1))) * W

    # Created on device — fixes original CPU tensor issue
    rate_init       = torch.zeros(1, len(H[0]), device=device, dtype=real_dtype)
    beam_error_init = torch.zeros(1, len(H[0]), device=device, dtype=real_dtype)
    rate_init[0, :]       = get_sum_rate(H, F, W, Pt)
    beam_error_init[0, :] = get_beam_error(H, F, W, R, Pt)

    return rate_init, beam_error_init, F, W


def safe_initialize(H, R, Pt, initial_normalization, device):
    """
    Calls initialize with inputs on device.
    Since initialize is now fully GPU-safe, this just ensures
    H and R are on device before calling.
    """
    return initialize(
        H.to(device), R.to(device), Pt, initial_normalization)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def array_response(N, phi, theta):
    a = np.zeros([N, 1], dtype='complex_')
    for n in range(N):
        a[n] = (1 / np.sqrt(N)) * np.exp(1j * np.pi * (n * np.sin(phi)))
    return a


def gen_channel(train_batch_size):
    batch_H = np.zeros([K, train_batch_size, M, Nt], dtype='complex64')
    for k in range(K):
        for ii in range(train_batch_size):
            AoD = np.zeros([2, Ncluster * Nray], dtype='complex64')
            AoA = np.zeros([2, Ncluster * Nray], dtype='complex64')
            for cc in range(Ncluster):
                AoD_m = np.random.uniform(0, 2 * np.pi, 2)
                AoA_m = np.random.uniform(0, 2 * np.pi, 2)
                AoD[0, cc*Nray:(cc+1)*Nray] = np.random.laplace(
                    AoD_m[0], angle_sigma, Nray)
                AoD[1, cc*Nray:(cc+1)*Nray] = np.random.laplace(
                    AoD_m[1], angle_sigma, Nray)
                AoA[0, cc*Nray:(cc+1)*Nray] = np.random.laplace(
                    AoA_m[0], angle_sigma, Nray)
                AoA[1, cc*Nray:(cc+1)*Nray] = np.random.laplace(
                    AoA_m[1], angle_sigma, Nray)
            alpha = np.sqrt(sigma / 2) * (
                np.random.normal(0, 1, Ncluster * Nray)
                + 1j * np.random.normal(0, 1, Ncluster * Nray))
            H = np.zeros([M, Nt], dtype='complex_')
            for j in range(Ncluster * Nray):
                at = array_response(Nt, AoD[0, j], AoD[1, j])
                ar = array_response(M,  AoA[0, j], AoA[1, j])
                H  = H + alpha[j] * ar * at.conj().T
            H = gamma * H
            batch_H[k, ii, :, :] = H
    return batch_H


def save_data(data_train, data_test):
    with h5py.File(data_path_train, 'w') as hf:
        hf.create_dataset('train_set', data=data_train)
    with h5py.File(data_path_test, 'w') as hf:
        hf.create_dataset('test_set', data=data_test)


def load_data_matlab():
    data_train = scipy.io.loadmat(data_path_train)
    data_test  = scipy.io.loadmat(data_path_test)
    return data_train['H_train'], data_test['H_test']


def load_data():
    with h5py.File(data_path_train, 'r') as hf:
        key  = list(hf.keys())[0]
        data_train_array = hf[key][()]
    with h5py.File(data_path_test, 'r') as hf:
        key  = list(hf.keys())[0]
        data_test_array  = hf[key][()]
    return data_train_array, data_test_array


def get_data_tensor(data_source):
    if data_source == 'python':
        data_train_array, data_test_array = load_data()
    else:
        data_train_array, data_test_array = load_data_matlab()
    return (torch.from_numpy(data_train_array),
            torch.from_numpy(data_test_array))


def get_radar_data(snr_dB, H):
    radar_data_file_name = directory_data + 'radar_data.mat'
    radar_data = scipy.io.loadmat(radar_data_file_name)
    idx_snr    = np.where(snr_dB_list == snr_dB)

    if K == 1:
        R0_4D  = radar_data['J']
        R0_2D  = np.squeeze(R0_4D[:, :, 0, idx_snr])
        R_array = np.tile(R0_2D, [1, len(H[0]), 1, 1])
        at_2D   = radar_data['a']
        at0     = np.expand_dims(at_2D, axis=0)
        at_array1 = np.tile(at0, (train_size, 1, 1, 1))
        at_array  = np.transpose(at_array1, (1, 0, 2, 3))
    else:
        R0_4D   = radar_data['J']
        R0_2D   = np.squeeze(R0_4D[:, :, :, idx_snr])
        R_array0 = np.transpose(R0_2D, (2, 0, 1))
        R_array1 = np.tile(R_array0, [len(H[0]), 1, 1, 1])
        R_array  = np.transpose(R_array1, (1, 0, 2, 3))
        at_2D    = radar_data['a']
        at0      = np.transpose(at_2D, (2, 0, 1))
        at_array1 = np.tile(at0, (train_size, 1, 1, 1))
        at_array  = np.transpose(at_array1, (1, 0, 2, 3))

    R          = torch.from_numpy(R_array)
    at         = torch.from_numpy(at_array)
    theta      = radar_data['theta']
    ideal_beam = radar_data['Pd_theta']
    return R, at, theta, ideal_beam[0, :]


# ══════════════════════════════════════════════════════════════════════════════
# RKD LOSSES
# ══════════════════════════════════════════════════════════════════════════════

def rkd_distance_loss(teacher, student):
    def _to_real(x):
        if torch.is_complex(x):
            return torch.view_as_real(x).flatten(-2)
        return x
    t = _to_real(teacher.detach())
    s = _to_real(student)
    with torch.no_grad():
        t_dist = torch.cdist(t, t, p=2)
        t_dist = t_dist / t_dist.mean().clamp(min=1e-12)
    s_dist = torch.cdist(s, s, p=2)
    s_dist = s_dist / s_dist.mean().clamp(min=1e-12)
    return nn.functional.smooth_l1_loss(s_dist, t_dist)


def rkd_angle_loss(teacher, student):
    def _to_real(x):
        if torch.is_complex(x):
            return torch.view_as_real(x).flatten(-2)
        return x
    t = _to_real(teacher.detach())
    s = _to_real(student)
    with torch.no_grad():
        t_e   = t.unsqueeze(0) - t.unsqueeze(1)
        t_e   = nn.functional.normalize(t_e, p=2, dim=-1)
        t_cos = torch.bmm(t_e, t_e.permute(0, 2, 1))
    s_e   = s.unsqueeze(0) - s.unsqueeze(1)
    s_e   = nn.functional.normalize(s_e, p=2, dim=-1)
    s_cos = torch.bmm(s_e, s_e.permute(0, 2, 1))
    return nn.functional.smooth_l1_loss(s_cos, t_cos)