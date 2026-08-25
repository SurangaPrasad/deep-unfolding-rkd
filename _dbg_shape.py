import sys
sys.path.insert(0, "/home/swengapp23/OneDrive/PHD Thesis/deep-unfolding-rkd")
import torch
from utility import initialize, get_radar_data, get_sum_rate, _pt_vec
from PGA_models import get_grad_F_com, get_grad_F_rad

device = torch.device('cpu')
K, B, M, Nt, Nrf = 1, 56, 4, 64, 4

torch.manual_seed(0)
H = (torch.randn(K, B, M, Nt) + 1j * torch.randn(K, B, M, Nt)).to(torch.complex64)
Pt = torch.full((B,), 15.0, dtype=torch.float32)

# Build a valid R (K,B,Nt,Nt)
R = (torch.randn(K, B, Nt, Nt) + 1j * torch.randn(K, B, Nt, Nt)).to(torch.complex64)
R = R @ R.conj().transpose(-1, -2)

import system_config as cfg
print("init_scheme", cfg.init_scheme, "Nrf==M?", cfg.Nrf == cfg.M)
rate_init, tau_init, F, W = initialize(H, R, Pt, 0)
print("F", F.shape, F.dtype, "W", W.shape, W.dtype)

try:
    g = get_grad_F_com(H, F, W)
    print("get_grad_F_com OK", g.shape, g.dtype)
except Exception as e:
    print("get_grad_F_com FAIL:", e)
    print("  H", H.shape, "F", F.shape, "W", W.shape)