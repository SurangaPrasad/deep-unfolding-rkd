import sys
sys.path.insert(0, "/home/swengapp23/OneDrive/PHD Thesis/deep-unfolding-rkd")
import torch
from utility import get_data_tensor, get_radar_data, initialize
from PGA_models import PGA_Unfold_J20, get_grad_F_com
import system_config as cfg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device", device)

H_train, H_test0 = get_data_tensor(cfg.data_source)
print("H_train dtype", H_train.dtype, "shape", H_train.shape)
# replicate the training batch slicing
Hs = torch.transpose(H_train, 0, 1)
cur_bs = 56
i_batch = 0
batch_size = cfg.batch_size
H = torch.transpose(Hs[i_batch:i_batch + batch_size], 0, 1).to(device)
print("batch H", H.shape, H.dtype, "B", H.shape[1])

import numpy as np
snr_dB_train = np.random.permutation(
    np.tile(cfg.snr_dB_list, (batch_size // len(cfg.snr_dB_list)) + 1)
)[:H.shape[1]]
snr_train = torch.tensor(10 ** (snr_dB_train / 10), dtype=torch.float32, device=device)

Rtrain, _, _, _ = get_radar_data(snr_dB_train, H)
Rtrain = Rtrain.to(device)
print("Rtrain", Rtrain.shape, Rtrain.dtype)

model = PGA_Unfold_J20(cfg.step_size_UPGA_J20).to(device)
try:
    _, _, F, W = model.execute_PGA(H, Rtrain, snr_train, cfg.n_iter_outer, cfg.n_iter_inner_J20)
    print("execute_PGA OK", F.shape, W.shape)
except Exception as e:
    print("execute_PGA FAIL:", type(e).__name__, e)
    # inspect after initialize
    ti, tt, F0, W0 = initialize(H, Rtrain, snr_train, 0)
    print("  init F", F0.shape, F0.dtype, "W", W0.shape, W0.dtype)
    try:
        get_grad_F_com(H, F0, W0)
        print("  get_grad_F_com at iter0 OK")
    except Exception as e2:
        print("  get_grad_F_com at iter0 FAIL:", e2)