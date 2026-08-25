from utility import *
from PGA_models import *

# load at for MSE-based models
_, H_test_tmp = get_data_tensor(data_source)
H_test_tmp1 = H_test_tmp[:, :test_size, :, :]
R_tmp, at0, _, ideal_beam = get_radar_data(snr_dB, H_test_tmp1)
at = at0[:, : test_size, :, :]


def execute_conv_PGA(model_conv_PGA, H_test, R, Pt):
    rate, tau, F, W = model_conv_PGA.execute_PGA(H_test, R, Pt, n_iter_outer)
    rate_avr = [r.detach().cpu().numpy() for r in (sum(rate) / len(H_test[0]))][-1]
    tau_avr = [r.detach().cpu().numpy() for r in (sum(tau) / len(H_test[0]))][-1]
    MSE_avr = get_MSE(F, W, at, R, Pt).detach().item()
    return rate_avr, tau_avr, MSE_avr


def execute_UPGA_J1(model_UPGA_J1, H_test, R, Pt):
    rate, tau, F, W = model_UPGA_J1.execute_PGA(H_test, R, Pt, n_iter_outer)
    rate_avr = [r.detach().cpu().numpy() for r in (sum(rate) / len(H_test[0]))][-1]
    tau_avr = [r.detach().cpu().numpy() for r in (sum(tau) / len(H_test[0]))][-1]
    MSE_avr = get_MSE(F, W, at, R, Pt).detach().item()
    return rate_avr, tau_avr, MSE_avr


def execute_UPGA_J10(model_UPGA_J10, H_test, R, Pt):
    rate, tau, F, W = model_UPGA_J10.execute_PGA(H_test, R, Pt, n_iter_outer, n_iter_inner_J10)
    rate_avr = [r.detach().cpu().numpy() for r in (sum(rate) / len(H_test[0]))][-1]
    tau_avr = [r.detach().cpu().numpy() for r in (sum(tau) / len(H_test[0]))][-1]
    MSE_avr = get_MSE(F, W, at, R, Pt).detach().item()
    return rate_avr, tau_avr, MSE_avr


def execute_student(model_student, H_test, R, Pt):
    """Run the distilled student (UPGA J=10, I=60, TGI+LRD) over the test set."""
    rate, tau, F, W = model_student.execute_PGA(H_test, R, Pt, I_student, n_iter_inner_J10)
    rate_avr = [r.detach().cpu().numpy() for r in (sum(rate) / len(H_test[0]))][-1]
    tau_avr = [r.detach().cpu().numpy() for r in (sum(tau) / len(H_test[0]))][-1]
    MSE_avr = get_MSE(F, W, at, R, Pt).detach().item()
    return rate_avr, tau_avr, MSE_avr


def execute_UPGA_J20(model_UPGA_J20, H_test, R, Pt):
    rate, tau, F, W = model_UPGA_J20.execute_PGA(H_test, R, Pt, n_iter_outer, n_iter_inner_J20)
    rate_avr = [r.detach().cpu().numpy() for r in (sum(rate) / len(H_test[0]))][-1]
    tau_avr = [r.detach().cpu().numpy() for r in (sum(tau) / len(H_test[0]))][-1]
    MSE_avr = get_MSE(F, W, at, R, Pt).detach().item()
    return rate_avr, tau_avr, MSE_avr