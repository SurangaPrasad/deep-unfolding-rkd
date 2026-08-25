# Project Logic — Deep Unfolding with Knowledge Distillation for Hybrid Beamforming in JCAS

This project implements **teacher-guided initialization (TGI)** and **layer-wise relational
distillation (LRD)** for a **joint communications and sensing (JCAS)** hybrid beamforming
system, following the paper
*"Knowledge Distillation Aided Deep Unfolding for Hybrid Beamforming in JCAS Systems"*.

The core idea: a large, accurate **teacher** model (UPGA, J=20, I=120) is trained first.
Its knowledge is then transferred to a much smaller **student** model (J=10, I=60) via
(1) **TGI** — compressing the teacher's learned step sizes to initialize the student, and
(2) **LRD** — matching the student's intermediate beamformer iterates to the teacher's
layer-by-layer. The student achieves ~75% lower complexity with near-teacher performance.

---

## 1. Problem Setup

A base station with `Nt` transmit antennas and `Nrf` RF chains serves `M` users while
simultaneously sensing `n_target` targets. The hybrid precoder is `F @ W` where:

- `F` — analog (RF) precoder, constant-modulus (unit-modulus entries), shape `(K, B, Nt, Nrf)`
- `W` — digital (baseband) precoder, shape `(K, B, Nrf, M)`
- `H` — communication channel, shape `(K, B, M, Nt)`
- `R` — radar covariance matrix (sensing target), shape `(K, B, Nt, Nt)`
- `Pt` — transmit power (per-sample SNR), shape `(B,)`

**Objective** (maximized): sum-rate minus a weighted beam-pattern error:

```
loss = -( sum_rate - OMEGA * beam_error )
```

- `sum_rate` — MU-MISO sum rate (communication quality)
- `beam_error` — Frobenius distance between the achieved beam pattern `F W (F W)^H`
  and the desired radar covariance `R`
- `OMEGA` — trade-off weight between communication and sensing

---

## 2. Core Algorithm: Projected Gradient Ascent (PGA)

The underlying optimization is **PGA**: iteratively ascend the objective by taking
gradient steps on `F` and `W`, then projecting back onto the feasible set
(unit-modulus for `F`, power constraint for `W`).

Each **outer iteration** `i` (of `I`) does:

1. **Analog update** — `J` inner steps on `F`:
   ```
   F += mu * ( grad_F_com * W_F_COM - grad_F_rad * W_F_RAD )
   ```
   then project to unit modulus.
2. **Digital update** — one gradient step on `W` per frequency band.
3. **Projection** — normalize `(F, W)` to satisfy the transmit-power constraint.

### Deep Unfolding

Instead of hand-tuning the step sizes, **deep unfolding** makes them **learnable
parameters**. Each model is an `nn.Module` whose `step_size` tensor
`[J, I, K+1]` is trained end-to-end by backpropagating through the PGA iterations.
The `J` inner steps and `I` outer steps become "layers" of the network.

---

## 3. Models (`PGA_models.py`)

| Class | Role | Inner `J` | Outer `I` | Complexity |
|-------|------|-----------|-----------|------------|
| `PGA_Conv` | Conventional PGA baseline | 1 | `I` | — |
| `PGA_Unfold_J10` | **Student** | 10 | 60 | ~4× cheaper |
| `PGA_Unfold_J20` | **Teacher** | 20 | 120 | reference |

Key methods:

- `execute_PGA(H, R, Pt, n_iter_outer, n_iter_inner)` — run the unfolded PGA,
  returns `(rates, taus, F, W)`.
- `execute_PGA_with_windows(H, R, Pt, n_iter_outer, n_iter_inner, L, le)` —
  like `execute_PGA` but also collects the `(F, W)` iterates inside the **early**
  and **late** windows needed for LRD. Returns
  `(rates, taus, F, W, FW_early, FW_late)`.
- `analog_block_pga(...)` — the inner `J`-step analog update, wrapped in
  `torch.utils.checkpoint` to save memory during backprop.
- `tgi_compress_step_sizes(teacher_step, J_S, I_S, K+1)` — **TGI** helper
  (see §5).

---

## 4. Training Pipeline (`main_train.py`)

The pipeline runs in two stages.

### Stage 1 — Train the Teacher (`train_teacher`)

- Model: `PGA_Unfold_J20` (J=20, I=120).
- Loss: **task loss only** (`get_sum_loss`).
- Optimizer: Adam, `lr = 1e-3`, `30` epochs, batch size `10`.
- Saves to `model_file_name_teacher`.

### Stage 2 — Train the Student (`train_student_lrd`)

1. **TGI** (optional): initialize the student's step sizes by compressing the
   teacher's trained step sizes (`tgi_init_student`).
2. **Teacher forward** (no grad): run the frozen teacher with windows to get
   `FW_t_early`, `FW_t_late` and its final task loss.
3. **Student forward**: run the student with windows to get `FW_s_early`,
   `FW_s_late`.
4. **Loss**:
   ```
   total = L_task + L_early + lambda_late * L_late
   ```
   where `L_early`/`L_late` are the LRD window losses (§6).
5. Saves to `UPGA_J10_student_TGI{1}_LRD{1}.pth`.

### Running the training loop

The whole pipeline is wrapped in `run_training()` in `main_train.py`. It is invoked
automatically when the script is run directly:

```
python main_train.py
```

- `run_training(use_tgi=True, use_lrd=True)` — the main entry point.
  1. Trains the teacher via `train_teacher()` (if `run_UPGA_J20 == 1`), or loads the
     saved teacher checkpoint otherwise.
  2. Trains the student via `train_student_lrd()` (if `run_UPGA_J10 == 1`).
  3. Calls `plot_training_losses()` and saves the curves.
  4. Returns the trained model(s).
- `train_teacher()` → returns `(model, epoch_losses)`.
- `train_student_lrd()` → returns `(model, epoch_losses)`.

### Plotting the training curves

`plot_training_losses(teacher_losses, student_losses)` in `main_train.py`:

- Plots the per-epoch **average loss** for the teacher vs. the student on one figure.
- x-axis: epoch index; y-axis: average loss.
- Saves `sim_results/<config>/training_loss_<system_config>.png` (and `.eps`).

---

## 5. Teacher-Guided Initialization (TGI)

The teacher's trained step sizes `[J_T, I_T, K+1]` are **compressed** to initialize
the student's `[J_S, I_S, K+1]` by averaging groups of consecutive step sizes:

- **Inner** reduction factor `r = J_T / J_S`: average each group of `r` teacher
  inner step sizes (paper Eq. 10).
- **Outer** reduction factor `r' = I_T / I_S`: average each group of `r'` teacher
  outer step sizes (paper Eq. 11).

Implemented in `tgi_compress_step_sizes()`.

---

## 6. Layer-wise Relational Distillation (LRD)

LRD transfers relational knowledge **layer by layer** by matching the student's
intermediate beamformer iterates to the teacher's, over two windows of outer layers:

- **Early window** — teacher iters `[le, le+L-1]` (start `le=20`), student iters `[0, L-1]`.
- **Late window** — last `L` iters of both (`L=15`).

For each outer layer in a window, the **effective beamformer** is
`phi = vecR(F @ W)` (real/imag stacked, shape `(B, 2*Nt*M)`). Two relational
potentials are matched (Huber loss):

- **Distance** (`lrd_distance_loss`, Eq. 12): pairwise L2 distances between samples.
- **Angle** (`lrd_angle_loss`, Eq. 13): pairwise cosine angles between samples.

The window loss (`lrd_window_loss`) sums these over the window:

```
L_window = sum over layers [ lambda_d * dist_loss + lambda_a * angle_loss ]
```

### Hyperparameters (paper Section IV)

| Param | Value | Meaning |
|---|---|---|
| `L` | 15 | window length |
| `le` | 20 | teacher early-window start |
| `lambda_d` | 25.0 | distance-loss weight |
| `lambda_a` | 50.0 | angle-loss weight |
| `lambda_late` | 0.8 | late-window weight |
| `lambda_log` | 0.0001 | (reserved) |
| `batch_size` | 10 | training batch |
| `lr` | 1e-3 | learning rate |
| `n_epoch` | 30 | epochs |

---

## 7. Loss Functions (`utility.py`)

- `get_sum_loss(F, W, H, R, Pt, batch_size)` — task loss
  `-(sum_rate - OMEGA * radar_error)`.
- `get_sum_rate(H, F, W, Pt)` — MU-MISO sum rate.
- `get_beam_error(H, F, W, R, Pt)` — radar beam-pattern error `tau`.
- `get_MSE(F, W, at, R, Pt)` — beampattern MSE (dB) vs. benchmark.
- `rkd_distance_loss` / `rkd_angle_loss` — classic RKD relational losses.
- `lrd_distance_loss` / `lrd_angle_loss` / `lrd_window_loss` — LRD losses (§6).
- `normalize` / `normalize_power` — power-constraint projection.
- `initialize` / `initialize_schemes` — `F`, `W` initialization (ZF / proposed / SVD).

---

## 8. Evaluation & Plotting

### Training-loss plot (`main_train.py`)

- `plot_training_losses(teacher_losses, student_losses)` — plots both training-loss
  curves vs. epoch and saves `training_loss_<system_config>.png/.eps`.

### Evaluation (`algorithms.py`)

- **`algorithms.py`** — thin wrappers that run a trained model over the test set and
  return `(rate_avr, tau_avr, MSE_avr)` for each scheme
  (`execute_conv_PGA`, `execute_UPGA_J1`, `execute_UPGA_J10`, `execute_UPGA_J20`).

### Plotting scripts

- **`main_SNR.py`** — sweeps SNR and plots sum rate / beam error / MSE vs. SNR,
  comparing teacher, student, and baselines. Figures saved to
  `<directory_result>/rate_vs_SNR_<Nt>_<OMEGA>.png`, `MSE_vs_SNR_...`, and
  `tradeoff_vs_SNR_...`.
- **`main_iter.py`** — plots convergence over PGA iterations (rate / beam error
  vs. iteration count), plus the beampattern. Figures saved to
  `<directory_result>/rate_vs_iter_<Nt>_<OMEGA>.png`, `obj_vs_iter_...`,
  `beampattern_error_vs_iter_...`, and `tradeoff_vs_iter_...`.

---

## 9. Configuration (`system_config.py`)

Central config: system size (`Nt`, `M`, `Nrf`, `K`), SNR settings, training
hyperparameters, step-size tensors, file paths, and figure labels. Key flags:

- `run_UPGA_J20` — train the teacher (1) or load it (0).
- `run_UPGA_J10` — train the student with TGI+LRD.
- `I_student = 60` — student outer layers.
- `model_file_name_teacher` / `model_file_name_student` — saved model paths.

---

## 10. Data Flow Summary

```
system_config.py  (hyperparameters, paths)
        │
        ▼
utility.py  ──►  get_radar_data, get_sum_rate, get_beam_error, get_MSE,
                 initialize, normalize, LRD/RKD losses
        │
        ▼
PGA_models.py  ──►  PGA_Unfold_J10 (student), PGA_Unfold_J20 (teacher),
                 execute_PGA(_with_windows), tgi_compress_step_sizes, get_sum_loss
        │
        ▼
main_train.py  ──►  train_teacher() → tgi_init_student() → train_student_lrd()
        │
        ▼
algorithms.py  ──►  execute_* wrappers (rate, tau, MSE)
        │
        ▼
main_SNR.py / main_iter.py  ──►  plots
```