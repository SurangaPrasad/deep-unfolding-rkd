"""
Interpretability visualisations for distilled deep unfolding networks
=====================================================================
Four complementary views that parallel CNN feature map analysis:

  View 1 — Beampattern strip
           F at each outer iteration shown as a spatial filter.
           Analogous to CNN feature maps at each layer.
           Shows how the array pattern sharpens from random to directed.

  View 2 — Singular value spectrum
           Eigenvalue decomposition of F^H F at each layer.
           Shows how energy concentrates across RF chains.
           Teacher vs student alignment in eigenspace.

  View 3 — Student-teacher F alignment
           Cosine similarity between teacher and student F per layer.
           Directly measures how well transfer learning worked at each depth.

  View 4 — Step size evolution heatmap
           step_size[j, i, k] shown as 2D heatmap.
           Inner axis (j) = row, outer axis (i) = column.
           Reveals what the model learned — analogous to visualising
           learned filter weights in a CNN.

  View 5 — Relational distance matrix
           Pairwise F distances across the batch at early vs late layers.
           Shows the RKD training signal and whether student matches teacher.

Usage
-----
Set model paths and run on a small test batch (B=8 recommended for clarity).
Requires: numpy, matplotlib, torch, your system_config/utility/PGA_models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from system_config import *
from utility import *
from PGA_models import *

# ── Config ────────────────────────────────────────────────────────────────────
TEACHER_PATH  = './model/64TX_4UE_4RF/UPGA_J20.pth'
STUDENT_PATH  = './model/64TX_4UE_4RF/UPGA_J10.pth_J5_I60_LAST5_INNER_RKD_Ko10_tsteep20.pth'

N_INNER_S     = 5
I_S           = 60
N_INNER_T     = 20
I_T           = 120
Nt            = 64   # transmit antennas
Nrf           = 4    # RF chains
B_VIZ         = 8    # batch size for visualisation (small for clarity)

# Angles for beampattern
angles        = np.linspace(-90, 90, 361)
steering_vec  = np.exp(1j * np.pi * np.outer(
    np.arange(Nt), np.sin(np.radians(angles))))   # [Nt, 361]

device = torch.device('cpu')

INNER_START_T = N_INNER_T - N_INNER_S  # = 15

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models...")

teacher_state = torch.load(TEACHER_PATH, map_location=device)
student_state = torch.load(STUDENT_PATH, map_location=device)

# Extract step_size tensors
ss_T = teacher_state.get('step_size',
       next(v for k,v in teacher_state.items() if 'step_size' in k)).detach()
ss_S = student_state.get('step_size',
       next(v for k,v in student_state.items() if 'step_size' in k)).detach()

print(f"Teacher step_size : {list(ss_T.shape)}")
print(f"Student step_size : {list(ss_S.shape)}")

J_T, I_T_ss, _ = ss_T.shape
J_S, I_S_ss, _ = ss_S.shape

# ── Load a small test batch ───────────────────────────────────────────────────
print("Loading test data...")
_, H_test0 = get_data_tensor(data_source)
H_test     = H_test0[:, :B_VIZ, :, :].to(device)
Rtest, _, _, _ = get_radar_data(snr_dB, H_test)

# ── Forward pass collecting F at every outer iteration ───────────────────────

class PGA_Teacher_Viz(PGA_Unfold_J20):
    """Teacher forward storing F at every outer iteration."""
    def forward_collect_all(self, H, R, Pt, n_outer, n_inner):
        _, _, F, W = initialize(H, R, Pt, initial_normalization)
        F_history = []
        for ii in range(n_outer):
            for jj in range(n_inner):
                grad_F_com  = get_grad_F_com(H, F, W)
                grad_F_rad  = get_grad_F_rad(F, W, R)
                delta_F_com = self.step_size[jj][ii][0] * grad_F_com
                delta_F_rad = self.step_size[jj][ii][0] * grad_F_rad
                F = F + delta_F_com * WEIGHT_F_COM - delta_F_rad * WEIGHT_F_RAD
                if sum(torch.abs(F[0,:,0,0])) > 1e3:
                    F = normalize_power(F, W, H, Pt)
            F = F / torch.abs(F)
            W_new = W.clone().detach()
            grad_W_com = get_grad_W_com(H, F, W)
            grad_W_rad = get_grad_W_rad(F, W, R)
            for k in range(K):
                delta_W_com = self.step_size[0][ii][k+1] * grad_W_com[k]
                delta_W_rad = self.step_size[0][ii][k+1] * grad_W_rad[k]
                W_new[k] = W[k].clone().detach() + delta_W_com*WEIGHT_W_COM - delta_W_rad*WEIGHT_W_RAD
            F, W = normalize(F, W_new, H, Pt)
            F_history.append(F.detach().clone())  # [K_u, B, Nt, Nrf]
        return F, W, F_history

class PGA_Student_Viz(torch.nn.Module):
    """Student forward storing F at every outer iteration."""
    def __init__(self, ss):
        super().__init__()
        self.step_size = torch.nn.Parameter(ss.float().clone())

    def forward_collect_all(self, H, R, Pt, n_outer, n_inner):
        _, _, F, W = initialize(H, R, Pt, initial_normalization)
        F_history = []
        for ii in range(n_outer):
            for jj in range(n_inner):
                grad_F_com  = get_grad_F_com(H, F, W)
                grad_F_rad  = get_grad_F_rad(F, W, R)
                delta_F_com = self.step_size[jj][ii][0] * grad_F_com
                delta_F_rad = self.step_size[jj][ii][0] * grad_F_rad
                F = F + delta_F_com * WEIGHT_F_COM - delta_F_rad * WEIGHT_F_RAD
                F = normalize_power(F, W, H, Pt)
            F = F / torch.abs(F)
            W_new = W.clone().detach()
            grad_W_com = get_grad_W_com(H, F, W)
            grad_W_rad = get_grad_W_rad(F, W, R)
            for k in range(K):
                delta_W_com = self.step_size[0][ii][k+1] * grad_W_com[k]
                delta_W_rad = self.step_size[0][ii][k+1] * grad_W_rad[k]
                W_new[k] = W[k].clone().detach() + delta_W_com*WEIGHT_W_COM - delta_W_rad*WEIGHT_W_RAD
            F, W = normalize(F, W_new, H, Pt)
            F_history.append(F.detach().clone())
        return F, W, F_history

print("Running teacher forward pass...")
model_T = PGA_Teacher_Viz(step_size_UPGA_J20).to(device)
model_T.load_state_dict(torch.load(TEACHER_PATH, map_location=device))
model_T.eval()
with torch.no_grad():
    F_T_final, W_T_final, F_T_history = model_T.forward_collect_all(
        H_test, Rtest.to(device), snr, I_T, N_INNER_T)

print("Running student forward pass...")
model_S = PGA_Student_Viz(ss_S).to(device)
model_S.eval()
with torch.no_grad():
    F_S_final, W_S_final, F_S_history = model_S.forward_collect_all(
        H_test, Rtest.to(device), snr, I_S_ss, J_S)

print(f"Teacher: {len(F_T_history)} outer iters collected")
print(f"Student: {len(F_S_history)} outer iters collected")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_beampattern(F_tensor, rf_chain=0, sample=0):
    """
    Compute beampattern for one RF chain of one sample.
    F_tensor : [K_u, B, Nt, Nrf] complex
    Returns  : [361] power in dB
    """
    f_vec = F_tensor[0, sample, :, rf_chain].numpy()   # [Nt] complex
    bp    = np.abs(steering_vec.conj().T @ f_vec) ** 2  # [361]
    bp_db = 10 * np.log10(bp / (bp.max() + 1e-12) + 1e-12)
    return bp_db

def get_singular_values(F_tensor, sample=0):
    """
    Singular values of F[:, sample, :, :] reshaped to [Nt, Nrf*K_u].
    Returns sorted descending.
    """
    F_mat = F_tensor[0, sample, :, :].numpy()   # [Nt, Nrf] complex
    sv    = np.linalg.svd(F_mat, compute_uv=False)
    return sv

def cosine_sim_F(F_A, F_B, sample=0):
    """
    Cosine similarity between F_A and F_B (flattened, real-valued).
    """
    a = F_A[0, sample, :, :].numpy().flatten()
    b = F_B[0, sample, :, :].numpy().flatten()
    # Stack real and imaginary
    a = np.concatenate([a.real, a.imag])
    b = np.concatenate([b.real, b.imag])
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

def pairwise_dist_matrix(F_tensor):
    """
    Pairwise L2 distance across batch samples.
    F_tensor : [K_u, B, Nt, Nrf] → [B, B] distance matrix
    """
    B   = F_tensor.shape[1]
    vecs = []
    for b in range(B):
        f  = F_tensor[0, b, :, :].numpy()
        rv = np.concatenate([f.real.flatten(), f.imag.flatten()])
        vecs.append(rv)
    vecs = np.stack(vecs)   # [B, D]
    diff = vecs[:, None, :] - vecs[None, :, :]  # [B, B, D]
    dist = np.linalg.norm(diff, axis=-1)         # [B, B]
    return dist / (dist.mean() + 1e-12)


# Selected outer iterations to visualise
T_iters_selected = [0, 10, 20, 40, 60, 80, 100, 119]
S_iters_selected = [0, 5, 10, 20, 30, 40, 50, 59]

SAMPLE = 0   # which test sample to visualise
RF     = 0   # which RF chain beampattern to show


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 1 — BEAMPATTERN STRIP
# Analogous to CNN feature map visualisation at each layer.
# Shows how the spatial filter (beamformer) sharpens from random to directed.
# ═══════════════════════════════════════════════════════════════════════════════

print("Plotting View 1: Beampattern strip...")

fig, axes = plt.subplots(2, len(T_iters_selected),
                          figsize=(2.5*len(T_iters_selected), 6))
fig.suptitle(
    'Beampattern evolution across outer iterations\n'
    '(analogous to CNN feature maps at each layer)',
    fontsize=13)

cmap_t = plt.cm.Blues
cmap_s = plt.cm.Reds

for col, (t_ii, s_ii) in enumerate(zip(T_iters_selected, S_iters_selected)):
    # Teacher beampattern
    bp_T = get_beampattern(F_T_history[t_ii], RF, SAMPLE)
    ax_t = axes[0][col]
    ax_t.plot(angles, bp_T, color='#1f77b4', linewidth=1.2)
    ax_t.set_ylim(-40, 2)
    ax_t.set_xlim(-90, 90)
    ax_t.set_title(f'Layer {t_ii}', fontsize=9)
    ax_t.set_xticks([-60, 0, 60])
    ax_t.grid(True, alpha=0.3, linewidth=0.5)
    if col == 0:
        ax_t.set_ylabel('Teacher\nPower (dB)', fontsize=9)
    else:
        ax_t.set_yticklabels([])

    # Student beampattern
    bp_S = get_beampattern(F_S_history[s_ii], RF, SAMPLE)
    ax_s = axes[1][col]
    ax_s.plot(angles, bp_S, color='#d62728', linewidth=1.2)
    ax_s.set_ylim(-40, 2)
    ax_s.set_xlim(-90, 90)
    ax_s.set_title(f'Layer {s_ii}', fontsize=9)
    ax_s.set_xticks([-60, 0, 60])
    ax_s.grid(True, alpha=0.3, linewidth=0.5)
    ax_s.set_xlabel('Angle (°)', fontsize=8)
    if col == 0:
        ax_s.set_ylabel('Student\nPower (dB)', fontsize=9)
    else:
        ax_s.set_yticklabels([])

plt.tight_layout()
plt.savefig('./beampattern_strip.png', dpi=150, bbox_inches='tight')
print("  Saved: beampattern_strip.png")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 2 — SINGULAR VALUE SPECTRUM ACROSS LAYERS
# Eigenspace interpretation: how energy distributes across RF chains.
# Teacher vs student eigenvalue alignment.
# ═══════════════════════════════════════════════════════════════════════════════

print("Plotting View 2: Singular value spectrum...")

# Compute SVs at every outer iteration for teacher and student
sv_T_all = np.array([get_singular_values(F_T_history[ii], SAMPLE)
                     for ii in range(len(F_T_history))])   # [I_T, Nrf]
sv_S_all = np.array([get_singular_values(F_S_history[ii], SAMPLE)
                     for ii in range(len(F_S_history))])   # [I_S, Nrf]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Singular value spectrum of F across outer iterations\n'
             '(eigenspace view — energy distribution across RF chains)',
             fontsize=12)

colors_sv = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # one per RF chain

# Left: teacher SV evolution
ax = axes[0]
for rf in range(Nrf):
    ax.plot(range(len(F_T_history)), sv_T_all[:, rf],
            color=colors_sv[rf], linewidth=1.5, label=f'σ_{rf+1}')
ax.set_xlabel('Outer iteration', fontsize=11)
ax.set_ylabel('Singular value', fontsize=11)
ax.set_title('Teacher — SV evolution', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Middle: student SV evolution
ax = axes[1]
for rf in range(Nrf):
    ax.plot(range(len(F_S_history)), sv_S_all[:, rf],
            color=colors_sv[rf], linewidth=1.5, linestyle='--',
            label=f'σ_{rf+1}')
ax.set_xlabel('Outer iteration', fontsize=11)
ax.set_ylabel('Singular value', fontsize=11)
ax.set_title('Student — SV evolution', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: final SV comparison (bar chart)
ax = axes[2]
x  = np.arange(Nrf)
w  = 0.35
ax.bar(x - w/2, sv_T_all[-1],  w, color='#1f77b4', label='Teacher (final)')
ax.bar(x + w/2, sv_S_all[-1],  w, color='#d62728', label='Student (final)')
ax.set_xticks(x)
ax.set_xticklabels([f'σ_{i+1}' for i in range(Nrf)])
ax.set_ylabel('Singular value', fontsize=11)
ax.set_title('Final layer SV comparison', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Annotate similarity
sv_sim = np.dot(sv_T_all[-1], sv_S_all[-1]) / (
    np.linalg.norm(sv_T_all[-1]) * np.linalg.norm(sv_S_all[-1]))
ax.text(0.05, 0.95, f'Cosine sim: {sv_sim:.3f}',
        transform=ax.transAxes, fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('./singular_value_spectrum.png', dpi=150, bbox_inches='tight')
print("  Saved: singular_value_spectrum.png")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 3 — STUDENT-TEACHER F ALIGNMENT PER LAYER
# Cosine similarity at each outer iteration.
# Directly shows where transfer learning is working — high similarity = good
# alignment. Should be highest in the first K_outer layers (RKD-supervised)
# and may drop in the middle (free development zone).
# ═══════════════════════════════════════════════════════════════════════════════

print("Plotting View 3: Student-teacher alignment...")

# Match student outer iter ii to teacher outer iter ii*2 (uniform subsampling)
n_compare  = min(len(F_S_history), len(F_T_history) // 2)
cos_sims   = []
subspace_angles = []  # principal angle between column spaces

for s_ii in range(n_compare):
    t_ii = s_ii * 2   # teacher iter matching student iter s_ii

    # Cosine similarity
    cs = cosine_sim_F(F_T_history[t_ii], F_S_history[s_ii], SAMPLE)
    cos_sims.append(cs)

    # Subspace angle: smallest principal angle between column spaces of F
    F_t_mat = F_T_history[t_ii][0, SAMPLE, :, :].numpy()   # [Nt, Nrf]
    F_s_mat = F_S_history[s_ii][0, SAMPLE, :, :].numpy()   # [Nt, Nrf]
    # QR decompose to get orthonormal bases
    Q_t, _ = np.linalg.qr(F_t_mat)
    Q_s, _ = np.linalg.qr(F_s_mat)
    # Singular values of Q_t^H Q_s give cosines of principal angles
    sv_pa   = np.linalg.svd(Q_t.conj().T @ Q_s, compute_uv=False)
    sv_pa   = np.clip(sv_pa, -1, 1)
    pa_min  = np.degrees(np.arccos(sv_pa[0]))   # smallest principal angle
    subspace_angles.append(pa_min)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Student-teacher F alignment per outer iteration\n'
             '(high cosine similarity = successful knowledge transfer)',
             fontsize=12)

ax = axes[0]
ax.plot(range(n_compare), cos_sims, color='#2ca02c', linewidth=2, marker='o',
        markersize=3, label='Cosine similarity')
ax.axvline(x=10, color='purple', linestyle='--', alpha=0.7,
           label='K_outer boundary\n(RKD supervision ends)')
ax.axhline(y=np.mean(cos_sims[-10:]), color='gray', linestyle=':',
           alpha=0.7, label=f'Late mean: {np.mean(cos_sims[-10:]):.3f}')
ax.set_xlabel('Student outer iteration', fontsize=11)
ax.set_ylabel('Cosine similarity', fontsize=11)
ax.set_title('F cosine similarity (teacher vs student)', fontsize=11)
ax.set_ylim([-1.05, 1.05])
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.fill_between(range(10), -1.05, 1.05, alpha=0.08, color='purple',
                label='RKD supervised region')

ax = axes[1]
ax.plot(range(n_compare), subspace_angles, color='#ff7f0e', linewidth=2,
        marker='s', markersize=3)
ax.axvline(x=10, color='purple', linestyle='--', alpha=0.7,
           label='K_outer boundary')
ax.set_xlabel('Student outer iteration', fontsize=11)
ax.set_ylabel('Principal angle (degrees)', fontsize=11)
ax.set_title('Subspace distance (0° = identical column spaces)', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.fill_between(range(10), 0, max(subspace_angles)*1.1,
                alpha=0.08, color='purple')

plt.tight_layout()
plt.savefig('./student_teacher_alignment.png', dpi=150, bbox_inches='tight')
print("  Saved: student_teacher_alignment.png")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 4 — STEP SIZE HEATMAP
# Visualises the learned parameters directly.
# Analogous to visualising CNN filter weights.
# ═══════════════════════════════════════════════════════════════════════════════

print("Plotting View 4: Step size heatmap...")

fig, axes = plt.subplots(2, 2, figsize=(16, 9))
fig.suptitle('Learned step-size parameters — step_size[j, i, k]\n'
             '(analogous to CNN filter weights: inner j = filter depth, '
             'outer i = layer index)',
             fontsize=12)

cmaps = ['Blues', 'Reds']
labels = ['F component (k=0)', 'W component (k=1)']

for comp in range(2):
    # Teacher heatmap
    ax = axes[0][comp]
    data_T = ss_T[:, :, comp].numpy()
    im = ax.imshow(data_T, aspect='auto', origin='upper',
                   cmap=cmaps[comp], interpolation='nearest')
    ax.set_title(f'Teacher — {labels[comp]}', fontsize=11)
    ax.set_xlabel('Outer iteration (i)', fontsize=10)
    ax.set_ylabel('Inner step (j)', fontsize=10)
    ax.set_yticks(range(J_T))
    x_ticks = np.linspace(0, I_T_ss-1, 10, dtype=int)
    ax.set_xticks(x_ticks)
    plt.colorbar(im, ax=ax, label='Step size')

    # Annotate max
    jmax, imax = np.unravel_index(data_T.argmax(), data_T.shape)
    ax.plot(imax, jmax, 'r*', markersize=10)
    ax.text(imax+1, jmax, f'{data_T.max():.3f}', color='red', fontsize=7)

    # Student heatmap
    ax = axes[1][comp]
    data_S = ss_S[:, :, comp].numpy()
    im = ax.imshow(data_S, aspect='auto', origin='upper',
                   cmap=cmaps[comp], interpolation='nearest')
    ax.set_title(f'Student — {labels[comp]}', fontsize=11)
    ax.set_xlabel('Outer iteration (i)', fontsize=10)
    ax.set_ylabel('Inner step (j)', fontsize=10)
    ax.set_yticks(range(J_S))
    x_ticks_s = np.linspace(0, I_S_ss-1, 10, dtype=int)
    ax.set_xticks(x_ticks_s)
    plt.colorbar(im, ax=ax, label='Step size')

    jmax_s, imax_s = np.unravel_index(data_S.argmax(), data_S.shape)
    ax.plot(imax_s, jmax_s, 'r*', markersize=10)
    ax.text(imax_s+1, jmax_s, f'{data_S.max():.3f}', color='red', fontsize=7)

plt.tight_layout()
plt.savefig('./step_size_heatmap.png', dpi=150, bbox_inches='tight')
print("  Saved: step_size_heatmap.png")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 5 — RELATIONAL DISTANCE MATRIX
# Pairwise F distances across the batch — the RKD training signal itself.
# Early layer: large distances (F still random), Late layer: structured.
# Student should match teacher's structure if RKD worked.
# ═══════════════════════════════════════════════════════════════════════════════

print("Plotting View 5: Relational distance matrix...")

selected_pairs = [
    (0,  0,  'Very early (iter 0)'),
    (10, 5,  'RKD boundary'),
    (40, 20, 'Mid trajectory'),
    (119, 59, 'Final layer'),
]

fig, axes = plt.subplots(3, len(selected_pairs),
                          figsize=(4*len(selected_pairs), 9))
fig.suptitle('Relational distance matrix F×F across batch\n'
             '(RKD training signal — student should match teacher structure)',
             fontsize=12)

for col, (t_ii, s_ii, label) in enumerate(selected_pairs):
    dist_T = pairwise_dist_matrix(F_T_history[t_ii])
    dist_S = pairwise_dist_matrix(F_S_history[s_ii])
    dist_diff = np.abs(dist_T - dist_S)

    vmax = max(dist_T.max(), dist_S.max())

    ax_t = axes[0][col]
    im = ax_t.imshow(dist_T, vmin=0, vmax=vmax, cmap='viridis')
    ax_t.set_title(f'Teacher\n{label}', fontsize=9)
    ax_t.set_xlabel('Sample', fontsize=8)
    if col == 0:
        ax_t.set_ylabel('Teacher\nSample', fontsize=9)
    plt.colorbar(im, ax=ax_t, fraction=0.046)

    ax_s = axes[1][col]
    im = ax_s.imshow(dist_S, vmin=0, vmax=vmax, cmap='viridis')
    ax_s.set_title(f'Student\n{label}', fontsize=9)
    ax_s.set_xlabel('Sample', fontsize=8)
    if col == 0:
        ax_s.set_ylabel('Student\nSample', fontsize=9)
    plt.colorbar(im, ax=ax_s, fraction=0.046)

    ax_d = axes[2][col]
    im = ax_d.imshow(dist_diff, cmap='hot')
    mae  = dist_diff.mean()
    ax_d.set_title(f'|T - S|\nMAE={mae:.3f}', fontsize=9)
    ax_d.set_xlabel('Sample', fontsize=8)
    if col == 0:
        ax_d.set_ylabel('|Difference|\nSample', fontsize=9)
    plt.colorbar(im, ax=ax_d, fraction=0.046)

plt.tight_layout()
plt.savefig('./relational_distance_matrix.png', dpi=150, bbox_inches='tight')
print("  Saved: relational_distance_matrix.png")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 6 — SUMMARY: distance-to-teacher per layer
# Single most interpretable plot for a paper/presentation.
# Shows how closely the student tracks the teacher's solution trajectory.
# ═══════════════════════════════════════════════════════════════════════════════

print("Plotting View 6: Distance-to-teacher summary...")

# Distance between student F_ii and teacher F_{ii*2}
# Also compute distance between student and teacher FINAL F
F_T_final_np = F_T_history[-1][0, SAMPLE, :, :].numpy()
sv_final_T   = np.linalg.svd(F_T_final_np, compute_uv=False)

dist_to_teacher_matched  = []   # student ii vs teacher ii*2
dist_to_teacher_final    = []   # student ii vs teacher FINAL
sv_gap_to_final          = []   # singular value gap to teacher's final SV

for s_ii in range(len(F_S_history)):
    t_ii   = min(s_ii * 2, len(F_T_history)-1)

    F_t_m  = F_T_history[t_ii][0, SAMPLE, :, :].numpy()
    F_s    = F_S_history[s_ii][0, SAMPLE, :, :].numpy()
    F_t_f  = F_T_final_np

    # Frobenius distance (real-valued after stacking)
    diff_m = np.abs(F_t_m - F_s)
    diff_f = np.abs(F_t_f - F_s)
    dist_to_teacher_matched.append(np.linalg.norm(diff_m, 'fro'))
    dist_to_teacher_final.append(np.linalg.norm(diff_f, 'fro'))

    # SV gap: how close is student's SV profile to teacher's final profile
    sv_s = np.linalg.svd(F_s, compute_uv=False)
    sv_gap_to_final.append(np.linalg.norm(sv_s - sv_final_T))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('How close is the student to the teacher at each layer?\n'
             '(lower = more aligned, vertical line = RKD supervision boundary)',
             fontsize=12)

x_s = range(len(F_S_history))

ax = axes[0]
ax.plot(x_s, dist_to_teacher_matched, color='#1f77b4', linewidth=2,
        label='Distance to matched teacher layer')
ax.axvline(x=10, color='purple', linestyle='--', alpha=0.7,
           label='K_outer=10 boundary')
ax.set_xlabel('Student outer iteration', fontsize=11)
ax.set_ylabel('Frobenius distance', fontsize=11)
ax.set_title('F distance: student vs matched teacher', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.fill_between(range(11), 0, max(dist_to_teacher_matched)*1.1,
                alpha=0.08, color='purple')

ax = axes[1]
ax.plot(x_s, dist_to_teacher_final, color='#d62728', linewidth=2,
        label='Distance to teacher FINAL F')
ax.axvline(x=10, color='purple', linestyle='--', alpha=0.7)
ax.set_xlabel('Student outer iteration', fontsize=11)
ax.set_ylabel('Frobenius distance', fontsize=11)
ax.set_title('F distance: student vs teacher final solution', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.fill_between(range(11), 0, max(dist_to_teacher_final)*1.1,
                alpha=0.08, color='purple')

ax = axes[2]
ax.plot(x_s, sv_gap_to_final, color='#2ca02c', linewidth=2,
        label='SV gap to teacher final')
ax.axvline(x=10, color='purple', linestyle='--', alpha=0.7)
ax.set_xlabel('Student outer iteration', fontsize=11)
ax.set_ylabel('||σ_student - σ_teacher_final||₂', fontsize=11)
ax.set_title('Eigenspace gap to teacher\'s final solution', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.fill_between(range(11), 0, max(sv_gap_to_final)*1.1,
                alpha=0.08, color='purple')

plt.tight_layout()
plt.savefig('./distance_to_teacher_summary.png', dpi=150, bbox_inches='tight')
print("  Saved: distance_to_teacher_summary.png")
plt.close()

print("\n" + "="*60)
print("All visualisations complete.")
print("Files saved:")
for f in ['beampattern_strip.png',
          'singular_value_spectrum.png',
          'student_teacher_alignment.png',
          'step_size_heatmap.png',
          'relational_distance_matrix.png',
          'distance_to_teacher_summary.png']:
    print(f"  {f}")
print("="*60)