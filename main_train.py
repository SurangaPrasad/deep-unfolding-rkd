import numpy as np
import matplotlib.pyplot as plt
from utility import *
from PGA_models import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---- training and test the models ----

# Load training data
H_train, H_test0 = get_data_tensor(data_source)
H_test = H_test0[:, :test_size, :, :]
torch.manual_seed(3407)

# =============================================================
#  Teacher: PGA_Unfold_J20 (J=20, I=120)
#  Student: PGA_Unfold_J10 (J=10, I=60)  -- 4x cheaper
# =============================================================
I_T = n_iter_outer          # teacher outer layers (120)
J_T = n_iter_inner_J20      # teacher inner layers (20)
I_S = I_student             # student outer layers (60)
J_S = n_iter_inner_J10      # student inner layers (10)

# LRD hyperparameters (paper Section IV)
L = 15          # window length
le = 20         # teacher early-window start
lambda_d = 25.0
lambda_a = 50.0
lambda_late = 0.8
lambda_log = 0.0001


def train_teacher():
    """Train the teacher model (J=20, I=120) with task loss only."""
    print('Training teacher (J=20, I=120)...')
    model = PGA_Unfold_J20(step_size_UPGA_J20).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_losses = []

    for i_epoch in range(n_epoch):
        batch_losses = []
        H_shuffled = torch.transpose(H_train, 0, 1)[np.random.permutation(len(H_train[0]))]
        for i_batch in range(0, len(H_train[0]), batch_size):
            H = torch.transpose(H_shuffled[i_batch:i_batch + batch_size], 0, 1).to(device)
            cur_bs = H.shape[1]
            snr_dB_train = np.random.permutation(
                np.tile(snr_dB_list, (batch_size // len(snr_dB_list)) + 1)
            )[:cur_bs]
            snr_train = torch.tensor(10 ** (snr_dB_train / 10), dtype=torch.float32, device=device)
            Rtrain, _, _, _ = get_radar_data(snr_dB_train, H)
            Rtrain = Rtrain.to(device)

            _, _, F, W = model.execute_PGA(H, Rtrain, snr_train, n_iter_outer, n_iter_inner_J20)
            loss = get_sum_loss(F, W, H, Rtrain, snr_train, cur_bs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        avg = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg)
        print(f"Teacher Epoch [{i_epoch+1}/{n_epoch}], Avg Loss: {avg:.4f}")

    torch.save(model.state_dict(), model_file_name_teacher)
    return model, epoch_losses


def tgi_init_student(teacher_model):
    """Teacher-Guided Initialization: compress teacher step sizes to student."""
    print('Applying TGI: compressing teacher step sizes to student...')
    teacher_step = teacher_model.step_size.detach()  # [J_T, I_T, K+1]
    student_step = tgi_compress_step_sizes(teacher_step, J_S, I_S, K + 1)
    return student_step


def train_student_lrd(teacher_model, use_tgi=True, use_lrd=True):
    """Train the student (J=10, I=60) with TGI and/or LRD."""
    print(f'Training student (J=10, I=60) | TGI={use_tgi}, LRD={use_lrd}...')

    if use_tgi:
        student_step = tgi_init_student(teacher_model)
    else:
        student_step = torch.full([J_S, I_S, K + 1], step_size_fixed, device=device, requires_grad=True)

    model = PGA_Unfold_J10(student_step).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_losses = []

    teacher_model.eval()

    for i_epoch in range(n_epoch):
        batch_losses = []
        H_shuffled = torch.transpose(H_train, 0, 1)[np.random.permutation(len(H_train[0]))]
        for i_batch in range(0, len(H_train[0]), batch_size):
            H = torch.transpose(H_shuffled[i_batch:i_batch + batch_size], 0, 1).to(device)
            cur_bs = H.shape[1]
            snr_dB_train = np.random.permutation(
                np.tile(snr_dB_list, (batch_size // len(snr_dB_list)) + 1)
            )[:cur_bs]
            snr_train = torch.tensor(10 ** (snr_dB_train / 10), dtype=torch.float32, device=device)
            Rtrain, _, _, _ = get_radar_data(snr_dB_train, H)
            Rtrain = Rtrain.to(device)

            # ---- Teacher forward (no grad) ----
            with torch.no_grad():
                _, _, F_t, W_t, FW_t_early, FW_t_late = teacher_model.execute_PGA_with_windows(
                    H, Rtrain, snr_train, n_iter_outer, J_T, L, le)
                J_teacher_final = get_sum_loss(F_t, W_t, H, Rtrain, snr_train, cur_bs)

            # ---- Student forward ----
            _, _, F_s, W_s, FW_s_early, FW_s_late = model.execute_PGA_with_windows(
                H, Rtrain, snr_train, I_S, J_S, L)

            # ---- Losses ----
            L_task = get_sum_loss(F_s, W_s, H, Rtrain, snr_train, cur_bs)

            total = L_task
            if use_lrd:
                L_early = lrd_window_loss(FW_t_early, FW_s_early, lambda_d, lambda_a)
                L_late = lrd_window_loss(FW_t_late, FW_s_late, lambda_d, lambda_a)
                total = total + L_early + lambda_late * L_late

            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            batch_losses.append(total.item())

        avg = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg)
        print(f"Student Epoch [{i_epoch+1}/{n_epoch}], Avg Loss: {avg:.4f}")

    tag = f"TGI{int(use_tgi)}_LRD{int(use_lrd)}"
    torch.save(model.state_dict(), directory_model + f'UPGA_J10_student_{tag}.pth')
    return model, epoch_losses


def plot_training_losses(teacher_losses, student_losses_dict):
    """Plot the training loss curves for the teacher and all 4 student variants.

    teacher_losses: list of per-epoch losses, or None (e.g. when the teacher is
                    loaded from a checkpoint instead of retrained).
    student_losses_dict: dict mapping (use_tgi, use_lrd) -> list of per-epoch
                         losses for each of the 4 student variants.
    """
    if teacher_losses is None and not student_losses_dict:
        print('No training-loss history to plot (teacher and students were loaded).')
        return

    plt.figure(figsize=(8, 6))
    if teacher_losses is not None:
        epochs_t = np.arange(1, len(teacher_losses) + 1)
        plt.plot(epochs_t, teacher_losses, '-o', color='red', linewidth=2,
                 markersize=4, label='Teacher (UPGA, J=20, I=120)')

    # Style for each of the 4 student variants
    student_styles = {
        (False, False): ('-',  'blue',   'Student (J=10, I=60)'),
        (True,  False): ('--', 'green',  'Student + TGI'),
        (False, True):  ('-.', 'orange', 'Student + LRD'),
        (True,  True):  (':',  'purple', 'Student + TGI + LRD'),
    }
    for (use_tgi, use_lrd), losses in student_losses_dict.items():
        if losses is None:
            continue
        linestyle, color, label = student_styles[(use_tgi, use_lrd)]
        epochs_s = np.arange(1, len(losses) + 1)
        plt.plot(epochs_s, losses, linestyle=linestyle, color=color, linewidth=2,
                 markersize=4, label=label)

    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss')
    plt.grid()
    plt.legend(loc='upper right')
    plt.savefig(directory_result + f'training_loss_{system_config}.png', dpi=200)
    plt.savefig(directory_result + f'training_loss_{system_config}.eps')
    plt.close()
    print(f"Saved training-loss plot to {directory_result}training_loss_{system_config}.png")


# =============================================================
#  Run the training pipeline
# =============================================================
def run_training():
    """Train the teacher (if enabled) then all 4 student variants
    (TGI/LRD on/off), and plot all training-loss curves in one diagram."""
    teacher_losses = None
    student_losses_dict = {}

    # ---- Stage 1: Teacher ----
    if run_UPGA_J20 == 1:
        teacher, teacher_losses = train_teacher()
    else:
        teacher = PGA_Unfold_J20(step_size_UPGA_J20).to(device)
        teacher.load_state_dict(torch.load(model_file_name_UPGA_J20, map_location=device))

    # ---- Stage 2: All 4 student variants ----
    if run_UPGA_J10 == 1:
        for use_tgi in [False, True]:
            for use_lrd in [False, True]:
                student, losses = train_student_lrd(teacher, use_tgi=use_tgi, use_lrd=use_lrd)
                student_losses_dict[(use_tgi, use_lrd)] = losses

    # ---- Plot training curves ----
    plot_training_losses(teacher_losses, student_losses_dict)
    return teacher, student_losses_dict


if __name__ == '__main__':
    run_training()
