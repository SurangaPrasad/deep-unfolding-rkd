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


def run_UPGA(step_size_UPGA, n_inner, run_id):
    """Train the unfolded PGA (UPGA) model with balanced per-SNR sampling."""
    model = PGA_Unfold_J10 if n_inner == n_iter_inner_J10 else PGA_Unfold_J20
    model_UPGA = model(step_size_UPGA).to(device)
    optimizer = torch.optim.Adam(model_UPGA.parameters(), lr=learning_rate)

    epoch_losses = []

    for i_epoch in range(n_epoch):
        batch_losses = []
        H_shuffled = torch.transpose(H_train, 0, 1)[np.random.permutation(len(H_train[0]))]

        for i_batch in range(0, len(H_train[0]), batch_size):
            H = torch.transpose(H_shuffled[i_batch:i_batch + batch_size], 0, 1).to(device)
            cur_bs = H.shape[1]
            # balanced per-SNR sampling
            snr_dB_train = np.random.permutation(
                np.tile(snr_dB_list, (batch_size // len(snr_dB_list)) + 1)
            )[:cur_bs]
            snr_train = torch.tensor(10 ** (snr_dB_train / 10), dtype=torch.float32, device=device)

            Rtrain, _, _, _ = get_radar_data(snr_dB_train, H)
            Rtrain = Rtrain.to(device)

            _, _, F, W = model_UPGA.execute_PGA(H, Rtrain, snr_train, n_iter_outer, n_inner)
            loss = get_sum_loss(F, W, H, Rtrain, snr_train, cur_bs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            print(f"Epoch [{i_epoch+1}/{n_epoch}], Batch [{i_batch//batch_size + 1}/{len(H_train[0])//batch_size}], Loss: {loss.item():.4f}")

        avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{i_epoch+1}/{n_epoch}], Average Loss: {avg_loss:.4f}")

    torch.save(model_UPGA.state_dict(), directory_model + f'UPGA_J{n_inner}_{run_id}.pth')

    # Plot and save loss over epochs
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title(f'Training Loss over Epochs (J={n_inner})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(directory_model + f'UPGA_J{n_inner}_{run_id}_loss.png', dpi=300)
    plt.show()


# ============================================================= proposed unfolding PGA =================================
if run_UPGA_J20 == 1:
    run_UPGA(step_size_UPGA_J20, n_iter_inner_J20, '320')

# ============================================================= proposed unfolding PGA =================================
if run_UPGA_J10 == 1:
    run_UPGA(step_size_UPGA_J10, n_iter_inner_J10, '320')
