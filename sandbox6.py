import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read the training log file assuming it is whitespace-delimited.
# The file should have a header like "step train_loss".
data = pd.read_csv("training.log", delim_whitespace=True)

# Extract training steps and loss values.
steps = data['step']
loss = data['train_loss']

# Compute a smoothed loss curve using a moving average.
def moving_average(data, window_size=3):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

window_size = 3  # You can adjust the window size as needed.
smoothed_loss = moving_average(loss, window_size)
smoothed_steps = steps[window_size - 1:]  # Adjust steps to match the smoothed data.

# Compute the pace of loss change: the difference between consecutive loss values.
loss_diff = np.diff(loss)
steps_diff = steps[1:]  # Corresponding steps for the differences.

# Create two subplots: one for the loss curves, one for the pace analysis.
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# Plot the raw training loss and the smoothed loss.
axes[0].plot(steps, loss, 'o-', label='Raw Training Loss', alpha=0.7)
axes[0].plot(smoothed_steps, smoothed_loss, 's--', label='Smoothed Loss (MA)', alpha=0.9)
axes[0].set_title('Training Loss vs. Steps')
axes[0].set_xlabel('Training Steps')
axes[0].set_ylabel('Loss')
axes[0].grid(True)
axes[0].legend()

# Plot the pace of training: the change (delta) in loss between successive steps.
axes[1].plot(steps_diff, loss_diff, 'o-', color='orange', label='Loss Change (Delta)')
axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
axes[1].set_title('Pace of Training Loss Change')
axes[1].set_xlabel('Training Steps')
axes[1].set_ylabel('Loss Difference')
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()
