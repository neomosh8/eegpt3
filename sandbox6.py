import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------
# 1. Read the data from file
# -----------------------------
data = pd.read_csv("training.log", delim_whitespace=True)
steps = data['step'].values
loss = data['train_loss'].values

# -----------------------------
# 2. Calculate cumulative percentage of losses ≤ 1.03
# -----------------------------
threshold = 1.03
cumulative_below = np.cumsum(loss <= threshold)
cumulative_percent = (cumulative_below / np.arange(1, len(loss) + 1)) * 100

# -----------------------------
# 3. Calculate moving average of actual loss
# -----------------------------
def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

window_size = 50
smoothed_loss = moving_average(loss, window_size)
smoothed_steps = steps[window_size - 1:]  # Align steps with smoothed data

# Print moving average values of loss
print(f"Moving Average of Loss Values (window size = {window_size}):")
for step, value in zip(smoothed_steps, smoothed_loss):
    print(f"Step {step}: {value:.4f}")

# -----------------------------
# 4. Plotting the Results (only cumulative percentage)
# -----------------------------
plt.figure(figsize=(12, 6))
plt.plot(steps, cumulative_percent, 'b.-', label=f'Cumulative % of Loss ≤ {threshold}')
plt.title('Cumulative Percentage of Losses ≤ 1.03 vs. Steps')
plt.xlabel('Training Step')
plt.ylabel('Cumulative Percentage (%)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
print(cumulative_percent[-1])