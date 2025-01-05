import math
import matplotlib.pyplot as plt

# Parameters
max_lr = 3e-3
min_lr = 4e-6
warmup_steps = 100
max_steps = 2400

# Function definition
def get_lr_dynamic_range(it, max_lr=max_lr, min_lr=min_lr, warmup_steps=warmup_steps, max_steps=max_steps, scale_factor=1.5, dynamic_boost=1.2, power=5):
    """
    Calculate the learning rate for a given iteration with dynamic range adjustments.

    Parameters:
        it (int): Current iteration.
        max_lr (float): Maximum learning rate.
        min_lr (float): Minimum learning rate.
        warmup_steps (int): Number of warmup steps.
        max_steps (int): Total number of steps.
        scale_factor (float): Factor to scale the cosine frequency.
        dynamic_boost (float): Boost factor for increasing the dynamic range.
        power (float): Power to amplify the decay effect.

    Returns:
        float: Learning rate at the given iteration.
    """
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = (0.5 * (1 + math.cos(math.pi * decay_ratio * scale_factor)) * dynamic_boost) ** power
    return min_lr + coeff * (max_lr - min_lr)

# Generate learning rate values for iterations
iterations = list(range(0, 2301))
learning_rates = [get_lr_dynamic_range(it) for it in iterations]

# Plot the learning rate schedule
plt.figure(figsize=(10, 6))
plt.plot(iterations, learning_rates, label="Dynamic Range Adjusted Learning Rate")
plt.xlabel("Iteration")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule with Increased Dynamic Range")
plt.grid(True)
plt.legend()
plt.show()
