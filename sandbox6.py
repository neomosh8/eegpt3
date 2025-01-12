import math
import matplotlib.pyplot as plt

epoch_num = 10
total_batch_size = 524288
max_lr = 3e-4
min_lr = 1e-7

max_steps = math.ceil(1e9/total_batch_size) * epoch_num
warmup_steps = ((max_steps)*0.04)
def get_lr(it, max_lr=max_lr, min_lr=min_lr, warmup_steps=warmup_steps, max_steps=1.7*max_steps):
    """
    Calculate the learning rate for a given iteration using simple exponential decay.

    Parameters:
        it (int): Current iteration.
        max_lr (float): Initial maximum learning rate.
        min_lr (float): Minimum learning rate after decay.
        warmup_steps (int): Number of warmup steps.
        max_steps (int): Total number of steps.

    Returns:
        float: Learning rate at the given iteration.
    """
    if it < warmup_steps:
        # Linear warmup
        lr = max_lr * (it + 1) / warmup_steps
    elif it > max_steps:
        # After max_steps, maintain min_lr
        lr = min_lr
    else:
        # Exponential decay
        decay_steps = it - warmup_steps
        total_decay_steps = max_steps - warmup_steps

        # Calculate decay rate to reach min_lr at max_steps
        decay_rate = math.log(min_lr / max_lr) / total_decay_steps

        # Apply exponential decay
        lr = max_lr * math.exp(decay_rate * decay_steps)

        # Ensure lr does not go below min_lr
        lr = max(lr, min_lr)

    return lr
# Generate learning rate values for iterations
iterations = list(range(0, max_steps))
learning_rates = [get_lr_exponential_decay(it) for it in iterations]

# Plot the learning rate schedule
plt.figure(figsize=(10, 6))
plt.plot(iterations, learning_rates, label="Dynamic Range Adjusted Learning Rate")
plt.xlabel("Iteration")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule with Increased Dynamic Range")
plt.grid(True)
plt.legend()
plt.show()


