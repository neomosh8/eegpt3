import math
import matplotlib.pyplot as plt




epoch_num = 20
total_batch_size = 524288
B = 16
T = 1024
plateau_count = 0
max_lr = 1e-1
min_lr = 1e-10
max_steps = math.ceil(1e9//total_batch_size) * epoch_num
warmup_steps =int(0.02*max_steps)
best_val_loss = float('inf')
no_improvement_count = 0
patience = 3

def get_lr(step, max_lr=max_lr, min_lr=min_lr, warmup_steps=warmup_steps, total_steps=2*max_steps):
    if step < warmup_steps:
        lr = max_lr * (step + 1) / warmup_steps
    else:
        ratio = (step - warmup_steps) / float(total_steps - warmup_steps)
        ratio = min(1.0, max(0.0, ratio))
        lr = max_lr * (min_lr / max_lr) ** ratio
        lr = max(lr, min_lr)
    # multiply by 0.1^plateau_count
    factor = 0.1 ** plateau_count
    lr_final = lr * factor
    return max(lr_final, min_lr)

# Generate learning rate values for iterations
iterations = list(range(0, max_steps))
learning_rates = [get_lr(it) for it in iterations]
print(learning_rates[27000])
# Plot the learning rate schedule
plt.figure(figsize=(10, 6))
plt.plot(iterations, learning_rates, label="Dynamic Range Adjusted Learning Rate")
plt.xlabel("Iteration")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule with Increased Dynamic Range")
plt.grid(True)
plt.legend()
plt.show()


