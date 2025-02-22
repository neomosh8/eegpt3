import torch
import matplotlib.pyplot as plt


class CustomLRScheduler:
    def __init__(self, optimizer, base_lr, constant_steps=100, decay_rate=0.99):
        """
        Custom LR scheduler with constant phase followed by exponential decay
        Args:
            optimizer: PyTorch optimizer
            base_lr: Initial learning rate
            constant_steps: Number of steps to maintain base_lr (default: 100)
            decay_rate: Exponential decay rate (default: 0.99)
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.constant_steps = constant_steps
        self.decay_rate = decay_rate
        self.step_count = 0

        # Initialize optimizer with base_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = base_lr

    def step(self):
        """Update learning rate for each step"""
        self.step_count += 1

        if self.step_count <= self.constant_steps:
            current_lr = self.base_lr
        else:
            # Exponential decay after constant_steps
            decay_steps = self.step_count - self.constant_steps
            current_lr = self.base_lr * (self.decay_rate ** decay_steps)

        # Update optimizer's learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr

        return current_lr


# Parameters
base_lr = 4e-4
max_steps = 16200
decay_rate = 1  # Adjustable decay rate (smaller value = faster decay)

# Dummy optimizer
optimizer = torch.optim.AdamW([torch.zeros(1)], lr=base_lr)

# Create custom scheduler
scheduler = CustomLRScheduler(optimizer, base_lr=base_lr, constant_steps=100, decay_rate=decay_rate)

# Track learning rate over steps
lrs = []

for step in range(max_steps):
    optimizer.step()
    lr = scheduler.step()
    lrs.append(lr)

# Plot learning rate schedule
plt.figure(figsize=(8, 5))
plt.plot(range(max_steps), lrs, label="Learning Rate")
plt.xlabel("Training Steps")
plt.ylabel("Learning Rate")
plt.title(f"Custom LR Schedule (Constant 100 steps, decay_rate={decay_rate})")
plt.legend()
plt.grid()
plt.show()