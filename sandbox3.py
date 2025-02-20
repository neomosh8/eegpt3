import torch
import matplotlib.pyplot as plt

# Parameters
base_lr = 1e-6
max_steps = 153280  # Total training steps
num_passes = 5
total_steps = max_steps // num_passes  # Correct total steps

# Dummy optimizer
optimizer = torch.optim.AdamW([torch.zeros(1)], lr=base_lr)

# Define the OneCycleLR scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=base_lr,
    total_steps=total_steps,  # Use correct total_steps
    pct_start=0.1,
    anneal_strategy='cos',
    cycle_momentum=False
)

# Track learning rate over steps
lrs = []

for step in range(total_steps):  # Use total_steps instead of max_steps
    optimizer.step()  # Ensure optimizer step is called before scheduler
    lrs.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

# Plot learning rate schedule
plt.figure(figsize=(8, 5))
plt.plot(range(total_steps), lrs, label="Learning Rate")
plt.xlabel("Training Steps")
plt.ylabel("Learning Rate")
plt.title("OneCycleLR Learning Rate Schedule")
plt.legend()
plt.grid()
plt.show()
