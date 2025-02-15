import torch
import matplotlib.pyplot as plt

# Parameters
base_lr = 6e-4
max_steps = 14835  # Total training steps

# Dummy optimizer (needed for the scheduler)
optimizer = torch.optim.AdamW([torch.zeros(1)], lr=base_lr)

# Define the OneCycleLR scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=base_lr,
    total_steps=max_steps,
    pct_start=0.10,
    anneal_strategy='cos',
    cycle_momentum=False
)

# Track learning rate over steps
lrs = []
for step in range(max_steps):
    lrs.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

# Plot learning rate schedule
plt.figure(figsize=(8, 5))
plt.plot(range(max_steps), lrs, label="Learning Rate")
plt.xlabel("Training Steps")
plt.ylabel("Learning Rate")
plt.title("OneCycleLR Learning Rate Schedule")
plt.legend()
plt.grid()
plt.show()
