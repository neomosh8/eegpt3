import matplotlib.pyplot as plt
import numpy as np

class LossPlotter:
    def __init__(self, plot_interval=500, window=50, filename_prefix="loss_plot"):
        """
        Args:
            plot_interval (int): How many steps between saving a new plot.
            window (int): Window size for the moving average.
            filename_prefix (str): Prefix for saved plot files.
        """
        self.plot_interval = plot_interval
        self.window = window
        self.filename_prefix = filename_prefix
        self.train_losses = []   # record of each training loss (per optimizer step)
        self.val_losses = []     # record of validation losses
        self.moving_avg = []     # moving average of train loss

    def update_train(self, loss):
        """Record the training loss and update moving average."""
        self.train_losses.append(loss)
        # Compute moving average over the last 'window' steps (or fewer if not available yet)
        current_window = self.train_losses[-self.window:]
        avg = np.mean(current_window)
        self.moving_avg.append(avg)

    def update_val(self, loss):
        """Record a validation loss."""
        self.val_losses.append(loss)

    def maybe_plot(self, step):
        """Save a plot if the current step is a multiple of plot_interval."""
        if step % self.plot_interval == 0:
            self.plot(step)

    def plot(self, step):
        """Generate and save the plot to a file."""
        plt.figure(figsize=(10, 5))
        # Train Loss: light blue and 30% transparent
        plt.plot(self.train_losses, label="Train Loss", color="lightblue", alpha=0.3)

        if self.val_losses:
            # Align validation loss with training steps
            val_steps = np.linspace(0, len(self.train_losses), len(self.val_losses))
            # Val Loss: dark blue and thick
            plt.plot(val_steps, self.val_losses, label="Val Loss", color="darkblue", linewidth=1)

        # Moving Average: thick and black
        plt.plot(self.moving_avg, label="Moving Avg Train Loss", color="black", linewidth=3)

        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"Loss Curves up to step {step}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.filename_prefix}.png")
        plt.close()

