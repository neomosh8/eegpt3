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
        """Generate and save the plot to a file with custom dark style."""

        # Create figure with dark background
        fig = plt.figure(figsize=(10, 5), facecolor="#000103")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#000103")

        # Customize grid
        ax.grid(True, color="#515052")

        # Set tick and label colors to white for contrast
        ax.tick_params(colors="white")
        plt.xlabel("Step", color="white")
        plt.ylabel("Loss", color="white")
        plt.title(f"Loss Curves up to step {step}", color="white")

        # Plot train loss in #011638
        plt.plot(self.train_losses, label="Train Loss", color="#20A4F3")

        # Plot validation loss in #EEC643 (if available) with a thick line
        if self.val_losses:
            val_steps = np.linspace(0, len(self.train_losses), len(self.val_losses))
            plt.plot(val_steps, self.val_losses, label="Val Loss", color="#EEC643", linewidth=1)

        # Plot moving average in #0D21A1 with a thick line
        plt.plot(self.moving_avg, label="Moving Avg Train Loss", color="#0D21A1", linewidth=3)

        # Create legend with custom styling: text in white, Helvetica, and matching background
        leg = plt.legend()
        for text in leg.get_texts():
            text.set_color("white")
        leg.get_frame().set_facecolor("#000103")
        leg.get_frame().set_edgecolor("white")

        # Save the figure using the dark background color
        plt.savefig(f"{self.filename_prefix}.png", facecolor="#000103")
        plt.close()


