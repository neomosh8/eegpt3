"""
checkpoint_manager.py

This module provides utilities for saving and loading PyTorch model checkpoints.
Each checkpoint includes:
  - Model state dictionary.
  - Optimizer state dictionary.
  - Model configuration.
  - Training step.
  - Validation loss.

Usage:
    from checkpoint_manager import save_checkpoint, load_checkpoint

    # To save a checkpoint:
    save_checkpoint(
        model=raw_model,
        optimizer=optimizer,
        config=raw_model.config,
        step=step,
        val_loss=val_loss_accum.item(),
        log_dir=log_dir
    )

    # To load a checkpoint:
    checkpoint = load_checkpoint(checkpoint_path, model=raw_model, optimizer=optimizer, device=device)
    # The returned checkpoint dict can be used to restore other variables if needed.
"""

import os
import torch


def save_checkpoint(model, optimizer, config, step, val_loss, log_dir):
    """
    Saves a checkpoint containing the model, optimizer, configuration, step, and validation loss.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer whose state will be saved.
        config (any): The configuration object or dict used to create the model.
        step (int): Current training step.
        val_loss (float): Current validation loss.
        log_dir (str): Directory where the checkpoint file will be saved.

    Returns:
        checkpoint_path (str): The path to the saved checkpoint file.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'step': step,
        'val_loss': val_loss,
    }

    checkpoint_path = os.path.join(log_dir, f"model_last_checkpoint.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    """
    Loads a checkpoint and restores the model (and optionally the optimizer).

    This version strips the "_orig_mod." prefix from state dict keys if present.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): The device on which to map the checkpoint.

    Returns:
        checkpoint (dict): The loaded checkpoint dictionary.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    # Remove the '_orig_mod.' prefix if it exists.
    new_state_dict = {}
    prefix = "_orig_mod."
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
        else:
            new_key = key
        new_state_dict[new_key] = value

    # Load the modified state dict.
    model.load_state_dict(new_state_dict)
    print(f"Model state loaded from {checkpoint_path}")

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded.")

    return checkpoint
