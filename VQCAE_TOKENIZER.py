import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, Optional


class VQCAETokenizer:
    """
    A simple tokenizer for VQCAE that encodes images to tokens and decodes tokens back to images.
    Includes EOS and PAD tokens for sequence handling.
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the VQCAE tokenizer

        Args:
            model_path: Path to the trained VQCAE model checkpoint
            device: Device to run inference on ('cuda', 'cpu', etc.)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load model
        self._load_model(model_path)

        # Define special tokens
        self.codebook_size = self.model.vq.codebook_size
        self.eos_token_idx = self.codebook_size
        self.pad_token_idx = self.codebook_size + 1

        # Set model to evaluation mode
        self.model.eval()

        print(f"Initialized VQCAE Tokenizer (Codebook Size: {self.codebook_size})")
        print(f"Special tokens: EOS={self.eos_token_idx}, PAD={self.pad_token_idx}")

    def _load_model(self, model_path: str):
        """Load the VQCAE model from checkpoint"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Extract model state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Clean up state dict keys if needed (from DDP or torch.compile)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.module.'):
                new_key = key[len('_orig_mod.module.'):]
                cleaned_state_dict[new_key] = value
            elif key.startswith('module.'):
                new_key = key[len('module.'):]
                cleaned_state_dict[new_key] = value
            else:
                cleaned_state_dict[key] = value

        # Get model parameters from state dict
        in_channels = None
        hidden_channels = None
        codebook_size = None

        # Look for VQ embedding to get codebook size and hidden channels
        for key, value in cleaned_state_dict.items():
            if key == 'vq.embedding.weight':
                codebook_size = value.shape[0]
                hidden_channels = value.shape[1]
                break

        # Look for encoder's first layer to get in_channels
        for key, value in cleaned_state_dict.items():
            if key == 'encoder.init_conv1.weight':
                in_channels = value.shape[1]
                break
            elif key == 'encoder.net.0.weight':  # Fallback for older models
                in_channels = value.shape[1]
                break

        # Use defaults if we couldn't extract values
        if in_channels is None:
            in_channels = 3  # Default to 3 channels (RGB)
        if hidden_channels is None:
            hidden_channels = 4096  # Default to 4096 hidden channels
        if codebook_size is None:
            codebook_size = 128  # Default to 128 codebook size

        # Import VQCAE model class
        from VQCAE_CLEAN_DDP import VQCAE

        # Create model instance
        self.model = VQCAE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            codebook_size=codebook_size
        ).to(self.device)

        # Load state dict
        self.model.load_state_dict(cleaned_state_dict, strict=False)

        # Add decode_indices method to model if needed
        if not hasattr(self.model, 'decode_indices'):
            self._add_decode_indices_method()

    def _add_decode_indices_method(self):
        """Add decode_indices method to the model instance"""

        def decode_indices(model_self, indices):
            """Decode indices to images using the VQCAE model"""
            batch_size = indices.shape[0]
            h, w = indices.shape[1], indices.shape[2]

            # Get embeddings from indices
            flat_indices = indices.reshape(-1)
            embeddings = model_self.vq.embedding(flat_indices)

            # Reshape to expected format
            z_q = embeddings.view(batch_size, h, w, model_self.vq.embedding_dim)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

            # Decode
            x_rec = model_self.decoder(z_q)
            return x_rec

        # Bind method to model instance
        import types
        self.model.decode_indices = types.MethodType(decode_indices, self.model)

    def encode(self, image: np.ndarray) -> torch.Tensor:
        """
        Encode an image to token indices

        Args:
            image: Input image as numpy array with shape [channels, height, width]
                  or [batch, channels, height, width]

        Returns:
            Token indices as a tensor
        """
        # Check input type
        if not isinstance(image, (np.ndarray, torch.Tensor)):
            raise TypeError("Image must be a numpy array or torch tensor")

        # Convert numpy to tensor if needed
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()

        # Check dimensions
        if image.ndim == 3:  # Single image [C, H, W]
            image = image.unsqueeze(0)  # Add batch dimension
            batch_input = False
        elif image.ndim == 4:  # Batch of images [B, C, H, W]
            batch_input = True
        else:
            raise ValueError(f"Invalid image shape: {image.shape}. Expected [C,H,W] or [B,C,H,W]")

        # Move to device
        image = image.to(self.device)

        # Encode
        with torch.no_grad():
            indices = self.model.encode_indices(image)

        # Remove batch dimension if input was a single image
        if not batch_input:
            indices = indices[0]

        return indices

    def encode_with_eos(self, image: np.ndarray) -> torch.Tensor:
        """
        Encode image to tokens and append EOS token

        Args:
            image: Input image as numpy array with shape [channels, height, width]
                  or [batch, channels, height, width]

        Returns:
            Token indices with EOS token appended
        """
        indices = self.encode(image)

        # Handle different shapes
        if indices.ndim == 2:  # Single image tokens [H, W]
            # Flatten and add EOS
            flat_indices = indices.flatten()
            result = torch.cat([
                flat_indices,
                torch.tensor([self.eos_token_idx], device=self.device)
            ])
        else:  # Batch of tokens [B, H, W]
            # Flatten each sample and add EOS
            batch_size = indices.shape[0]
            flattened = indices.reshape(batch_size, -1)
            eos_tokens = torch.full((batch_size, 1), self.eos_token_idx,
                                    device=self.device, dtype=indices.dtype)
            result = torch.cat([flattened, eos_tokens], dim=1)

        return result

    def decode(self, indices: torch.Tensor) -> np.ndarray:
        """
        Decode token indices back to an image

        Args:
            indices: Token indices with shape [height, width] or [batch, height, width]

        Returns:
            Reconstructed image(s) as numpy array
        """
        # Convert to tensor if numpy
        if isinstance(indices, np.ndarray):
            indices = torch.from_numpy(indices).long()

        # Move to device
        indices = indices.to(self.device)

        # Check dimensions
        if indices.ndim == 2:  # Single set of indices [H, W]
            indices = indices.unsqueeze(0)  # Add batch dimension
            batch_input = False
        elif indices.ndim == 3:  # Batch of indices [B, H, W]
            batch_input = True
        else:
            raise ValueError(f"Invalid indices shape: {indices.shape}. Expected [H,W] or [B,H,W]")

        # Replace special tokens with the last regular token
        mask = (indices == self.eos_token_idx) | (indices == self.pad_token_idx)
        if mask.any():
            indices = indices.clone()  # Create a copy
            indices[mask] = self.codebook_size - 1

        # Decode
        with torch.no_grad():
            images = self.model.decode_indices(indices)

        # Convert to numpy
        images_np = images.cpu().numpy()

        # Remove batch dimension if input was a single image
        if not batch_input:
            images_np = images_np[0]

        return images_np

    def get_codebook(self) -> np.ndarray:
        """Get the codebook embeddings"""
        with torch.no_grad():
            codebook = self.model.vq.embedding.weight.cpu().numpy()
        return codebook

    def get_eos_token(self) -> int:
        """Get the EOS token index"""
        return self.eos_token_idx

    def get_pad_token(self) -> int:
        """Get the PAD token index"""
        return self.pad_token_idx

    def get_vocab_size(self) -> int:
        """Get total vocabulary size (codebook + special tokens)"""
        return self.codebook_size + 2  # Regular tokens + EOS + PAD

    def __len__(self) -> int:
        """Get vocabulary size"""
        return self.get_vocab_size()