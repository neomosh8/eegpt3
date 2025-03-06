import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, List, Optional


class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, hidden_channels, 3, 2, 1), nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, out_channels=3, hidden_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, 64, 3, 2, 1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, out_channels, 3, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class VectorQuantizerEMA(nn.Module):
    def __init__(self, codebook_size, embedding_dim, decay=0.9, eps=1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps

        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())

    def forward(self, z):
        device = z.device

        flat_z = z.reshape(-1, self.embedding_dim)

        # Calculate distances between inputs and embeddings
        dist = torch.sum(flat_z.pow(2), dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight.pow(2), dim=1) - \
               2 * torch.matmul(flat_z, self.embedding.weight.t())

        # Get nearest embedding indices
        encoding_indices = torch.argmin(dist, dim=1)

        # Create one-hot encodings on the same device
        encodings = torch.zeros(encoding_indices.shape[0], self.codebook_size, device=device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized.view_as(z)

        if self.training:
            # EMA update
            cluster_size_new = encodings.sum(0)
            self.cluster_size.data.mul_(self.decay).add_(cluster_size_new, alpha=1 - self.decay)

            flat_z_t = flat_z.t()
            embed_sums = torch.matmul(flat_z_t, encodings)
            self.ema_w.data.mul_(self.decay).add_(embed_sums.t(), alpha=1 - self.decay)

            n = self.cluster_size.sum()
            cluster_size = ((self.cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n)
            embed_normalized = self.ema_w / cluster_size.unsqueeze(1)
            self.embedding.weight.data.copy_(embed_normalized)

        # Straight-through estimator
        quantized_st = z + (quantized - z).detach()

        # Compute loss
        e_latent_loss = torch.mean((quantized.detach() - z).pow(2))
        q_latent_loss = torch.mean((quantized - z.detach()).pow(2))
        vq_loss = e_latent_loss + q_latent_loss

        return quantized_st, encoding_indices.view(z.shape[:-1]), vq_loss

    def get_codebook_entry(self, indices):
        """Transforms indices to quantized embeddings"""
        # Get embeddings for indices
        quantized = self.embedding(indices)
        return quantized


class VQCAE(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128, codebook_size=128, decay=0.9, commitment_beta=0.3):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_channels)
        self.vq = VectorQuantizerEMA(codebook_size, hidden_channels, decay=decay)
        self.decoder = Decoder(in_channels, hidden_channels)
        self.commitment_beta = commitment_beta

    def forward(self, x):
        z_e = self.encoder(x)
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        z_q, idxs, vq_loss = self.vq(z_e)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        x_rec = self.decoder(z_q)
        loss = F.mse_loss(x_rec, x) + self.commitment_beta * vq_loss
        return x_rec, loss

    def encode_indices(self, x):
        z_e = self.encoder(x)
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        _, idxs, _ = self.vq(z_e)
        return idxs

    def decode_indices(self, indices):
        # Get spatial dimensions from indices shape
        batch_size = indices.shape[0]
        height = indices.shape[1]
        width = indices.shape[2]

        # Convert indices to embeddings
        flat_indices = indices.view(-1)
        quantized = self.vq.get_codebook_entry(flat_indices)
        quantized = quantized.view(batch_size, height, width, -1)

        # Permute to channel-first and decode
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        x_rec = self.decoder(quantized)
        return x_rec


class VQCAETokenizer:
    """
    A tokenizer class for VQCAE that handles encoding images to token indices and decoding back.
    Includes an EOS token for sequence completion.
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the VQCAE tokenizer

        Args:
            model_path: Path to the trained VQCAE model checkpoint
            device: Device to run the model on (e.g., 'cuda:0', 'cpu')
                   If None, will use CUDA if available, otherwise CPU
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load checkpoint
        self._load_model(model_path)

        # Set up EOS token (assign it to the next index after the codebook)
        self.codebook_size = self.model.vq.codebook_size
        self.eos_token_idx = self.codebook_size

        # Set evaluation mode
        self.model.eval()

    def _load_model(self, model_path: str):
        """
        Load the VQCAE model from a checkpoint

        Args:
            model_path: Path to the model checkpoint
        """
        # Check if file exists
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Extract model parameters
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Clean up state dict keys if they have prefixes from DDP or torch.compile
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

        # Get model parameters from the state dict
        in_channels = None
        hidden_channels = None
        codebook_size = None

        # Try to extract parameters from the state dict
        for key in cleaned_state_dict.keys():
            if key == 'encoder.net.0.weight':
                in_channels = cleaned_state_dict[key].shape[1]
            if key == 'encoder.net.6.weight':
                hidden_channels = cleaned_state_dict[key].shape[0]
            if key == 'vq.embedding.weight':
                codebook_size = cleaned_state_dict[key].shape[0]

        # Check if we found all parameters
        if None in (in_channels, hidden_channels, codebook_size):
            raise ValueError("Could not infer model architecture from state dict")

        # Create model
        self.model = VQCAE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            codebook_size=codebook_size
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(cleaned_state_dict)

        print(f"Loaded VQCAE model with:")
        print(f"  - In channels: {in_channels}")
        print(f"  - Hidden channels: {hidden_channels}")
        print(f"  - Codebook size: {codebook_size}")
        print(f"  - EOS token index: {codebook_size} (added by tokenizer)")

    def encode(self, image: np.ndarray) -> torch.Tensor:
        """
        Encode an image to token indices

        Args:
            image: Input image as a numpy array with shape [channels, height, width]

        Returns:
            indices: Tensor of token indices with shape [height, width]
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a numpy array")

        # Check dimensions
        if image.ndim == 3:
            # Single image [channels, height, width]
            batch_input = False
        elif image.ndim == 4:
            # Batch of images [batch, channels, height, width]
            batch_input = True
        else:
            raise ValueError(f"Invalid image dimensions: {image.shape}. Expected [C,H,W] or [B,C,H,W]")

        # Convert to batch if single image
        if not batch_input:
            image = np.expand_dims(image, 0)

        # Convert to tensor
        with torch.no_grad():
            x = torch.from_numpy(image).float().to(self.device)
            indices = self.model.encode_indices(x)

        # Return indices (remove batch dimension if input was a single image)
        if not batch_input:
            return indices[0]
        else:
            return indices

    def decode(self, indices: torch.Tensor) -> np.ndarray:
        """
        Decode token indices back to an image

        Args:
            indices: Tensor of token indices with shape [height, width] or [batch, height, width]

        Returns:
            image: Reconstructed image as a numpy array with shape [channels, height, width]
                   or [batch, channels, height, width]
        """
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, device=self.device)

        # Check dimensions
        if indices.ndim == 2:
            # Single set of indices [height, width]
            batch_input = False
            indices = indices.unsqueeze(0)
        elif indices.ndim == 3:
            # Batch of indices [batch, height, width]
            batch_input = True
        else:
            raise ValueError(f"Invalid indices dimensions: {indices.shape}. Expected [H,W] or [B,H,W]")

        # Move to correct device
        indices = indices.to(self.device)

        # Replace any EOS tokens with the last regular token from the codebook
        mask = indices == self.eos_token_idx
        if mask.any():
            indices[mask] = self.codebook_size - 1

        # Decode to image
        with torch.no_grad():
            x_rec = self.model.decode_indices(indices)

        # Convert to numpy
        x_rec = x_rec.cpu().numpy()

        # Return (remove batch dimension if input was a single image)
        if not batch_input:
            return x_rec[0]
        else:
            return x_rec

    def encode_with_eos(self, image: np.ndarray) -> torch.Tensor:
        """
        Encode an image to token indices and append an EOS token

        Args:
            image: Input image as a numpy array with shape [channels, height, width]

        Returns:
            indices: Tensor of token indices with EOS token appended
        """
        indices = self.encode(image)

        # Create a new tensor with an extra element for EOS
        if indices.ndim == 2:
            # For a single image [height, width]
            # Add EOS as an extra token at the end (flattened)
            flat_indices = indices.flatten()
            with_eos = torch.cat([flat_indices, torch.tensor([self.eos_token_idx], device=self.device)])
            return with_eos
        elif indices.ndim == 3:
            # For a batch of images [batch, height, width]
            # Add EOS as an extra token at the end of each sequence
            batch_size = indices.shape[0]
            flat_indices = indices.reshape(batch_size, -1)
            eos_tokens = torch.full((batch_size, 1), self.eos_token_idx, device=self.device)
            with_eos = torch.cat([flat_indices, eos_tokens], dim=1)
            return with_eos

    def decode_with_eos(self, indices: torch.Tensor, height: int, width: int) -> np.ndarray:
        """
        Decode token indices that include an EOS token

        Args:
            indices: Tensor of token indices with EOS token
            height: Original height of the image
            width: Original width of the image

        Returns:
            image: Reconstructed image as a numpy array
        """
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, device=self.device)

        # Check if batch or single sequence
        if indices.ndim == 1:
            # Single sequence with EOS
            # Find EOS token position
            eos_pos = torch.where(indices == self.eos_token_idx)[0]
            if len(eos_pos) > 0:
                # Keep only tokens before EOS
                indices = indices[:eos_pos[0]]

            # Reshape to 2D (height, width)
            if len(indices) < height * width:
                # Pad with zeros if needed
                padded = torch.zeros(height * width, device=self.device, dtype=indices.dtype)
                padded[:len(indices)] = indices
                indices = padded
            else:
                # Truncate if too long
                indices = indices[:height * width]

            indices = indices.reshape(height, width)

        elif indices.ndim == 2:
            # Batch of sequences with EOS [batch, sequence]
            batch_size = indices.shape[0]

            # Process each sequence in the batch
            reshaped_indices = []
            for i in range(batch_size):
                seq = indices[i]

                # Find EOS token position
                eos_pos = torch.where(seq == self.eos_token_idx)[0]
                if len(eos_pos) > 0:
                    # Keep only tokens before EOS
                    seq = seq[:eos_pos[0]]

                # Reshape to 2D (height, width)
                if len(seq) < height * width:
                    # Pad with zeros if needed
                    padded = torch.zeros(height * width, device=self.device, dtype=seq.dtype)
                    padded[:len(seq)] = seq
                    seq = padded
                else:
                    # Truncate if too long
                    seq = seq[:height * width]

                reshaped_indices.append(seq.reshape(height, width))

            # Stack back into a batch
            indices = torch.stack(reshaped_indices)

        # Decode the processed indices
        return self.decode(indices)

    def get_codebook(self) -> np.ndarray:
        """
        Get the codebook embeddings

        Returns:
            codebook: Numpy array of codebook embeddings
        """
        with torch.no_grad():
            codebook = self.model.vq.embedding.weight.cpu().numpy()
        return codebook

    def get_eos_token(self) -> int:
        """
        Get the EOS token index

        Returns:
            eos_token_idx: Index of the EOS token
        """
        return self.eos_token_idx

    def __len__(self) -> int:
        """
        Get the size of the vocabulary (codebook size + EOS token)

        Returns:
            vocab_size: Size of the vocabulary
        """
        return self.codebook_size + 1  # +1 for EOS token


# Example usage
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="VQCAE Tokenizer Demo")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--image_path", type=str, help="Path to an image file to encode/decode")

    args = parser.parse_args()

    # Initialize tokenizer
    tokenizer = VQCAETokenizer(model_path=args.model_path)

    # Create or load an image
    if args.image_path is None:
        # Create a random example image
        img = np.random.randn(3, 64, 64).astype(np.float32)
    else:
        # Load the image
        img = np.load(args.image_path).astype(np.float32)

        # Ensure it's in [C, H, W] format
        if img.ndim == 3 and img.shape[0] not in [1, 3]:
            img = np.transpose(img, (2, 0, 1))

    # Encode image to tokens
    tokens = tokenizer.encode(img)
    print(f"Encoded tokens shape: {tokens.shape}")

    # Encode with EOS token
    tokens_with_eos = tokenizer.encode_with_eos(img)
    print(f"Tokens with EOS shape: {tokens_with_eos.shape}")

    # Decode tokens back to image
    reconstructed_img = tokenizer.decode(tokens)
    print(f"Reconstructed image shape: {reconstructed_img.shape}")

    # Show a sample of the tokens
    print("\nSample of tokens (5x5 section):")
    h, w = min(5, tokens.shape[0]), min(5, tokens.shape[1])
    print(tokens[:h, :w].cpu().numpy())

    # Plot original and reconstructed image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    if img.shape[0] == 1:
        plt.imshow(img[0], cmap='gray')
    else:
        plt.imshow(np.transpose(img, (1, 2, 0)))

    plt.subplot(1, 2, 2)
    plt.title("Reconstructed")
    if reconstructed_img.shape[0] == 1:
        plt.imshow(reconstructed_img[0], cmap='gray')
    else:
        plt.imshow(np.transpose(reconstructed_img, (1, 2, 0)))

    plt.tight_layout()
    plt.savefig("reconstruction_demo.png")
    plt.show()