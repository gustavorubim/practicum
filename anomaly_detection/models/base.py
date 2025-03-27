import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
from pathlib import Path
import yaml

class BaseAutoencoder(nn.Module, ABC):
    """Abstract base class for all autoencoder implementations."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input image to latent space representation.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Latent representation tensor
        """
        pass
        
    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to image space.
        
        Args:
            z: Latent tensor
            
        Returns:
            Reconstructed image tensor of shape (B, C, H, W)
        """
        pass
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Reconstructed image tensor
        """
        z = self.encode(x)
        return self.decode(z)
        
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward pass."""
        return self.forward(x)
        
    def calculate_reconstruction_error(
        self, 
        x: torch.Tensor, 
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """Calculate reconstruction error between input and output.
        
        Args:
            x: Input tensor
            reduction: How to reduce the error ('mean', 'sum', 'none')
            
        Returns:
            Reconstruction error tensor
        """
        x_recon = self.reconstruct(x)
        error = torch.abs(x - x_recon)
        if reduction == 'mean':
            return error.mean()
        elif reduction == 'sum':
            return error.sum()
        return error
        
    def save(self, path: str) -> None:
        """Save model weights and config to disk.
        
        Args:
            path: Path to save directory
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / "model.pt")
        with open(Path(path) / "config.yaml", 'w') as f:
            yaml.dump(self.config, f)
            
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'BaseAutoencoder':
        """Load model from disk.
        
        Args:
            path: Path to model directory
            device: Device to load model onto
            
        Returns:
            Loaded model instance
        """
        with open(Path(path) / "config.yaml", 'r') as f:
            config = yaml.safe_load(f)
            
        model = cls(config).to(device)
        model.load_state_dict(torch.load(Path(path) / "model.pt", map_location=device))
        return model