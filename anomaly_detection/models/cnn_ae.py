import torch
import torch.nn as nn
from typing import Dict, Any

from .base import BaseAutoencoder


class CNNAutoencoder(BaseAutoencoder):
    """CNN-based autoencoder for anomaly detection."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize CNN autoencoder.

        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def _build_encoder(self) -> nn.Sequential:
        """Build encoder network.

        Returns:
            Encoder network as nn.Sequential
        """
        channels = self.config["channels"]
        layers = []
        in_channels = 3  # RGB input

        # Build encoder layers
        for out_channels in channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(self.config.get("dropout", 0.1))
            ])
            in_channels = out_channels

        # Final encoding to latent space
        layers.append(
            nn.Conv2d(in_channels, self.config["latent_dim"], kernel_size=4, stride=1, padding=0)
        )

        return nn.Sequential(*layers)

    def _build_decoder(self) -> nn.Sequential:
        """Build decoder network.

        Returns:
            Decoder network as nn.Sequential
        """
        channels = self.config["channels"][::-1]  # Reverse encoder channels
        layers = []
        in_channels = self.config["latent_dim"]

        # Initial upsampling from latent space
        layers.append(
            nn.ConvTranspose2d(in_channels, channels[0], kernel_size=4, stride=1, padding=0)
        )
        layers.append(nn.BatchNorm2d(channels[0]))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Dropout2d(self.config.get("dropout", 0.1)))
        in_channels = channels[0]

        # Build decoder layers
        for out_channels in channels[1:]:
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(self.config.get("dropout", 0.1))
            ])
            in_channels = out_channels

        # Final layer to reconstruct image
        layers.extend([
            nn.ConvTranspose2d(in_channels, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in [-1, 1] range
        ])

        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input image to latent space.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Latent representation tensor
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to image space.

        Args:
            z: Latent tensor

        Returns:
            Reconstructed image tensor of shape (B, 3, H, W)
        """
        return self.decoder(z)