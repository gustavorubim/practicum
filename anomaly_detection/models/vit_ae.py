import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import Dict, Any

from .base import BaseAutoencoder


class PatchEmbedding(nn.Module):
    """Convert input image to sequence of patch embeddings."""
    def __init__(self, image_size: int = 256, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, embed_dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.position_embeddings
        return x


class TransformerEncoder(nn.Module):
    """Multi-layer transformer encoder."""
    def __init__(self, dim: int, depth: int, heads: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                activation='gelu'
            )
            for _ in range(depth)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ViTAutoencoder(BaseAutoencoder):
    """Vision Transformer based autoencoder for anomaly detection."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        vit_config = config["models"]["vit_ae"]
        
        # Encoder components
        self.patch_embed = PatchEmbedding(
            image_size=config["data"]["image_size"][0],
            patch_size=vit_config["patch_size"],
            in_channels=3,
            embed_dim=vit_config["dim"]
        )
        self.encoder = TransformerEncoder(
            dim=vit_config["dim"],
            depth=vit_config["depth"],
            heads=vit_config["heads"],
            mlp_dim=vit_config["dim"] * 4,
            dropout=vit_config["dropout"]
        )
        
        # Decoder components
        self.decoder_embed = nn.Linear(vit_config["dim"], vit_config["dim"])
        self.decoder_pos_embed = nn.Parameter(
            torch.randn(1, self.patch_embed.num_patches + 1, vit_config["dim"])
        )
        self.decoder = TransformerEncoder(
            dim=vit_config["dim"],
            depth=vit_config["depth"] // 2,
            heads=vit_config["heads"],
            mlp_dim=vit_config["dim"] * 4,
            dropout=vit_config["dropout"]
        )
        self.decoder_pred = nn.Linear(
            vit_config["dim"],
            vit_config["patch_size"]**2 * 3,
            bias=True
        )
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input image to latent space representation."""
        latent = self.patch_embed(x)
        latent = self.encoder(latent)
        return latent
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to image space."""
        # Project to decoder dimension
        x = self.decoder_embed(z)
        
        # Add position embeddings
        x = x + self.decoder_pos_embed
        
        # Transformer decoder
        x = self.decoder(x)
        
        # Predict pixel values for each patch
        x = self.decoder_pred(x)
        
        # Remove class token and reshape to image
        patches = x[:, 1:, :]
        h = w = int(patches.shape[1] ** 0.5)
        img = rearrange(
            patches,
            'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            h=h, w=w,
            p1=self.patch_embed.patch_size,
            p2=self.patch_embed.patch_size
        )
        return torch.sigmoid(img)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass through the autoencoder."""
        z = self.encode(x)
        return self.decode(z)