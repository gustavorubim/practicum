import torch
import torch.nn as nn
from transformers import ViTModel
from .base import BaseAutoencoder

class ViTAE(BaseAutoencoder):
    def __init__(self, config):
        super().__init__(config)
        self.patch_size = config.models.vit_ae.patch_size
        self.dim = config.models.vit_ae.dim
        self.depth = config.models.vit_ae.depth
        self.heads = config.models.vit_ae.heads
        
        # Encoder
        self.encoder = ViTModel.from_pretrained(
            'google/vit-base-patch16-224',
            hidden_size=self.dim,
            num_hidden_layers=self.depth,
            num_attention_heads=self.heads
        )
        
        # Decoder (simple linear projection)
        self.decoder = nn.Sequential(
            nn.Linear(self.dim, 3 * self.patch_size**2),
            nn.Unflatten(1, (3, self.patch_size, self.patch_size)),
            nn.Upsample(scale_factor=16, mode='bilinear'),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Reshape and encode
        outputs = self.encoder(x)
        z = outputs.last_hidden_state[:, 0]  # CLS token
        
        # Decode
        x_recon = self.decoder(z)
        return x_recon