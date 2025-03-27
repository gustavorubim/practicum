import torch
import torch.nn as nn
from torchvision.models import resnet18
from .base import BaseAutoencoder

class ResNetAE(BaseAutoencoder):
    def __init__(self, config):
        super().__init__(config)
        self.latent_dim = config.models.resnet_ae.latent_dim
        self.base_model = config.models.resnet_ae.base_model
        
        # Encoder
        self.encoder = self._build_encoder()
        
        # Decoder
        self.decoder = self._build_decoder()
        
    def _build_encoder(self):
        """Create encoder from pretrained ResNet"""
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(512, self.latent_dim)
        return model
        
    def _build_decoder(self):
        """Create decoder with transposed convolutions"""
        return nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.Unflatten(1, (512, 1, 1)),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon