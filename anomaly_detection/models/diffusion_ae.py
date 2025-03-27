import torch
import torch.nn as nn
from .base import BaseAutoencoder
from diffusers import UNet2DModel
from huggingface_hub import hf_hub_download

class DiffusionAE(BaseAutoencoder):
    def __init__(self, config):
        super().__init__(config)
        self.timesteps = config.models.diffusion_ae.timesteps
        self.dim = config.models.diffusion_ae.dim
        self.dim_mults = config.models.diffusion_ae.dim_mults
        
        # U-Net for diffusion
        self.model = UNet2DModel(
            sample_size=config.data.image_size[0],
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=[self.dim * m for m in self.dim_mults],
            norm_num_groups=8
        )
        
        # Noise scheduler
        self.betas = torch.linspace(1e-4, 0.02, self.timesteps)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
    def forward(self, x, t=None):
        if t is None:
            t = torch.randint(0, self.timesteps, (x.shape[0],)).to(x.device)
            
        # Add noise
        noise = torch.randn_like(x)
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        noisy = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * noise
        
        # Predict noise
        pred_noise = self.model(noisy, t).sample
        
        # Return both reconstruction and noise prediction
        return pred_noise, noise