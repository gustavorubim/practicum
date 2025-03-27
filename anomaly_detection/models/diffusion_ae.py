import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from .base import BaseAutoencoder


class UNetBlock(nn.Module):
    """Basic building block for U-Net architecture."""
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        # Reshape time embedding to match spatial dimensions
        time_emb = self.activation(self.time_mlp(t.unsqueeze(-1)))
        time_emb = time_emb.view(time_emb.size(0), -1, 1, 1)
        h = h + time_emb
        h = self.norm(h)
        h = self.activation(h)
        h = self.conv2(h)
        return h


class UNet(nn.Module):
    """U-Net architecture for noise prediction."""
    def __init__(self, dim: int, channels: int = 3, dim_mults: Tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__()
        self.time_emb_dim = dim * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )
        
        # Downsample blocks
        self.down_blocks = nn.ModuleList()
        in_ch = channels
        for mult in dim_mults:
            out_ch = dim * mult
            self.down_blocks.append(UNetBlock(in_ch, out_ch, self.time_emb_dim))
            in_ch = out_ch
            
        # Middle block
        self.mid_block = UNetBlock(in_ch, in_ch, self.time_emb_dim)
        
        # Upsample blocks
        self.up_blocks = nn.ModuleList()
        for mult in reversed(dim_mults):
            out_ch = dim * mult
            self.up_blocks.append(UNetBlock(in_ch + out_ch, out_ch, self.time_emb_dim))
            in_ch = out_ch
            
        # Final convolution - ensure output matches input channels
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, channels, 1)
        )
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Time embedding
        t = self.time_mlp(t)
        
        # Downsample path
        h = x
        hs = []
        for block in self.down_blocks:
            h = block(h, t)
            hs.append(h)
            h = F.avg_pool2d(h, 2)
            
        # Middle block
        h = self.mid_block(h, t)
        
        # Upsample path
        for block in self.up_blocks:
            h = F.interpolate(h, scale_factor=2, mode='nearest')
            h = torch.cat([h, hs.pop()], dim=1)
            h = block(h, t)
            
        return self.final_conv(h)


class DiffusionAutoencoder(BaseAutoencoder):
    """Diffusion model-based autoencoder."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        diff_config = config["models"]["diffusion_ae"]
        self.timesteps = diff_config["timesteps"]
        self.dim = diff_config["dim"]
        
        # Noise prediction network
        self.unet = UNet(
            dim=self.dim,
            channels=3,
            dim_mults=diff_config["dim_mults"]
        )
        
        # Beta schedule
        self.betas = self._get_beta_schedule(diff_config["beta_schedule"])
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def _get_beta_schedule(self, schedule_type: str) -> torch.Tensor:
        """Generate beta schedule for diffusion process."""
        if schedule_type == "linear":
            scale = 1000 / self.timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, self.timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {schedule_type}")
            
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """For diffusion models, encoding is identity."""
        return x
        
    def decode(self, z: torch.Tensor, timesteps: Optional[int] = None) -> torch.Tensor:
        """Sample from the diffusion model."""
        if timesteps is None:
            timesteps = self.timesteps
            
        # Start from random noise matching input dimensions
        img = torch.randn_like(z)
        
        # Reverse diffusion process
        for t in reversed(range(timesteps)):
            t_tensor = torch.full((z.shape[0],), t, device=z.device, dtype=torch.float32)
            pred_noise = self.unet(img, t_tensor)
            img = self._p_sample(img, pred_noise, t)
            
        # Return only the first sample to match input batch size
        return img[:z.shape[0]]
        
    def _p_sample(self, x: torch.Tensor, noise_pred: torch.Tensor, t: int) -> torch.Tensor:
        """Single reverse diffusion step."""
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]
        
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
            
        mean = (x - beta_t * noise_pred / torch.sqrt(1 - alpha_cumprod_t)) / torch.sqrt(alpha_t)
        variance = torch.sqrt(beta_t) * noise
        
        return mean + variance
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass through the autoencoder (reconstruction via diffusion)."""
        return self.decode(self.encode(x))
    
    