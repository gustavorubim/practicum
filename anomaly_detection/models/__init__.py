from typing import Dict, Any
from .base import BaseAutoencoder
from .cnn_ae import CNNAutoencoder
from .resnet_ae import ResNetAE
from .vit_ae import ViTAutoencoder
from .diffusion_ae import DiffusionAutoencoder

def get_model(config: Dict[str, Any]) -> BaseAutoencoder:
    """Model factory that creates appropriate autoencoder based on config.
    
    Args:
        config: Configuration dictionary containing model parameters
        
    Returns:
        Instance of the requested autoencoder model
    """
    if not isinstance(config, dict):
        raise ValueError("Config must be a dictionary")
        
    model_type = config.get("model_type", "cnn").lower()
    
    models = {
        "cnn": CNNAutoencoder,
        "resnet": ResNetAE,
        "vit": ViTAutoencoder,
        "diffusion": DiffusionAutoencoder,
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return models[model_type](config)