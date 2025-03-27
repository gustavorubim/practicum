from .base import BaseAutoencoder
from .cnn_ae import CNNAutoencoder as CNNAE
from .resnet_ae import ResNetAE
from .vit_ae import ViTAE
# Temporarily disabled due to dependency issues
# from .diffusion_ae import DiffusionAE

def get_model(config):
    if not isinstance(config, dict):
        raise ValueError("Config must be a dictionary")
        
    model_type = config.get("model_type", "cnn").lower()
    
    models = {
        "cnn": CNNAE,
        "resnet": ResNetAE,
        "vit": ViTAE,
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return models[model_type](config["models"][f"{model_type}_ae"])