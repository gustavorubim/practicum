import torch
import yaml
from anomaly_detection.models import get_model

def test_diffusion_autoencoder():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """Test the diffusion autoencoder implementation."""
    # Load config
    with open("anomaly_detection/config/default.yaml") as f:
        config = yaml.safe_load(f)
    
    # Create model - explicitly specify diffusion model and move to device
    config["model_type"] = "diffusion"
    model = get_model(config).to(device)
    print(f"Created {model.__class__.__name__} successfully")
    
    # Test forward pass with GPU support and memory management
    with torch.no_grad():
        x = torch.randn(1, 3, 64, 64, device=device)  # Smaller test image on correct device
        out = model(x)
    
    # Verify output shape matches input
    assert out.shape == x.shape, f"Expected output shape {x.shape}, got {out.shape}"
    print("Forward pass test passed!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

if __name__ == "__main__":
    test_diffusion_autoencoder()