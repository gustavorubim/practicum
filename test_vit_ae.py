import torch
import yaml
from anomaly_detection.models import get_model

def test_vit_autoencoder():
    # Create test config matching the default.yaml structure
    config = {
        "model_type": "vit",
        "data": {
            "image_size": [256, 256]
        },
        "models": {
            "vit_ae": {
                "patch_size": 16,
                "dim": 768,
                "depth": 6,
                "heads": 8,
                "dropout": 0.1
            }
        }
    }

    # Create model
    model = get_model(config)
    print(f"Created {model.__class__.__name__} successfully")

    # Test forward pass
    dummy_input = torch.randn(1, 3, 256, 256)  # Batch of 1 RGB image
    output = model(dummy_input)
    
    # Verify output shape matches input
    assert output.shape == dummy_input.shape, \
        f"Output shape {output.shape} doesn't match input {dummy_input.shape}"
    
    print("Forward pass test passed!")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    test_vit_autoencoder()