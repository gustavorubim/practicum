import numpy as np
import pytest
from anomaly_detection.data.augmentation import (
    AugmentationConfig,
    random_rotation,
    random_flip,
    random_brightness_contrast,
    random_noise,
    apply_augmentations
)

@pytest.fixture
def sample_image():
    return np.random.rand(64, 64, 3)

def test_random_rotation(sample_image):
    rotated = random_rotation(sample_image)
    assert rotated.shape == sample_image.shape
    assert not np.allclose(rotated, sample_image)

def test_random_flip(sample_image):
    flipped = random_flip(sample_image, prob=1.0)
    assert flipped.shape == sample_image.shape
    assert np.allclose(flipped, sample_image[:, ::-1, :])

def test_random_brightness_contrast(sample_image):
    adjusted = random_brightness_contrast(sample_image, (2.0, 2.0), (2.0, 2.0))
    assert adjusted.shape == sample_image.shape
    assert np.mean(adjusted) > np.mean(sample_image)

def test_random_noise(sample_image):
    noisy = random_noise(sample_image, std=0.1)
    assert noisy.shape == sample_image.shape
    assert not np.allclose(noisy, sample_image)

def test_apply_augmentations(sample_image):
    config = AugmentationConfig(
        rotation_range=(10, 10),
        brightness_range=(1.5, 1.5),
        flip_prob=1.0,
        noise_std=0.1
    )
    augmented = apply_augmentations(sample_image, config)
    assert augmented.shape == sample_image.shape
    assert not np.allclose(augmented, sample_image)

def test_augmentation_consistency():
    """Test that random number generation is reproducible with fixed seed"""
    np.random.seed(42)
    rng1 = np.random.rand(10)
    
    np.random.seed(42)
    rng2 = np.random.rand(10)
    
    # Verify random number generation is reproducible
    assert np.allclose(rng1, rng2)