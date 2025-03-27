import pytest
import numpy as np
from pathlib import Path
import sys
import os
import cv2
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from anomaly_detection.data.preprocessing import (
    normalize_image,
    resize_image,
    to_grayscale,
    load_image
)

@pytest.fixture
def sample_image():
    """Generate a sample RGB test image."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

@pytest.fixture
def sample_grayscale():
    """Generate a sample grayscale test image."""
    return np.random.randint(0, 255, (100, 100), dtype=np.uint8)

def test_normalize_image(sample_image):
    """Test image normalization functionality."""
    # Test default normalization to [0,1]
    normalized = normalize_image(sample_image)
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
    assert normalized.dtype == np.float32
    
    # Test custom mean/std normalization
    normalized = normalize_image(sample_image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # Check per-channel means are close to 0
    assert np.allclose(normalized.mean(axis=(0,1)), [0.0, 0.0, 0.0], atol=0.1)
    
def test_resize_image(sample_image):
    """Test image resizing functionality."""
    # Test simple resize
    resized = resize_image(sample_image, (50, 50), keep_aspect=False)
    assert resized.shape == (50, 50, 3)
    

def test_preprocessing_pipeline(sample_image):
    """Test the preprocessing pipeline functionality."""
    from anomaly_detection.data.preprocessing import PreprocessingPipeline, normalize_image, resize_image
    
    # Create pipeline with multiple steps
    pipeline = (
        PreprocessingPipeline()
        .add_step(lambda x: resize_image(x, (50, 50)))
        .add_step(normalize_image)
    )
    
    # Process image through pipeline
    processed = pipeline(sample_image)
    
    # Verify results
    assert processed.shape == (50, 50, 3)
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0
    
    # Test empty pipeline
    empty_pipeline = PreprocessingPipeline()
    assert np.allclose(empty_pipeline(sample_image), sample_image)
    # Test aspect ratio preservation
    resized = resize_image(sample_image, (50, 100), keep_aspect=True)
    assert resized.shape == (50, 100, 3)
    
    # Test with padding
    resized = resize_image(
        sample_image,
        (200, 200),
        keep_aspect=True,
        padding_mode='reflect'
    )
    assert resized.shape == (200, 200, 3)
    assert not np.allclose(resized[0,0], 0)  # Should have reflected content
    
    # Test different interpolation methods
    resized = resize_image(
        sample_image,
        (50, 50),
        interpolation=cv2.INTER_NEAREST
    )
    assert resized.shape == (50, 50, 3)
    
    # Test aspect ratio preservation
    resized = resize_image(sample_image, (50, 30), keep_aspect=True)
    assert resized.shape == (50, 30, 3)
    # Check padding was added
    assert np.any(resized == 0)
    
def test_to_grayscale(sample_image, sample_grayscale):
    """Test grayscale conversion."""
    # Test RGB conversion
    gray = to_grayscale(sample_image)
    assert gray.ndim == 2
    assert gray.shape == (100, 100)
    
    # Test passthrough for grayscale
    gray = to_grayscale(sample_grayscale)
    assert gray.ndim == 2
    assert gray.shape == (100, 100)
    
def test_load_image(tmp_path):
    """Test image loading functionality."""
    # Create a test image file
    test_img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    test_path = tmp_path / "test_img.png"
    cv2.imwrite(str(test_path), test_img)
    
    # Test loading
    loaded = load_image(test_path)
    assert loaded.shape == (50, 50, 3)
    assert loaded.dtype == np.uint8
    # Verify RGB conversion
    assert not np.array_equal(loaded, test_img)  # OpenCV loads as BGR