import numpy as np
import cv2
import random
from typing import Tuple, Union, Callable
from dataclasses import dataclass

@dataclass
class AugmentationConfig:
    """Configuration for image augmentations"""
    rotation_range: Tuple[float, float] = (-15, 15)
    scale_range: Tuple[float, float] = (0.9, 1.1)
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    flip_prob: float = 0.5
    noise_std: float = 0.02

def random_rotation(
    image: np.ndarray,
    angle_range: Tuple[float, float] = (-15, 15)
) -> np.ndarray:
    """Apply random rotation to image.
    
    Args:
        image: Input image (H,W,C)
        angle_range: Min/max rotation angle in degrees
        
    Returns:
        Rotated image
    """
    angle = random.uniform(*angle_range)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def random_flip(
    image: np.ndarray,
    prob: float = 0.5
) -> np.ndarray:
    """Randomly flip image horizontally.
    
    Args:
        image: Input image
        prob: Probability of flipping
        
    Returns:
        Flipped or original image
    """
    if random.random() < prob:
        return cv2.flip(image, 1)
    return image

def random_brightness_contrast(
    image: np.ndarray,
    brightness_range: Tuple[float, float] = (0.8, 1.2),
    contrast_range: Tuple[float, float] = (0.8, 1.2)
) -> np.ndarray:
    """Adjust brightness and contrast randomly.
    
    Args:
        image: Input image
        brightness_range: Min/max brightness multiplier
        contrast_range: Min/max contrast multiplier
        
    Returns:
        Adjusted image
    """
    brightness = random.uniform(*brightness_range)
    contrast = random.uniform(*contrast_range)
    return cv2.addWeighted(image, contrast, np.zeros_like(image), 0, brightness-1)

def random_noise(
    image: np.ndarray,
    std: float = 0.02
) -> np.ndarray:
    """Add Gaussian noise to image.
    
    Args:
        image: Input image
        std: Standard deviation of noise
        
    Returns:
        Noisy image
    """
    noise = np.random.normal(0, std, image.shape)
    return np.clip(image + noise, 0, 1)

def apply_augmentations(
    image: np.ndarray,
    config: AugmentationConfig
) -> np.ndarray:
    """Apply all configured augmentations to image.
    
    Args:
        image: Input image
        config: Augmentation configuration
        
    Returns:
        Augmented image
    """
    if len(image.shape) == 2:
        image = np.expand_dims(image, -1)
        
    # Apply augmentations in fixed order
    image = random_rotation(image, config.rotation_range)
    image = random_flip(image, config.flip_prob)
    image = random_brightness_contrast(
        image,
        config.brightness_range,
        config.contrast_range
    )
    if config.noise_std > 0:
        image = random_noise(image, config.noise_std)
        
    return np.clip(image, 0, 1)