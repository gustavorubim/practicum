import cv2
import numpy as np
from typing import Tuple, Optional, Union, List
from pathlib import Path

def normalize_image(
    image: np.ndarray, 
    mean: Optional[List[float]] = None, 
    std: Optional[List[float]] = None
) -> np.ndarray:
    """Normalize image with given mean and std or to [0,1] range.
    
    Args:
        image: Input image as numpy array (H,W,C)
        mean: Optional mean values for each channel
        std: Optional std values for each channel
        
    Returns:
        Normalized image
    """
    if mean is not None and std is not None:
        # First normalize to [0,1] then apply mean/std
        normalized = image.astype(np.float32) / 255.0
        return (normalized - np.array(mean)) / np.array(std)
    return image.astype(np.float32) / 255.0

def resize_image(
    image: np.ndarray, 
    size: Tuple[int, int], 
    keep_aspect: bool = False
) -> np.ndarray:
    """Resize image to target size.
    
    Args:
        image: Input image
        size: Target (height, width)
        keep_aspect: Whether to maintain aspect ratio with padding
        
    Returns:
        Resized image
    """
    if keep_aspect:
        # Calculate aspect ratio and pad if needed
        h, w = image.shape[:2]
        target_h, target_w = size
        
        ratio = min(target_h/h, target_w/w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Pad to target size
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2
        return cv2.copyMakeBorder(
            resized, 
            pad_h, target_h - new_h - pad_h,
            pad_w, target_w - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=0
        )
        
    return cv2.resize(image, (size[1], size[0]))

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale if it has 3 channels.
    
    Args:
        image: Input image (H,W) or (H,W,3)
        
    Returns:
        Grayscale image (H,W)
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load image from path with OpenCV and convert to RGB.
    
    Args:
        path: Path to image file
        
    Returns:
        Image as numpy array in RGB format
    """
    image = cv2.imread(str(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)