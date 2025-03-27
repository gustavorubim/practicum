import cv2
import numpy as np
from typing import Tuple, Optional, Union, List
from pathlib import Path

class PreprocessingPipeline:
    """A pipeline for applying multiple preprocessing steps sequentially."""
    
    def __init__(self):
        self.steps = []
    
    def add_step(self, func: callable) -> 'PreprocessingPipeline':
        """Add a processing step to the pipeline.
        
        Args:
            func: A callable that takes and returns a numpy array
            
        Returns:
            self for method chaining
        """
        self.steps.append(func)
        return self
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply all processing steps sequentially.
        
        Args:
            image: Input image array
            
        Returns:
            Processed image array
        """
        processed = image
        for step in self.steps:
            processed = step(processed)
        return processed

__all__ = [
    'normalize_image',
    'resize_image',
    'to_grayscale',
    'load_image',
    'PreprocessingPipeline'
]

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
    keep_aspect: bool = False,
    padding_mode: str = 'constant',
    padding_value: Union[int, float, Tuple] = 0,
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """Resize image with flexible aspect ratio and padding options.
    
    Args:
        image: Input image array (H,W) or (H,W,C)
        size: Target (height, width)
        keep_aspect: Whether to maintain aspect ratio
        padding_mode: One of ['constant', 'edge', 'reflect', 'symmetric']
        padding_value: Value for constant padding or tuple for per-channel values
        interpolation: OpenCV interpolation method
        
    Returns:
        Resized image with optional padding
    """
    if not keep_aspect:
        return cv2.resize(image, (size[1], size[0]), interpolation=interpolation)
        
    # Calculate aspect ratio preserving dimensions
    h, w = image.shape[:2]
    target_h, target_w = size
    ratio = min(target_h/h, target_w/w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    
    # Resize with specified interpolation
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    # Apply padding if needed
    if new_h < target_h or new_w < target_w:
        border_type = getattr(cv2, f'BORDER_{padding_mode.upper()}')
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left
        
        return cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            border_type, value=padding_value
        )
    return resized

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

class PreprocessingPipeline:
    def __init__(self):
        self.steps = []

    def add_step(self, func):
        self.steps.append(func)
        return self

    def __call__(self, image):
        result = image.copy()
        for step in self.steps:
            result = step(result)
        return result
    """
    image = cv2.imread(str(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)