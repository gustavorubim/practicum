import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, Any
from torch.utils.data import DataLoader


def calculate_metrics(
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    device: torch.device
) -> Dict[str, float]:
    """Calculate various anomaly detection metrics.
    
    Args:
        model: Autoencoder model
        dataloader: DataLoader for evaluation
        device: Device to run calculations on
        
    Returns:
        Dictionary of metric names and values
    """
    model.eval()
    image_scores = []
    pixel_scores = []
    labels = []
    pixel_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            labels.extend(batch["label"].cpu().numpy())
            
            # Get reconstructions and errors
            reconstructions = model(images)
            errors = torch.abs(images - reconstructions)
            
            # Image-level scores (mean error per image)
            image_errors = errors.view(errors.size(0), -1).mean(dim=1)
            image_scores.extend(image_errors.cpu().numpy())
            
            # Pixel-level scores (only for anomalous images)
            if masks.sum() > 0:
                pixel_scores.extend(errors[masks.bool()].cpu().numpy())
                pixel_labels.extend(masks[masks.bool()].cpu().numpy())

    # Convert to numpy arrays
    labels = np.array(labels)
    image_scores = np.array(image_scores)
    
    # Calculate image-level metrics
    metrics = {
        "image_auroc": roc_auc_score(labels, image_scores),
        "image_aupr": average_precision_score(labels, image_scores),
    }

    # Calculate pixel-level metrics if we have anomalous samples
    if len(pixel_scores) > 0:
        pixel_scores = np.array(pixel_scores)
        pixel_labels = np.array(pixel_labels)
        
        metrics.update({
            "pixel_auroc": roc_auc_score(pixel_labels, pixel_scores),
            "pixel_aupr": average_precision_score(pixel_labels, pixel_scores),
        })

    return metrics


def calculate_reconstruction_metrics(
    original: torch.Tensor, 
    reconstructed: torch.Tensor
) -> Dict[str, float]:
    """Calculate reconstruction quality metrics.
    
    Args:
        original: Original images
        reconstructed: Reconstructed images
        
    Returns:
        Dictionary of reconstruction metrics
    """
    mse = torch.nn.functional.mse_loss(original, reconstructed)
    l1 = torch.nn.functional.l1_loss(original, reconstructed)
    
    return {
        "mse": mse.item(),
        "l1": l1.item(),
    }