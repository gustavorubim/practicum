import argparse
import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from anomaly_detection.data.dataset import MVTecDataset
from anomaly_detection.models.base import BaseAutoencoder
from anomaly_detection.utils.metrics import calculate_metrics
from anomaly_detection.utils.logger import setup_logger


class AnomalyDetector:
    """Class for performing inference with trained autoencoder models."""

    def __init__(self, config: Dict[str, Any], model: BaseAutoencoder):
        """Initialize detector.

        Args:
            config: Configuration dictionary
            model: Trained autoencoder model
        """
        self.config = config
        self.model = model
        self.device = torch.device(config["training"]["device"])
        self.model.to(self.device)
        self.model.eval()
        self.logger = setup_logger()

        # Create test dataset
        self.test_dataset = MVTecDataset(
            root=config["data"]["root"],
            split="test",
            image_size=config["data"]["image_size"],
            good_only=False,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config["inference"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["num_workers"],
        )

    def detect_anomalies(self) -> Dict[str, Any]:
        """Run anomaly detection on test set.

        Returns:
            Dictionary containing:
                - metrics: Evaluation metrics
                - samples: List of sample results with images, errors, etc.
        """
        results = {
            "metrics": None,
            "samples": []
        }

        # Calculate metrics
        results["metrics"] = calculate_metrics(
            self.model, self.test_loader, self.device
        )

        # Process individual samples
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                if i >= 10:  # Limit number of samples to process for visualization
                    break

                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                labels = batch["label"].cpu().numpy()
                image_paths = batch["image_path"]

                # Get reconstructions and errors
                reconstructions = self.model(images)
                errors = torch.abs(images - reconstructions)

                # Convert tensors to numpy for visualization
                images_np = images.cpu().numpy().transpose(0, 2, 3, 1)
                reconstructions_np = reconstructions.cpu().numpy().transpose(0, 2, 3, 1)
                errors_np = errors.cpu().numpy().transpose(0, 2, 3, 1)
                masks_np = masks.cpu().numpy().transpose(0, 2, 3, 1)

                # Store sample results
                for j in range(images.size(0)):
                    results["samples"].append({
                        "image_path": image_paths[j],
                        "original": images_np[j],
                        "reconstruction": reconstructions_np[j],
                        "error": errors_np[j],
                        "mask": masks_np[j],
                        "label": labels[j],
                        "anomaly_score": errors[j].mean().item(),
                    })

        return results

    def visualize_results(self, results: Dict[str, Any], output_dir: str) -> None:
        """Generate visualizations of anomaly detection results.

        Args:
            results: Results from detect_anomalies()
            output_dir: Directory to save visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(results["samples"]):
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            # Original image
            axes[0].imshow(sample["original"])
            axes[0].set_title("Original")
            axes[0].axis("off")
            
            # Reconstruction
            axes[1].imshow(sample["reconstruction"])
            axes[1].set_title("Reconstruction")
            axes[1].axis("off")
            
            # Error map
            error_img = axes[2].imshow(sample["error"].mean(axis=-1), cmap="hot")
            plt.colorbar(error_img, ax=axes[2])
            axes[2].set_title("Error Map")
            axes[2].axis("off")
            
            # Ground truth mask (if available)
            if sample["mask"].sum() > 0:
                axes[3].imshow(sample["mask"].squeeze(), cmap="gray")
                axes[3].set_title("Ground Truth")
            else:
                axes[3].imshow(sample["original"])
                axes[3].set_title("Normal Sample")
            axes[3].axis("off")
            
            # Save figure
            fig.suptitle(f"Anomaly Score: {sample['anomaly_score']:.4f}")
            plt.tight_layout()
            plt.savefig(output_path / f"sample_{i}.png")
            plt.close()

    def generate_report(self, results: Dict[str, Any], output_path: str) -> None:
        """Generate a text report of the evaluation metrics.

        Args:
            results: Results from detect_anomalies()
            output_path: Path to save report
        """
        with open(output_path, "w") as f:
            f.write("Anomaly Detection Evaluation Report\n")
            f.write("=================================\n\n")
            
            f.write("Metrics:\n")
            for name, value in results["metrics"].items():
                f.write(f"{name}: {value:.4f}\n")
            
            f.write("\nSample Anomaly Scores:\n")
            for sample in results["samples"]:
                f.write(f"{sample['image_path']}: {sample['anomaly_score']:.4f}\n")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_path: str, config: Dict[str, Any]) -> BaseAutoencoder:
    """Load trained model from checkpoint.

    Args:
        model_path: Path to model checkpoint
        config: Configuration dictionary

    Returns:
        Loaded model instance
    """
    # Determine model type from config
    model_type = config["model_type"] if "model_type" in config else "cnn_ae"
    
    if model_type == "cnn_ae":
        from anomaly_detection.models.cnn_ae import CNNAutoencoder
        model_class = CNNAutoencoder
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model_class.load(model_path)


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, 
                        help="Path to trained model directory")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to config file")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for results")
    args = parser.parse_args()

    # Load config and model
    config = load_config(args.config)
    model = load_model(args.model, config)

    # Run inference
    detector = AnomalyDetector(config, model)
    results = detector.detect_anomalies()

    # Save results
    detector.visualize_results(results, os.path.join(args.output, "visualizations"))
    detector.generate_report(results, os.path.join(args.output, "report.txt"))

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()