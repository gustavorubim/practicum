import os
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from anomaly_detection.data.dataset import create_datasets
from anomaly_detection.models.base import BaseAutoencoder
from anomaly_detection.utils.logger import setup_logger
from anomaly_detection.utils.metrics import calculate_metrics


class Trainer:
    """Class to handle training of autoencoder models."""

    def __init__(self, config: Dict[str, Any], model: BaseAutoencoder):
        """Initialize trainer.

        Args:
            config: Configuration dictionary
            model: Autoencoder model to train
        """
        self.config = config
        self.model = model
        self.device = torch.device(config["training"]["device"])
        self.model.to(self.device)

        # Setup logging
        self.logger = setup_logger()
        self.writer = SummaryWriter(log_dir=config["training"]["log_dir"])

        # Create datasets
        self.train_dataset, self.val_dataset, _ = create_datasets(config)

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=True,
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
        )

        # Setup optimizer and loss
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        self.criterion = nn.MSELoss()

        # Setup scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config["training"]["epochs"],
            eta_min=1e-6,
        )

        # Training state
        self.best_loss = float("inf")
        self.epochs_no_improve = 0
        self.checkpoint_dir = Path(config["training"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int) -> float:
        """Train model for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            reconstructions = self.model(images)
            loss = self.criterion(reconstructions, images)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Log batch progress
            if batch_idx % 10 == 0:
                self.logger.info(
                    f"Train Epoch: {epoch} [{batch_idx * len(images)}/{len(self.train_loader.dataset)} "
                    f"({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )

        avg_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar("Loss/train", avg_loss, epoch)
        return avg_loss

    def validate(self, epoch: int) -> float:
        """Validate model on validation set.

        Args:
            epoch: Current epoch number

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                reconstructions = self.model(images)
                loss = self.criterion(reconstructions, images)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar("Loss/val", avg_loss, epoch)

        # Calculate additional metrics
        metrics = calculate_metrics(self.model, self.val_loader, self.device)
        for name, value in metrics.items():
            self.writer.add_scalar(f"Metrics/{name}", value, epoch)

        return avg_loss

    def early_stopping(self, val_loss: float) -> bool:
        """Check if training should stop early.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.epochs_no_improve = 0
            # Save best model
            self.model.save(self.checkpoint_dir / "best")
            return False
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.config["training"]["patience"]:
                return True
            return False

    def train(self):
        """Main training loop."""
        start_time = time.time()

        for epoch in range(1, self.config["training"]["epochs"] + 1):
            # Train and validate
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            # Update learning rate
            self.scheduler.step()

            # Check for early stopping
            if self.early_stopping(val_loss):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

            # Save checkpoint
            if epoch % 10 == 0:
                self.model.save(self.checkpoint_dir / f"epoch_{epoch}")

            self.logger.info(
                f"Epoch: {epoch}\tTrain Loss: {train_loss:.6f}\tVal Loss: {val_loss:.6f}"
            )

        # Save final model
        self.model.save(self.checkpoint_dir / "final")
        self.writer.close()

        total_time = time.time() - start_time
        self.logger.info(f"Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")


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


def main():
    """Main training function."""
    # Load configuration
    config = load_config("anomaly_detection/config/default.yaml")

    # Initialize model (replace with actual model class)
    from anomaly_detection.models.cnn_ae import CNNAutoencoder
    model = CNNAutoencoder(config["models"]["cnn_ae"])

    # Train model
    trainer = Trainer(config, model)
    trainer.train()


if __name__ == "__main__":
    main()