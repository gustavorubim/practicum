import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2


class MVTecDataset(Dataset):
    """Dataset class for loading MVTec anomaly detection data."""

    def __init__(
        self,
        root: str,
        category: str = "zipper",
        split: str = "train",
        transform: Optional[Compose] = None,
        good_only: bool = True,
        image_size: Tuple[int, int] = (256, 256),
    ):
        """Initialize MVTec dataset.

        Args:
            root: Path to dataset root directory
            category: MVTec category (e.g., 'zipper')
            split: 'train' or 'test'
            transform: Albumentations transforms
            good_only: Whether to only load good/normal samples
            image_size: Target image size (H, W)
        """
        self.root = Path(root) / category
        self.split = split
        self.good_only = good_only
        self.image_size = image_size
        self.transform = transform or self.default_transform()
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict[str, str]]:
        """Load dataset samples with paths to images and masks.

        Returns:
            List of sample dictionaries with 'image_path' and 'mask_path'
        """
        samples = []

        if self.split == "train":
            # Training only uses good samples
            good_dir = self.root / "train" / "good"
            for img_path in good_dir.glob("*.png"):
                samples.append({
                    "image_path": str(img_path),
                    "mask_path": None,
                    "label": 0,  # 0 = normal
                })
        else:
            # Test set includes both good and anomalous samples
            good_dir = self.root / "test" / "good"
            if not self.good_only:
                # Add anomalous samples
                for defect_type in os.listdir(self.root / "test"):
                    if defect_type == "good":
                        continue
                    defect_dir = self.root / "test" / defect_type
                    mask_dir = self.root / "ground_truth" / defect_type
                    for img_path in defect_dir.glob("*.png"):
                        mask_path = mask_dir / f"{img_path.stem}_mask.png"
                        samples.append({
                            "image_path": str(img_path),
                            "mask_path": str(mask_path) if mask_path.exists() else None,
                            "label": 1,  # 1 = anomalous
                        })

            # Add good test samples
            for img_path in good_dir.glob("*.png"):
                samples.append({
                    "image_path": str(img_path),
                    "mask_path": None,
                    "label": 0,
                })

        return samples

    def default_transform(self) -> Compose:
        """Create default transform pipeline.

        Returns:
            Albumentations Compose object
        """
        return Compose([
            Resize(height=self.image_size[0], width=self.image_size[1]),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load image
        image = cv2.imread(sample["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        transformed = self.transform(image=image)
        image = transformed["image"]

        # Load mask if available
        mask = torch.zeros(1, *self.image_size, dtype=torch.float32)
        if sample["mask_path"] is not None:
            mask_img = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
            mask_img = cv2.resize(mask_img, self.image_size[::-1])
            mask[0] = torch.from_numpy(mask_img / 255.0)

        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "image_path": sample["image_path"],
        }


def create_datasets(
    config: Dict[str, Any]
) -> Tuple[MVTecDataset, MVTecDataset, MVTecDataset]:
    """Create train, validation, and test datasets.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_cfg = config["data"]
    
    # Full training dataset (will be split into train/val)
    full_train = MVTecDataset(
        root=data_cfg["root"],
        split="train",
        image_size=data_cfg["image_size"],
        good_only=True,
    )

    # Split into train and validation
    train_size = int(data_cfg["train_split"] * len(full_train))
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(data_cfg["seed"]),
    )

    # Test dataset
    test_dataset = MVTecDataset(
        root=data_cfg["root"],
        split="test",
        image_size=data_cfg["image_size"],
        good_only=False,
    )

    return train_dataset, val_dataset, test_dataset