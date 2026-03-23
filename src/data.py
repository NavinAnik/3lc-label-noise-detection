"""Data loading, noise injection, and sample extraction for CIFAR-10."""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def load_cifar10(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Dataset, Dataset]:
    """Load CIFAR-10 with transforms.

    Returns:
        train_loader, val_loader, train_dataset, val_dataset
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    )

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = CIFAR10(
        root=str(data_path),
        train=True,
        download=True,
        transform=train_transform,
    )

    val_dataset = CIFAR10(
        root=str(data_path),
        train=False,
        download=True,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, train_dataset, val_dataset


def inject_label_noise(
    labels: np.ndarray,
    noise_rate: float,
    num_classes: int = 10,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly flip a fraction of labels to other classes.

    Args:
        labels: Ground-truth labels (int array).
        noise_rate: Fraction of labels to corrupt (0.0 to 1.0).
        num_classes: Number of classes.
        seed: Random seed.

    Returns:
        noisy_labels: Corrupted labels.
        noise_mask: Boolean array, True where label was corrupted.
    """
    rng = np.random.default_rng(seed)
    noisy_labels = labels.copy()
    n = len(labels)

    n_noisy = int(n * noise_rate)
    if n_noisy == 0:
        return noisy_labels, np.zeros(n, dtype=bool)

    noisy_indices = rng.choice(n, size=n_noisy, replace=False)
    noise_mask = np.zeros(n, dtype=bool)
    noise_mask[noisy_indices] = True

    for idx in noisy_indices:
        true_label = labels[idx]
        # Pick a different class uniformly at random
        other_classes = [c for c in range(num_classes) if c != true_label]
        noisy_labels[idx] = rng.choice(other_classes)

    return noisy_labels, noise_mask


def get_sample_images(
    dataset: Dataset,
    indices: list[int],
    denormalize: bool = True,
) -> list[Image.Image]:
    """Extract PIL images from dataset for display.

    Args:
        dataset: CIFAR-10 dataset (or Subset).
        indices: Sample indices.
        denormalize: If True, reverse normalization for display.

    Returns:
        List of PIL Images.
    """
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    images = []
    for idx in indices:
        item = dataset[idx]
        if isinstance(item, (tuple, list)):
            x = item[0]
        else:
            x = item

        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        else:
            x = np.array(x)

        # C, H, W -> H, W, C
        if x.shape[0] == 3:
            x = np.transpose(x, (1, 2, 0))

        if denormalize:
            x = x * std + mean
            x = np.clip(x, 0, 1)

        x = (x * 255).astype(np.uint8)
        images.append(Image.fromarray(x))

    return images


def create_noisy_dataset(
    dataset: Dataset,
    noisy_labels: np.ndarray,
) -> "NoisyLabelDataset":
    """Wrap dataset with noisy labels for training."""
    return NoisyLabelDataset(dataset, noisy_labels)


class NoisyLabelDataset(Dataset):
    """Dataset that overrides labels with noisy versions."""

    def __init__(self, base_dataset: Dataset, noisy_labels: np.ndarray):
        self.base_dataset = base_dataset
        self.noisy_labels = noisy_labels

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        item = self.base_dataset[idx]
        if isinstance(item, (tuple, list)):
            x, _ = item
            return x, int(self.noisy_labels[idx])
        return item
