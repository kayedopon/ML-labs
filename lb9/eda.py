import os
import torch
from pathlib import Path
from PIL import Image
from collections import Counter

from torchvision import transforms


def get_dist(train_path: Path):
    counts = {}

    for dir in train_path.iterdir():
        if dir.is_dir():
            counts[dir.name] = len(os.listdir(dir))

    return counts

def get_sizes(train_path: Path):
    sizes = []
    for cls in train_path.iterdir():
        if cls.is_dir():
            for img_path in cls.glob(r"*"):
                with Image.open(img_path) as img:
                    sizes.append(img.size)
    return sizes

def get_statistics(train_path: Path):
    transform = transforms.ToTensor()

    mean = 0
    std = 0
    total_images = 0

    for cls in train_path.iterdir():
        if cls.is_dir():
            for img_path in cls.glob(r"*"):
                with Image.open(img_path) as img:
                    img = transform(img)

                    mean += img.mean()
                    std += img.mean()
                    total_images += 1

    mean /= total_images
    std /= total_images

    return mean, std

def get_class_weights(y_train):
    counts = Counter(y_train)

    class_counts = torch.tensor(
        [counts[i] for i in range(len(counts))],
        dtype=torch.float
    )

    weights = 1.0 / class_counts
    print(weights)
    weights = weights / weights.sum() * len(class_counts)
    print(weights)
    return weights

