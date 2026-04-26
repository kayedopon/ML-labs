from collections import Counter
from PIL import Image
from torch.utils.data import WeightedRandomSampler

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import models
import preprocessing

from eda import get_dist, get_sizes
from loss import FocalLoss


def plot_charts(train_dir):
    counts = get_dist(train_dir)
    sizes = get_sizes(train_dir)

    plt.bar(counts.keys(), counts.values())
    plt.xticks(rotation=45)
    plt.title("Class Distribution (Train)")
    plt.show()

    width = [s[0] for s in sizes]
    height = [s[1] for s in sizes]

    plt.hist(width, bins=20, alpha=0.5, label="Width")
    plt.hist(height, bins=20, alpha=0.5, label="Height")
    plt.legend()
    plt.title("Image resolution distribution")
    plt.show()

def plot_results(results, epochs, save=False, path=None):
    keys = [k for k in results.keys() if k not in ["report", "duration"]]
    n = len(keys)
    cols = 2
    rows = (n + 1) // 2

    fig, axis = plt.subplots(rows, cols, figsize=(17, 10))

    for i, (k, v) in enumerate(results.items()):
        if k in ["report", "duration"]:
            continue
        row = i // cols
        col = i % cols

        axis[row, col].plot(list(range(epochs)), v)
        axis[row, col].set_xlabel('Epoch')
        axis[row, col].set_ylabel(f'{k} score')
        axis[row, col].set_xticks(range(epochs))
        axis[row, col].set_title(f"{k} ({'Train' if 'train' in k else 'Eval'})")
    plt.tight_layout()
    
    if save == True:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()

def get_set(path, class_to_idx):
    X, y = [], []
    for cls in os.listdir(path):
        cls_path = os.path.join(path, cls)
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            X.append(Image.open(img_path).convert("RGB"))
            y.append(class_to_idx[cls])
    return X, y

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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_model(model_name, num_classes):
    if model_name == "resnet18":
        return models.get_resNet18(num_classes)

    elif model_name == "resnet50":
        return models.get_resNet50(num_classes)

    elif model_name == "effnet_b0":
        return models.get_effNet_b0(num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")
    
def get_transforms(transform_name):
    if transform_name == "resnet18":
        return preprocessing.get_resNet18_transforms()

    elif transform_name == "resnet50":
        return preprocessing.get_resNet50_transforms()

    elif transform_name == "effnet_b0":
        return preprocessing.get_effnet_b0_transforms()

    else:
        raise ValueError(f"Unknown transforms: {transform_name}")
    
def get_loss(loss_name, class_weights, device):
    if loss_name == "ce":
        return torch.nn.CrossEntropyLoss()

    elif loss_name == "weighted_ce":
        return torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    elif loss_name == "focal":
        return FocalLoss(alpha=None, gamma=2.0)

    elif loss_name == "weighted_focal":
        return FocalLoss(alpha=class_weights.to(device), gamma=2.0)

    else:
        raise ValueError(f"Unknown loss: {loss_name}")
    
def get_sampler(y_train):
    class_counts = Counter(y_train)

    sample_weights = torch.tensor(
        [1.0 / class_counts[label] for label in y_train],
        dtype=torch.float
    )

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )