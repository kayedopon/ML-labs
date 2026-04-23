import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from torchvision import transforms
from torchsummary import summary
from PIL import Image

from eda import get_dist, get_sizes
import models


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

def get_set(path):
    X, y = [], []
    for cls in os.listdir(path):
        cls_path = os.path.join(path, cls)
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            X.append(Image.open(img_path)), y.append(cls)

    return X, y

def main():
    torch.manual_seed(42)

    path = Path(r"C:\studies\uni\semester 4\Machine Learning\lab works\lb8\data")

    train_dir = path / "train"
    test_dir = path / "test"

    num_cls = len(os.listdir(train_dir))

    X_train, y_train= get_set(train_dir)
    X_test, y_test = get_set(test_dir)

    # kf = KFold(n_splits=5, random_state=42) 

    X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train)

    resnet18 = models.get_resNet50(num_cls)
    


if __name__ == "__main__":
    main()