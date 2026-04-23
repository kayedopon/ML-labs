import matplotlib.pyplot as plt
import os

from PIL import Image

from eda import get_dist, get_sizes


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

def get_set(path, class_to_idx):
    X, y = [], []
    for cls in os.listdir(path):
        cls_path = os.path.join(path, cls)
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            X.append(Image.open(img_path).convert("RGB"))
            y.append(class_to_idx[cls])
    return X, y