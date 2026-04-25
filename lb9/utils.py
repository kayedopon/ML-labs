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

def plot_results(results, epochs):
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