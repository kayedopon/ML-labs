import torch
import os

from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from torchsummary import summary
from torch.utils.data.dataloader import DataLoader

import models
import preprocessing
from utils import get_set
from dataset import SkinDataset


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    path = Path(r"C:\studies\uni\semester 4\Machine Learning\lab works\lb9\data")

    train_dir = path / "train"
    test_dir = path / "test"

    classes = os.listdir(train_dir)
    num_cls = len(classes)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    X_train, y_train= get_set(train_dir, class_to_idx)
    X_test, y_test = get_set(test_dir, class_to_idx)

    # kf = KFold(n_splits=5, random_state=42) 

    X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train)
    
    resnet18 = models.get_resNet18(num_cls)
    train_transforms, test_transforms = preprocessing.get_resNet18_transforms()
    
    train_data = SkinDataset(X_train, y_train, train_transforms)
    eval_data = SkinDataset(X_eval, y_eval, test_transforms)
    test_data = SkinDataset(X_test, y_test, test_transforms)

    train_loader = DataLoader(train_data,
                              batch_size=32,
                              shuffle=True,)
    eval_loader = DataLoader(eval_data,
                             batch_size=32)

    print(next(iter(train_loader)))

if __name__ == "__main__":
    main()