import torch
import os
import utils
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from torchsummary import summary
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from experiment import ExperimentConfig, run_experiment
from utils import get_set, plot_results, set_seed
from models import get_resNet18
from preprocessing import get_resNet18_transforms
from dataset import SkinDataset
from engine import inference


def start_experiment():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    data_path = Path(
        r"C:\studies\uni\semester 4\Machine Learning\lab works\lb9\data"
    )

    save_dir = r"C:\studies\uni\semester 4\Machine Learning\lab works\lb9\results"

    train_dir = data_path / "train"
    test_dir = data_path / "test"

    classes = sorted(os.listdir(train_dir))
    num_classes = len(classes)

    class_to_idx = {
        cls_name: i
        for i, cls_name in enumerate(classes)
    }

    X_train, y_train = get_set(train_dir, class_to_idx)
    X_test, y_test = get_set(test_dir, class_to_idx)

    X_train, X_eval, y_train, y_eval = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        stratify=y_train,
        random_state=42
    )

    experiments = [
        ExperimentConfig(
            name="effnet_b0_weighted_ce",
            model_name="effnet_b0",
            transform_name="effnet_b0",
            loss_name="weighted_ce",
            use_sampler=False,
            use_mixup=False,
            use_cutmix=False,
            epochs=20,
            lr=1e-3
        ),

        ExperimentConfig(
            name="effnet_b0_mixup_ce",
            model_name="effnet_b0",
            transform_name="effnet_b0",
            loss_name="ce",
            use_sampler=False,
            use_mixup=True,
            use_cutmix=False,
            epochs=20,
            lr=1e-3
        ),

        ExperimentConfig(
            name="resnet18_weighted_focal",
            model_name="resnet18",
            transform_name="resnet18",
            loss_name="weighted_focal",
            use_sampler=False,
            use_mixup=False,
            use_cutmix=False,
            epochs=20,
            lr=1e-3
        ),

        ExperimentConfig(
            name="resnet18_sampler_ce",
            model_name="resnet18",
            transform_name="resnet18",
            loss_name="ce",
            use_sampler=True,
            use_mixup=False,
            use_cutmix=False,
            epochs=20,
            lr=1e-3
        ),
    ]

    all_results = []

    for config in experiments:
        results, test_results = run_experiment(
            config,
            X_train,
            y_train,
            X_eval,
            y_eval,
            X_test,
            y_test,
            num_classes,
            device,
            save_dir
        )

        all_results.append({
            "name": config.name,
            "best_eval_f1": max(results["f1_eval"]),
            "best_eval_auc": max(results["roc_auc_eval"]),
            "test_f1": test_results["f1"],
            "test_auc": test_results["roc_auc"],
            "test_accuracy": test_results["accuracy"],
            "duration": results["duration"]
        })

    print("\n===== Summary =====")
    for r in all_results:
        print(r)

def main():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    
    model_path = r"results\resnet18\resnet18_weighted_focal.pth"

    test_dir = r"data\Test"
    classes = sorted(os.listdir(test_dir))
    num_classes = len(classes)

    class_to_idx = {
        cls_name: i
        for i, cls_name in enumerate(classes)
    }

    X_test, y_test = get_set(test_dir, class_to_idx) 
    model = get_resNet18(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)

    _, test_transforms = utils.get_transforms("resnet18")

    test_data = SkinDataset(X_test, y_test, test_transforms)

    test_loader = DataLoader(
        test_data,
        batch_size=32,
    )

    preds, labels = inference(model, test_loader, device)

    utils.plot_confusion_matrix(labels, preds, classes) 

if __name__ == "__main__":
    main()