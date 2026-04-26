from dataclasses import dataclass
import os
from typing import Optional
import json

import torch
import utils

from torch.utils.data import DataLoader
from torchvision.transforms.v2 import MixUp, CutMix
from dataset import SkinDataset
from engine import eval_step, train


@dataclass
class ExperimentConfig:
    name: str
    model_name: str
    transform_name: str
    loss_name: str
    use_sampler: bool = False
    use_mixup: bool = False
    use_cutmix: bool = False
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3


def run_experiment(config, X_train, y_train, X_eval, y_eval, X_test, y_test,
                   num_classes, device, save_dir):

    print(f"\n===== Running experiment: {config.name} =====")

    model = utils.get_model(config.model_name, num_classes)
    train_transforms, test_transforms = utils.get_transforms(config.transform_name)

    train_data = SkinDataset(X_train, y_train, train_transforms)
    eval_data = SkinDataset(X_eval, y_eval, test_transforms)
    test_data = SkinDataset(X_test, y_test, test_transforms)

    if config.use_sampler:
        sampler = utils.get_sampler(y_train)
        train_loader = DataLoader(
            train_data,
            batch_size=config.batch_size,
            sampler=sampler
        )
    else:
        train_loader = DataLoader(
            train_data,
            batch_size=config.batch_size,
            shuffle=True
        )

    eval_loader = DataLoader(
        eval_data,
        batch_size=config.batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_data,
        batch_size=config.batch_size,
        shuffle=False
    )

    class_weights = utils.get_class_weights(y_train)
    loss_fn = utils.get_loss(config.loss_name, class_weights, device)

    mix_aug = None
    if config.use_mixup:
        mix_aug = MixUp(num_classes=num_classes, alpha=0.2)
    elif config.use_cutmix:
        mix_aug = CutMix(num_classes=num_classes, alpha=0.2)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    results = train(
        model,
        train_loader,
        eval_loader,
        loss_fn,
        optimizer,
        scheduler,
        mix_aug,
        config.epochs,
        device
    )

    charts_path = os.path.join(save_dir, "images", f"{config.name}.png")
    os.makedirs(charts_path, exist_ok=True)

    utils.plot_results(results, config.epochs, save=True, path=charts_path)

    test_results = eval_step(model, test_loader, loss_fn, device)

    save_dir = os.path.join(save_dir, f"{config.model_name}")
    os.makedirs(os.path.join(save_dir), exist_ok=True)

    model_path = os.path.join(save_dir, f"{config.name}.pth")
    torch.save(model.state_dict(), model_path)

    results_path = os.path.join(save_dir, f"{config.name}_results.json")

    save_results = {
        "config": config.__dict__,
        "accuracy_eval": results["accuracy_eval"],
        "f1_eval": results["f1_eval"],
        "roc_auc_eval": results["roc_auc_eval"],
        "loss_eval": results["loss_eval"],
        "accuracy_train": results["accuracy_train"],
        "f1_train": results["f1_train"],
        "loss_train": results["loss_train"],
        "duration": results["duration"],
        "test_accuracy": test_results["accuracy"],
        "test_f1": test_results["f1"],
        "test_roc_auc": test_results["roc_auc"],
        "test_loss": test_results["loss"],
        "test_report": test_results["report"]
    }

    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=4)

    print(f"Saved model to: {model_path}")
    print(f"Saved results to: {results_path}")

    print("\nTest results:")
    print("Accuracy:", test_results["accuracy"])
    print("F1:", test_results["f1"])
    print("ROC-AUC:", test_results["roc_auc"])
    print(test_results["report"])

    return results, test_results