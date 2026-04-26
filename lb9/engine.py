import torch
import time

from torch.utils.data.dataloader import DataLoader
from torch import nn

from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, classification_report


def train_step(model: nn.Module,
               loader: DataLoader,
               loss_fn: nn.Module,
               optim: torch.optim,
               device: str):

    results = {
        "accuracy": 0.0,
        "f1": 0.0,
        "roc_auc": 0.0,
        "loss": 0.0,
    }

    all_preds = []
    all_probs = []
    all_labels = []

    model.train()
    for X, y in loader:
        X, y = X.to(device), y.to(device)

        logits = model(X)
        loss = loss_fn(logits, y)
        optim.zero_grad()
        loss.backward()
        optim.step()

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        results["loss"] += loss.item()
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())
        all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).detach().cpu().numpy()
    all_labels = torch.cat(all_labels).numpy()

    results["accuracy"] = balanced_accuracy_score(all_labels, all_preds)
    results["loss"] /= len(loader)
    results["roc_auc"] = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    results["f1"] = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return results

def eval_step(model: nn.Module,
               loader: DataLoader,
               loss_fn: nn.Module,
               device: str):

    results = {
        "accuracy": 0.0,
        "f1": 0.0,
        "roc_auc": 0.0,
        "loss": 0.0,
        "report": None
    }

    all_preds = []
    all_probs = []
    all_labels = []

    model.eval()
    with torch.inference_mode():
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            loss = loss_fn(logits, y)

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            results["loss"] += loss.item()
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())

        all_preds = torch.cat(all_preds).cpu().numpy()
        all_probs = torch.cat(all_probs).detach().cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()

        results["accuracy"] = balanced_accuracy_score(all_labels, all_preds)
        results["loss"] /= len(loader)
        results["f1"] = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        results["roc_auc"] = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        results["report"] = classification_report(all_labels, all_preds, zero_division=0)

    return results

def train(model: nn.Module,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        loss: nn.Module,
        optim: torch.optim,
        scheduler: torch.optim.lr_scheduler,
        mixup,
        epochs: int,
        device: str):
    results = {
        "accuracy_train": [],
        "f1_train": [],
        "roc_auc_train": [],
        "loss_train": [],
        "accuracy_eval": [],
        "f1_eval": [],
        "roc_auc_eval": [],
        "loss_eval": [],
        "report": []
    }

    model.to(device)
    start = time.time()
    for epoch in range(epochs):
        train_res = train_step(model, train_loader, loss, optim, device)
        eval_res = eval_step(model, eval_loader, loss, device)
        scheduler.step(eval_res["f1"])

        print(f"Epoch: {epoch+1} | f1_train: {train_res['f1']:.2f} | loss_train: {train_res['loss']:.4f} | "
              f"f1_eval: {eval_res['f1']:.2f} | roc_auc_eval: {eval_res['roc_auc']:.4f} | loss_eval: {eval_res['loss']:.4f}")


        results["accuracy_train"].append(train_res["accuracy"])
        results["f1_train"].append(train_res["f1"])
        results["loss_train"].append(train_res["loss"])
        results["roc_auc_train"].append(train_res["roc_auc"])

        results["accuracy_eval"].append(eval_res["accuracy"])
        results["f1_eval"].append(eval_res["f1"])
        results["roc_auc_eval"].append(eval_res["roc_auc"])
        results["loss_eval"].append(eval_res["loss"])
        
        results["report"].append(eval_res["report"])
    results["duration"] = time.time() - start

    return results

def inference(model, test_loader, device):
    model.eval()
    
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for X, y in test_loader:
            X = X.to(device)

            logits = model(X)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

    return all_preds, all_labels