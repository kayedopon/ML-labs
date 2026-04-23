import torch

from torch.utils.data.dataloader import DataLoader
from torch import nn

def train(model: nn.Module,
          train_loader: DataLoader,
          eval_loader: DataLoader,
          loss: nn.Module,
          optim: torch.optim,
          epochs: int,
          device: str):
    pass