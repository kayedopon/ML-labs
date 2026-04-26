import torch

import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        """
        alpha: tensor of shape (num_classes,) for class weighting (optional)
        gamma: focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        logits: (B, C)
        targets: (B,) with class indices
        """

        log_probs = F.log_softmax(logits, dim=1)   
        probs = torch.exp(log_probs)               

        targets = targets.long()
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)          

        loss = -((1 - pt) ** self.gamma) * log_pt

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        return loss.mean()