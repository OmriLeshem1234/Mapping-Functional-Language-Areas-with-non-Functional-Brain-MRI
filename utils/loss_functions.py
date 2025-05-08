import torch
import torch.nn as nn
from monai.losses import DiceCELoss


class WeightDecayL1(nn.Module):
    """Computes L1 weight decay for a model."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(model: nn.Module) -> torch.Tensor:
        norm = sum(torch.linalg.norm(p.flatten(), 1) for p in model.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        return norm / total_params if total_params > 0 else torch.tensor(0.0, device=model.device)


class GMLoss(nn.Module):
    """Generalized Loss for gray matter segmentation."""

    def __init__(self, p: int = 2, epsilon: float = 1e-6, include_background: bool = False):
        super().__init__()
        self.p = p
        self.epsilon = epsilon
        self.include_background = include_background

    def forward(self, pred: torch.Tensor, gm: torch.Tensor) -> torch.Tensor:
        _pred = pred if self.include_background else pred[:, 1:, ...]
        numerator = torch.sum((_pred ** self.p) * gm)
        denominator = torch.sum(_pred ** self.p) + self.epsilon
        return 1 - (numerator / denominator)


class AnatomyLoss(nn.Module):
    """Combines DiceCELoss and GMLoss for anatomy-aware segmentation."""

    def __init__(self, include_background: bool = False, _lambda: float = 1.0):
        super().__init__()
        self.dice_ce = DiceCELoss(include_background=include_background)
        self.gm_loss = GMLoss(include_background=include_background)
        self._lambda = _lambda

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, gm: torch.Tensor) -> torch.Tensor:
        gm_loss = self.gm_loss(pred=pred, gm=gm) if self._lambda > 0 else torch.tensor(0.0, device=pred.device)
        return self.dice_ce(pred, gt) + self._lambda * gm_loss


class TrainingLoss(nn.Module):
    """Full training loss combining anatomy loss and weight decay."""

    def __init__(self, include_background: bool = False, _lambda: float = 1.0, weight_decay_factor: float = 0.0):
        super().__init__()
        self.anatomy_loss = AnatomyLoss(include_background=include_background, _lambda=_lambda)
        self.weight_decay = WeightDecayL1()
        self.weight_decay_factor = weight_decay_factor

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, gm: torch.Tensor, model: nn.Module) -> torch.Tensor:
        weight_decay = self.weight_decay(model) if self.weight_decay_factor > 0 else torch.tensor(0.0,
                                                                                                  device=pred.device)
        return self.anatomy_loss(pred, gt, gm) + self.weight_decay_factor * weight_decay
