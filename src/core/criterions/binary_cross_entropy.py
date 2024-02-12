import torch 
import torch.nn as nn

from torch.nn import functional as F


def bce_loss(probs, targets, reduction='none'):
    """
    cross entropy loss in pytorch.
    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
    # loss = - (targets * torch.log(probs) + (1. - targets) * torch.log(1. - probs))
    # if reduction == 'none':
    #     return loss 
    # else:
    #     return loss.mean()
    # probs = torch.clamp(probs, min=1e-6, max=1. - 1e-6)
    return F.binary_cross_entropy(probs, targets, reduction=reduction)


class BCELoss(nn.Module):
    """
    Wrapper for ce loss
    """
    def forward(self, logits, targets, reduction='none'):
        return bce_loss(logits, targets, reduction)