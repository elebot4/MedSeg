"""
Functions and classes for losses and metrics related to evaluation of segmentation models performance for optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(p, gt_onehot, smooth=1e-6, do_bg=False):
    """
    dice loss function for 2d & 3d segmentation.
    """

    if not do_bg:
        gt_onehot = gt_onehot[:, 1:] # remove background class for dice loss
        p = p[:, 1:] 

    axes = tuple(range(2, len(p.shape)))
    intersection = (p * gt_onehot).sum(dim = axes)
    p_sum = p.sum(dim = axes)
    gt_sum = gt_onehot.sum(dim = axes)
    d =  (2 * intersection + smooth) / (p_sum + gt_sum + smooth)
    return 1 - d.mean() 


#def get_loss_fn(args=None):
#    """
#    Simple loss function factory. Currently just returns weighted CE + Dice.
#    """
#    def loss_fn(logits, targets):
#        # Standard medical segmentation: weighted CrossEntropy + Dice
#        ce_loss = F.cross_entropy(logits, targets) 
#        dice_loss_val = dice_loss(logits, F.one_hot(targets, num_classes=logits.shape[1]).permute(0, -1, *range(1, targets.ndim)).float())
#        return 0.5 * ce_loss + 0.5 * dice_loss_val
#    
#    return loss_fn


if __name__ == "__main__":
    # test the loss functions with dummy data
    #loss_fn = get_loss_fn()
    
    logits = torch.randn(2, 2, 4, 4)
    targets = torch.randint(0, 2, (2, 4, 4)).long()
    
    #loss_value = loss_fn(logits, targets)
    #print("Loss value:", loss_value.item())

