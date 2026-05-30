"""
Functions and classes for losses and metrics related to evaluation of segmentation models performance for optimization
"""


from torch import Tensor


def dice_loss(p: Tensor, gt_onehot: Tensor, smooth : float =1e-6 , do_bg=False):
    """
    dice loss function for 2d & 3d segmentation.
    """

    if not do_bg:
        gt_onehot = gt_onehot[:, 1:]  # remove background class for dice loss
        p = p[:, 1:]

    axes = tuple(range(2, len(p.shape)))
    intersection = (p * gt_onehot).sum(dim=axes)
    p_sum = p.sum(dim=axes)
    gt_sum = gt_onehot.sum(dim=axes)
    d = (2 * intersection + smooth) / (p_sum + gt_sum + smooth)
    return 1 - d.mean()

