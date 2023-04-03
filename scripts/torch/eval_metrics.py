# Same as medical-image-registration/eval/metrics.py

import torch
import numpy as np
from monai.metrics import compute_hausdorff_distance
from monai.metrics import DiceMetric, HausdorffDistanceMetric


def dice_coef(y_true, y_pred):
        # smooth = 1.
        # iflat = y_true.view(-1)        
        # tflat = y_pred.view(-1)        
        # intersection = (iflat * tflat).sum()
    
        # return ((2. * intersection + smooth) /
        #       (iflat.sum() + tflat.sum() + smooth))
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return dice

# def compute_hd95(fixed,moving,moving_warped,labels):
#     hd95 = []
#     for i in labels:
#         if ((fixed==i).sum()==0) or ((moving==i).sum()==0):
#             hd95.append(np.NAN)
#         else:
#             hd95.append(compute_robust_hausdorff(compute_surface_distances((fixed==i), (moving_warped==i), np.ones(3)), 95.))
#     mean_hd95 =  np.nanmean(hd95)
#     return mean_hd95,hd95

"""
Computes the Hausdorff distance between two one-hot encoded tensors.
target and pred are of shape (N, C, H, W, D), C is the number of classes.

Returns: a tensor of shape (N, C) containing the Hausdorff distance for each class.
"""
def compute_hd95(pred, target ,percentile=95, include_background=True):
    return compute_hausdorff_distance(pred, target, percentile=percentile, include_background=include_background)

"""
Class to compute the evaluation metrics for a batch of images.
"""
class EvaluationMetrics():
    def __init__(self, include_background=False):
        
        self.dice_metric = DiceMetric(include_background=include_background, reduction="mean", get_not_nans=False)

        self.hd95 = HausdorffDistanceMetric(include_background=include_background, distance_metric='euclidean', 
                                            percentile=95, directed=False, get_not_nans=False)
    
    """
    Function to compute the evaluation metrics for a batch of images.
    pred and target are of shape (N, C, H, W, D), C is the number of classes or list of (C, H, W, D).
    """
    def evaluate(self, pred, target):
        self.dice_metric(y_pred=pred, y=target)
        self.hd95(y_pred=pred, y=target)
    
    """
    Returns: a tuple of the dice and hd95 scores.
    """
    def aggregate(self):
        dice = self.dice_metric.aggregate().item()
        hd95 = self.hd95.aggregate().item()
        
        self.dice_metric.reset()
        self.dice_metric.reset()
        
        return dice, hd95
        
        
        