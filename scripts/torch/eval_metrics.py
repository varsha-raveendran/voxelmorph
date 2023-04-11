# Same as medical-image-registration/eval/metrics.py

import torch
import numpy as np
from monai.metrics import compute_hausdorff_distance
from monai.metrics import DiceMetric, HausdorffDistanceMetric

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import matplotlib.pyplot as plt
import monai
import torch


def preview_image(image_array, normalize_by="volume", cmap=None, figsize=(12, 12), threshold=None, title=""):
    """
    Display three orthogonal slices of the given 3D image.

    image_array is assumed to be of shape (H,W,D)

    If a number is provided for threshold, then pixels for which the value
    is below the threshold will be shown in red
    """
    if normalize_by == "slice":
        vmin = None
        vmax = None
    elif normalize_by == "volume":
        vmin = 0
        vmax = image_array.max().item()
    else:
        raise (ValueError(f"Invalid value '{normalize_by}' given for normalize_by"))

    # half-way slices
    x, y, z = np.array(image_array.shape) // 2
    imgs = (image_array[x, :, :], image_array[:, y, :], image_array[:, :, z])

    fig, axs = plt.subplots(1, 3, figsize=figsize)
#     fig.suptitle(title)
    for ax, im in zip(axs, imgs):
        ax.axis("off")
        ax.imshow(im, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)

        # threshold will be useful when displaying jacobian determinant images;
        # we will want to clearly see where the jacobian determinant is negative
        if threshold is not None:
            red = np.zeros(im.shape + (4,))  # RGBA array
            red[im <= threshold] = [1, 0, 0, 1]
            ax.imshow(red, origin="lower")
    
    
    plt.show()


def plot_2D_vector_field(vector_field, downsampling):
    """Plot a 2D vector field given as a tensor of shape (2,H,W).

    The plot origin will be in the lower left.
    Using "x" and "y" for the rightward and upward directions respectively,
      the vector at location (x,y) in the plot image will have
      vector_field[1,y,x] as its x-component and
      vector_field[0,y,x] as its y-component.
    """
    downsample2D = monai.networks.layers.factories.Pool["AVG", 2](kernel_size=downsampling)
    vf_downsampled = downsample2D(vector_field.unsqueeze(0))[0]
    plt.quiver(
        vf_downsampled[1, :, :],
        vf_downsampled[0, :, :],
        angles="xy",
        scale_units="xy",
        scale=downsampling,
        headwidth=4.0,
    )


def preview_3D_vector_field(vector_field, downsampling=None):
    """
    Display three orthogonal slices of the given 3D vector field.

    vector_field should be a tensor of shape (3,H,W,D)

    Vectors are projected into the viewing plane, so you are only seeing
    their components in the viewing plane.
    """

    if downsampling is None:
        # guess a reasonable downsampling value to make a nice plot
        downsampling = max(1, int(max(vector_field.shape[1:])) >> 5)

    x, y, z = np.array(vector_field.shape[1:]) // 2  # half-way slices
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.axis("off")
    plot_2D_vector_field(vector_field[[1, 2], x, :, :], downsampling)
    plt.subplot(1, 3, 2)
    plt.axis("off")
    plot_2D_vector_field(vector_field[[0, 2], :, y, :], downsampling)
    plt.subplot(1, 3, 3)
    plt.axis("off")
    plot_2D_vector_field(vector_field[[0, 1], :, :, z], downsampling)
    plt.show()


def plot_2D_deformation(vector_field, grid_spacing, **kwargs):
    """
    Interpret vector_field as a displacement vector field defining a deformation,
    and plot an x-y grid warped by this deformation.

    vector_field should be a tensor of shape (2,H,W)
    """
    _, H, W = vector_field.shape
    grid_img = np.zeros((H, W))
    grid_img[np.arange(0, H, grid_spacing), :] = 1
    grid_img[:, np.arange(0, W, grid_spacing)] = 1
    grid_img = torch.tensor(grid_img, dtype=vector_field.dtype).unsqueeze(0)  # adds channel dimension, now (C,H,W)
    warp = monai.networks.blocks.Warp(mode="bilinear", padding_mode="zeros")
    grid_img_warped = warp(grid_img.unsqueeze(0), vector_field.unsqueeze(0))[0]
    plt.imshow(grid_img_warped[0], origin="lower", cmap="gist_gray")


def preview_3D_deformation(vector_field, grid_spacing, **kwargs):
    """
    Interpret vector_field as a displacement vector field defining a deformation,
    and plot warped grids along three orthogonal slices.

    vector_field should be a tensor of shape (3,H,W,D)
    kwargs are passed to matplotlib plotting

    Deformations are projected into the viewing plane, so you are only seeing
    their components in the viewing plane.
    """
    x, y, z = np.array(vector_field.shape[1:]) // 2  # half-way slices
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.axis("off")
    plot_2D_deformation(vector_field[[1, 2], x, :, :], grid_spacing, **kwargs)
    plt.subplot(1, 3, 2)
    plt.axis("off")
    plot_2D_deformation(vector_field[[0, 2], :, y, :], grid_spacing, **kwargs)
    plt.subplot(1, 3, 3)
    plt.axis("off")
    plot_2D_deformation(vector_field[[0, 1], :, :, z], grid_spacing, **kwargs)
    plt.show()


def jacobian_determinant(vf):
    """
    Given a displacement vector field vf, compute the jacobian determinant scalar field.

    vf is assumed to be a vector field of shape (3,H,W,D),
    and it is interpreted as the displacement field.
    So it is defining a discretely sampled map from a subset of 3-space into 3-space,
    namely the map that sends point (x,y,z) to the point (x,y,z)+vf[:,x,y,z].
    This function computes a jacobian determinant by taking discrete differences in each spatial direction.

    Returns a numpy array of shape (H-1,W-1,D-1).
    """

    _, H, W, D = vf.shape

    # Compute discrete spatial derivatives
    def diff_and_trim(array, axis):
        return np.diff(array, axis=axis)[:, : (H - 1), : (W - 1), : (D - 1)]

    dx = diff_and_trim(vf, 1)
    dy = diff_and_trim(vf, 2)
    dz = diff_and_trim(vf, 3)

    # Add derivative of identity map
    dx[0] += 1
    dy[1] += 1
    dz[2] += 1

    # Compute determinant at each spatial location
    det = (
        dx[0] * (dy[1] * dz[2] - dz[1] * dy[2])
        - dy[0] * (dx[1] * dz[2] - dz[1] * dx[2])
        + dz[0] * (dx[1] * dy[2] - dy[1] * dx[2])
    )

    return det

#https://github.com/adalca/pystrum/blob/0e7a47e5cc62725dfadc728351b89162defca696/pystrum/pynd/ndutils.py#L111

def ndgrid(*args, **kwargs):
    """
    Disclaimer: This code is taken directly from the scitools package [1]
    Since at the time of writing scitools predominantly requires python 2.7 while we work with 3.5+
    To avoid issues, we copy the quick code here.
    Same as calling ``meshgrid`` with *indexing* = ``'ij'`` (see
    ``meshgrid`` for documentation).
    Ref : https://github.com/adalca/pystrum/blob/0e7a47e5cc62725dfadc728351b89162defca696/pystrum/pynd/ndutils.py#L208
    """
    kwargs['indexing'] = 'ij'
    return np.meshgrid(*args, **kwargs)

def volsize2ndgrid(volsize):
    """
    return the dense nd-grid for the volume with size volsize
    essentially return the ndgrid fpr
    """
    ranges = [np.arange(e) for e in volsize]
    return ndgrid(*ranges)



def flow_jacdet(flow):

    vol_size = flow.shape[:-1]
    grid = np.stack(volsize2ndgrid(vol_size), len(vol_size))  
    J = np.gradient(flow + grid)

    dx = J[0]
    dy = J[1]
    dz = J[2]

    Jdet0 = dx[:,:,:,0] * (dy[:,:,:,1] * dz[:,:,:,2] - dy[:,:,:,2] * dz[:,:,:,1])
    Jdet1 = dx[:,:,:,1] * (dy[:,:,:,0] * dz[:,:,:,2] - dy[:,:,:,2] * dz[:,:,:,0])
    Jdet2 = dx[:,:,:,2] * (dy[:,:,:,0] * dz[:,:,:,1] - dy[:,:,:,1] * dz[:,:,:,0])

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet


def plot_against_epoch_numbers(epoch_and_value_pairs, **kwargs):
    """
    Helper to reduce code duplication when plotting quantities that vary over training epochs

    epoch_and_value_pairs: An array_like consisting of pairs of the form (<epoch number>, <value of thing to plot>)
    kwargs are forwarded to matplotlib.pyplot.plot
    """
    array = np.array(epoch_and_value_pairs)
    plt.plot(array[:, 0], array[:, 1], **kwargs)
    plt.xlabel("epochs")

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
    pred and target are of shape (N, C, H, W, D), 
    C is the number of classes or list of (C, H, W, D).
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
        
        
        