#!/usr/bin/env python

"""
Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register 
a scan (moving) to an atlas (fixed). To register a scan to the atlas and save the warp field, run:

    register.py --moving moving.nii.gz --fixed fixed.nii.gz --model model.pt 
        --moved moved.nii.gz --warp warp.nii.gz

The source and target input images are expected to be affinely registered.

If you use this code, please cite the following, and read function docs for further info/citations
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration 
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in 
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or 
implied. See the License for the specific language governing permissions and limitations under 
the License.
"""

import os
import argparse

# third party
import numpy as np
import nibabel as nib
import torch
from eval_metrics import EvaluationMetrics


# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm   # nopep8
import torchio as tio

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
# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--moving', required=True, help='moving image (source) filename')
parser.add_argument('--fixed', required=True, help='fixed image (target) filename')
parser.add_argument('--moving_mask', required=True, help='moving image (source) filename')
parser.add_argument('--fixed_mask', required=True, help='fixed image (target) filename')
parser.add_argument('--moved', required=False, help='warped image output filename')
parser.add_argument('--model', required=True, help='pytorch model for nonlinear registration')
parser.add_argument('--warp', help='output warp deformation filename')
parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
args = parser.parse_args()

# device handling
if args.gpu and (args.gpu != '-1'):
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load moving and fixed images
add_feat_axis = not args.multichannel

def image_norm(img):
    max_v = np.max(img)
    min_v = np.min(img)

    norm_img = (img - min_v) / (max_v - min_v)
    return norm_img

moving = vxm.py.utils.load_volfile(args.moving, add_batch_axis=False, add_feat_axis=add_feat_axis)
fixed, fixed_affine = vxm.py.utils.load_volfile(
    args.fixed, add_batch_axis=False, add_feat_axis=add_feat_axis, ret_affine=True)

moving_mask = vxm.py.utils.load_volfile(args.moving_mask, add_batch_axis=False, add_feat_axis=add_feat_axis)
fixed_mask, fixed_affine = vxm.py.utils.load_volfile(
    args.fixed_mask, add_batch_axis=False, add_feat_axis=add_feat_axis, ret_affine=True)

# rescale = tio.RescaleIntensity(out_min_max=(0, 1))
# transforms = [rescale]
# transform = tio.Compose(transforms)

rescale = tio.RescaleIntensity(percentiles=(0.5, 99.5))

# HOUNSFIELD_AIR, HOUNSFIELD_BONE = -1000, 1000
# clamp = tio.Clamp(out_min=HOUNSFIELD_AIR, out_max=HOUNSFIELD_BONE)


preprocess_intensity = tio.Compose([
rescale,
])


# load and set up model
model = vxm.networks.VxmDense.load(args.model, device)
model.to(device)
model.eval()



# set up tensors and permute
#shape of moving: (1, 224, 192, 224, 1)
# input_moving = torch.from_numpy(moving).to(device).float().permute(0, 4, 1, 2, 3)
# input_fixed = torch.from_numpy(fixed).to(device).float().permute(0, 4, 1, 2, 3)

# moving_mask = torch.from_numpy(moving_mask).to(device).permute(0, 4, 1, 2, 3)
# fixed_mask = torch.from_numpy(fixed_mask).to(device).permute(0, 4, 1, 2, 3)
print(moving.shape)

input_fixed = image_norm(fixed)
input_moving = image_norm(moving)

input_moving = torch.from_numpy(input_moving).to(device).float().permute(3,0, 1, 2)
input_fixed = torch.from_numpy(input_fixed).to(device).float().permute(3,0, 1, 2)

moving_mask = torch.from_numpy(moving_mask).to(device).permute(3,0, 1, 2)
fixed_mask = torch.from_numpy(fixed_mask).to(device).permute(3,0, 1, 2)




# input_moving =  preprocess_intensity(input_moving)
# input_fixed =  preprocess_intensity(input_fixed)
print(torch.max(input_moving))
print(torch.min(input_moving))
print(torch.max(input_fixed))
print(torch.min(input_fixed))
input_moving = input_moving.unsqueeze(0)
input_fixed = input_fixed.unsqueeze(0)

# input_moving = input_moving * moving_mask
# input_fixed = input_fixed * fixed_mask

# predict
moved, warp = model(input_moving, input_fixed, registration=True)

det = jacobian_determinant(warp.squeeze(0).detach().numpy())

print(torch.max(moved))
print(torch.min(moved))
print(torch.max(warp))
print(torch.min(warp))

print(f"number of folds: {(det<=0).sum()}")
# save moved image
if args.moved:
    moved = moved.detach().cpu().numpy().squeeze()
    
    vxm.py.utils.save_volfile(moved, args.moved, fixed_affine)

# save warp
if args.warp:
    warp = warp.detach().cpu().numpy().squeeze()
    vxm.py.utils.save_volfile(warp, args.warp, fixed_affine)

