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
import torchshow as ts


# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm   # nopep8
import torchio as tio
from torch.utils.data import DataLoader

# parse commandline args
parser = argparse.ArgumentParser()

parser.add_argument('--model', required=True, help='pytorch model for nonlinear registration')
# parser.add_argument('--warp', help='output warp deformation filename')
# parser.add_argument('--use-mask', help='Set to true to use mask for evaluation')
parser.add_argument('--output_path', help='Path to results')
parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
args = parser.parse_args()

def convert_pytorch_grid2scipy(flow_):
    '''
        Convert from the pytorch grid_sample formulation to the scipy formulation
    '''
    # _, H, W, D = grid.shape
    # # grid_x  = (grid[0, ...] + 1) * (D -1)/2
    # # grid_y = (grid[1, ...] + 1) * (W -1)/2
    # # grid_z = (grid[2, ...] + 1) * (H -1)/2
    
    # grid_x  = grid[0, ...] 
    # grid_y = grid[1, ...] 
    # grid_z = grid[2, ...] 
    
    # # grid = np.stack([grid_z, grid_y, grid_x])
    # grid = np.stack([grid_x, grid_y, grid_z])
    # identity_grid = np.meshgrid(np.arange(H), np.arange(W), np.arange(D), indexing='ij')
    # grid = grid - identity_grid
    
    # # Simple ITK to nibabel grid
    # grid = grid[::-1, ...] 
    # # grid = grid.swapaxes(1, 3)
    
    flow = torch.flip(flow_, [1, 3])
    # flow[0] = -flow[0]
    # flow[2] = -flow[2]
    # flow = flow.permute(0, 1, 3, 2)
    flow = flow[[0, 2, 1]]
    flow = flow.numpy()
    return flow

# device handling
if args.gpu and (args.gpu != '-1'):
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load moving and fixed images
add_feat_axis = not args.multichannel
#https://nipy.org/nibabel/coordinate_systems.html

test_dataset =  vxm.nlst.NLST("/vol/pluto/users/raveendr/data/NLST/", "NLST_dataset_train_test_v1.json",
                                downsampled=False, masked=False,
                            train_transform=None, is_norm=True, train=False)

val_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# load and set up model
model = vxm.networks.VxmDense.load(args.model, device)
model.to(device)
model.eval()
out_path = args.output_path

os.makedirs(out_path, exist_ok=True)
os.makedirs(out_path + "/" + "moved_imgs", exist_ok=True)
os.makedirs(out_path + "/" + "disp_field", exist_ok=True)


for batch_idx, batch in enumerate(val_dataloader):
    
    val1 = batch["fixed_name"][0][-16:-12]
    val2 = batch["moving_name"][0][-16:-12]   
    

    input_fixed = batch["fixed_img"].to(device)
    
    input_moving = batch["moving_img"].to(device)
    fixed_affine = batch["fixed_affine"][0]

    # predict
    moved, warp = model(input_moving, input_fixed, registration=True)

    # save results
    moved = moved.detach().cpu().numpy().squeeze()
    
    moved_path = os.path.join(out_path + '/moved_imgs', f'moved_{str(val1).zfill(4)}_{str(val2).zfill(4)}.nii.gz')
    warped_path = os.path.join(out_path + '/disp_field', f'flow_{str(val1).zfill(4)}_{str(val2).zfill(4)}.nii.gz')
    vxm.py.utils.save_volfile(moved, moved_path, fixed_affine)
    flow = warp.detach().cpu().squeeze().numpy()
    vxm.py.utils.save_volfile(flow, warped_path, fixed_affine)
    
    #Ref: https://learn2reg.grand-challenge.org/Submission/ 
    # The convention used for displacement fields depends on scipy's map_coordinates() 
    # function, expecting displacement fields in the format [ X, Y, Z,[x, y, z],], 
    # where X, Y, Z and x, y, z represent voxel displacements and image dimensions, respectively.
    # Pytorch's grid_sample expects displacement fields in the format [X, Y, Z, [z, y, x]] and normalized coordinates between -1 and 1. 
    # warp shape: (1, 3, H, W, D), scipy's map_coordinates expects (H, W, D, 3) 
    # so we need to permute and flip
    
    # shape = warp.shape[2:]
    # nib_disp_field = warp.detach()
    # print(shape)
    # for i in range(len(shape)):
    #     nib_disp_field[:, i, ...] = nib_disp_field[ :,i, ...] / 2
    #     nib_disp_field[ :,i, ...] = nib_disp_field[:, i, ...] + 0.5
    #     nib_disp_field[:, i, ...] = nib_disp_field[:, i, ...] * (shape[i] - 1)

    # nib_disp_field = nib_disp_field.permute(0,2,3,4,1).flip(-1).float().squeeze().cpu()
    # nib_disp_field=((warp.detach().permute(0,2,3,4,1))).flip(-1).float().squeeze().cpu()
    # nib.save(nib.Nifti1Image(nib_disp_field.numpy(), np.eye(4)), os.path.join(out_path, f'disp_{str(val1).zfill(4)}_{str(val2).zfill(4)}.nii.gz'))
    # print(f'Saved disp_{str(val1).zfill(4)}_{str(val2).zfill(4)}.nii.gz')
    
    del input_fixed, input_moving, moved, warp


