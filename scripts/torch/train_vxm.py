#!/usr/bin/env python

"""
Example script to train a VoxelMorph model.

You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8). pp
    1788-1800. 2019. 

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
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import os
import random
import argparse
import time
import numpy as np
import torch
import math
import json 
import nibabel as nib
# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchio as tio
import monai

import wandb

wandb.init(project="Vxm")
# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--img-list', help='line-seperated list of training files')
parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--model-dir', default='models',
                    help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=15,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--load-model', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

parser.add_argument('--data_json',  help='name of dataset json file')
parser.add_argument('--masked', action='store_true', help='mask input?', default=True)


# loss hyperparameters
parser.add_argument('--image-loss', default='ncc',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                    help='weight of deformation loss (default: 0.01)')
args = parser.parse_args()

bidir = args.bidir

json_file_name = args.data_json

train_dataset =  vxm.nlst.NLST("/vol/pluto/users/raveendr/data/NLST/", json_file_name,
                                downsampled=True, 
                                masked=False,
                             is_norm=True)
valid_set =  vxm.nlst.NLST("/vol/pluto/users/raveendr/data/NLST/", json_file_name,
                                downsampled=True, 
                                masked=False,train=False,
                             is_norm=True)
# train_set_size = int(len(train_dataset) * 0.9)

# valid_set_size = len(train_dataset) - train_set_size
# print("train_set_size: ", train_set_size)
# print("valid_set_size: ", valid_set_size)

# split the train set into two
seed = torch.Generator().manual_seed(42)
# train_set, valid_set = data.random_split(train_dataset, [train_set_size, valid_set_size], 
                                        #  generator=seed)

# overfit_set = torch.utils.data.Subset(train_set, [2])

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
val_dataloader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=4)

print("Number of training samples: ", len(train_dataset))
print("Number of val samples: ", len(valid_set))

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
assert np.mod(args.batch_size, nb_gpus) == 0, \
    'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

if args.load_model:
    # load initial model (if specified)
    model = vxm.networks.VxmDense.load(args.load_model, device)
else:
    # otherwise configure new model
    model = vxm.networks.VxmDense(
        inshape=(224/2,192/2,224/2),
        nb_unet_features=[enc_nf, dec_nf],
        bidir=bidir,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize
    )

if nb_gpus > 1:
    # use multiple GPUs via DataParallel
    model = torch.nn.DataParallel(model)
    model.save = model.module.save

# prepare the model for training and send to device
model.to(device)
model.train()

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = vxm.losses.NCC().loss 

elif args.image_loss == 'mse':
    image_loss_func = vxm.losses.MSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

lncc_loss = monai.losses.LocalNormalizedCrossCorrelationLoss(
    spatial_dims=3, kernel_size=3, kernel_type="rectangular", reduction="mean"
    )
def similarity_loss(lncc_loss,warped_img2, image_pair):
    """Accepts a batch of displacement fields, shape (B,3,H,W,D),
    and a batch of image pairs, shape (B,2,H,W,D)."""
    
    return lncc_loss(warped_img2, image_pair) 

# need two image loss functions if bidirectional
if bidir:
    losses = [image_loss_func, image_loss_func]
    weights = [0.5, 0.5]
else:
    losses = [image_loss_func]
    weights = [1]

# prepare deformation loss
losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
weights += [args.weight]
print("lamda: ", args.weight)
best_validation_loss = float("inf")
# training loops
for epoch in range(args.initial_epoch, args.epochs):
    model.train()    

    epoch_loss = []
    epoch_total_loss = []
    epoch_step_time = []

    # for step in range(args.steps_per_epoch):
    for batch_idx, batch in enumerate(train_dataloader):
        fixed_img = batch["fixed_img"].to(device)
        moving_img = batch["moving_img"].to(device)
        fixed_mask = batch["fixed_mask"].to(device)
        moving_mask = batch["moving_mask"].to(device)
        zero_ff = batch["zero_flow_field"].to(device)
        step_start_time = time.time()
        
        y_pred = model(moving_img, fixed_img) 
        y_true = (fixed_img, zero_ff ) 

        # calculate total loss
        loss = 0
        loss_list = []
        print(losses[1](y_true[1], y_pred[1 ]))
        for n, loss_function in enumerate(losses):
            
            curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
            if math.isnan(curr_loss) == True:
                breakpoint()
            print(n, "curr_loss: ", loss_function)
            print(n, "curr_loss: ", curr_loss)
            
            loss_list.append(curr_loss.item())
            loss += curr_loss
        
        breakpoint()
        # loss = similarity_loss(lncc_loss, y_pred[0], y_true[0])
        loss_list.append(loss.item())
        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.detach().item())
        wandb.log({"train/loss": loss.detach().item()})
        # wandb.log({"train/grad_loss": loss_list[1]})
        wandb.log({"train/ncc_loss": loss_list[0]})
        
        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # get compute time
        epoch_step_time.append(time.time() - step_start_time)

    model.eval()
    with torch.no_grad():
        val_epoch_loss = []
        val_epoch_total_loss = []
        for batch_idx, batch in enumerate(val_dataloader):
            fixed_img = batch["fixed_img"].to(device)
            moving_img = batch["moving_img"].to(device)
            zero_ff = batch["zero_flow_field"].to(device)
            y_pred = model(moving_img, fixed_img, registration=True) 
            y_true = (fixed_img, zero_ff ) 

            # calculate total loss
            val_loss = 0
            val_loss_list = []
            # for n, loss_function in enumerate(losses):
                
            #     curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
            #     if math.isnan(curr_loss) == True:
            #         breakpoint()
                    
            #     val_loss_list.append(curr_loss.item())
            #     val_loss += curr_loss
            
            val_loss = similarity_loss(lncc_loss, y_pred[0], y_true[0])
            val_loss_list.append(val_loss.item())
            val_epoch_loss.append(val_loss_list)
            val_epoch_total_loss.append(val_loss.item())
            wandb.log({"val/loss": val_loss.detach().item()})
        # save model checkpoint
    if np.mean(val_epoch_total_loss) < best_validation_loss:
        best_validation_loss = np.mean(val_epoch_total_loss)
        model.save(os.path.join(model_dir, '%04d.pt' % epoch))
        
        
    # if epoch % 20 == 0:
    #     model.save(os.path.join(model_dir, '%04d.pt' % epoch))
    # print epoch info
    epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
    # wandb.log({"train/epoch": epoch + 1})
    time_info = '%.4f sec/step' % np.mean(epoch_step_time)
    losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = 'train_loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
    
    # val_losses_info = ', '.join(['%.4e' % f for f in np.mean(val_epoch_loss, axis=0)])
    # val_loss_info = 'val_loss: %.4e  (%s)' % (np.mean(val_epoch_total_loss), val_losses_info)
    
    print(' - '.join((epoch_info, time_info, loss_info)), flush=True)
    wandb.log({"train/epoch_loss": np.mean(epoch_total_loss), 'epoch': epoch + 1})
    wandb.log({"val/epoch_loss": np.mean(val_epoch_total_loss), 'epoch': epoch + 1})
    
# final model save
model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))
