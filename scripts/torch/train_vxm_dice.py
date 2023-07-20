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
import torchinfo
import bad_grad_viz as bad_grad


import wandb


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

# loss hyperparameters
parser.add_argument('--image-loss', default='ncc',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                    help='weight of deformation loss (default: 0.01)')
parser.add_argument('--masked', action='store_true', help='mask input?')
parser.add_argument('--dice_weight', type=float, dest='dice_weight', default=0.01,
                    help='weight of dice loss (default: 0.01)')
args = parser.parse_args()

wandb.init(project="Vxm - dice", config=args)

bidir = args.bidir

json_file_name = args.data_json

train_dataset =  vxm.nlst.NLST("/vol/pluto/users/raveendr/data/NLST/", json_file_name,
                                downsampled=True, 
                                masked=args.masked,
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
# overfit_set = torch.utils.data.Subset(train_dataset, [2]*20)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
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

print(enc_nf)
print(dec_nf)

if args.load_model:
    # load initial model (if specified)
    model = vxm.networks_dice.VxmDenseSemisupervised.load(args.load_model, device)
else:
    # otherwise configure new model
    model = vxm.networks_dice.VxmDenseSemisupervised(
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
    

text_file = open(model_dir + "/model_summary.txt", "w")
 
#write string to file
summary = str(torchinfo.summary(model, [(1, 224//2,192//2,224//2), (1, 224//2,192//2,224//2),(1, 224//2,192//2,224//2)], batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose = 1))
text_file.write(summary) 
#close file
text_file.close()
# prepare the model for training and send to device
model.to(device)
model.train()

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
lncc_loss = monai.losses.LocalNormalizedCrossCorrelationLoss(
    spatial_dims=3, kernel_size=3, kernel_type="rectangular", reduction="mean"
    )
def similarity_loss(lncc_loss,warped_img2, image_pair):
    """Accepts a batch of displacement fields, shape (B,3,H,W,D),
    and a batch of image pairs, shape (B,2,H,W,D)."""
    
    return lncc_loss(warped_img2, image_pair) 

# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = vxm.losses.NCC().loss 
    # image_loss_func = monai.losses.LocalNormalizedCrossCorrelationLoss(
    #     spatial_dims=3,
    #     kernel_size=3,
    #     kernel_type='rectangular',
    #     reduction="mean", smooth_nr=0.0, smooth_dr=1e-6
    # )
elif args.image_loss == 'mse':
    image_loss_func = vxm.losses.MSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

# need two image loss functions if bidirectional
if bidir:
    losses = [image_loss_func, image_loss_func]
    weights = [0.5, 0.5]
else:
    losses = [image_loss_func]
    weights = [1]

best_validation_loss = float("inf")

# prepare deformation loss
# losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
# weights += [args.weight]

grad_loss = vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss
grad_wt = args.weight
print("grad_wt: ", grad_wt)
dice_loss = vxm.losses.Dice().loss

# dice_loss = monai.losses.DiceLoss(include_background=True, to_onehot_y=True)
dice_weight = args.dice_weight
# dice_loss = monai.losses.MultiScaleLoss(dice_loss, scales=[0, 1, 2])
transformer = vxm.layers.SpatialTransformer((224//2,192//2,224//2), mode="nearest").to(device)
# torch.backends.cudnn.enabled = False
# warp_nearest = monai.networks.blocks.Warp(mode="nearest", padding_mode="border")
# warp_nearest_val = monai.networks.blocks.Warp(mode="nearest", padding_mode="border")
# wandb.watch( model,log = 'all',
#  log_freq = 10,
 
#  log_graph = True
# )

# training loops
for epoch in range(args.initial_epoch, args.epochs):
    model.train()
    epoch_loss = []
    epoch_total_loss = []
    epoch_step_time = []

 
    for batch_idx, batch in enumerate(train_dataloader):
        # print(batch["fixed_name"])
        # print(batch["moving_name"])
        
        fixed_img = batch["fixed_img"].to(device)
        moving_img = batch["moving_img"].to(device)
        fixed_mask = batch["fixed_mask"].to(device)
        moving_mask = batch["moving_mask"].to(device)
        zero_ff = batch["zero_flow_field"].to(device)
        step_start_time = time.time()

        y_pred = model(moving_img, fixed_img,moving_mask) 
        y_true = (fixed_img, zero_ff ) 

        # calculate total loss
        loss = 0
        loss_list = []
        # for n, loss_function in enumerate(losses):
            
        #     curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
            
        #     if math.isnan(curr_loss) == True:
        #         print(n)
        #         breakpoint()
                
        #     loss_list.append(curr_loss.item())
        #     loss += curr_loss
            
        sim_loss = similarity_loss(lncc_loss, y_pred[0], y_true[0])
        
        # sim_loss = image_loss_func(y_true[0], y_pred[0])
        
        
        grad_loss_val = grad_loss(y_true[1],y_pred[1])   
        
        y_pred[2][y_pred[2] > 1] = 1
        
       
        dice_loss_val = dice_loss(fixed_mask, y_pred[2]) 
        
        loss = sim_loss + grad_loss_val  * grad_wt + dice_loss_val * dice_weight
        # get_dot = bad_grad.register_hooks(loss)
       
        # backpropagate and optimize
        optimizer.zero_grad()
        
        loss.backward()
        # dot = get_dot()
        # dot.save('dot_files/tmp' + batch_idx +'.dot')
        
        optimizer.step()
        loss_list.append(sim_loss.detach().item())
        loss_list.append(grad_loss_val.detach().item())
        loss_list.append(dice_loss_val.detach().item())
        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.detach().item())
   
        wandb.log({"train/loss": loss.detach().item(),"train/ncc": loss_list[0],
                  "train/grad": loss_list[1],"train/dice_loss":loss_list[2]})
   
   
        epoch_step_time.append(time.time() - step_start_time)
        
    
    model.eval()
    with torch.no_grad():
        val_epoch_loss = []
        val_epoch_total_loss = []
        dice_all = []
        for batch_idx, batch in enumerate(val_dataloader):
            fixed_img = batch["fixed_img"].to(device)
            moving_img = batch["moving_img"].to(device)
            zero_ff = batch["zero_flow_field"].to(device)
            fixed_mask = batch["fixed_mask"].to(device)
            moving_mask = batch["moving_mask"].to(device)
            
            y_pred = model(moving_img, fixed_img,moving_mask, registration=True) 
            y_true = (fixed_img, zero_ff ) 

            # calculate total loss
            val_loss = 0
            val_loss_list = []
           
            # val_sim_loss = image_loss_func(y_true[0], y_pred[0])
            val_sim_loss = similarity_loss(lncc_loss, y_pred[0], y_true[0])
            
            val_grad_loss = grad_loss(y_true[1],y_pred[1])   
            y_pred[2][y_pred[2] > 1] = 1
            
            
            val_dice_loss = dice_loss(fixed_mask, y_pred[2]) 
        
            val_loss = val_sim_loss + (val_grad_loss  * grad_wt )+ (val_dice_loss * dice_weight)
            
            val_loss_list.append(val_loss.item())

            val_epoch_loss.append(val_loss_list)
            val_epoch_total_loss.append(val_loss.item())
            dice_all.append(val_dice_loss.item())
            wandb.log({"val/ncc": val_sim_loss.detach().item(),
                  "val/grad": val_grad_loss.detach().item(),"val/loss": val_loss.detach().item(), "val/dice": val_dice_loss.detach().item()})
            
            # if epoch % 30 == 0:
            #     test_data_at = wandb.Artifact("test_samples_" + str(wandb.run.id), type="predictions")            

            #     table_columns = [ 'moving - source', 'fixed - target', 'warped', 'flow_x', 'flow_y', 
            #                     'mask_warped', 'mask_fixed', 'dice']
            #     #'displacement_magn', *list(metrics.keys())
            #     table_results = wandb.Table(columns = table_columns)
            #     fixed = fixed_img.to('cpu').detach().numpy()
            #     fixed = fixed[0,0,:,48,:]
            #     moving = moving_img.to('cpu').detach().numpy()
            #     moving = moving[0,0,:,48,:]
            #     warped = y_pred[0].to('cpu').detach().numpy()
            #     warped = warped[0,0,:,48,:]
                
            #     target_fixed = fixed_mask.to('cpu').detach().numpy()
            #     target_fixed = target_fixed[0,0,:,48,:]
            #     mask_fixed = wandb.Image(fixed, masks={
            #                 "predictions": {
            #                     "mask_data": target_fixed
                                
            #                 }
            #                 })
                
            #     warped_seg = y_pred[2].to('cpu').detach().numpy()
            #     warped_seg = warped_seg[0,0,:,48,:]

            #     mask_warped = wandb.Image(warped, masks={
            #                 "predictions": {
            #                     "mask_data": warped_seg
                                
            #                 }
            #                 })

            #     flow_x = y_pred[1][0,0,:,48,:].to('cpu').detach().numpy()
            #     flow_y = y_pred[1][0,1,:,48,:].to('cpu').detach().numpy()
                
            #     table_results.add_data(wandb.Image(moving), wandb.Image(fixed),
            #                            wandb.Image(warped),wandb.Image(flow_x), 
            #                            wandb.Image(flow_y), mask_warped ,mask_fixed, val_dice_loss.item())
                
            #     test_data_at.add(table_results, "predictions")
            #     wandb.run.log_artifact(test_data_at) 
        # save model checkpoint
    if (np.mean(val_epoch_total_loss) - best_validation_loss) < 0.0001:
        best_validation_loss = np.mean(val_epoch_total_loss)
        model.save(os.path.join(model_dir, '%04d.pt' % epoch))

    epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
    # wandb.log({"train/epoch": epoch + 1})
    time_info = '%.4f sec/step' % np.mean(epoch_step_time)
    losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = 'train_loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
       
    print(' - '.join((epoch_info, time_info, loss_info)), flush=True)
    wandb.log({"train/epoch_loss": np.mean(epoch_total_loss), "val/epoch_loss": np.mean(val_epoch_total_loss),
               "val/val_epoch_dice": np.mean(dice_all), 'epoch': epoch})
    
# final model save
model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))
