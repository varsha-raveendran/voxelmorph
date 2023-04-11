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
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchio as tio
import monai
from sklearn.model_selection import KFold
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureType,
)
from monai.data import  decollate_batch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from eval_metrics import *
import wandb

wandb.init(project="Vxm")

torch.manual_seed(42)
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
parser.add_argument('--epochs', type=int, default=20,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=150,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--load-model', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=1,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

# loss hyperparameters
parser.add_argument('--image-loss', default='ncc',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0,
                    help='weight of deformation loss (default: 0.01)')
args = parser.parse_args()

bidir = args.bidir

# load and prepare training data
train_dataset =  vxm.nlst.NLST("/vol/pluto/users/raveendr/data/NLST/", 
                               "NLST_dataset_train_test.json",
                                downsampled=False, 
                                masked=True,
                            train_transform=True, is_norm=True)

k=1
splits=KFold(n_splits=k,shuffle=True,random_state=42)
foldperf={}
     
# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = 'cuda:0'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
assert np.mod(args.batch_size, nb_gpus) == 0, \
    'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet
if args.cudnn_nondet:
    print(args.cudnn_nondet)
    print('WARNING: CUDNN nondeterminism is enabled. This may slow down training.')
# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = vxm.losses.NCC_2(win=3)
   
elif args.image_loss == 'mse':
    image_loss_func = vxm.losses.MSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

# need two image loss functions if bidirectional
if bidir:
    print("Is bidir")
    losses = [image_loss_func, image_loss_func]
    weights = [0.5, 0.5]
else:
    losses = [image_loss_func]
    weights = [1]

# prepare deformation loss
# losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
grad_loss_func = vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss
weights += [args.weight]

transformer = vxm.layers.SpatialTransformer(((224,192,224)), mode="nearest").to(device)
post_label = Compose([EnsureType("tensor", device="cuda:0"), AsDiscrete(to_onehot=2)])
# dice_loss = vxm.losses.Dice().loss
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

hd95 = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', 
                                            percentile=95, directed=False, get_not_nans=False)
        
history = {'train_loss': [], 'val_loss': []}
print("Number of training samples: {}".format(len(train_dataset)))
      
for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_dataset)))):
    # training loops
    print('Fold {}'.format(fold + 1))
    os.makedirs(os.path.join(model_dir , 'fold_'+ str(fold)), exist_ok=True)
    
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(train_dataset, batch_size=1, sampler=test_sampler, num_workers=4)
    print("Number of training samples in fold: {}".format(len(train_loader)))
    print("Number of validation samples: {}".format(len(val_loader)))
    model = vxm.networks.VxmDense(
        inshape=(224,192,224),
        nb_unet_features=[enc_nf, dec_nf],
        bidir=bidir,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize
    )
    model.to(device)
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(args.initial_epoch, args.epochs):
        
        model.train()
        # save model checkpoint
        if epoch % 20 == 0:
            model.save(os.path.join(model_dir + '/fold_'+ str(fold), '%04d.pt' % epoch))

        print("Setting epoch losses to zero")
        epoch_loss = []
        epoch_total_loss = []
        epoch_step_time = []

        # for step in range(args.steps_per_epoch):
        for batch_idx, batch in enumerate(train_loader):
            fixed_img = batch["fixed_img"].to(device)
            moving_img = batch["moving_img"].to(device)
            zero_ff = batch["zero_flow_field"].to(device)
            step_start_time = time.time()
           
            y_pred = model(moving_img, fixed_img) 
            y_true = (fixed_img, zero_ff ) 

            # calculate total loss
            loss = 0
            loss_list = []
            # for n, loss_function in enumerate(losses):                
            #     curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
            #     if math.isnan(curr_loss) == True:
            #         breakpoint()
                    
            #     loss_list.append(curr_loss.item())
            #     loss += curr_loss
                
            ncc_loss = image_loss_func(y_true[0], y_pred[0]) 
            grad_loss = grad_loss_func(y_true[1], y_pred[1])
            
            loss = ncc_loss * weights[0] + grad_loss * weights[1]
            
            with torch.no_grad():
                # breakpoint()
                # det = jacobian_determinant(y_pred[1].squeeze(0).detach().to('cpu', non_blocking=True).numpy())

                # moved_seg = transformer(batch["moving_mask"].detach().to(device), y_pred[1].detach())
                # warped_seg_labels = [post_label(i) for i in decollate_batch(moved_seg.detach())]
                # target_labels = [post_label(i) for i in decollate_batch( batch["fixed_mask"].to(device).detach())]
            
                # dice_metric(y_pred=warped_seg_labels, y=target_labels)
                # hd95(y_pred=warped_seg_labels, y=target_labels)
                # dice = dice_metric.aggregate().item()
                # hd95_val = hd95.aggregate().item()
                
                # dice_metric.reset()
                # hd95.reset()
                # wandb.log({"train/dice": dice})
                # wandb.log({"train/hd95": hd95_val})
                # wandb.log({"train/folds": (det<=0).sum()})
                wandb.log({"train/loss": loss.detach().item()})
                wandb.log({"train/loss_ncc":ncc_loss.detach().item()})
                wandb.log({"train/loss_grad": grad_loss.detach().item()})
            
                # del det
                # del warped_seg_labels
                # del target_labels
        
            loss_list.append(ncc_loss.detach().item())
            
            print("adding to epoch losses for batch", str(batch_idx))
            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.detach().item())
            
            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # get compute time
            epoch_step_time.append(time.time() - step_start_time)
            
            del batch
            
            
        model.eval()
        with torch.no_grad():
            val_epoch_loss = []
            val_epoch_total_loss = []
            for batch_idx, batch in enumerate(val_loader):
                fixed_img = batch["fixed_img"].to(device)
                moving_img = batch["moving_img"].to(device)
                zero_ff = batch["zero_flow_field"].to(device)
                y_pred = model(moving_img, fixed_img) 
                y_true = (fixed_img, zero_ff ) 

                # calculate total loss
                val_loss = 0
                val_loss_list = []
                for n, loss_function in enumerate(losses):
                    
                    curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
                    if math.isnan(curr_loss) == True:
                        breakpoint()
                        
                    val_loss_list.append(curr_loss.item())
                    val_loss += curr_loss
                
                del batch
            val_epoch_loss.append(val_loss_list)
            val_epoch_total_loss.append(val_loss.item())
            wandb.log({"val/loss": val_loss.detach().item()})
                
        # print epoch info
        epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
        time_info = '%.4f sec/step' % np.mean(epoch_step_time)
        print("mean of epoch losses for epoch", str(epoch), "is", str(np.mean(epoch_total_loss)))
        losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
        loss_info = 'train_loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
        
        val_losses_info = ', '.join(['%.4e' % f for f in np.mean(val_epoch_loss, axis=0)])
        val_loss_info = 'val_loss: %.4e  (%s)' % (np.mean(val_epoch_total_loss), val_losses_info)
        
        print(' - '.join((epoch_info, time_info, loss_info, val_loss_info)), flush=True)
        wandb.log({"train/epoch": epoch + 1})
        wandb.log({"train/epoch_loss": np.mean(epoch_total_loss)})
        wandb.log({"val/epoch_loss": np.mean(val_epoch_total_loss)})
        history['train_loss'].append(np.mean(epoch_total_loss))
        history['val_loss'].append(np.mean(val_epoch_total_loss))
     
        
# final model save
model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))
avg_train_loss = np.mean(history['train_loss'])
avg_val_loss = np.mean(history['val_loss'])


print('Performance of {} fold cross validation'.format(k))
print("Average Training Loss: {:.4f} \t Average Test Loss: {:.4f} ".format(avg_train_loss,avg_val_loss)) 
