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

import wandb

wandb.init(project="Vxm")
# def image_norm(img):
#     max_v = np.max(img)
#     min_v = np.min(img)

#     norm_img = (img - min_v) / (max_v - min_v)
#     return norm_img
torch.manual_seed(42)
class NLST(torch.utils.data.Dataset):
    def __init__(self, root_dir, json_conf='dataset.json', masked=False, downsampled=False, train_transform = False, train=True, is_norm=False):
       
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir,'imagesTr')
        self.keypoint_dir = os.path.join(root_dir,'keypointsTr')
        self.masked = masked
        with open(os.path.join(root_dir,json_conf)) as f:
            self.dataset_json = json.load(f)
        self.shape = self.dataset_json['tensorImageShape']['0']
        self.H, self.W, self.D = self.shape
        self.downsampled = downsampled
        self.train = train
        #self.transforms = transforms.Resize((192, 192))
        rescale = tio.RescaleIntensity(out_min_max=(0, 1))
        transforms = [rescale]
        self.transform = tio.Compose(transforms)
        self.is_norm = False
        if self.train :
            self.type_data = 'training_paired_images'
        
        else:
            self.type_data = 'registration_val'
        
    def __len__(self):
        
        if self.train:
            return self.dataset_json['numPairedTraining']
        else:
            return len(self.dataset_json['registration_val'])

    def get_shape(self):
        if self.downsampled:
            return [x//2 for x in self.shape]
        else:
            return self.shape
    
    def __getitem__(self, idx):
        fix_idx = self.dataset_json[self.type_data][idx]['fixed']
        mov_idx = self.dataset_json[self.type_data][idx]['moving']
        
        fix_path=os.path.join(self.root_dir,fix_idx)
        
        mov_path=os.path.join(self.root_dir,mov_idx)
        fixed_img = nib.load(fix_path).get_fdata()
        moving_img = nib.load(mov_path).get_fdata()
    
        # if self.is_norm:
        #     fixed_img = image_norm(fixed_img)
        #     moving_img = image_norm(moving_img)

        
        fixed_img=torch.from_numpy(fixed_img).float()
        moving_img=torch.from_numpy(moving_img).float()
        
        
        fixed_mask=torch.from_numpy(nib.load(fix_path.replace('images', 'masks')).get_fdata()).float()
        moving_mask=torch.from_numpy(nib.load(mov_path.replace('images', 'masks')).get_fdata()).float()
        
        
        # fixed_kp=torch.from_numpy(np.genfromtxt(fix_path.replace('images','keypoints').replace('nii.gz','csv'),delimiter=','))
        # moving_kp=torch.from_numpy(np.genfromtxt(mov_path.replace('images','keypoints').replace('nii.gz','csv'),delimiter=','))
        # fixed_kp=(fixed_kp.flip(-1)/torch.tensor(self.shape))*2-1
        # moving_kp=(moving_kp.flip(-1)/torch.tensor(self.shape))*2-1

        if self.masked:
            #fixed_img=torch.from_numpy(nib.load(fix_path.replace('images', 'masks')).get_fdata())*fixed_img
            fixed_img = fixed_img * fixed_mask
            moving_img = moving_img * moving_mask
            #moving_img=torch.from_numpy(nib.load(mov_path.replace('images', 'masks')).get_fdata())*moving_img
        
        # if self.downsampled:
        #     fixed_img=F.interpolate(fixed_img.view(1,1,self.H,self.W,self.D),size=(self.H//2,self.W//2,self.D//2),mode='trilinear').squeeze()
        #     moving_img=F.interpolate(moving_img.view(1,1,self.H,self.W,self.D), size=(self.H//2,self.W//2,self.D//2), mode='trilinear').squeeze()
        #     if self.masked:
        #         fixed_img*=F.interpolate(torch.from_numpy(nib.load(fix_path.replace('images', 'masks')).get_fdata()).view(1,1,self.H,self.W,self.D),size=(self.H//2,self.W//2,self.D//2),mode='nearest').squeeze()
        #         moving_img*=F.interpolate(torch.from_numpy(nib.load(mov_path.replace('images', 'masks')).get_fdata()).view(1,1,self.H,self.W,self.D),size=(self.H//2,self.W//2,self.D//2),mode='nearest').squeeze()

     
        if self.transform is not None:
            fixed_img = self.transform(fixed_img.float().unsqueeze(0))
            moving_img = self.transform(moving_img.float().unsqueeze(0))
       
        
        shape = fixed_img.shape[1:-1]
        
        zeros = torch.zeros((1, *shape, len(shape)))
        
        return { "fixed_name" : fix_idx,
                "moving_name" : mov_idx,
                "fixed_img" : fixed_img, 
                "moving_img" : moving_img, 
                "fixed_mask" : fixed_mask.unsqueeze(0), 
                "moving_mask" : moving_mask.unsqueeze(0),
                "zero_flow_field" : zeros}
        
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
parser.add_argument('--epochs', type=int, default=1000,
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
parser.add_argument('--lambda', type=float, dest='weight', default=1,
                    help='weight of deformation loss (default: 0.01)')
args = parser.parse_args()

bidir = args.bidir
print(bidir)
# load and prepare training data
# train_files = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix,
#                                           suffix=args.img_suffix)
# assert len(train_files) > 0, 'Could not find any training data.'

# no need to append an extra feature axis if data is multichannel
# add_feat_axis = not args.multichannel

# if args.atlas:
#     # scan-to-atlas generator
#     atlas = vxm.py.utils.load_volfile(args.atlas, np_var='vol',
#                                       add_batch_axis=True, add_feat_axis=add_feat_axis)
#     generator = vxm.generators.scan_to_atlas(train_files, atlas,
#                                              batch_size=args.batch_size, bidir=args.bidir,
#                                              add_feat_axis=add_feat_axis)
# else:
#     # scan-to-scan generator
#     generator = vxm.generators.scan_to_scan(
#         train_files, batch_size=args.batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)

# extract shape from sampled input
# inshape = next(generator)[0][0].shape[1:-1]


train_dataset =  NLST("/vol/pluto/users/raveendr/data/NLST/", "NLST_dataset_train_test.json",
                                downsampled=False, 
                                masked=True,
                            train_transform=True, is_norm=True)

k=5
splits=KFold(n_splits=k,shuffle=True,random_state=42)
foldperf={}
     
# train_set_size = int(len(train_dataset) * 0.8)

# valid_set_size = len(train_dataset) - train_set_size
# print("train_set_size: ", train_set_size)
# print("valid_set_size: ", valid_set_size)

# # split the train set into two
# seed = torch.Generator().manual_seed(42)
# train_set, valid_set = data.random_split(train_dataset, [train_set_size, valid_set_size], generator=seed)


# train_dataloader = DataLoader(train_set, batch_size=1, shuffle=True)
# val_dataloader = DataLoader(valid_set, batch_size=1, shuffle=False)

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

# if args.load_model:
#     # load initial model (if specified)
#     model = vxm.networks.VxmDense.load(args.load_model, device)
# else:
#     # otherwise configure new model
#     model = vxm.networks.VxmDense(
#         inshape=(224,192,224),
#         nb_unet_features=[enc_nf, dec_nf],
#         bidir=bidir,
#         int_steps=args.int_steps,
#         int_downsize=args.int_downsize
#     )

# if nb_gpus > 1:
#     # use multiple GPUs via DataParallel
#     model = torch.nn.DataParallel(model)
#     model.save = model.module.save

# prepare the model for training and send to device

# model.to(device)
# model.train()

# set optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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

# prepare deformation loss
losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
weights += [1]

history = {'train_loss': [], 'val_loss': []}

for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_dataset)))):
    # training loops
    print('Fold {}'.format(fold + 1))
    os.makedirs(os.path.join(model_dir , 'fold_'+ str(fold)), exist_ok=True)
    
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler)
    val_loader = DataLoader(train_dataset, batch_size=1, sampler=test_sampler)
    
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

        epoch_loss = []
        epoch_total_loss = []
        epoch_step_time = []

        # for step in range(args.steps_per_epoch):
        for batch_idx, batch in enumerate(train_loader):
            

            fixed_img = batch["fixed_img"].to(device)
            moving_img = batch["moving_img"].to(device)
            zero_ff = batch["zero_flow_field"].to(device)
            step_start_time = time.time()

            # generate inputs (and true outputs) and convert them to tensors
            #inputs, y_true = next(generator)
            
            # inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
            # y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]

            # run inputs through the model to produce a warped image and flow field
            # y_pred = model(*inputs)
            
            y_pred = model(moving_img, fixed_img) 
            y_true = (fixed_img, zero_ff ) 

            # calculate total loss
            loss = 0
            loss_list = []
            for n, loss_function in enumerate(losses):
                
                curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
                if math.isnan(curr_loss) == True:
                    breakpoint()
                    
                loss_list.append(curr_loss.item())
                loss += curr_loss
                

            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())
            wandb.log({"train/loss": loss.detach().item()})
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
                

            val_epoch_loss.append(val_loss_list)
            val_epoch_total_loss.append(val_loss.item())
            wandb.log({"val/loss": val_loss.detach().item()})
                
        # print epoch info
        epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
        time_info = '%.4f sec/step' % np.mean(epoch_step_time)
        losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
        loss_info = 'train_loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
        
        val_losses_info = ', '.join(['%.4e' % f for f in np.mean(val_epoch_loss, axis=0)])
        val_loss_info = 'val_loss: %.4e  (%s)' % (np.mean(val_epoch_total_loss), val_losses_info)
        
        print(' - '.join((epoch_info, time_info, loss_info, val_loss_info)), flush=True)
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
