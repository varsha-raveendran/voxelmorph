import json
import os

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import pytorch_lightning as pl
from typing import Optional
import numpy as np
import nibabel as nib
import torchio as tio
import voxelmorph as vxm 

def image_norm(img):
    max_v = np.max(img)
    min_v = np.min(img)

    norm_img = (img - min_v) / (max_v - min_v)
    return norm_img
class NLSTDataModule(pl.LightningDataModule):
    """
    DataModule for NLST dataset

    """

    def __init__(self, data_dir, json_conf="dataset.json", batch_size: int = 1, is_norm=True,num_workers = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.is_norm = is_norm
        self.json_conf = json_conf
        
    # def prepare_data(self):
    #     pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        """ 
        
        train_dataset =  NLST(self.data_dir, self.json_conf,
                                downsampled=False, 
                                masked=True,
                            train_transform=True, is_norm=self.is_norm)
        
        train_set_size = int(len(train_dataset) * 0.9)
        
        valid_set_size = len(train_dataset) - train_set_size

        # split the train set into two
        seed = torch.Generator().manual_seed(42)
        self.train_set, self.valid_set = data.random_split(train_dataset, [train_set_size, valid_set_size], generator=seed)
        
        self.test_dataset = NLST(self.data_dir, self.json_conf, downsampled=False, masked=True, train=False,
                                train_transform=True, is_norm=self.is_norm)

        
            
    def train_dataloader(self):
        return data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )


    def val_dataloader(self):
        return data.DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    #TODO add test dataloader


# Ref: https://github.com/MDL-UzL/L2R/blob/main/examples/task_specific/NLST/Example_NLST.ipynb
class NLST(torch.utils.data.Dataset):
    def __init__(self, root_dir, json_conf='NLST_dataset.json', masked=False, 
                 downsampled=False, train_transform = None, train=True, is_norm=False):
       
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
        
        self.is_norm = is_norm
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
        

        fixed_img = nib.load(fix_path)
        fixed_affine = fixed_img.affine
        fixed_img = fixed_img.get_fdata()
        
        moving_img = nib.load(mov_path).get_fdata()
        
        if self.is_norm:
            fixed_img = image_norm(fixed_img)
            moving_img = image_norm(moving_img)
        
        fixed_img=torch.from_numpy(fixed_img).float()
        moving_img=torch.from_numpy(moving_img).float()
        fixed_img = fixed_img.unsqueeze(0)
        moving_img = moving_img.unsqueeze(0)
        fixed_mask=torch.from_numpy(nib.load(fix_path.replace('images', 'masks')).get_fdata()).float()
        fixed_mask = fixed_mask.unsqueeze(0)
        moving_mask=torch.from_numpy(nib.load(mov_path.replace('images', 'masks')).get_fdata()).float()
        moving_mask = moving_mask.unsqueeze(0)
                 
        if self.masked:            
            fixed_img = fixed_img * fixed_mask
            moving_img = moving_img * moving_mask

        
        shape = fixed_img.shape[1:-1]
        
        zeros = torch.zeros((1, *shape, len(shape)))
        
        return { "fixed_name" : fix_idx,
                "moving_name" : mov_idx,
                "fixed_img" : fixed_img, 
                "moving_img" : moving_img, 
                "fixed_mask" : fixed_mask, 
                "moving_mask" : moving_mask,
                "zero_flow_field" : zeros,
                "fixed_affine" : fixed_affine}

class RandomDataModule(pl.LightningDataModule):
    """
    DataModule for NLST dataset

    """

    def __init__(self, data_dir, batch_size: int = 1, is_norm=False,num_workers = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.is_norm = is_norm
        
    # def prepare_data(self):
    #     pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        """ 
        
        self.train_dataset =  RandomData(self.data_dir, 
                                downsampled=False, 
                                masked=True,
                            train_transform=True, is_norm=self.is_norm)
        
        self.val_dataset = RandomData(self.data_dir, downsampled=False, masked=True, train=False, train_transform=True, is_norm=self.is_norm)

        
            
    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )


    def val_dataloader(self):
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        
class RandomData(torch.utils.data.Dataset):
    def __init__(self, root_dir, masked=False, downsampled=False, train_transform = False, train=True, is_norm=False):
        """
        
        """
        print("INITIALIZE DATASET")
        self.fixed_img =  torch.rand(1,  64,64, 64)
        self.moving_img = torch.rand(1,  64, 64, 64)
        self.is_norm = is_norm
    def __len__(self):
        
        return 1
    
    def __getitem__(self, idx):
        fixed_img = self.fixed_img.float().to("cuda")
        moving_img = self.moving_img.float().to("cuda")
        
        shape = fixed_img.shape[1:-1]
        
        zeros = torch.zeros((1, *shape, len(shape)))
        
        # if self.is_norm:
        #     fixed_img = image_norm(fixed_img.to("cpu").numpy())
        #     moving_img = image_norm(moving_img.to("cpu").numpy())
            
            
        # fixed_img=torch.from_numpy(fixed_img).float().to("cuda")
        # moving_img=torch.from_numpy(moving_img).float().to("cuda")
        
        
            
        return { 
                "fixed_img" : fixed_img, 
                "moving_img" : moving_img, 
                "zero_flow_field" : zeros}