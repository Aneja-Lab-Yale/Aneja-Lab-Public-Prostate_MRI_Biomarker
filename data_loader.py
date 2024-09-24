#--------------------------------------------------data-loader------------------------------------------------------------
# Author: Sajid Hossain
# This script contains the data loader class to load data for inference using pretrained models or for training models
# 4/23/2023
# Aneja Lab
# Yale School of Medicine
#-------------------------------------------------------------------------------------------------------------------------

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchio as tio
import numpy as np
import math
import os
import pickle
import random

# this is the data loader class which inherets Dataset from torch.utils.data
class ProstateDataset(Dataset):
    '''
    Inputs when creating this ProstateDataset class object:
    patient_IDs: list of unique patient ids that will be used as keys to access images and epe/svi labels
    labels: dictionary where keys are patient ids and values are 0's and 1's corresponding to binary labels (either for EPE or SVI)
    aug: boolean for whether to apply augmentation methods to the images; default should be False
    data_dir: string for path to directory with images stored as numpy arrays
    batch_size: integer value for number of patients to process at a time; default is 4
    dim: tuple representing the (z, x, y) dimensions of each image; default is (50, 200, 200) which is good for SVI-Net
    n_channels: integer; only 1 channel
    n_classes: binary model requires 0 classes

    This data loader takes preprocessed numpy arrays that are stored as .npy with a prefix corresponding to the unique patient id
    in the list of patient ids that is provided. The numpy arrays should be generated beforehand and stored in the data directory 
    specified as an input when calling this class.
    '''
    def __init__(self, 
                 patient_IDs,
                 labels, 
                 aug,
                 data_dir, 
                 batch_size=4, 
                 dim=(50,200,200), # change this to (60, 200, 200) if running EPE-Net 
                 n_channels=1,
                 n_classes=0):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.aug = aug
        self.patient_IDs = patient_IDs
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.data_dir = data_dir
        self.random_affine_5 = tio.RandomAffine(degrees=(5,15,0,0,0,0))

        # defines data augmentations that will be performed on images if aug is set to TRUE 
        self.augment = tio.Compose([
            tio.RandomAnisotropy(p=0.25), # this sheers the image
            tio.RandomAffine(degrees=(5,15,0,0,0,0)), # this applys random rotation between 5 and 15 degrees in either direction
            tio.RandomFlip(axes=['inferior-superior'], flip_probability=0.25), # this flips the image left to right 25% of the time
            tio.RandomNoise(p=0.25), # this adds random noise 25% of the time
            tio.RandomGamma(p=1), # this stretches the image 100% of time
        ])
        self.augment_1 = tio.Compose([
            tio.RandomAnisotropy(p=0.25),
            tio.RandomAffine(degrees=(5,15,0,0,0,0)),
            tio.RandomFlip(axes=['inferior-superior'], flip_probability=0.25),
            tio.RandomNoise(p=0.25),
            tio.RandomBiasField(p=1), # this adds random bias field 100% of time
        ])
        self.augment_2 = tio.Compose([
            tio.RandomAnisotropy(p=0.25),
            tio.RandomAffine(degrees=(5,15,0,0,0,0)),
            tio.RandomFlip(axes=['inferior-superior'], flip_probability=0.25),
            tio.RandomNoise(p=0.25),
            tio.RandomBlur(p=1), # this adds a random blur 100% of time
        ])
        self.augment_3 = tio.Compose([
            tio.RandomAnisotropy(p=0.25),
            tio.RandomAffine(degrees=(5,15,0,0,0,0)),
            tio.RandomFlip(axes=['inferior-superior'], flip_probability=0.25),
            tio.RandomNoise(p=1), # this adds random noise 100% of time
        ])

        # list of transformations that will be performed 
        self.trans_list = [self.augment, self.augment_1, self.augment_2, self.augment_3, self.random_affine_5]

    def __getitem__(self, index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size), dtype=int)

        patient = self.patient_IDs[index]
        array = np.load(os.path.join(self.data_dir, patient + '.npy'))
        #x = np.expand_dims(array, axis=0) # this line may need to be commented back in depending on how the arrays were constructed, it adds a dimension to the images

        # loop to run through list of augmentations and apply them to original image
        if self.aug:
            do = random.randint(0, 1)
            if do == 1:
                i = random.randint(0, 4)
                transform = self.trans_list[i]
                array = transform(array)
        y = np.array(self.labels[patient], dtype=int)
        x = torch.from_numpy(array)
        y = torch.from_numpy(y)

        return x, y

    def __len__(self):
        return len(self.patient_IDs)