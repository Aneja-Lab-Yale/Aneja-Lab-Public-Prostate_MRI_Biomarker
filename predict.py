#------------------------------------------------------Predict------------------------------------------------------------
# Author: Sajid Hossain
# This script contains an executable python file to run a prediction on batches of preprocessed images
# 4/23/2023
# Aneja Lab
# Yale School of Medicine
#-------------------------------------------------------------------------------------------------------------------------

import os
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
from torch.nn import Conv3d
from torch.nn import Linear
from torch.nn import MaxPool3d
from torch.nn import LeakyReLU
from torch.nn import BatchNorm1d
from torch.nn import BatchNorm3d
from torch.nn import Dropout
from torch.nn import Sigmoid
from torch.nn import Flatten
from torch.nn import AvgPool3d
from torch.nn import Dropout3d
import torch.optim.lr_scheduler as lr_scheduler
from data_loader import ProstateDataset
from models.SVI_Net_final import BoxNet, binary_acc, sigmoid_acc, cross_matrix, roc_auc, f1_score
# use the following line if running EPE-Net and comment out the previous line:
#from models.EPE_Net_final import BoxNet, binary_acc, sigmoid_acc, cross_matrix, roc_auc, f1_score
import csv
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
import pandas as pd

#---------------------------------------Setup-----------------------------------------------------------------------
# Set the device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set project root directory
# Change this path to your cloned github directory
#project_root = '/home/shossain/Aneja-Lab-Public-Prostate-MRI-Biomarkers'
project_root = '/mnt/c/Users/shasa/Desktop/Aneja-Lab-Public-Prostate-MRI-Biomarkers'
# Change this path to the directory with the arrays of images in reference to your project directory
#data_directory = 'data/prostate_dx/arrays/seminal_vesicles'
data_directory = 'data/prostate_dx/arrays/prostates'
# Use the following line if running EPE-Net and comment out the previous line:
#data_directory = 'data/prostate_dx/arrays/prostates'

# select the name of the model that you want to run the inference with
# change to "SVI_Net_final" to "EPE_Net_final" if you want to run EPE-Net
model_name = 'EPE_Net_final'

# activation functions and loss criterion
act = Sigmoid()
criterion = nn.BCEWithLogitsLoss()
#---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    # opens the stored dictionary with svi labels
    # change to "tcia_svi_labels.pkl" to "tcia_epe_labels.pkl" if running EPE-Net
    with open(os.path.join(project_root, data_directory, 'tcia_epe_labels.pkl'), 'rb') as handle:
        labels = pickle.load(handle)

    # opens the stored dictionary with svi partition of patients, they are all used for testing
    # change to "tcia_svi_partition.pkl" to "tcia_epe_paritition.pkl" if running EPE-Net
    with open(os.path.join(project_root, data_directory, 'tcia_epe_partition.pkl'), 'rb') as handle:
        partition_patients = pickle.load(handle)

    test_normal = [i for i in partition_patients['eval']] # creates separate list of patients if they are designated as testing 
    train_patients = [i for i in partition_patients['train']] # creates separate list of patients if they are designated as training
    full_list = test_normal + train_patients # full list of patients is just train + test; all patients in TCIA Prostate-Diagnosis are testing

    test = ProstateDataset(full_list, labels, aug=False, data_dir=os.path.join(project_root, data_directory)) # initates the dataset class

    test_loader = DataLoader(test, batch_size = 4, shuffle = False) # initiates the data loader

    # loads the model weights and model
    model = torch.load(os.path.join(project_root, 'models', model_name + '.pt'), map_location=device)

    # this sets the model to evaluate mode
    model.eval()

    # important to run through model without changes weights to it says torch.no_grad (weights will not be updates)
    with torch.no_grad():
        t_loss = 0
        t_acc = 0

        t_output = torch.empty(0)
        t_output = t_output.to(device)
        
        t_preds = torch.empty(0)
        t_preds = t_preds.to(device)

        # Iterate through test dataset
        for images, labels in test_loader:
            images = images.to(device)

            # Forward propagation
            outputs = model(images)
            preds = act(outputs) # for AUC and F1

            t_output = torch.cat((t_output, outputs))
            t_preds = torch.cat((t_preds, preds))

            # store loss and iteration

        t_output = sigmoid_acc(t_output)
        t_output = t_output.to('cpu')
        t_preds = t_preds.to('cpu')

        t_output = t_output.detach().numpy()
        t_preds = t_preds.detach().numpy()

        t_output = [int(i[0]) for i in t_output]
        t_preds = [i[0] for i in t_preds]

        full_list.insert(0, 'Image_ID')
        t_preds.insert(0, 'Logits Pred')
        t_output.insert(0, 'Binary Pred')

        results_output = list(zip(full_list, t_output, t_preds))

        # saves a csv of results to the experiments folder
        with open(os.path.join(project_root, 'output', model_name + '_results.csv'), 'w+', newline='') as f:
            write = csv.writer(f)
            write.writerows(results_output)


