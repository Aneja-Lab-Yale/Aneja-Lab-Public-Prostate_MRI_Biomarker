#------------------------------------------------------EPE-Net------------------------------------------------------------
# Author: Sajid Hossain
# This script contains the final EPE-Net Model (BoxNet class) architecture and performance function metric functions
# 9/1/2022
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
from data_loader import ProstateDataset
from sklearn.metrics import roc_auc_score
import pandas as pd
import random
from random import shuffle
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from torchinfo import summary

#---------------------------------------Device Configuration--------------------------------------------------------------
# Set the device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set project root directory
# Change this path to your cloned github directory
project_root = '/home/shossain/Aneja-Lab-Public-Prostate-MRI-Biomarkers'

#---------------------------------------Set Hyperparameters---------------------------------------------------------------
# Define training parameters
num_epochs = 100
batch_size = 3
learning_rate = 0.01

#---------------------------------------Define Model (BoxNet)-------------------------------------------------------------
# Define the BoxNet model class
class BoxNet(Module):
    def __init__(self, n_dims=60, n_channels=1, n_classes=1, feats=[16, 32, 64, 128, 256, 2304, 4208], filt=[3], stride=[1], p_filt=[2], p_stride=[2]):
        super(BoxNet, self).__init__()
       # Define max pooling layers
        self.mp1 = MaxPool3d((1, 2, 2), (1, 2, 2))
        self.mp2 = MaxPool3d(p_filt[0], p_stride[0])
        # Define convolutional layers
        self.conv_layer1 = self._conv_layer_set(n_channels, feats[0], filt[0], stride[0])
        self.conv_layer2 = self._conv_layer_set(feats[0], feats[1], filt[0], stride[0])
        self.conv_layer3 = self._conv_layer_set(feats[1], feats[2], filt[0], stride[0])
        self.conv_layer4 = self._conv_layer_set(feats[2], feats[2], filt[0], stride[0])
        self.conv_layer5 = self._conv_layer_set(feats[2], feats[3], filt[0], stride[0])
        self.conv_layer6 = self._conv_layer_set(feats[3], feats[3], filt[0], stride[0])
        self.conv_layer7 = self._conv_layer_set(feats[3], feats[4], filt[0], stride[0])
        self.conv_layer8 = self._conv_layer_set(feats[4], feats[4], filt[0], stride[0])
        # Define dropout layer
        self.s_dp = Dropout3d(p=0)
        self.dp = Dropout(p=0.6)
        # Define fully connected layers
        self.fl1 = Flatten()
        self.fc1 = self._full_connect_set(feats[5], feats[5])
        self.fc2 = self._full_connect_set(feats[5], feats[6])
        self.fc3 = Linear(feats[5], n_classes, bias=True)
        self.act = Sigmoid()

    # Define the convolutional layer set
    def _conv_layer_set(self, in_channels, out_channels, k, s):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, k, s, 1, bias=False),
            BatchNorm3d(out_channels),
            LeakyReLU(negative_slope=3e-2, inplace=True),
        )
        return conv_layer

    # Define the fully connected layer set
    def _full_connect_set(self, in_channels, out_channels):
        conv_layer = nn.Sequential(
            Linear(in_channels, out_channels, bias=False),
            BatchNorm1d(out_channels),
            LeakyReLU(negative_slope=3e-2, inplace=True),
        )
        return conv_layer

    # defines the forward pass through the deep convolutional neural network
    def forward(self, input):
        input = input.float()
        out = self.conv_layer1(input)
        out = self.mp1(out)
        out = self.conv_layer2(out)
        out = self.mp2(out)
        out = self.conv_layer3(out)
        out = self.mp2(out)
        out = self.conv_layer4(out)
        out = self.mp2(out)
        out = self.conv_layer5(out)
        out = self.conv_layer6(out)
        out = self.mp2(out)
        out = self.conv_layer7(out)
        out = self.conv_layer8(out)
        out = self.mp2(out)
        out = self.fl1(out)
        out = self.fc1(out)
        out = self.dp(out)
        out = self.fc3(out)
        return out

#-----------------------------------------------Callable Performance Metrics Functions-------------------------------------- 
# Define sigmoid accuracy function
def sigmoid_acc(y_pred):
    act = Sigmoid()
    y_pred_tag = act(y_pred)
    y_pred_tag = torch.round(y_pred_tag)

    return y_pred_tag

# Define binary accuracy function
def binary_acc(y_pred, y_test):
    act = Sigmoid()
    y_pred_tag = act(y_pred)
    y_pred_tag = torch.round(y_pred_tag)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

# Define confusion matrix function
def cross_matrix(y_pred, y_actual):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_actual[i]==y_pred[i]==1:
           TP += 1
        if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
           FP += 1
        if y_actual[i]==y_pred[i]==0:
           TN += 1
        if y_pred[i]==0 and y_actual[i]!=y_pred[i]:
           FN += 1

    return TP, FP, TN, FN

# Define ROC AUC score function
def roc_auc(y_pred, y_actual):
    a = roc_auc_score(y_actual, y_pred)

    return a

# Define F1 score function
def f1_score(tp, fp, fn):
    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    return f1
