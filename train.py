#------------------------------------------------------Model Trainer-------------------------------------------
# Author: Sajid Hossain
# This script can be used to train models from scratch using EPE-Net or SVI-Net architecture
# 9/1/2022
# Aneja Lab
# Yale School of Medicine
#--------------------------------------------------------------------------------------------------------------
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
from models.EPE_Net import BoxNet
# from models.SVI_Net import BoxNet # use this line if you want to train EPE_Net 
from sklearn.metrics import roc_auc_score
import pandas as pd
import random
from random import shuffle
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from F1_score_loss import F1_Loss
from torchinfo import summary

#---------------------------------------Device Configuration--------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # use GPU if available for training
project_root = '/home/shossain/Aneja-Lab-Prostate-MRI-Public' # define root of project, change to your root

#---------------------------------------Define Model----------------------------------------------------------------------
'''
# this chunk commented out as BoxNet class is imported from SVI_Net or EPE_Net models for training
# however this chunk can be used to make your own modifications to model for training instead of importing prior SVI_Net or EPE_Net
class BoxNet(Module):
    def __init__(self, n_dims=50, n_channels=1, n_classes=1, feats=[16, 32, 64, 128, 256, 2304, 4208], filt=[3], stride=[1], p_filt=[2], p_stride=[2]):
        super(BoxNet, self).__init__()
        #self.avgpool1 = AvgPool3d((2, 1, 1), (2, 1, 1))
        self.mp1 = MaxPool3d((1, 2, 2), (1, 2, 2))
        self.mp2 = MaxPool3d(p_filt[0], p_stride[0])
        self.conv_layer1 = self._conv_layer_set(n_channels, feats[0], filt[0], stride[0])
        self.conv_layer2 = self._conv_layer_set(feats[0], feats[1], filt[0], stride[0])
        self.conv_layer3 = self._conv_layer_set(feats[1], feats[2], filt[0], stride[0])
        self.conv_layer4 = self._conv_layer_set(feats[2], feats[2], filt[0], stride[0])
        self.conv_layer5 = self._conv_layer_set(feats[2], feats[3], filt[0], stride[0])
        self.conv_layer6 = self._conv_layer_set(feats[3], feats[3], filt[0], stride[0])
        self.conv_layer7 = self._conv_layer_set(feats[3], feats[4], filt[0], stride[0])
        self.conv_layer8 = self._conv_layer_set(feats[4], feats[4], filt[0], stride[0])
        self.s_dp = Dropout3d(p=0)
        self.fl1 = Flatten()
        self.fc1 = self._full_connect_set(feats[5], feats[5])
        self.fc2 = self._full_connect_set(feats[5], feats[6])
        self.fc3 = Linear(feats[5], n_classes, bias=True)
        self.act = Sigmoid()

    def _conv_layer_set(self, in_channels, out_channels, k, s):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, k, s, 1, bias=False),
            BatchNorm3d(out_channels),
            LeakyReLU(negative_slope=3e-2, inplace=True),
        )
        return conv_layer

    def _full_connect_set(self, in_channels, out_channels):
        conv_layer = nn.Sequential(
            Linear(in_channels, out_channels, bias=False),
            BatchNorm1d(out_channels),
            LeakyReLU(negative_slope=3e-2, inplace=True),
            #Dropout(p=0),
        )
        return conv_layer

    def forward(self, input):
        #out = self.avgpool1(input)
        input = input.float()
        out = self.conv_layer1(input)
        out = self.mp1(out)
        out = self.conv_layer2(out)
        out = self.mp2(out)
        #out = self.s_dp(out)
        out = self.conv_layer3(out)
        out = self.mp2(out)
        out = self.conv_layer4(out)
        out = self.mp2(out)
        #out = self.s_dp(out)
        out = self.conv_layer5(out)
        out = self.conv_layer6(out)
        out = self.mp2(out)
        #out = self.s_dp(out)
        out = self.conv_layer7(out)
        out = self.conv_layer8(out)
        out = self.mp2(out)
        #out = self.s_dp(out)
        out = self.fl1(out)
        out = self.fc1(out)
        #out = self.fc2(out)
        out = self.fc3(out)
        #out = self.act(out)
        return out
'''

#----------------------------------------Define Performance Metrics Functions---------------------------------------
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

if __name__ == "__main__":
    
    #--------------------------Configure/Compile Model---&----Set Hyperparameters-------------------------------------------------
    # opens prelaoded labels and partition dictionary as determined from the mri preprocessing code
    with open(os.path.join(project_root, 'data/prostate_dx/arrays/labels_80_20.pkl'), 'rb') as handle:
        labels = pickle.load(handle)

    with open(os.path.join(project_root, 'data/prostate_dx/arrays/partition_80_20.pkl'), 'rb') as handle:
        partition_patients = pickle.load(handle)

    # lists the test and train subject for this run
    test = [i for i in partition_patients['eval']]
    train = [i for i in partition_patients['train']]

    model_name = 'SVI_Net_R001' # name the model and include the run number to help with organization and reference after training
    model = BoxNet().to(device) # load model to CUDA device
    preload_path = '/home/shossain/Aneja-Lab-Prostate-MRI-Public/experiments/SVI_Net_R001_best_auc_state.pt' # path to a preloaded weights from prior training, can be 'none' if training from scratch
    model.load_state_dict(torch.load(preload_path)) # load the weights to the model from the preload path
    description = 'Model to train on Seminal Vesicle Invasion with Prostate_Dx data. First Run.' # short description to job your memory of what this run is about
    
    num_epochs = 100 # number of epochs to train
    batch_size = 5 # number of images to load in each run
    input_size = (batch_size, 1, 50, 200, 200) # input size will vary between SVI_Net (batch_size, 1, 50, 200, 200) and EPE_Net (batch_size, 1, 60, 200, 200) 
    learning_rate = 0.1 # initial learning rate to start training
    c_weight = 0.9 # this weight can be used when there is a class imbalance while training, comment out as 'none' if not necessary
    weights = torch.FloatTensor([c_weight]) # loads the class weight as a tensor
    crit = 'BCE' # can be 'BCE' (binary cross entropy), 'AUCM' (area under curve)  
    if crit == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001) # we use ADAM optimizer for BCE Loss
    elif crit == 'AUCM':
        criterion = AUCMLoss()
        optimizer = PESG(model, loss_fn=criterion, lr=learning_rate, weight_decay=0.00001, epoch_decay=0.003) # PESG optimizer for AUCM Loss
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True) # this scheduler applies a simple learning rate scheduler, every 15 epochs the LR decrases by factor of 10
    scheduler = ReduceLROnPlateau(optimizer, factor = 0.5, patience = 4, threshold = 0.001, threshold_mode='abs', min_lr=1e-9) # this scheduler adjusts the LR only when there is a plateau in loss optimization
    
    architecture = summary(model, input_size)
    print(architecture) # prints the architecture of the DCNN to the console
    hyperparameters_file = model_name + '_hyperparameters.csv'

    # creates a hyperparameters data frame to store and save all the hyperparameters for this run for future reference
    hyperparameters = pd.DataFrame(index=['date and time',
                                         'model name',
                                         'model description',
                                         'preload model',
                                         'input size',
                                         '-------------------------------------------------',
                                         'num epochs trained',
                                         'batch size',
                                         'initial learning rate',
                                         'class weight',
                                         'loss function',
                                         '-------------------------------------------------',
                                         'patients train',
                                         'patients test',
                                         'architecture',
                                         '-------------------------------------------------'],
                                  data=[datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                                        model_name,
                                        description,
                                        preload_path,
                                        input_size,
                                        '--------------------------------------------------',
                                        num_epochs,
                                        batch_size,
                                        learning_rate,
                                        c_weight,
                                        criterion,
                                        '--------------------------------------------------',
                                        train,
                                        test,
                                        architecture,
                                        '--------------------------------------------------'])

    # saves hyperparemeters of this run to a csv file for future reference
    hyperparameters.to_csv(os.path.join(project_root, 'experiments', model_name + '_hyperparameters.csv'), index=True)

    #-------------------------------------Load Data----------------------------------------------------------------------------

    # these lines create a class called prostate dataset and initiate the data loader for both train and test subjects
    train_split = ProstateDataset(partition_patients['train'], labels)
    test_split = ProstateDataset(partition_patients['eval'], labels)

    train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test, batch_size = batch_size, shuffle = True)

    #---------------------------------------Training Loops---------------------------------------------------------------------
    # creates a bunch of empty lists to store values of performance metrics across train and test data after each epoch
    count = 0
    epoch_list = []

    train_loss_list = []
    train_accuracy_list = []
    train_tp = []
    train_fp = []
    train_tn = []
    train_fn = []
    train_auc = []
    train_sn = []
    train_sp = []

    test_loss_list = []
    test_accuracy_list = []
    test_tp = []
    test_fp = []
    test_tn = []
    test_fn = []
    test_auc = []
    test_sn = []
    test_sp = []

    max_test_acc = 0
    max_test_auc = 0
    max_test_sn = 0
    max_test_sp = 0

    # iterates through the specified number of epochs to train the model and update weights with each epoch
    for epoch in range(num_epochs):
        epoch_list.append(epoch+1)
        print(f'Epoch {epoch + 1}/{num_epochs}')
        model.train()

        # initializes variables to store the current performance metrics for each new epoch
        epoch_loss = 0
        epoch_acc = 0
        epoch_output = torch.empty(0)
        epoch_output = epoch_output.to(device)
        epoch_labels = torch.empty(0)
        epoch_labels = epoch_labels.to(device)

        # calls images from the train data loader and passes through the neural network
        for images, labels in tqdm(train_loader):
            #print(images.shape)
            images = images.to(device)
            labels = labels.float()
            labels = labels.to(device)
            optimizer.zero_grad()

            #Forward pass
            output = model(images)

            #Calculate loss and accuracy
            criterion.cuda()
            loss = criterion(output, labels.unsqueeze(1))
            acc = binary_acc(output, labels.unsqueeze(1))

            #Backward and optimize
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_output = torch.cat((epoch_output, output))
            epoch_labels = torch.cat((epoch_labels, labels.unsqueeze(1)))

        # prints the performance metrics during this epoch of training for user to monitor
        print(f'Epoch {epoch+1:03}: | Train Loss: {epoch_loss/len(train_loader):.5f} | Train Acc: {epoch_acc/len(train_loader):.3f} \n')
        train_loss_list.append(epoch_loss/len(train_loader))
        train_accuracy_list.append(epoch_acc/len(train_loader))
        epoch_output = sigmoid_acc(epoch_output)
        epoch_output = epoch_output.to('cpu')
        epoch_labels = epoch_labels.to('cpu')
        epoch_output = epoch_output.detach().numpy()
        epoch_labels = epoch_labels.detach().numpy()
        train_auc.append(roc_auc(epoch_output, epoch_labels))
        train_cross_matrix = cross_matrix(epoch_output, epoch_labels)
        train_tp.append(train_cross_matrix[0])
        train_fp.append(train_cross_matrix[1])
        train_tn.append(train_cross_matrix[2])
        train_fn.append(train_cross_matrix[3])
        train_sn.append(train_cross_matrix[0]/(train_cross_matrix[0]+train_cross_matrix[3]))
        train_sp.append(train_cross_matrix[2]/(train_cross_matrix[2]+train_cross_matrix[1]))

        # no gradient changes when evaluating the model on the test data
        model.eval()
        with torch.no_grad():

            # initializes performance metric variables prior to running on test data
            t_loss = 0
            t_acc = 0
            t_output = torch.empty(0)
            t_output = t_output.to(device)
            t_labels = torch.empty(0)
            t_labels = t_labels.to(device)

            # Iterate through test dataset
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.float()
                labels = labels.to(device)

                # Forward propagation
                outputs = model(images)

                # Calculate loss and accuracy
                criterion.cuda()
                loss = criterion(outputs, labels.unsqueeze(1))
                acc = binary_acc(outputs, labels.unsqueeze(1))

                t_loss += loss.item()
                t_acc += acc.item()
                t_output = torch.cat((t_output, outputs))
                t_labels = torch.cat((t_labels, labels.unsqueeze(1)))

                # store loss and iteration

            # prints performance metrics on the test dataset
            print(f'Epoch {epoch + 1:03}: | Test Loss: {t_loss / len(test_loader):.5f} | Test Acc: {t_acc / len(test_loader):.3f} \n')
            test_loss_list.append(t_loss / len(test_loader))
            test_accuracy_list.append(t_acc / len(test_loader))
            t_output = sigmoid_acc(t_output)
            t_output = t_output.to('cpu')
            t_labels = t_labels.to('cpu')
            t_output = t_output.detach().numpy()
            t_labels = t_labels.detach().numpy()
            test_auc.append(roc_auc(t_output, t_labels))
            test_cross_matrix = cross_matrix(t_output, t_labels)
            test_tp.append(test_cross_matrix[0])
            test_fp.append(test_cross_matrix[1])
            test_tn.append(test_cross_matrix[2])
            test_fn.append(test_cross_matrix[3])
            test_sn.append(test_cross_matrix[0]/(test_cross_matrix[0] + test_cross_matrix[3]))
            test_sp.append(test_cross_matrix[2]/(test_cross_matrix[2] + test_cross_matrix[1]))

            # a number of if statements that will save the models in the experiments folder depending on if they are the best AUC, ACC, Sn, Sp, etc
            if (t_acc / len(test_loader)) >= max_test_acc:
                max_test_acc = t_acc / len(test_loader)
                print('Highest Validation Accuracy.. Saving Model to Experiments Folder')
                torch.save(model, os.path.join(project_root, 'experiments', model_name + '_best_acc.pt'))
                torch.save(model.state_dict(), os.path.join(project_root, 'experiments', model_name + '_best_acc_state.pt'))

            if test_auc[-1] >= max_test_auc:
                max_test_auc = test_auc[-1]
                print('Highest Validation AUC.. Saving Model to Experiments Folder')
                torch.save(model, os.path.join(project_root, 'experiments', model_name + '_best_auc.pt'))
                torch.save(model.state_dict(), os.path.join(project_root, 'experiments', model_name + '_best_auc_state.pt'))

            if test_sn[-1] >= max_test_sn and not test_sn[-1] == 1:
                max_test_sn = test_sn[-1]
                print('Highest Validations Sn.. Saving Model to Experiments Folder')
                torch.save(model, os.path.join(project_root, 'experiments', model_name + '_best_sn.pt'))
                torch.save(model.state_dict(), os.path.join(project_root, 'experiments', model_name + '_best_sn_state.pt'))

            if test_sp[-1] >= max_test_sp and not test_sp[-1] == 1:
                max_test_sp = test_sp[-1]
                print('Highest Validations Sp.. Saving Model to Experiments Folder')
                torch.save(model, os.path.join(project_root, 'experiments', model_name + '_best_sp.pt'))
                torch.save(model.state_dict(), os.path.join(project_root, 'experiments', model_name + '_best_sp_state.pt'))
        
        # advances the scheduler one step
        scheduler.step()

        # every ten epochs, saves images and csv of the training results such that while training occurs, user can view results in real time 
        if (epoch+1)%10 == 0:
            fig, ax = plt.subplots()
            plt.plot(epoch_list, test_loss_list, 'g', label='validation')
            plt.plot(epoch_list, train_loss_list, 'b', label='training')
            plt.xlabel("Epoch")
            plt.ylabel("Binary Cross Entropy Loss")
            plt.title('Loss')
            plt.legend()
            plt.savefig(os.path.join(project_root, 'experiments', model_name + '_loss.png'))

            plt.plot(epoch_list, test_accuracy_list, 'g', label='validation')
            plt.plot(epoch_list, train_accuracy_list, 'b', label='training')
            plt.xlabel("Epoch")
            plt.ylabel("Binary Accuracy")
            plt.title('Accuracy')
            plt.legend()
            plt.savefig(os.path.join(project_root, 'experiments', model_name + '_accuracy.png'))

            df = pd.DataFrame()
            df['epoch'] = epoch_list
            df['train_loss'] = train_loss_list
            df['train_acc'] = train_accuracy_list
            df['train_auc'] = train_auc
            df['train_TP'] = train_tp
            df['train_FP'] = train_fp
            df['train_TN'] = train_tn
            df['train_FN'] = train_fn
            df['train_Sn'] = train_sn
            df['train_Sp'] = train_sp
            df['test_loss'] = test_normal_loss
            df['test_acc'] = test_normal_accuracy
            df['test_auc'] = test_normal_auc
            df['test_TP'] = test_normal_tp
            df['test_FP'] = test_normal_fp
            df['test_TN'] = test_normal_tn
            df['test_FN'] = test_normal_fn
            df['test_Sn'] = test_normal_sn
            df['test_Sp'] = test_normal_sp

            df.to_csv(os.path.join(project_root, 'experiments', model_name + '_metrics.csv'), index=False)