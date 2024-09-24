# -------------------------------------------------------------------------------
# Prostate Cancer MRI Preprocessing Tool Modified from Masoudi et al. (2019)
# Author: Sajid Hossain
# Date: 05/17/2022
# Yale University School of Medicine
# Aneja-Lab
# -------------------------------------------------------------------------------

#-----------------------------------Imports--------------------------------------
from __future__ import print_function
import argparse
import sys
import os
from os.path import join, exists

import SimpleITK
from tqdm import tqdm
import json
import glob
import random
from preprocess.Dicom_Tools import *
from preprocess.utils import *
import csv
from preprocess.Annotation_utils import *
from preprocess.Nyul_preprocessing import *
from registration.mri_orient import *
from registration.mri_registration import *
from preprocess.resampler import *
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import seaborn as sns
import pickle
import torch
import torchio as tio
import time
from matplotlib import gridspec
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
from matplotlib.path import Path
from matplotlib.patches import *
import seaborn as sns

#--------------------------------------------------------------------------------

#-----------------------------------Preprocess class----------------------------
class Preprocess:
    def __init__(self, data_root):
        #########################################################################
        #            SET DIRECTORIES AND PARAMETERS FOR PREPROCESSING           #
        #########################################################################
        # Set data root:
        self.data_root = data_root
        # Set patients DICOM folder:
        self.dicom_folder = join(self.data_root, 'raw')
        # Set folder for LAS corrected niftis:
        self.corrected_folder = join(self.data_root, 'corrected')
        # Set folder with manual prostate masks:
        self.mask_folder = join(self.data_root, 'segmentations/prostates') 
        # ^this needs to be changes to 'segmentations/seminal_vesicles' if working with seminal vesicles
        # Set split_train_test specifications folder:
        self.split_folder = join(self.data_root, 'split_train_test')
        # Set bias field corrected folder:
        self.bfc_folder = join(self.data_root, 'bias_field_corrected')
        # Set nyul normalized folder:
        self.norm_folder = join(self.data_root, 'normalised')
        # set resampled folder:
        self.resampled_folder = join(self.data_root, 'resampled')
        # Set final preprocessed folder:
        self.processed_folder = join(self.data_root, 'niftis/prostates')
        # ^this needs to be changes to 'niftis/seminal_vesicles' if working with seminal vesicles
        # Set processed arrays folder
        self.array_folder = join(self.data_root, 'arrays/prostates')
        # ^this needs to be changes to 'arrays/seminal_vesicles' if working with seminal vesicles
        # Set patient list from directories:
        self.patient_list = [f for f in listdir(self.dicom_folder) if f.endswith('.nii.gz')]
        # Set path to lookup EPE values
        self.epe_lookup = join(self.array_folder, 'tcia_epe_svi_labels.csv')

        # Set parameters for splitting patients in preparation for nyul standardization
        # Default paths to json files with split patients is empty but can be initialized here
        self.split_ratio = [.6,.2,0.2]
        self.path_to_train_set = []
        self.path_to_validation_set = []
        self.path_to_test_set = []
        # Default list of train, test, and validate patients is set to null
        self.train_patients = []
        self.validation_patients = []
        self.test_patients = []

        # Set output pixel type for writing images:
        self.output_pixel_type = 'Uint16'
        # Set desired output spacing, default is [1, 1, 1]
        self.spacing = [0.5, 0.5, 0.5]
        # Set desired zero padding for images, default is [120, 120, 100]
        self.zero_pad = [200, 200, 50]
        # Set prostates with seminal vesicles or seperated; combined vs separated
        self.combine = True

        # Set parameters for registration
        self.patient_list_path = 'ex_patient_list.csv'
        self.nbins = 32
        self.pipeline = ["rigid"]
        self.sampling_prop = None
        self.metric = 'MI'
        self.level_iters = [10000, 1000, 100]
        self.sigmas = [3.0, 1.0, 0.0]
        self.factors = [4, 2, 1]

        print(
            'Pre-processing for MR images includes N4-Bias Correction, histogram normalization, prostate and seminal-vesicle masking, space-matching followed by zero padding. Images are going to be:')
        print('                          processed by N4 bias correction to lose their bias field')

        print('                          normalized using Nyul et al. (2000) standardization method')

        print('                          resampled to have the target spacing:', self.spacing)

        print('                          zero-padded to be of size: ', self.zero_pad)

    def change_root_directory(self, path):
        '''
        This function can be used to change the data directory after the class has been created if your so need
        '''
        self.data_root = path
        self.dicom_folder = join(self.data_root, 'raw')
        self.corrected_folder = join(self.data_root, 'corrected')
        self.mask_folder = join(self.data_root, 'segmentations/prostates') # <- change to 'segmentations/seminal_vesicles'
        self.split_folder = join(self.data_root, 'split_train_test')
        self.bfc_folder = join(self.data_root, 'bias_field_corrected')
        self.norm_folder = join(self.data_root, 'normalised')
        self.resampled_folder = join(self.data_root, 'resampled')
        self.processed_folder = join(self.data_root, 'niftis/prostates') # <- change to 'niftis/seminal_vesicles'
        self.array_folder = join(self.data_root, 'arrays/prostates') # <- change to 'arrays/seminal_vesicles'
        self.patient_list = [f for f in listdir(self.dicom_folder) if f.endswith('.nii.gz')]
        self.epe_lookup = join(self.array_folder, 'tcia_epe_svi_labels.csv')
        return 0

    def reorient(self):
        '''
        This function corrects the orientation of all input image files to LAS coords
        '''
        patient_list = self.patient_list

        for patient in tqdm(patient_list, desc='Correcting MRI orientations'):
            try:
                image = nib.load(join(self.dicom_folder, patient))
                orientation = nib.io_orientation(image.affine)
                orientation[orientation[:, 0] == 0, 1] = - orientation[orientation[:, 0] == 0, 1]
                image = image.as_reoriented(orientation)

            except Exception as e:
                print(e)
                print('Unable to correct: ', patient)
                print('Skipping file...')
                continue

            os.makedirs(self.corrected_folder, exist_ok=True)
            nib.save(image, join(self.corrected_folder, patient))

    def mriorient(self):
        '''
        This function identifies all the different coordinate orientations in the raw data
        '''
        self.file_list = make_files_list(join(self.data_root, self.dicom_folder))
        self.file_list = filter_files(self.file_list)

        self.coords_list = make_coords_list(self.file_list)

        self.unique_coords = set(self.coords_list)

        print(f'Unique coordinates in the entire MRI volumes: \n 'f'{self.unique_coords} \n')

    def mriorientcalc(self):
        '''
        This function prints the files that are not in standard LAS orientation and prints them
        '''
        self.file_list = make_files_list(join(self.dicom_folder))
        self.file_list = filter_files(self.file_list)
        for file in self.file_list:
            _, _, coords = load_nifti(file, return_coords=True)
            if not coords == ('L','A','S'):
                print('incorrect coordinates for:')
                print(file)
                print(coords)

    def change_patient_lists(self, path = None):
        '''
        This function changes the patient list based on an input path to a csv where the first column of 
        csv contains a list of image identifiers
        '''
        if path == None:
            self.patient_list = [f for f in listdir(self.dicom_folder) if f.endswith('.nii.gz')]
        elif os.path.isfile(path):
            self.patient_list = make_patient_list(path)
        else:
            print('path to new patient list specified could not be found')
            return 0

    def split_train_test_validate(self):
        '''
        This function randomly splits the batch of input images into train, test, and validation 
        sets that is used for the Nyul normalisation step
        '''
        patient_list = self.patient_list
        all_patients = random.shuffle(patient_list)

        with open(join(self.split_folder, 'all_patients' + '.json'), 'w') as output:
            json.dump(all_patients, output)
        try:
            self.train_patients = json.load(open(self.path_to_train_set, 'r'))
        except Exception:
            self.train_patients = self.patient_list[:int(self.split_ratio[0] * len(self.patient_list))]
            with open(join(self.split_folder, 'train_set' + '.json'), 'w') as output:
                json.dump(self.train_patients, output)
        try:
            self.validation_patients = json.load(open(self.path_to_validation_set, 'r'))
        except Exception:
            self.validation_patients = self.patient_list[int(self.split_ratio[0] * len(self.patient_list)):int(
                (self.split_ratio[0] + self.split_ratio[1]) * len(self.patient_list))]
            with open(join(self.split_folder, 'validation_set' + '.json'), 'w') as output:
                json.dump(self.validation_patients, output)
        try:
            self.test_patients = json.load(open(self.test_set, 'r'))
        except Exception:
            self.test_patients = self.patient_list[int((self.split_ratio[0] + self.split_ratio[1]) * len(self.patient_list)):]
            with open(join(self.split_folder, 'test_set' + '.json'), 'w') as output:
                json.dump(self.test_patients, output)

        with open(join(self.split_folder, 'image_specifications.csv'), 'w', newline='') as csvfile:
            fieldnames = ['Image', 'Original_Spacing', 'Original_Size', 'Organ_Size', 'Split']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        return 0

    def bias_field_correct(self, path = None):
        '''
        This function corrects images for B-field variations within each image
        '''
        if path == None:
            patient_list = self.patient_list
        elif os.path.isfile(path):
            patient_list = make_patient_list(path)
        else:
            print('path to patient list for bias field corrections could not be found')
            return 0
    
        for patient in tqdm(patient_list, desc='Performing Bias Field Correction'):
            print(patient)
            input_file = join(self.corrected_folder, patient)
            output_check = join(self.bfc_folder, patient)
            if os.path.isfile(input_file) and not os.path.isfile(output_check):
                image1 = nifti_read(input_file)
                # shift the data up so that all intensity values turn positive
                # removes the outliers with a probability of occurring less than 5e-3 through histogram computation
                data = histo_normalize_against_self(image1)
                image = new_image(data, image1)
                # performs bias correction (this step takes a lot of time)
                image_b = dicom_bias_correct(image)
                nifti_write(image_b, self.bfc_folder,
                                output_name=patient, OutputPixelType=self.output_pixel_type)
                print('Bias field correction completed on ' + patient)
            elif os.path.isfile(output_check):
                print('Bias field correction already performed on ' + patient)
            else:
                print('Unexpected error, skipping bias correction on ' + patient)
        return 0

    def nyul_normalize(self, path = None):
        '''
        This function uses nyul histogram standardisation technique accross multiple images by learning a set
        of landmark histogram features from a set of images and standardising those features across all images
        '''
        if path == None:
            patient_list = self.patient_list
        elif os.path.isfile(path):
            patient_list = make_patient_list(path)
        else:
            print('path to normalize list specified is not available for processing')
            return 0

        model_path = join(self.split_folder, trained_model + '.npz')
        train(train_patients, dir1=self.bfc_folder, dir2=model_path)
        f = np.load(model_path, allow_pickle=True)
        model = f['trainedModel'].all()
        mean_landmarks = model['meanLandmarks']

        error_patients=[]

        for patient in tqdm(patient_list, desc='Performing Nyul Standardization\n'):
            input_file = join(self.bfc_folder, patient)
            output_check = join(self.norm_folder, patient)
            try:
                if os.path.isfile(input_file) and not os.path.isfile(output_check):
                    print('Standardizing ...', basename(input_file))
                    image_b = nifti_read(input_file)
                    image_b_s = transform(image_b, meanLandmarks=mean_landmarks)

                    nifti_write(image_b_s, self.norm_folder,
                            output_name=patient, OutputPixelType='Uint16')
                    print('Standardization performed on ' + patient)
                elif os.path.isfile(output_check):
                    print('Standardization already performed on ' + patient)
                else:
                    print('Unexpected error, skipping nyul normalization on ' + patient)
            except Exception as err:
                print(err)
                print(patient)
                error_patients.append(patient)

        print(error_patients)

        return 0

    def resample_images(self, path = None):
        '''
        This function is used to reample all images to a standard voxel size 
        '''
        if path == None:
            patient_list = self.patient_list
        elif os.path.isfile(path):
            patient_list = make_patient_list(path)
        else:
            print('path to resampling list specified is not available for processing')
            return 0

        for patient in tqdm(patient_list, desc="resampling images"):
            input_file = join(self.norm_folder, patient)
            output_check = join(self.resampled_folder, patient)
            if os.path.isfile(input_file): # and not os.path.isfile(output_check):
                img, affine, voxsize, coords = load_nifti(input_file, return_voxsize=True, return_coords=True)
                #imshow(img, voxsize)

                print('image shape:', img.shape)
                print('voxel size', voxsize)
                img = img[None]

                target_spacing = (0.5, 0.5, 1)
                z_anisotropy_threshold = 2 

                img2, seg2 = resample_patient(data=img,
                                              seg=None,
                                              original_spacing=voxsize,
                                              target_spacing=target_spacing,
                                              force_separate_z=None,
                                              order_z_data = 3,
                                              separate_z_anisotropy_threshold=z_anisotropy_threshold)

                x_ratio = target_spacing[0] / voxsize[0]
                y_ratio = target_spacing[1] / voxsize[1]
                z_ratio = target_spacing[2] / voxsize [2]

                S_matrix = np.array([[x_ratio, 0, 0, 0],
                                    [0, y_ratio, 0, 0],
                                    [0, 0, z_ratio, 0],
                                    [0, 0, 0, 1]])

                affine_tx = affine @ S_matrix
                img2 = img2[0]

                #print('image2 shape', img2.shape)

                save_nifti(output_check, img2, affine_tx)
                print('resampling completed on ' + patient)

            elif os.path.isfile(output_check):
                print('resampling already completed on ' + patient)
            else:
                print('Unexpected error, skipping resampling on ' + patient)

    def apply_mask_to_images(self, path = None):
        '''
        This function is used to apply prostate or seminal vesicle segmentations from nnUNet to resample images
        '''
        if path == None:
            patient_list = self.patient_list
        elif os.path.isfile(path):
            patient_list = make_patient_list(path)
        else:
            print('path to mask list specified is not available for processing')
            return 0

        error_pts = []
        for patient in tqdm(patient_list, desc='Applying Masks to T2'):
            image_b_s = nifti_read(join(self.resampled_folder, patient + '.nii.gz'))
            mask = nifti_read(join(self.mask_folder, patient+'.nii.gz'))
            image_b_s_m = dicom_apply_mask(mask, image_b_s)
            image_b_s_m_c = crop_2D_mask(image_b_s_m)
            try:
                image_b_s_m_c_z = zero_pad_3D(image_b_s_m_c[0], [self.zero_pad[0], self.zero_pad[1], self.zero_pad[2]], mask=None)
            except Exception:
                error_pts.append(patient)
                image_b_s_m_c_z1 = zero_pad_3D(image_b_s_m_c[0],
                                                [500, 500, 500], mask=None)
                image_b_s_m_c_z = crop_3D(image_b_s_m_c_z1, [self.zero_pad[0], self.zero_pad[1], self.zero_pad[2]], [250, 250, 250])
            save_dicom_as_NPY(image_b_s_m_c_z, self.array_folder, patient)
            nifti_write(image_b_s_m_c_z, self.processed_folder,
                            output_name=patient+'.nii.gz', OutputPixelType=self.output_pixel_type)
        print(error_pts)

        return 0

    def split_train_eval(self, path = None):
        '''
        From a set of images, this function can be used to randomly split a train and test set for purposes of 
        training deep learning models
        '''
        if path == None:
            patient_list = self.patient_list
        elif os.path.isfile(path):
            patient_list = make_patient_list(path)
        else:
            print('path to split train test list specified is not available for processing')
            return 0

        indices = np.random.permutation(len(self.patient_list))
        split = int(len(self.patient_list) * .75)
        train_idx, eval_idx = indices[:split], indices[split:]
        self.train_pts = [self.patient_list[i] for i in train_idx]
        self.eval_pts = [self.patient_list[i] for i in eval_idx]

        return 0

    def find_split(self, path = None):
        '''
        This function is used to find prior split train and test images from a csv where these parameters are defined
        '''
        if path == None:
            patient_list = self.patient_list
        elif os.path.isfile(path):
            patient_list = make_patient_list(path)
        else:
            print('path to split train test list specified is not available for processing')
            return 0

        self.df_lookup = pd.read_csv(self.epe_lookup)

        patient_list_2 = list(self.df_lookup['Vis_Accession_Number'])

        self.patient_list = [i for i in self.patient_list if i in patient_list_2]
        self.df_lookup = self.df_lookup[self.df_lookup['Vis_Accession_Number'].isin(self.patient_list)]
        self.train_pts = list(self.df_lookup[self.df_lookup['train'] == 1]['Vis_Accession_Number'])
        self.eval_pts = list(self.df_lookup[self.df_lookup['train'] == 0]['Vis_Accession_Number'])

    def add_dimension(self, path):
        '''
        This function adds a dimension to the final preprocessed images to denote the modality
        '''
        epe_eval = list(self.df_lookup[self.df_lookup['Vis_Accession_Number'].isin(self.eval_pts)]['EPE_Sensitive'].astype(int))
        # 'EPE_Sensitive' can be changed to 'SVI_Sensitive' when running this create SVI labels

        patient_list_aug = []
        epe_aug = []
        train_pts_aug = []
        eval_pts_aug = []

        for j, patient in enumerate(self.eval_pts):
            patient = str(patient)

            ori_image = np.load(os.path.join(self.array_folder, patient + '.npy'))
            ori_image = np.expand_dims(ori_image, axis=0)
            np.save(os.path.join(self.array_folder, patient + '.npy'), ori_image)

            patient_list_aug.append(patient)
            eval_pts_aug.append(patient)
            epe_aug.append(epe_eval[j])

        labels = dict(zip(patient_list_aug, epe_aug))
        partition_IDs = dict({'train': train_pts_aug, 'eval': eval_pts_aug})

        print(labels)
        print(partition_IDs)

        with open(os.path.join(self.array_folder, 'tcia_epe_labels.pkl'),'wb') as handle:
            pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.array_folder, 'tcia_epe_partition.pkl'),'wb') as handle:
            pickle.dump(partition_IDs, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return 0


    def torch_augment_data(self, path):
        '''
        This function applies multiple data augmentations to one image to generate a 10-fold increase in data to 
        train deep learning models. It also generates the necessary labels and partition files used in the data loader
        to train deep learning models with this augmented data
        '''
        epe_train = list(self.df_lookup[self.df_lookup['Vis_Accession_Number'].isin(self.train_pts)]['EPE_Sensitive'].astype(int))
        epe_eval = list(self.df_lookup[self.df_lookup['Vis_Accession_Number'].isin(self.eval_pts)]['EPE_Sensitive'].astype(int))
        # 'EPE_Sensitive' can be changed to 'SVI_Sensitive' when running this create SVI labels

        random_noise = tio.RandomNoise()
        random_affine_5 = tio.RandomAffine(degrees=(5,15,0,0,0,0))
        random_affine_10 = tio.RandomAffine(degrees=(5,15,0,0,0,0))
        random_affine_15 = tio.RandomAffine(degrees=(5,15,0,0,0,0))
        random_anisotropy = tio.RandomAnisotropy()
        random_flip = tio.RandomFlip(axes=['inferior-superior'], flip_probability=1)
        random_elastic = tio.RandomElasticDeformation(max_displacement=(2, 2, 2))
        random_bias = tio.RandomBiasField(coefficients=1)
        random_blur = tio.RandomBlur()

        augment = tio.Compose([
            tio.RandomAnisotropy(p=0.25),
            tio.RandomAffine(),
            tio.RandomFlip(),
            tio.RandomNoise(p=0.25),
            tio.RandomGamma(p=0.5),
        ])

        augment_1 = tio.Compose([
            tio.RandomAnisotropy(p=0.25),
            tio.RandomAffine(),
            tio.RandomFlip(),
            tio.RandomNoise(p=0.25),
            tio.RandomBiasField(p=0.5),
        ])

        augment_2 = tio.Compose([
            tio.RandomAnisotropy(p=0.25),
            tio.RandomAffine(),
            tio.RandomFlip(),
            tio.RandomNoise(p=0.25),
            tio.RandomBlur(p=0.5),
        ])

        augment_3 = tio.Compose([
            tio.RandomAnisotropy(p=0.25),
            tio.RandomAffine(),
            tio.RandomFlip(),
            tio.RandomNoise(p=0.25),
        ])

        trans_list = []
        trans_list_svi = [random_affine_5, random_affine_10, random_noise, random_bias, random_blur, augment, augment_1, augment_2, augment_3]


        patient_list_aug = []
        epe_aug = []
        train_pts_aug = []
        eval_pts_aug = []

        for j, patient in enumerate(self.train_pts):
            print(epe_train[j])
            ori_image = np.load(os.path.join(self.array_folder, patient + '.npy'))
            np.save(os.path.join(self.array_folder, patient + '.npy'), ori_image)
            patient_list_aug.append(patient)
            train_pts_aug.append(patient)
            epe_aug.append(epe_train[j])

            if epe_train[j] == 0:
                for i, transform in enumerate(trans_list):
                    transformed = transform(ori_image)
                    np.save(os.path.join(self.array_folder, patient+'_'+str(i)+'.npy'), transformed)
                    patient_list_aug.append(patient + '_'+ str(i))
                    train_pts_aug.append(patient + '_' + str(i))
                    epe_aug.append(epe_train[j])

            elif epe_train[j] == 1:
                for i, transform in enumerate(trans_list_svi):
                    transformed = transform(ori_image)
                    np.save(os.path.join(self.array_folder, patient + '_' + str(i) + '.npy'), transformed)
                    patient_list_aug.append(patient + '_' + str(i))
                    train_pts_aug.append(patient + '_' + str(i))
                    epe_aug.append(epe_train[j])

        for j, patient in enumerate(self.eval_pts):
            print(epe_eval[j])
            ori_image = np.load(os.path.join(self.array_folder, patient + '.npy'))
            np.save(os.path.join(self.array_folder, patient + '.npy'), ori_image)
            patient_list_aug.append(patient)
            eval_pts_aug.append(patient)
            epe_aug.append(epe_eval[j])

            if epe_eval[j] == 0:
                for i, transform in enumerate(trans_list):
                    transformed = transform(ori_image)
                    np.save(os.path.join(self.array_folder, patient + '_' + str(i) + '.npy'), transformed)
                    patient_list_aug.append(patient + '_' + str(i))
                    eval_pts_aug.append(patient + '_' + str(i))
                    epe_aug.append(epe_eval[j])

            elif epe_eval[j] == 1:
                for i, transform in enumerate(trans_list_svi):
                    transformed = transform(ori_image)
                    np.save(os.path.join(self.array_folder, patient + '_' + str(i) + '.npy'), transformed)
                    patient_list_aug.append(patient + '_' + str(i))
                    eval_pts_aug.append(patient + '_' + str(i))
                    epe_aug.append(epe_eval[j])

        labels = dict(zip(patient_list_aug, epe_aug))
        partition_IDs = dict({'train': train_pts_aug, 'eval': eval_pts_aug})

        print(labels)
        print(partition_IDs)

        with open(os.path.join(self.array_folder, 'tcia_epe_labels.pkl'),'wb') as handle:
            pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.array_folder, 'tcia_epe_partition.pkl'),'wb') as handle:
            pickle.dump(partition_IDs, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return 0

    def generate_images(self):
        '''
        Code to generate online supplement of example of preprocessing steps
        '''
        mpl.rcParams['font.family'] = 'Arial'
        mpl.rcParams['xtick.labelsize'] = 12
        mpl.rcParams['ytick.labelsize'] = 12
        mpl.rcParams['axes.labelsize'] = 14
        mpl.rcParams['axes.titlesize'] = 14

        self.patient_list = ['ProstateDx-01-0014', 'ProstateDx-01-0038', 'ProstateDx-01-0054']
        n_patients = len(self.patient_list)
        mri_fig, mri_axarr = plt.subplots(n_patients, 5, figsize=(8, n_patients * 1.5))
        hist_fig, hist_axarr = plt.subplots(n_patients, 5, figsize=(16, n_patients * 2 + 2))
        mri_hist_fig, mri_hist_axarr = plt.subplots(n_patients*2, 5, figsize=(8, n_patients * 3.5))
        regenerate = False

        mri_hist_fig = plt.figure()
        mri_hist_fig.set_figheight(n_patients*2)
        mri_hist_fig.set_figwidth(8)

        spec = gridspec.GridSpec(ncols=5, nrows=n_patients*2,
                                 height_ratios=[1, 0.4, 1, 0.4, 1, 0.4],
                                 wspace=0.1,
                                 hspace=0.1)

        nrows, ncols = spec.get_geometry()
        mri_hist_axarr = np.array([[mri_hist_fig.add_subplot(spec[i, j]) for j in range(ncols)] for i in range(nrows)])


        for row, patient in enumerate(self.patient_list):
            self.seminal_folder = join(self.data_root, 'niftis/seminal_vesicles')
            folder_list = [self.dicom_folder, self.bfc_folder, self.resampled_folder, self.processed_folder, self.seminal_folder]
            hist_list = ['raw', 'bfc', 'normalised', 'processed', 'seminal']
            for col, folder in enumerate(folder_list):
                if folder == self.processed_folder or folder == self.seminal_folder:
                    image = nifti_read(join(folder, patient + '.nii.gz'))
                else:
                    image = nifti_read(join(folder, patient + '.nii.gz'))
                data = sitk.GetArrayFromImage(image)
                z = data.shape[0] // 2
                if folder == self.dicom_folder:
                    mri_axarr[row, col].imshow(data[z], cmap='gray')
                    mri_hist_axarr[row*2, col].imshow(data[z], cmap='gray')
                else:
                    mri_axarr[row, col].imshow(np.flipud(data[z]), cmap='gray')
                    mri_hist_axarr[row*2, col].imshow(np.flipud(data[z]), cmap='gray')
                mri_hist_axarr[row*2, col].set_xticks([])
                mri_hist_axarr[row*2, col].set_yticks([])
                mri_axarr[row, col].set_xticks([])
                mri_axarr[row, col].set_yticks([])

                data_f = data.ravel()
                data_f = np.ma.masked_equal(data_f, 0)
                # plt.sca(axarr2[x, y])
                # plt.hist(data_f, bins='auto', color = colors[patient])

                # Check if histogram has been generated
                hist_path = join(self.images_folder, patient + "_" + hist_list[col] + ".png")
                if not exists(hist_path) or regenerate:
                    fig, ax = plt.subplots(figsize=(3, 1), constrained_layout=True)
                    sns.histplot(data_f, stat='density', kde=True,
                                 linewidth=0, color='C' + str(row % 10), alpha=0.5,
                                 line_kws=dict(color='C' + str(row % 10), alpha=1, linewidth=1.5, label='KDE'),
                                 ax=ax)
                    if folder == self.resampled_folder or folder == self.processed_folder or folder == self.seminal_folder:
                        ax.set_xlim([0, 150])
                    ax.set_ylabel(None)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    fig.savefig(hist_path)
                img = mpimg.imread(hist_path)
                hist_axarr[row, col].imshow(img)
                mri_hist_axarr[row*2+1, col].imshow(img)
                hist_axarr[row, col].set_xticks([])
                hist_axarr[row, col].set_yticks([])
                hist_axarr[row, col].set_frame_on(False)
                mri_hist_axarr[row*2+1, col].set_xticks([])
                mri_hist_axarr[row*2+1, col].set_yticks([])
                mri_hist_axarr[row*2+1, col].set_frame_on(False)

            #mri_axarr[row, 0].set_ylabel(patient, rotation=0, size='small', ha='right')
            #hist_axarr[row, 0].set_ylabel(patient, rotation=0, size='small', ha='right')
            #mri_hist_axarr[row, 0].set_ylabel(patient, rotation=0, size='small', ha='right')

        cols = ['Input', 'Orient Corrected\n& N4 Bias Corrected', 'Normalised\n& Resampled', 'P: Masked\n& Padded', 'SV: Masked\n& Padded']
        for ax, col in zip(mri_axarr[0], cols):
            ax.set_title(col)
        mri_fig.subplots_adjust(wspace=0, hspace=0)
        mri_fig.tight_layout()
        plt.show()
        mri_fig.savefig(join(self.images_folder, 'pdx_preprocessing_mris.png'), dpi=400)

        cols = ['Input', 'Orient Corrected\nN4 Bias Corrected', 'Normalised\nResampled', 'P: Masked\n& Padded', 'SV: Masked\n& Padded']
        for ax, col in zip(hist_axarr[0], cols):
            ax.set_title(col)
        hist_fig.tight_layout()
        plt.show()
        hist_fig.savefig(join(self.images_folder, 'pdx_preprocessing_histograms.png'), dpi=400)

        cols = ['Input', 'LAS Oriented\n& N4 Bias Corrected', 'Intensity Normalised\n& Resampled', 'P: nnUNet Masked\n& Padded', 'SV: nnUNet Masked\n& Padded']
        for ax, col in zip(mri_hist_axarr[0], cols):
            ax.set_title(col, size='small')
        mri_hist_fig.subplots_adjust(wspace=0.1, hspace=0)
        #mri_hist_fig.tight_layout()
        plt.show()
        mri_hist_fig.savefig(join(self.images_folder, 'pdx_preprocessing_mri_histograms.png'), dpi=400)

        return mri_fig, hist_fig

    def mri_data_aug(self, patient):
        '''
        Code to generate online supplement of example of data augmentations
        '''
        mri_fig, mri_axarr = plt.subplots(nrows=3, ncols=3)

        row = {0:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2}
        col = {0:0, 1:1, 2:2, 3:0, 4:1, 5:2, 6:0, 7:1, 8:2}

        for i in range(0,9):
            array = np.load(os.path.join(self.array_folder, patient + '_' + str(i) + '.npy'))
            mri_axarr[row[i], col[i]].imshow(array[1, 25, :, :], cmap='gray')

        mri_fig.subplots_adjust(wspace=0, hspace=0)
        mri_fig.tight_layout()
        plt.show()
        mri_fig.savefig(join(self.data_root, 'data_augmentation.png'), dpi=300)

        return 0

def make_patient_list(patients_csv):
    images_df = pd.read_csv(patients_csv, header=None)
    patient_list = list(images_df.iloc[:, 0])
    return patient_list

if __name__ == "__main__":
    #creates a preprocess class and runs functions on batches of images in sequential operation
    x = Preprocess('./data/prostate_dx')
    x.reorient()
    x.mriorient()
    x.split_train_test_validate()
    x.bias_field_correct()
    x.nyul_normalize()
    x.resample_images()
    # after resampling images, look to the github for creating prostate and 
    # seminal vesicle segmentations with nnUNet prior to running the next few steps 
    x.apply_mask_to_images()
    x.split_train_eval()
    x.find_split()
    x.add_dimension()
    x.torch_augment_data()