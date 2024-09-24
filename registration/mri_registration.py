# Prostate MRI Project
# This class performs affine registration and rigid registration over images
# Aneja Lab | Yale School of Medicine
# Developed by Sajid Hossain
# Created (7/13/2022)

import os
from os.path import join, exists
from tqdm import tqdm
import pandas as pd
import numpy as np
from dipy.io.image import load_nifti, save_nifti
from dipy.align import affine_registration

class MRIRegistration:
    """
    This class scans the entire input MRI images and performs an affine and rigid transformation,
    based on the affine function
    """
    def __init__(self):
        ############################################################################################
        # SET THESE:
        self.project_root = '/Users/shasa/Desktop/Prostate Cancer Research/Prostate MRI'
        self.corrected_folder = 'data/preprocess/normalized_2'
        self.patient_list_path = 'patient_list_c+r+t.csv'
        self.registered_folder = 'data/preprocess/registered_2'
        self.pickle_path = 'data/preprocess/affine_maps'
        self.nbins = 32
        self.pipeline = ["center_of_mass", "translation", "rigid"]
        self.sampling_prop = None
        self.metric = 'MI'
        self.level_iters = [10000, 1000, 100]
        self.sigmas = [3.0, 1.0, 0.0]
        self.factors = [4, 2, 1]
        ###########################################################################################

        self.patient_list = make_patient_list(join(self.project_root, self.corrected_folder, self.patient_list_path))

        for patient in tqdm(self.patient_list, desc = 'Registering DWI, ADC, and T2'):
            path_to_t2 = join(self.project_root, self.corrected_folder, patient + '_0000.nii.gz')
            path_to_adc = join(self.project_root, self.corrected_folder, patient + '_0001.nii.gz')
            path_to_low_b = join(self.project_root, self.corrected_folder, patient + '_0002.nii.gz')
            path_to_high_b = join(self.project_root, self.corrected_folder, patient + '_0003.nii.gz')


            t2_registered_path = join(self.project_root, self.registered_folder, patient + '_0000.nii.gz')
            adc_registered_path = join(self.project_root, self.registered_folder, patient + '_0001.nii.gz')
            low_b_registered_path = join(self.project_root, self.registered_folder, patient + '_0002.nii.gz')
            high_b_registered_path = join(self.project_root, self.registered_folder, patient + '_0003.nii.gz')

            affine_map_path = join(self.project_root, self.pickle_path, patient + '.pkl')

            t2_data, t2_affine = load_nifti(path_to_t2)
            adc_data, adc_affine = load_nifti(path_to_adc)
            low_b_data, low_b_affine = load_nifti(path_to_low_b)
            high_b_data, high_b_affine = load_nifti(path_to_high_b)

            low_b_xformed_img, high_b_xformed_img, adc_xformed_img, reg_affine = affine_registration(affine_map_path, low_b_data, high_b_data, adc_data, t2_data,
                moving_affine=low_b_affine,
                static_affine=t2_affine,
                nbins=self.nbins,
                metric=self.metric,
                pipeline=self.pipeline,
                level_iters=self.level_iters,
                sigmas=self.sigmas,
                factors=self.factors)

            os.makedirs(join(self.project_root, self.registered_folder), exist_ok=True)
            save_nifti(t2_registered_path, t2_data, t2_affine)
            save_nifti(low_b_registered_path, low_b_xformed_img, reg_affine)
            save_nifti(high_b_registered_path, high_b_xformed_img, reg_affine)
            save_nifti(adc_registered_path, adc_xformed_img, reg_affine)

#----------------------------------------------Helper Functions-------------------------------------------------------#
def make_patient_list(patients_csv):
    images_df = pd.read_csv(patients_csv, header=None)
    patient_list = list(images_df.iloc[:, 0])
    return patient_list

if __name__ == '__main__':

    mriregister = MRIRegistration()