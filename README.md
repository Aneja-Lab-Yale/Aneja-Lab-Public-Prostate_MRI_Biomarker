# Deep Learning for Prostate MRI Biomarkers

**Preprint: **
Deep Learning Identified Extra-Prostatic Extension and Seminal Vesicle Invasion as an MRI Biomarker for Prostate Cancer Outcomes
Sajid Hossain, Saeed Hossain, Durga Sritharan, Daniel Fu, Aishwarya Nene, Jahid Hossain, Saahil Chadha, Issac Kim, MingDe Lin, Mariam Aboian, Sanjay Aneja
medRxiv 2024.12.31.24319822; doi: https://doi.org/10.1101/2024.12.31.24319822

https://www.medrxiv.org/content/10.1101/2024.12.31.24319822v1

In this project, we use Deep Learning (DL) on Prostate Cancer (PCA) Magnetic Resonance Imaging (MRI) to develop and validate two prognostic biomarkers that independently predict Biochemical Recurrence (BCR).

<i><b>TL;DR:</b> We hypothesized that features derived from diagnostic imaging of Prostate Cancer are associated with outcomes and can inform clincal treatment decisions, particulary useful in the growing and dynamic landscape of PCa therapies. We trained two deep convolutional neural networks, EPE-Net and SVI-Net, respectively on radiographic evidence of Extraprostatic Extension (EPE) and Seminal Vesicle Invasion (SVI), two high-risk pathological phenotypes of PCa. We demonstrate that these models perform well in identifying radiographic and pathologic EPE and SVI, respectively, and demonstrate the utility of their outputs in stratifying patients' risk for BCR using survival analysis. Finally, we develop a simple points based methodology to integrate these biomarkers into existing clinical guidelines, NCCN and CAPRA, and demonstrate improved risk stratification from baseline.</i>

## Contents:
Use these links to easily navigate through this readme:

1. [Installation](#getting-started)
2. [File Structure](#file-structure)
3. [Public Data](#data)
4. [Preprocessing](#preprocessing-images)
5. [Segmentations](#segmentations)
6. [Pretrained Models](#pretrained-models)
7. [SVI-Net](#svi-net)
8. [EPE-Net](#epe-net)
9. [Training Models](#training-models)
10. [References](#references)

## Installation:
### Getting Started:
1) First clone this repository:

```bash
git clone https://github.com/Aneja-Lab-Yale/Aneja-Lab-Public-Prostate_MRI_Biomarker
```

2) Navigate to the cloned repository and run the following commands to create the appropriate environment and download packages as specified:

```bash
conda create -n prostate_env python=3.7
conda activate prostate_env
# source activate prostate_env # use this command if above command does not work
pip install -r requirements.txt
```

### File Structure:
The cloned repository will have the following file structure. Descriptions of major folders and files are below:

```bash
.
├── data                   # contains image and clinical data for preprocessing, experimentation, and results
    ├── prostate_dx             # contains all images from public PROSTATE-DIAGNOSIS dataset
        ├── raw                 # raw images from PROSTATE-DIAGNOSIS as .nifti files
        ├── corrected           # coordinate system standardised images
        ├── bias_field_corrected    # bfc corrected images
        ├── normalised          # histogram feature normalized images
        ├── resampled           # resampled images to standard voxel size [0.5, 0,5, 1] 
        ├── segmentations       # segmentations inferred from nnUNet on resampled PROSTATE-DIAGNOSIS images
            ├── prostates               # prostate gland nnUNet segmentations  
            ├── seminal_vesicles        # seminal vesicles nnUNet segmentations
        ├── niftis             # segmentations inferred from nnUNet on resampled PROSTATE-DIAGNOSIS images
            ├── prostates               # prostate gland nnUNet segmentations  
            ├── seminal_vesicles        # seminal vesicles nnUNet segmentations
        ├── arrays             # fully preprocessed and masked numpy arrays used as input to models
            ├── prostates               # numpy arrays for prostate glands
            ├── seminal_vesicles        # numpy arrays for seminal vesicles

├── models                 # contains final pretrained EPE-Net and SVI-Net models for download and use
    ├── EPE_Net_final.pt            # EPE-Net model saved weights
    ├── EPE_Net_final.py            # NON-EXECUTABLE/CALLABLE: final EPE-Net model class in Pytorch
    ├── EPE_Net_final_hyper.csv     # EPE-Net model saved hyperparameters
    ├── SVI_Net_final.pt            # SVI-Net model saved weights
    ├── SVI_Net_final.py            # NON-EXECUTABLE/CALLABLE: final SVI-Net model class in Pytorch
    ├── SVI_Net_final_hyper.csv     # SVI-Net model saved hyperparameters
    
├── nnUNet                 # contains pretrained nnUNet models for segmenting prostates and seminal vesicles
├── preprocess             # contains nonexecutable python code and functions called by mri_preprocessing.py
├── registration           # contains more nonexecutable python code and functions called by mri_processing.py
├── mri_preprocessing.py   # EXECUTABLE: python file to preprocess mri images and generate data for training and testing models
├── predict.py             # EXECUTABLE: python file to run an inference with a pretrained model
├── train.py               # EXECUTABLE: python file to train a new model or continue training previous models
├── data_loader.py         # NON-EXECUTABLE/CALLABLE: loads data for inference; called in predict.py and train.py
├── requirements.txt       # basic package requirements for usage.. execute with "pip install -r requirements.txt" in venv
├── .gitignore             # instructs git to ignore python cache files and virtual environment files
```
## Data:
Images and clinical data used to train and test our pretrained EPE-Net and SVI-Net models are only available through a Data Use Agreement (DUA).

The [PROSTATE-DIAGNOSIS](https://www.cancerimagingarchive.net/collection/prostate-diagnosis/) dataset--including MRI images, clinical metadata, and prostate and seminal vesicle segmentations--are publically available through [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net). 

The images from the **PROSTATE-DIAGNOSIS** dataset are included in the <b>*./data/prostate_dx/raw*</b> directory in this public repository to use as a toy dataset. This dataset is also used as a low resolution external validation set in our manuscript. 

## Preprocessing Images:
Our MRI image preprocessing pipeline includes coordinate standardization to LAS system, N4 bias field correction, histogram normalisation, resampling to standard voxel size, [nnUNet segmentation](#segmentations), masking, zero-padding, and data augmentation. 

As an example, the MRI images from the PROSTATE-DIAGNOSIS dataset from TCIA are included in the <b>*./data/prostate_dx*</b> folder. The <b>*mri_preprocessing.py*</b> file contains a preprocessing class with self referencing executable functions for each preprocessing step. Run the following command in a terminal to execute the preprocessing pipeline on the PROSTATE-DIAGNOSIS data:

```bash
python mri_preprocessing.py
```

To run this pipeline on new images, simply create a new subdirectory in the **data** folder and reproduce the same folder structure as within the <b>*./data/prostate_dx*</b> directory. Next, store your images as <b>*.nifti*</b> files in the folder titled <b>raw</b>. Next, modify the <b>*mri_preprocessing.py*</b> source code to create a **Preprocess** class that accepts the PATH to your new data directory as an argument:

```python
# lines 742 to 757
if __name__ == "__main__":
    #creates a preprocess class and runs functions on batches of images in sequential operation
    x = Preprocess('./data/prostate_dx') # <- Modify this line to the PATH to your new data subdirectory
    x.reorient()
    x.mriorient()
    x.split_train_test_validate()
    x.bias_field_correct()
    x.nyul_normalize()
    x.resample_images()
    # after resampling images, look to the github for creating prostate and 
    # seminal vesicle segmentations with nnUNet prior to running the next few steps 
    x.apply_mask_to_images() # <- this line & below can be commented out prior to nnUNet segmentations
    x.split_train_eval()
    x.find_split()
    x.add_dimension()
    x.torch_augment_data()
```

Be aware that [nnUNet segmentations](#segmentations) should be derived from the resampled images and stored in the segmentations folder prior to running the steps that follow resampling. To run the preprocessing pipeline after making any adjustments to the <b>*mri_preprocessing.py*</b> source code, simply run the command ```python mri_preprocessing.py```

## Segmentations:
This project uses nnUNet, a state-of-the-art medical segmentation algorithm, for automatic segmentations of whole organ prostates and seminal vesicles. We trained nnUNet using a random selection of 100 MRI images from our main clinical cohort of patients and used the trained models to infer the segmentations for all other images. Using these models elimates the need for laborious and expert organ delineation prior to running our EPE-Net and SVI-Net models. These models can be found in the <b>*Task500_Prostate*</b>, <b>*Task501_SV*</b>, and <b>*Task503_3DProstate*</b> subdirectories in the <b>*Aneja-Lab-Public-Prostate-MRI-Biomarkers/nnUNet/3d_fullres* directory</b>. To use our pretrained nnUNet models:

1) Activate the [prostate_env](#getting-started) virtual environment by running ```conda activate prostate_env``` or ```source activate prostate_env``` in your terminal. With the activated virtual environment, follow the directions in the [nnUNet Github](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) to [install nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1?tab=readme-ov-file#installation). 

2) As described in this [nnUNet documentation](https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/documentation/setting_up_paths.md), environmental variables need to be configured for nnUNet to recognize and use our pretrained models for inference. This can be done in force by setting paths in your .bashrc file. Modify your .bashrc file in your favorite text editor to include the following line:

```bash
export RESULTS_FOLDER="/home/shossain/Aneja-Lab-Public-Prostate-MRI-Biomarkers/nnUNet"
```

Run ```source .bashrc``` or restart your terminal for changes to be implemented. This command sets an enviornmental variable, RESULTS_FOLDER, as a path to the nnUNet directory of this cloned Github. If you wish not to change your .bashrc file, you may also run this command in your terminal each time prior to running a nnUNet inference, but the variable will be lost once you close your terminal. 

You can type the command ```echo $RESULTS_FOLDER ``` to verify that the variable has been set correctly.

3) The general command structure to run an inference with nnUNet is:

```bash
nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -t TASK_ID -m MODEL_CONFIGURATION
```

- **INPUT_FOLDER**: is the path to all the input images in .nifti format 
- **OUTPUT_FOLDER**: is the path to where nnUNet will save the segmentations as binary masks in .nifti format 
- **TASK_ID**: is the ID of the specific task you want to run; 501 calls our seminal vesicle model and 503 calls our prostate model
- **MODEL_CONFIGURATION**: can either be 2d, 3d_fullres, or ensemble; the models we pretrained are 3d_fullres 

Below is an example to generate prostate segmentations for a batch of PROSTATE-DIAGNOSIS images using our pretrained model and TASK_ID 503. Prior to running this command, navigate to the cloned Github directory in your terminal:

```bash
nnUNet_predict -i ./data/prostate_dx/resampled -o ./data/prostate_dx/segmentations/prostates -t 503 -m 3d_fullres --save_npz
```

Example to generate prostate segmentations for a batch of PROSTATE-DIAGNOSIS images using our pretrained model and TASK_ID 501 Prior to running this command, navigate to the cloned Github directory in your terminal:

```bash
nnUNet_predict -i ./data/prostate_dx/resampled -o ./data/prostate_dx/segmentations/seminal_vesicles -t 501 -m 3d_fullres --save_npz
```

## Pretrained Models:
Running an inference with our pretrained SVI-Net or EPE-Net requires running the executable <b>*predict.py*</b> file. The source code of this file can be modified to run these models on your data and <b>PROSTATE-DIAGNOSIS</b> dataset can be used as a toy dataset; examples are included below:

### SVI-Net:
As an example the <b>*predict.py*</b> executable is pre-configured to run the SVI-Net model on the <b>PROSTATE-DIAGNOSIS</b> toy data set. Run the following command in your terminal:

```bash
python predict.py
```

The prediction results, including an ROC-AUC curve and a .csv with predicted biomarkers for each image, will be generated in the <b>*Aneja-Lab-Public-Prostate-MRI-Biomakers/experiments/*</b> directory.

### EPE-Net:
Running pretrained EPE-Net on the PROSTATE-DIAGNOSIS toy data set requires some adjustments in the <b>*predict.py*</b> file. These adjusments are commented in the source code and are shown below:

```python
# lines 34 through 36:
from models.SVI_Net_final import BoxNet, binary_acc, sigmoid_acc, cross_matrix, roc_auc, f1_score
# use the following line if running EPE-Net and comment out the previous line:
#from models.EPE_Net_final import BoxNet, binary_acc, sigmoid_acc, cross_matrix, roc_auc, f1_score
```

```python
# lines 50 through 54:
# Change this path to the directory with the arrays of images in reference to your project directory
data_directory = 'data/prostate_dx/arrays/seminal_vesicles' 

# Use the following line if running EPE-Net and comment out the previous line:
#data_directory = 'data/prostate_dx/arrays/prostates'
```

```python
# lines 56 through 58
# select the name of the model that you want to run the inference with
# change to "SVI_Net_final" to "EPE_Net_final" if you want to run EPE-Net
model_name = 'SVI_Net_final'
```

```python
# lines 67 through 75
    # opens the stored dictionary with svi labels
    # change to "tcia_svi_labels.pkl" to "tcia_epe_labels.pkl" if running EPE-Net
    with open(os.path.join(project_root, data_directory, 'tcia_svi_labels.pkl'), 'rb') as handle:
        labels = pickle.load(handle)

    # opens the stored dictionary with svi partition of patients, they are all used for testing
    # change to "tcia_svi_partition.pkl" to "tcia_epe_paritition.pkl" if running EPE-Net
    with open(os.path.join(project_root, data_directory, 'tcia_svi_partition.pkl'), 'rb') as handle:
        partition_patients = pickle.load(handle)
```
After these changes are made, run the following command in your terminal:

```bash
python predict.py
```

The prediction results, including an ROC-AUC curve and a .csv with predicted labels, will be generated in the <b>*Aneja-Lab-Public-Prostate-MRI-Biomakers/experiments/*</b> directory.

## Training Models:
Training a model requires running the executable <b>*train.py*</b> file. The source code of this file can be modified to run these models on your data and <b>PROSTATE-DIAGNOSIS</b> dataset can be used as a toy dataset. Run the following command in your terminal:

```bash
python train.py
```

## References:
Hossain, S., Hossain, S., Sritharan, D., Fu, D., Nene, A., Hossain, J., Chadha, S., Kim, I., Lin, M., Aboian, M. and Aneja, S., 2025. Deep Learning Identified Extra-Prostatic Extension and Seminal Vesicle Invasion as an MRI Biomarker for Prostate Cancer Outcomes. medRxiv, pp.2024-12.(https://www.medrxiv.org/content/10.1101/2024.12.31.24319822v1)

[Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.](https://www.nature.com/articles/s41592-020-01008-z)
 
