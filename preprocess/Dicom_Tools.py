#!/usr/bin/env python3
# -------------------------------------------------------------------------------
# Author: Samira Masoudi
# Date:   11.07.2019
# -------------------------------------------------------------------------------
from __future__ import print_function
import SimpleITK as sitk
import glob
import sys
import os
from os.path import exists
import json
from utils import *
from skimage.morphology import disk, dilation
import warnings
import numpy as np
import cv2
import matplotlib as plt
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def Get_the_image_list(Input_path):
    image_list = []
    series_reader = sitk.ImageSeriesReader()
    try:
        series_IDs = series_reader.GetGDCMSeriesIDs(Input_path)
        series_file_names = series_reader.GetGDCMSeriesFileNames(Input_path, series_IDs[0])
        image_reader = sitk.ImageFileReader()
        image_reader.LoadPrivateTagsOn()
        for file_name in series_file_names:
            image_reader.SetFileName(file_name)
            image_list.append(image_reader.Execute())
        return image_list
    except Exception:
        return image_list
def dicom_read (Input_path):
    """
    Reading Dicom files from Input path to dicom image series
    :param Input_path: path to dicom folder which contains dicom series
    :return: 3D Array of the dicom image
    """
    print("Reading Dicom file from:", Input_path )
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(Input_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image
def nifti_read (Input_path):
    """
    Reading Dicom files from Input path to dicom image series
    :param Input_path: path to dicom folder which contains dicom series
    :return: 3D Array of the dicom image
    """
    #print("Reading Nifti from: ", Input_path)
    image = sitk.ReadImage(Input_path, sitk.sitkFloat32)
    return image
def nifti_write(image,output_dir,output_name=None,OutputPixelType='Uint16'):
    """
    Saving an image in either formats Uint8, Uint16
    :param Input_path: path to dicom folder which contains dicom series
    :return: Saving an image in either formats Uint8, Uint16
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if output_name is None:
            output_name='dicom_image.nii'
    # castImageFilter = sitk.CastImageFilter()
    # castImageFilter.SetOutputPixelType(sitk.sitkUInt8)
    # image = castImageFilter.Execute(image)
    if OutputPixelType=='Uint16':
        cast_type = sitk.sitkInt16
    else:
        cast_type = sitk.sitkFloat32
    sitk.WriteImage(sitk.Cast(image, cast_type), os.path.join(output_dir,output_name))
    #sitk.WriteImage(image, os.path.join(output_dir,output_name))
    return 1
def DicomWrite(Image,output_dir,OutputPixelType='Uint16',Referenced_dicom_image_directory='',Modality='MRI',Anonymized=False,Image_Type='Secondary'):
    """
    Saving an image in dicom format at output_dir
    :param image: could be "a dicom Image" or "a numpy array with Z direction as in ITS FIRST DIMENSION"
    :return:
    """
    if OutputPixelType=='Uint16':
        OutputPixelType=16
    else:
        OutputPixelType=8
    if not exists(output_dir):
        makedirs(output_dir)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    try:
        array= sitk.GetArrayFromImage(Image)
        Spacing = Image.GetSpacing()
        Origin = Image.GetOrigin()
        Direction=Image.GetDirection()
        SIZE=array.shape
        # print(SIZE)
    except Exception:
        array=Image
        Spacing = [1,1,1]
        Origin = [0,0,0]
        Direction=[1,0,0,0,0,1]
        SIZE=array.shape
        Path = getcwd()
        # with open(join(Path,'Met_Data.json'),'w') as Meta_out:
        #         json.dump(Meta_data,Meta_out)
        with open(join(Path, 'Meta_Data.json'), 'r') as Meta_out:
            Meta_data = json.load(Meta_out)
    modification_date = time.strftime("%Y%m%d")
    modification_time = time.strftime("%H%M%S")

    image_list = Get_the_image_list(Referenced_dicom_image_directory)

    Meta_data={}
    annonymized_tags = {"0010|0010":"Patient Unknown",
                            "0010|0020": str(random.randint(0,10000000)),
                            "0010|0030": "19700707",
                            # "0020|000D": Study Instance UID, for machine consumption
                            "0020|0010": str(random.randint(0,10000000)),
                            "0008|0012":  modification_date,
                            "0008|0020": modification_date,
                            "0008|0021": modification_date,
                            "0008|0022": modification_date,
                            "0008|0023": modification_date,
                            "0010|0030": modification_date,
                            "0040|0244": modification_date,
                            "0040|0250": modification_date,
                            "0008|0030": '080808',
                            "0008|0031": '090909',
                            "0008|0032": '090909',
                            "0008|0033": '090909',
                            "0040|0009": '6000.0123456789',
                            "0040|0245": '080808',
                            "0040|0251": '080808',
                            "0040|0253": '123456789123',
                            "0018|0080": '4000.123456789',
                            "0018|0080": '120',
                            "0008|0070" : "X Medical Sys",
                            "0008|0080" : "Institution_A",
                            "0008|0090" : "Dr. X",
                            "0008|1010" : "XYZ",
                            "0032|1033": "Department A",
                            "0008|1090" : "Philips",
                            "2005|0014" : "Philips",
                            "0010|0040" : "Unknown",
                            "0010|1020": "2",
                            "0010|1030": "87",
                            "0010|2110": "Other",
                            "0010|21C0": "None",
                            "0020|0011": str(random.randint(0,1000)),
                            "0018|0050": str(Spacing[2]),
                            "0018|0088": str(Spacing[2]),
                            "0028|0010": str(SIZE[1]),
                            "0028|0011": str(SIZE[2]),
                            "6000|0010": str(SIZE[1]),
                            "6000|0011": str(SIZE[2]),
                            "0028|0100": str(OutputPixelType),
                            "0028|0101": str(12),
                            "0028|0102": str(11),
                            "0028|0103": str(0),
                            "07a1|1002":str(SIZE[0])
                            }


    for j in range(Image.GetDepth()):
        image_slice = Image[:, :, j]
        if len(image_list):
            Meta_data={}
            i = np.int((j * Spacing[2]) / int(image_list[0].GetMetaData("0018|0050")))
            original_slice = image_list[i]
            for k in original_slice.GetMetaDataKeys():
                Meta_data.update({k: original_slice.GetMetaData(k)})
        if Anonymized:
            for k,tag in annonymized_tags.items():
                Meta_data.update({k: tag})
        # if flag:
        for k in Meta_data.keys():
            image_slice.SetMetaData(k, Meta_data[k])
        Phys_loc = (j * Spacing[2]) + Origin[2]

        image_slice.SetMetaData("0028|0100", str(OutputPixelType))
        image_slice.SetMetaData("0028|0101", str(12))
        image_slice.SetMetaData("0028|0102", "11")
        image_slice.SetMetaData("0028|0103",  str(0))

        image_slice.SetMetaData("0008|0013", modification_time)
        image_slice.SetMetaData("0008|0030", modification_time)
        image_slice.SetMetaData("0008|0021", modification_date)

        image_slice.SetMetaData('0020|0013', str(Image.GetDepth() - j))
        # Meta_data.update({'Slice_number': })
        # image_slice.SetMetaData('0020|0032', Position)
        # Position = (original_slice.GetMetaData('0020|0032')).replace(
        #     (((original_slice.GetMetaData('0020|0032')).split("\\"))[-1]), str(Phys_loc))
        image_slice.SetMetaData('0020|1041', str(Phys_loc))

        Window_Center = (np.min(array)+np.max(array))/2
        Window_Width = np.max(array) - np.min(array)

        image_slice.SetMetaData('0028|1050', str(Window_Center))
        image_slice.SetMetaData('0028|1051', str(Window_Width) )

        # image_slice.SetMetaData('0028|0106', str())
        # image_slice.SetMetaData('0020|0107',  str(np.max(array)))

        image_slice.SetMetaData("0008|103E", Modality+'_'+Image_Type)
        image_slice.SetMetaData("0008|0008", Image_Type)
        image_slice.SetMetaData("0020|000e",
                                "1.2.826.0.1.3680043.2.1125." + modification_date+ ".1" + modification_time)
        # image_slice.SetMetaData("0020|000e", "3.3.333.113704.7.32." + modification_date + ".1" + modification_time),
        # Series Instance UID
        image_slice.SetMetaData("0020|0032",'\\'.join(map(str, (0,0,Phys_loc))))
        # image_slice.SetMetaData('0020|0032', (original_slice.GetMetaData('0020|0032')).replace(
        #                 (((original_slice.GetMetaData('0020|0032')).split("\\"))[-1]), str(Phys_loc)))
        image_slice.SetMetaData("0020|0037",'\\'.join(map(str, (Direction[0], Direction[3], Direction[6],  # Image Orientation (Patient)
                             Direction[1], Direction[4], Direction[7]))))
        image_slice.SetMetaData("0028|0030", '\\'.join(map(str, (Spacing[0],Spacing[1]))))
        # print('works')
        writer.SetFileName(join(output_dir, str(j) + '.dcm'))
        writer.Execute(image_slice)


    return 1
def dicom_bias_correct(image):
    """"
    For more information please see: https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html
    """
    inputImage = sitk.Cast(image, sitk.sitkFloat32)
    maskImage = sitk.OtsuThreshold(image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    # numberFittingLevels = 4
    imageB = corrector.Execute(inputImage, maskImage)
    imageB.SetSpacing(image.GetSpacing())
    imageB.SetOrigin(image.GetOrigin())
    imageB.SetDirection(image.GetDirection())
    return imageB
def save_dicom_as_png_slices(Image, path, patient, normalize=False, OutputPixelType='Uint16'):
    """
    :param Image:
    :param path: path to save the slices
    :param patient: Image name
    :return: Saves the image slices in png format
    """
    if OutputPixelType=='Uint16':
        OutputPixelType=16
    else:
        OutputPixelType=8
    if not os.path.isdir(path):
        os.makedirs(path)
    array=sitk.GetArrayFromImage(Image)
    if normalize:
        array = normalize(array, N=(2**int(OutputPixelType))-1)
    for i in range(array.shape[0]):
        cv2.imwrite(os.path.join(path, patient + '_' + str(i) + '.png'),array[i, ...])
def save_dicom_as_NPY(image,path,patient):

    array=sitk.GetArrayFromImage(image)
    #array[array>200.0]=200.0
    #array = np.expand_dims(array, axis=-1)
    # array=normalize(array,N=256*256-1)
    np.save(os.path.join(path, patient+'.npy'), array)
def Slice_matching(New,Old):
    """Old and New are different versions (sampled from teh same image)
    Old is the address to version A of the slides
    New is the address to the version B of the  slides
    Produces a dictioanry that shows which instance number
    in version A should be assigned to which instance in version B
    Dic={A==>B}"""
    file_reader = sitk.ImageFileReader()
    Old_slides=glob.glob(join(Old,'*'))
    Old_slds=[f for f in Old_slides if ('VERSION' not in f)]
    z_book=np.zeros((len(Old_slds),1))
    Ins_book=np.zeros((len(Old_slds),1))
    for i0,slide in enumerate(Old_slds):
      # if 'VERSION' not in slide:
        file_name = slide
        file_reader.SetFileName(file_name)
        file_reader.ReadImageInformation()
        Instance_number= int(file_reader.GetMetaData('0020|0013'))
        Slice_location= round(np.float(file_reader.GetMetaData('0020|1041')),1)
        z_book[i0]=Slice_location
        Ins_book[i0] = Instance_number-1

    New_slides=glob.glob(join(New,'*'))
    New_slds=[f for f in New_slides if ('VERSION' not in f)]
    z_book1 = np.zeros((len(New_slds),1))
    Ins_book1 = np.zeros((len(New_slds),1))
    for i1,slide in enumerate(New_slds):
      # if 'VERSION' not in slide:
        file_name = slide
        # data_directory = '.'

        # Read the file's meta-information without reading bulk pixel data
        file_reader.SetFileName(file_name)
        file_reader.ReadImageInformation()
        Instance_number= int(file_reader.GetMetaData('0020|0013'))
        Slice_location= round(np.float(file_reader.GetMetaData('0020|1041')),1)
        z_book1[i1] = Slice_location
        Ins_book1[i1] = Instance_number-1

    Dic={}
    #Remove the bias on Instance numbers
    Ins_book=Ins_book-np.tile(np.min(Ins_book,axis=0),(len(Old_slds),1))
    for i,ins in enumerate(Ins_book):
        df_ind = pd.Index(np.abs(z_book1-np.tile(z_book[i],(len(New_slds),1))))
        j=df_ind.argmin()
        Dic.update({int(ins): int(Ins_book1[j])})
    return Dic
def clip_dicom(image,WL,WW):
    """
    Clipping or windowing the Input 2D slice
    :param Slice: Input 2D array
    :param WL: Wl is the center of the threshold interval
    :param WW: WW is half the length of the threshold interval
    :return: Input 2D array which values are clipped at [WL-WW, WL+WW]
    """
    scan=sitk.GetArrayFromImage(image)
    scan1=np.zeros(scan.shape)
    for i in range(scan.shape[0]):
        Slice=scan[i,...]
        Slice[Slice < (WL - WW)] = np.floor(WL - WW)
        Slice[Slice > (WL + WW)] = np.floor(WL + WW)
        scan1[i,...]=Slice
    image_clipped=sitk.GetImageFromArray(scan1)
    image_clipped.SetSpacing(image.GetSpacing())
    image_clipped.SetOrigin(image.GetOrigin())
    image_clipped.SetDirection(image.GetDirection())

    return image_clipped
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm
def clip_1(scan,WW):
    """
    Clipping or windowing the Input 2D slice
    :param Slice: Input 2D array
    :param WL: Wl is the center of the threshold interval
    :param WW: WW is half the length of the threshold interval
    :return: Input 2D array which values are clipped at [WL-WW, WL+WW]
    """
    # scan=sitk.GetArrayFromImage(image)
    scan1=np.zeros(scan.shape)
    # if (WL-WW > np.max(Slice):
    for i in range(scan.shape[0]):
        Slice=scan[i,...]
        # Slice[Slice < ( WW)] = WW
        Slice[Slice > (WW)] = WW
        scan1[i,...]=Slice
    return scan1
def dicom_apply_mask(mask, image, OutputPixelType='Uint16'):
    """
        :param dcm1: Should be in harmony with
        :param dcm2: and vice versa.
        :return:dcm3
        """
    if OutputPixelType == 'Uint16':
        OutputPixelType = np.uint16
    else:
        OutputPixelType = np.uint16
    kernel = np.ones((6, 6, 6), OutputPixelType)
    img1 = sitk.GetArrayFromImage(mask)
    img2 = sitk.GetArrayFromImage(image)
    img1[img1 == 2] = 1
    img1 = dilation(img1, selem=kernel)
    img3 = img2 * img1
    #img3[img1 == 1] = stats.zscore(img3[img1 == 1])
    dcm3 = sitk.GetImageFromArray(img3)
    dcm3 = sitk.Cast(dcm3, sitk.sitkInt16)
    dcm3.SetSpacing(image.GetSpacing())
    dcm3.SetOrigin(image.GetOrigin())
    dcm3.SetDirection(image.GetDirection())
    return dcm3

def dicom_apply_mask_2(mask, image, OutputPixelType='Uint16'):
    """
        :param dcm1: Should be in harmony with
        :param dcm2: and vice versa.
        :return:dcm3
        """
    if OutputPixelType == 'Uint16':
        OutputPixelType = np.uint16
    else:
        OutputPixelType = np.uint16
    kernel = np.ones((6, 6, 6), OutputPixelType)
    img1 = sitk.GetArrayFromImage(mask)
    img2 = sitk.GetArrayFromImage(image)
    x_ori =img2.shape[1]
    y_ori =img2.shape[2]
    z_ori =img2.shape[0]
    img1[img1 == 2] = 1
    img1 = dilation(img1, selem=kernel)

    x = np.where(img1 != 0)
    print(img1.shape)
    print(img2.shape)
    x_min = min(x[1])
    x_max = max(x[1])
    y_min = min(x[2])
    y_max = max(x[2])
    z_min = min(x[0])
    z_max = max(x[0])

    x_diff = x_max - x_min
    y_diff = y_max - y_min
    z_diff = z_max - z_min

    z_sides = (60 - z_diff) / 2
    x_sides = (200 - x_diff) / 2
    y_sides = (200 - y_diff) / 2

    bb_x_min = round(x_min - x_sides)
    bb_x_max = round(x_max + x_sides)
    bb_y_min = round(y_min - y_sides)
    bb_y_max = round(y_max + y_sides)
    bb_z_min = round(z_min - z_sides)
    bb_z_max = round(z_max + z_sides)

    x_ori =img2.shape[1]
    y_ori =img2.shape[2]
    z_ori =img2.shape[0]

    x_over = False
    x_under = False
    y_over = False
    y_under = False
    z_over = False
    z_under = False

    if bb_x_max > x_ori:
        bb_x_max = x_ori
        x_right = 200 - (x_ori - bb_x_min)
        x_over = True

    if bb_x_min < 0:
        bb_x_min = 0
        x_under = True
        x_left = 200 - (bb_x_max)

    if bb_y_max > y_ori:
        bb_y_max = y_ori
        y_top = 200 - (y_ori - bb_y_min)
        y_over = True

    if bb_y_min < 0:
        bb_y_min = 0
        y_under = True
        y_bottom = 200 - (bb_x_max)

    if bb_z_max > z_ori:
        bb_z_max = z_ori
        z_in = 60 - (z_ori - bb_z_min)
        z_over = True

    if bb_z_min < 0:
        bb_z_min = 0
        z_under = True
        z_out = 60 - (bb_z_max)

    out_arr = img2[bb_z_min:bb_z_max, bb_x_min:bb_x_max, bb_y_min:bb_y_max]

    if z_over:
        pad_width = ((0, z_in), (0, 0), (0, 0))
        out_arr = np.pad(out_arr, pad_width, mode='constant', constant_values=0)

    if z_under:
        pad_width = ((z_out, 0), (0, 0), (0, 0))
        out_arr = np.pad(out_arr, pad_width, mode='constant', constant_values=0)

    if x_over:
        pad_width = ((0, 0), (0, x_right), (0, 0))
        out_arr = np.pad(out_arr, pad_width, mode='constant', constant_values=0)

    if x_under:
        pad_width = ((0, 0), (x_left, 0), (0, 0))
        out_arr = np.pad(out_arr, pad_width, mode='constant', constant_values=0)

    if y_over:
        pad_width = ((0, 0), (0, 0), (0, y_top))
        out_arr = np.pad(out_arr, pad_width, mode='constant', constant_values=0)

    if y_under:
        pad_width = ((0, 0), (0, 0), (y_bottom, 0))
        out_arr = np.pad(out_arr, pad_width, mode='constant', constant_values=0)

    print(out_arr.shape)

    '''Output_arr = img1[x_min:x_max, y_min:y_max, z_min:z_max]
    Image=sitk.GetImageFromArray(Output_arr)
    Image.SetSpacing(image.GetSpacing())
    Image.SetOrigin(image.GetOrigin())
    Image.SetDirection(image.GetDirection())

    array = sitk.GetArrayFromImage(Image)
    x=array.shape[1]
    y=array.shape[2]
    z=array.shape[0]
    New_size=[200,200,50]
    Old_size =[y,x,z]
    constant=1
    filt = sitk.ConstantPadImageFilter()
    filt.SetConstant(constant)
    try:
        Upper_bound=[(New_size[0]-Old_size[0])//2,(New_size[1]-Old_size[1])//2,(New_size[2]-Old_size[2])//2]
        Lower_bound=[(New_size[0]-Old_size[0])//2,(New_size[1]-Old_size[1])//2,(New_size[2]-Old_size[2])//2]
        if (2*Upper_bound[0]+Old_size[0]) == (New_size[1]-1):
            Upper_bound[0] += 1
        if (2*Upper_bound[1]+Old_size[1]) == (New_size[0]-1):
            Upper_bound[1] += 1
        if (2*Upper_bound[2]+Old_size[2]) == (New_size[2]-1):
            Upper_bound[2] += 1
        print(Upper_bound)
        filt.SetPadUpperBound(Upper_bound)
        filt.SetPadLowerBound(Lower_bound)
        padded_img = filt.Execute(Image)
        #sitk.sitkFloat32
        padded_img = sitk.Cast(padded_img, sitk.sitkInt16)
    except Exception:
        New_size = [1000, 1000, 1000]
        Upper_bound=[(New_size[0]-Old_size[0])//2,(New_size[1]-Old_size[1])//2,(New_size[2]-Old_size[2])//2]
        Lower_bound=[(New_size[0]-Old_size[0])//2,(New_size[1]-Old_size[1])//2,(New_size[2]-Old_size[2])//2]
        if (2*Upper_bound[0]+Old_size[0]) == (New_size[1]-1):
            Upper_bound[0] += 1
        if (2*Upper_bound[1]+Old_size[1]) == (New_size[0]-1):
            Upper_bound[1] += 1
        if (2*Upper_bound[2]+Old_size[2]) == (New_size[2]-1):
            Upper_bound[2] += 1
        print(Upper_bound)
        filt.SetPadUpperBound(Upper_bound)
        filt.SetPadLowerBound(Lower_bound)
        padded_img = filt.Execute(Image)
        #sitk.sitkFloat32
        padded_img = sitk.Cast(padded_img, sitk.sitkInt16)
        padded_img = crop_3D(padded_img, [200, 200, 50], [500, 500, 500])

    array = sitk.GetArrayFromImage(padded_img)
    x=array.shape[1]
    y=array.shape[2]
    z=array.shape[0]
    New_size=[y_ori,x_ori,z_ori]
    Old_size =[y,x,z]
    constant=0
    filt = sitk.ConstantPadImageFilter()
    filt.SetConstant(constant)

    Upper_bound=[(New_size[0]-Old_size[0])//2,(New_size[1]-Old_size[1])//2,(New_size[2]-Old_size[2])//2]
    Lower_bound=[(New_size[0]-Old_size[0])//2,(New_size[1]-Old_size[1])//2,(New_size[2]-Old_size[2])//2]
    if (2*Upper_bound[0]+Old_size[0]) == (New_size[1]-1):
        Upper_bound[0] += 1
    if (2*Upper_bound[1]+Old_size[1]) == (New_size[0]-1):
        Upper_bound[1] += 1
    if (2*Upper_bound[2]+Old_size[2]) == (New_size[2]-1):
        Upper_bound[2] += 1
    print(Upper_bound)
    filt.SetPadUpperBound(Upper_bound)
    filt.SetPadLowerBound(Lower_bound)
    padded_img = filt.Execute(padded_img)
    #sitk.sitkFloat32
    img1 = sitk.Cast(padded_img, sitk.sitkInt16)

    img1 = sitk.GetArrayFromImage(img1)
    x = np.where(img1 != 0)
    x_min = min(x[0])
    x_max = max(x[0])
    y_min = min(x[1])
    y_max = max(x[1])
    z_min = min(x[2])
    z_max = max(x[2])
    Output_arr = img2[x_min:x_max, y_min:y_max, z_min:z_max]
    Image=sitk.GetImageFromArray(Output_arr)
    Image.SetSpacing(image.GetSpacing())
    Image.SetOrigin(image.GetOrigin())
    Image.SetDirection(image.GetDirection())'''

    '''img3 = img2 * img1
    #img3[img1 == 1] = stats.zscore(img3[img1 == 1])'''
    dcm3 = sitk.GetImageFromArray(out_arr)
    dcm3 = sitk.Cast(dcm3, sitk.sitkFloat64)
    dcm3.SetSpacing(image.GetSpacing())
    dcm3.SetOrigin(image.GetOrigin())
    dcm3.SetDirection(image.GetDirection())
    return dcm3


def dicom_apply_mask_adc(mask, image, OutputPixelType='Uint16'):
    """
        :param dcm1: Should be in harmony with
        :param dcm2: and vice versa.
        :return:dcm3
        """
    if OutputPixelType == 'Uint16':
        OutputPixelType = np.uint16
    else:
        OutputPixelType = np.uint16
    kernel = np.ones((6, 6, 6), OutputPixelType)
    img1 = sitk.GetArrayFromImage(mask)
    img2 = sitk.GetArrayFromImage(image)
    img1[img1 == 2] = 1
    img1 = dilation(img1, selem=kernel)
    img3 = img2 * img1
    mini = np.min(img3[img1 == 1])
    maxi = np.max(img3[img1 == 1])
    img3[img1 == 1] = (img3[img1 == 1]-mini)/(maxi-mini+1e-6)
    dcm3 = sitk.GetImageFromArray(img3)
    dcm3 = sitk.Cast(dcm3, sitk.sitkFloat32)
    dcm3.SetSpacing(mask.GetSpacing())
    dcm3.SetOrigin(mask.GetOrigin())
    dcm3.SetDirection(mask.GetDirection())
    return dcm3

def dicom_matmul(mask,image,organ_i,OutputPixelType='Uint16'):
    """
    :param dcm1: Should be in harmony with
    :param dcm2: and vice versa.
    :return:dcm3
    """
    if OutputPixelType=='Uint16':
        OutputPixelType=np.uint16
    else:
        OutputPixelType=np.uint16
    kernel = np.ones((3, 3, 3), OutputPixelType)
    img1 = sitk.GetArrayFromImage(mask)
    img2 = sitk.GetArrayFromImage(image)
    # histo, bins = np.histogram(img2.flatten(), 10)
    # print(bins,histo)
    img2 = dilation(img2, selem=kernel)
    # histo, bins = np.histogram(img1.flatten(), 10)
    # print(bins,histo)
    # print(np.max(img1))
    organ = (img1 == organ_i).astype(int)
    img3 = img2*organ
    # print(img3.max(),img3.min())
    # histo, bins = np.histogram(img3.flatten(), 10)
    # histo=normalize(histo)
    # print(histo,bins)
    # Bin = bins[np.min(np.where(histo<5e-3))]
    # img3=clip_1(img3,Bin)
    # print(img1.shape,img3.shape)
    dcm3 = sitk.GetImageFromArray(img3)
    dcm3.SetSpacing(mask.GetSpacing())
    dcm3.SetOrigin(mask.GetOrigin())
    dcm3.SetDirection(mask.GetDirection())
    return dcm3
def organ_iso(mask, organ_i, OutputPixelType='Uint16'):
    if OutputPixelType=='Uint16':
        OutputPixelType=np.uint16
    else:
        OutputPixelType=np.uint16
    img1 = sitk.GetArrayFromImage(mask)
    organ = (img1 == organ_i).astype(int)
    dcm = sitk.GetImageFromArray(organ)
    dcm.SetSpacing(mask.GetSpacing())
    dcm.SetOrigin(mask.GetOrigin())
    dcm.SetDirection(mask.GetDirection())
    return dcm
def Dicom_dilate(dcm1,kernel,OutputPixelType='Uint16'):
    """

    :param dcm1: Should be in harmony with
    :param dcm2: and vice versa.
    :return:dcm3
    """
    if OutputPixelType=='Uint16':
        OutputPixelType=np.uint16
    else:
        OutputPixelType=np.uint16
    # kernel = np.ones((3, 3, 3), OutputPixelType)
    img1 = sitk.GetArrayFromImage(dcm1)
    # img2 = sitk.GetArrayFromImage(dcm2)
    # histo, bins = np.histogram(img2.flatten(), 100)
    # print(bins,histo)
    img3 = dilation(img1, selem=kernel)
    # histo, bins = np.histogram(img2.flatten(), 100)
    # print(bins,histo)
    # img3 = img1*img2
    # print(img3.max(),img3.min())
    # histo, bins = np.histogram(img3.flatten(), 10)
    # histo=normalize(histo)
    # print(histo,bins)
    # Bin = bins[np.min(np.where(histo<5e-3))]
    # img3=clip_1(img3,Bin)
    # print(img1.shape,img3.shape)
    dcm3 = sitk.GetImageFromArray(img3)
    dcm3.SetSpacing(dcm1.GetSpacing())
    dcm3.SetOrigin(dcm1.GetOrigin())
    dcm3.SetDirection(dcm1.GetDirection())
    return dcm3
def dicom_concatenate(dcm1, dcm2, OutputPixelType='Uint16'):
    if OutputPixelType=='Uint16':
        OutputPixelType=np.uint16
    else:
        OutputPixelType=np.uint16
    img1 = sitk.GetArrayFromImage(dcm1)
    img2 = sitk.GetArrayFromImage(dcm2)
    img3 = np.vstack((img1,img2))
    img3 = img3[~(img3==0).all((2,1))]
    dcm3 = sitk.GetImageFromArray(img3)
    dcm3.SetSpacing(dcm1.GetSpacing())
    dcm3.SetOrigin(dcm1.GetOrigin())
    dcm3.SetDirection(dcm1.GetDirection())
    return dcm3
def dicom_combine(dcm1, dcm2, OutputPixelType='Uint16'):
    if OutputPixelType=='Uint16':
        OutputPixelType=np.uint16
    else:
        OutputPixelType=np.uint16
    img1 = sitk.GetArrayFromImage(dcm1)
    img2 = sitk.GetArrayFromImage(dcm2)
    print(img1.shape)
    print(img2.shape)
    img3 = np.stack((img1,img2), axis=-1)
    dcm3 = sitk.GetImageFromArray(img3)
    dcm3.SetSpacing(dcm1.GetSpacing())
    dcm3.SetOrigin(dcm1.GetOrigin())
    dcm3.SetDirection(dcm1.GetDirection())
    return dcm3
def dicom_list_to_npy(dcm_list, OutputPixelType='Uint16'):
    if OutputPixelType=='Uint16':
        OutputPixelType=np.uint16
    else:
        OutputPixelType=np.uint16
    list = []
    for f in dcm_list:
        img = sitk.GetArrayFromImage(f)
        list.append(img)
    array = np.stack(list, axis=0)
    dcm = sitk.GetImageFromArray(array)
    return array, dcm
def Resample(img0,img_i):
    """Resmlaing image_i accoring to image0"""
    Filter=sitk.ResampleImageFilter()
    Filter.SetReferenceImage(img0)
    Filter.SetInterpolator(sitk.sitkBSpline)
    img_o=Filter.Execute(img_i)
    return img_o
def resample_3D(image,New_Spacing,New_Size=None,OutputPixelType='Uint16',mask=None):
    """
    Image will be resampled to the New_Spacing as the desired size
                    or
    Image will be resampled to the New_Size as the desired size
    :param image:
    :param New_Spacing: is a triple in form of [x,y,z] for spacing in x, y, and z
    :param New_Size: is a triple in form of [x,y,z] for diemsnions in x, y, and z
    :param
    :return:
    """
    if OutputPixelType=='Uint16':
        OutputPixelType=16
    else:
        OutputPixelType=8
    resample = sitk.ResampleImageFilter()
    orig_size = np.array(image.GetSize(), dtype=np.int)
    orig_spacing = image.GetSpacing()
    if len(New_Spacing)==3:
        z=New_Spacing[2]
        x=New_Spacing[0]
        y=New_Spacing[1]
    elif len(New_Spacing)==2:
        z=None
        y=New_Spacing[1]
        x=New_Spacing[0]
    elif len(New_Spacing)==0 and len(New_Size)==3:
        z = orig_size[2] * orig_spacing[2] / New_Size[2]
        x = orig_size[0] * orig_spacing[0] / New_Size[0]
        y = orig_size[1] * orig_spacing[1] / New_Size[1]
    elif len(New_Size)==2:
        z=None
        x = orig_size[0] * orig_spacing[0] / New_Size[0]
        y = orig_size[1] * orig_spacing[1] / New_Size[1]
    else:
        warnings.warn("Warning!!! Potentially wrong arguments for size and spacing... Thus no Resizing")
        z = None
        x=orig_spacing[0]
        y=orig_spacing[1]
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    if z is None:
        z=image.GetSpacing()[-1]
        max_num_slices = orig_size[-1]#
    else:
        max_num_slices = (orig_size[2]*orig_spacing[2]/z).astype(np.int)
    if New_Size is None:
        New_Size=[int(orig_size[0] * orig_spacing[0] /x),int(orig_size[1] * orig_spacing[1] /y)]
    new_size = [New_Size[0], New_Size[1],
                max_num_slices]
    new_size = [int(s) for s in new_size]
    new_spacing=[x, y, z]
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    newimage = resample.Execute(sitk.Cast(image, sitk.sitkFloat32))
    output_type = sitk.sitkFloat32
    if OutputPixelType==8:
        output_type = sitk.sitkInt8
    if mask is not None:
        newimagemask =  resample.Execute(sitk.Cast(mask, sitk.sitkFloat32))
        return sitk.Cast(newimage, output_type), sitk.Cast(newimagemask, output_type)
    return sitk.Cast(newimage, output_type)
def Resample_3D_at_slice(image,Size,z=None):
    """
    Size is the desired size so that image will be resampled to this Size
    z is the new spacing that the image should be adapted to this z-spaceing
    :param image:
    :param mask:
    :param size: is a scalar or a tuple for size in x and y
    :param z:
    :param max_num_slices:
    :return:
    """
    resample = sitk.ResampleImageFilter()
    orig_size =  np.array(image.GetSize(), dtype=np.int)
    # print(orig_size)
    orig_spacing = image.GetSpacing()
    resample.SetInterpolator(sitk.sitkLinear)
    max_num_slices = (orig_size[2]-1)*int(orig_spacing[2]/z)+1
    if isinstance(Size, list):
        size=Size
    else:
        size=[Size, Size]
    new_size = [size[0], size[1], max_num_slices] #np.rint(new_size_z).astype(np.int)]  # Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    x=orig_size[0] * orig_spacing[0] / size[0]
    y=orig_size[1] * orig_spacing[1] / size[1]
    new_spacing=[x, y, z]
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    newimage = resample.Execute(sitk.Cast(image, sitk.sitkInt16))
    return sitk.Cast(newimage, sitk.sitkInt16)
def zero_pad_3D(image,Size,mask=None):
    """
    Size is the desired size so that image will be resampled to this Size
    z is the new spacing that the image should be adapted to this z-spaceing
    :param image:
    :param mask:
    :param size: is a scalar or a tuple for size in x and y

    :return:
    """
    array = sitk.GetArrayFromImage(image)
    x=array.shape[1]
    y=array.shape[2]
    z=array.shape[0]
    New_size=[Size[1],Size[0],Size[2]]
    Old_size =[y,x,z]
    constant=0
    # constant=constant.astype('int')
    filt = sitk.ConstantPadImageFilter()
    Upper_bound=[(New_size[0]-Old_size[0])//2,(New_size[1]-Old_size[1])//2,(New_size[2]-Old_size[2])//2]
    Lower_bound=[(New_size[0]-Old_size[0])//2,(New_size[1]-Old_size[1])//2,(New_size[2]-Old_size[2])//2]
    if (2*Upper_bound[0]+Old_size[0]) == (New_size[1]-1):
        Upper_bound[0] += 1
    if (2*Upper_bound[1]+Old_size[1]) == (New_size[0]-1):
        Upper_bound[1] += 1
    if (2*Upper_bound[2]+Old_size[2]) == (New_size[2]-1):
        Upper_bound[2] += 1
    print(Upper_bound)
    filt.SetPadUpperBound(Upper_bound)
    filt.SetPadLowerBound(Lower_bound)
    padded_img = filt.Execute(image)
    #sitk.sitkFloat32
    padded_img = sitk.Cast(padded_img, sitk.sitkInt16)
    return padded_img
def crop_3D(image,Size,Center1):
    """
    Size is the desired size so that image will be resampled to this Size
    z is the new spacing that the image should be adapted to this z-spaceing
    :param image:
    :param mask:
    :param size: is a scalar or a tuple for size in x and y

    :return:
    """
    array=sitk.GetArrayFromImage(image)
    x=array.shape[1]
    y=array.shape[2]
    z=array.shape[0]

    #x_diff = x-size[0]
    #y_diff =

    Output_arr=np.zeros((Size[2],Size[0],Size[1]))
    Center=(np.array([Size[2],Size[0],Size[1]])/2).astype(int)
    Output_arr[max(0, Center[0] - Center1[0]):min(Size[2], Center[0] - Center1[0]+z),max(Center[1] - Center1[1], 0):min(Center[1] - Center1[1]+x, Size[0]),
    max(0, Center[2] - Center1[2]):min(Size[1], Center[2] -Center1[2]+y)] = array[max(0, -Center[0] + Center1[0]):min(z, Center[0] + Center1[0]),max(Center1[1] - Center[1], 0):min(
        Center[1] + Center1[1], x), max(0, -Center[2] + Center1[2]):min(y, Center[2] + Center1[2])]
    # # print(Center[0]-orig_Size[0],Center[0]+orig_Size[0],Center[1]-orig_Size[1],Center[1]+orig_Size[1],Center[2]-orig_Size[2],Center[2]+orig_Size[2])
    # Output_arr[Center[0]-orig_Size0[0]:Center[0]+orig_Size1[0],Center[1]-orig_Size0[1]:Center[1]+orig_Size1[1],Center[2]-orig_Size0[2]:Center[2]+orig_Size1[2]]=array[:,:,:]
    Image=sitk.GetImageFromArray(Output_arr)
    Image.SetSpacing(image.GetSpacing())
    Image.SetOrigin(image.GetOrigin())
    Image.SetDirection(image.GetDirection())
    return sitk.Cast(Image, sitk.sitkInt16)
def crop_2D_mask(image):
    array = sitk.GetArrayFromImage(image)
    x = np.where(array != 0)
    x_min = min(x[0])
    x_max = max(x[0])
    y_min = min(x[1])
    y_max = max(x[1])
    z_min = min(x[2])
    z_max = max(x[2])
    Output_arr = array[x_min:x_max, y_min:y_max, z_min:z_max]
    #print(Output_arr.shape)
    Image=sitk.GetImageFromArray(Output_arr)
    Image.SetSpacing(image.GetSpacing())
    Image.SetOrigin(image.GetOrigin())
    Image.SetDirection(image.GetDirection())
    return sitk.Cast(Image, sitk.sitkInt16), Output_arr.shape #, sitk.sitkInt16
def new_image(array, image):
    image1 = sitk.GetImageFromArray(array)
    image1.SetSpacing(image.GetSpacing())
    image1.SetOrigin(image.GetOrigin())
    image1.SetDirection(image.GetDirection())
    return image1
def histo_normalize_against_self(image):
    data = sitk.GetArrayFromImage(image)
    # shift the data up so that all intensity values turn positive
    data -= np.min(data)
    # Removing the outliers with a probability of occuring less than 5e-3 through histogram computation
    histo, bins = np.histogram(data.flatten(), 10)
    histo = normalize(histo)
    Bin = bins[np.min(np.where(histo < 5e-3))]
    data = np.clip(data, 0, Bin)
    return data
def unit_normalize(image):
    data = sitk.GetArrayFromImage(image)
    mean = data.mean()
    std = data.std()
    var = data.var()
    data_1 = (data - mean) / std
    return data_1
def normalize_min_max(data_arr, n=100, eps=1e-6):
    output = n*(data_arr-np.min(data_arr))/(np.max(data_arr)-np.min(data_arr)+eps)
    return output
def image_convert_dtype(image):
    data = sitk.GetArrayFromImage(image)
    image2 = sitk.GetImageFromArray(data)
    image2 = sitk.Cast(image2, sitk.sitkFloat32)
    image2.SetSpacing(image.GetSpacing())
    image2.SetOrigin(image.GetOrigin())
    image2.SetDirection(image.GetDirection())
    return image2
def register(fixed_path, moving_path_1, out_dir, patient, nbins = 34, sampling_percentage = 0.10, OutputPixelType='Uint16', moving_path_2=None, moving_path_3=None):
    if OutputPixelType == 'Uint16':
        OutputPixelType = 16
    else:
        OutputPixelType = 8

    pixelType = sitk.sitkFloat32

    fixed = sitk.ReadImage(fixed_path, pixelType)
    moving = sitk.ReadImage(moving_path_1, pixelType)
    if moving_path_2 is not None:
        moving_2 = sitk.ReadImage(moving_path_2, pixelType)
    if moving_path_3 is not None:
        moving_3 = sitk.ReadImage(moving_path_3, pixelType)

    #transformDomainMeshSize = [10] * moving.GetDimension()
    #print("transform domain mesh size: ")
    #print(transformDomainMeshSize)
    #tx = sitk.BSplineTransformInitializer(fixed, transformDomainMeshSize)

    #print("Initial Number of Parameters:")
    #print(tx.GetNumberOfParameters())

    #R = sitk.ImageRegistrationMethod()
    #R.SetMetricAsMattesMutualInformation(50)
    #R.SetOptimizerAsGradientDescentLineSearch(
    #    5.0, 100, convergenceMinimumValue=1e-4, convergenceWindowSize=5
    #)

    #R.SetOptimizerScalesFromPhysicalShift()
    #R.SetInitialTransform(tx)
    #R.SetInterpolator(sitk.sitkLinear)
    #R.SetShrinkFactorsPerLevel([6, 2, 1])
    #R.SetSmoothingSigmasPerLevel([6, 2, 1])

    #R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
    #R.AddCommand(
    #    sitk.sitkMultiResolutionIterationEvent, lambda: command_multi_iteration(R)
    #)

    #outTx1 = R.Execute(fixed, moving)

    #print("-------")
    #print(tx)
    #print(outTx1)
    #print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    #print(f" Iteration: {R.GetOptimizerIteration()}")
    #print(f" Metric value: {R.GetMetricValue()}")

    #sitk.WriteTransform(outTx, transform_path)

    #res1 = sitk.ResampleImageFilter()
    #res1.SetReferenceImage(fixed)
    #res1.SetInterpolator(sitk.sitkLinear)
    #res1.SetDefaultPixelValue(100)
    #res1.SetTransform(outTx1)

    #out1 = res1.Execute(moving)
    #out2 = res1.Execute(moving_2)
    #out3 = res1.Execute(moving_3)

    #outim1 = sitk.Cast(out1, pixelType)
    #outim2 = sitk.Cast(out2, pixelType)
    #outim3 = sitk.Cast(out3, pixelType)

    #reg1 = sitk.ImageRegistrationMethod()
    #reg1.SetMetricAsJointHistogramMutualInformation(nbins)
    #reg1.SetMetricSamplingPercentage(sampling_percentage, sitk.sitkWallClock)
    #reg1.SetMetricSamplingStrategy(reg1.RANDOM)
    #reg1.SetOptimizerAsRegularStepGradientDescent(1.0, .001, 200)
    #reg1.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    #reg1.SetInterpolator(sitk.sitkLinear)
    #reg1.SetOptimizerAsRegularStepGradientDescent(
    #    learningRate=1.0,
    #    numberOfIterations=200,
    #    convergenceMinimumValue=1e-5,
    #    convergenceWindowSize=5,
    #)
    #reg1.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    #reg1.SetInterpolator(sitk.sitkLinear)

    #outTx1 = reg1.Execute(fixed, moving)

    #res1 = sitk.ResampleImageFilter()
    #res1.SetReferenceImage(fixed)
    #res1.SetInterpolator(sitk.sitkLinear)
    #res1.SetDefaultPixelValue(100)
    #res1.SetTransform(outTx1)

    #outim1 = res1.Execute(moving)
    #outim2 = res1.Execute(moving_2)
    #outim3 = res1.Execute(moving_3)

    reg2 = sitk.ImageRegistrationMethod()
    reg2.SetMetricAsMattesMutualInformation(nbins)
    reg2.SetMetricSamplingPercentage(sampling_percentage, sitk.sitkWallClock)
    reg2.SetMetricSamplingStrategy(reg2.RANDOM)
    reg2.SetOptimizerAsRegularStepGradientDescent(1.0, .001, 200)
    reg2.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    reg2.SetInterpolator(sitk.sitkLinear)

    outTx2 = reg2.Execute(fixed, moving)
    sitk.WriteTransform(outTx2, os.path.join(out_dir, patient+'.tfm'))

    res2 = sitk.ResampleImageFilter()
    res2.SetReferenceImage(fixed)
    res2.SetInterpolator(sitk.sitkLinear)
    res2.SetDefaultPixelValue(100)
    res2.SetTransform(outTx2)

    out1 = res2.Execute(moving)
    if moving_path_2 is not None:
        out2 = res2.Execute(moving_2)
    if moving_path_3 is not None:
        out3 = res2.Execute(moving_3)

    #translate = sitk.TranslationTransform(fixed.GetDimension())
    #translate.SetOffset((0, -50, 0))

    #res2 = sitk.ResampleImageFilter()
    #res2.SetReferenceImage(fixed)
    #res2.SetInterpolator(sitk.sitkLinear)
    #res2.SetDefaultPixelValue(100)
    #res2.SetTransform(translate)

    #out1_1 = res2.Execute(out1)
    #if moving_path_2 is not None:
    #    out2_2 = res2.Execute(out2)
    #if moving_path_3 is not None:
    #    out3_3 = res2.Execute(out3)

    outim_1 = sitk.Cast(out1, OutputPixelType)
    if moving_path_2 is not None:
        outim_2 = sitk.Cast(out2, OutputPixelType)
    if moving_path_3 is not None:
        outim_3 = sitk.Cast(out3, OutputPixelType)

    if moving_path_2 is None and moving_path_3 is None:
        return outim_1
    if moving_path_2 is not None and moving_path_3 is None:
        return outim_1, outim_2

    return outim_1, outim_2, outim_3

def command_iteration(method):
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():10.5f}"
    )
    print("\t#: ", len(method.GetOptimizerPosition()))


def command_multi_iteration(method):
    print("--------- Resolution Changing ---------")








#----------------------------------------------Test Functions------------------------------------------------------------
##fixed_path = "C:\\Users\shasa\Desktop\Prostate Cancer Research\Prostate MRI\data\preprocess\\normalized_2\YG_1CJSN6ILETKB_0000.nii.gz"
##moving_path = "C:\\Users\shasa\Desktop\Prostate Cancer Research\Prostate MRI\data\preprocess\\normalized_2\YG_1CJSN6ILETKB_0002.nii.gz"
#transform_path = "C:\\Users\shasa\Desktop\Prostate Cancer Research\Prostate MRI\data\preprocess\\transforms\YG_0MMTG52CMAQ5"
#nbins = 32
#sampling_percentage = 0.10


#tx_image = register(fixed_path, moving_path, nbins, sampling_percentage)
#nifti_write(tx_image, "C:\\Users\shasa\Desktop\Prostate Cancer Research\Prostate MRI\data\preprocess\\registered_2", "YG_1CJSN6ILETKB_0002.nii.gz")

