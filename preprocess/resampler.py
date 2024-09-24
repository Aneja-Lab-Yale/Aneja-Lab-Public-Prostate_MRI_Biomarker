# CapsNet Project
# Image re-sampler
# Aneja Lab | Yale School of Medicine
# Developed by Arman Avesta, MD
# Created (10/20/22)
# Updated (--/--/--)

# ----------------------------------------------------- Imports ----------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from dipy.io.image import load_nifti
from dipy.io.image import save_nifti
from collections import OrderedDict
import nibabel as nib
from copy import deepcopy

from skimage.transform import resize
from batchgenerators.augmentations.utils import resize_segmentation

# from tools.mri_slicer import imshow

# --------------------------------------------- Resampling Functions ------------------------------------------------
def resample_patient(data, seg, original_spacing, target_spacing,
                     order_data=3, order_seg=0,
                     force_separate_z=False, order_z_data=0, order_z_seg=0,
                     separate_z_anisotropy_threshold=3):
    """
    :param data: (numpy.array) image data with shape [c, x, y, z]; c: channels
    :param seg: (numpy.array) segmentation with shape [c, x, y, z]
    :param original_spacing: (tuple, list or numpy.array) pixel spacing. e.g. (.7, .7, 3)
    :param target_spacing: (tuple, list or numpy.array) target pixel spacing. e.g. (1, 1, 1)
    :param order_data: (int) order of spline interpolation for data. Default: 3.
    :param order_seg: (int) oder of spline interpolation for segmentation. Default: 0.
    :param force_separate_z: (bool or None)
        - None: dynamically decide how to resample along z, if anisotropy is more than the threshold
        - True: do separately resample along z
        - False: do not separately resample along z
    :param order_z_data: (int) order of spline interpolation, only applies if do_separate_z is True
    :param order_z_seg: (int) order of spline interpolation, only applies if do_separate_z is True
    :param separate_z_anisotropy_threshold: if max_spacing > separate_z_anisotropy_threshold * min_spacing (per axis)
        then resample along lowres axis with order_z_data/order_z_seg instead of order_data/order_seg
    :return: reshaped_data (numpy.array) with new shape [c, xn, yn, zn].
    Notes:
    ------
    Even when force_separate_z=True, if there is more than one high-spacing axis ,e.g. (.5, 1, 1), the
    function does NOT separately resample along z or any axis, because there is not one prominent high-spacing axis.
    """
    assert (data is not None) or (seg is not None)
    if (data is not None) and (seg is not None):
        assert data[0].shape == seg[0].shape

    if data is not None:
        assert data.ndim == 4, "data must be c x y z"
        original_shape = np.array(data[0].shape)
    if seg is not None:
        assert seg.ndim == 4, "seg must be c x y z"
        original_shape = np.array(seg[0].shape)

    original_spacing, target_spacing = np.array(original_spacing), np.array(target_spacing)
    new_shape = np.round(original_spacing / target_spacing * original_shape).astype(int)

    if force_separate_z is None:
        force_separate_z = (max(original_spacing) / min(original_spacing)) > separate_z_anisotropy_threshold

    axis = get_lowres_axis(original_spacing)    # returns None if there is more than one high-spacing axis.
    if axis is None:    # if there is more than one high-spacing axis, don't separately resample along any axis.
        force_separate_z = False


    if data is not None:
        data_reshaped = resample_data_or_seg(data, new_shape,
                                             is_seg=False, axis=axis, order=order_data,
                                             do_separate_z=force_separate_z, order_z=order_z_data)
    else:
        data_reshaped = None

    if seg is not None:
        seg_reshaped = resample_data_or_seg(seg, new_shape,
                                            is_seg=True, axis=axis, order=order_seg,
                                            do_separate_z=force_separate_z, order_z=order_z_seg)
    else:
        seg_reshaped = None

    return data_reshaped, seg_reshaped

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def resample_data_or_seg(data, new_shape, is_seg, axis=None, order=3, do_separate_z=False, order_z=0):
    """
    :param data: shape [c, x, y, z]; c: channels
    :param new_shape: shape [xn, yn, zn]; same number of channels c as input data --> new shape: (c, xn, yn, zn)
    :param is_seg: (bool) is the first argument image data or segmentation data?
    :param axis: (int or None)
    :param order: (int) interpolation order
    :param do_separate_z: True --> resample z with different order (default order_z=0)
    :param order_z: only applies if do_separate_z=True
    :return: resampled data; shape [c, xn, yn, zn]
    """
    assert data.ndim == 4, "data must have the shape [c, x, y, z]"
    assert len(new_shape) == data.ndim - 1, "new_shape must have the shape [x_new, y_new, z_new]"
    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}

    data_dtype = data.dtype
    c, x, y, z = data.shape
    xn, yn, zn = new_shape
    orig_shape = np.array(data[0].shape)    # [x, y, z]
    new_shape = np.array(new_shape)         # [xn, yn, zn]

    if (orig_shape == new_shape).all():
        print("no resampling necessary")
        return data

    data = data.astype(float)

    if not do_separate_z:
        print("no separate z, order", order)
        reshaped_data = np.zeros([c, xn, yn, zn], dtype=data_dtype)
        for ci in range(c):
            reshaped_data[ci, ...] = resize_fn(data[ci, ...], new_shape, order, **kwargs).astype(data_dtype)
        return reshaped_data

    else:       # this else is unnecessary after the return above, but I left it here for code clarity.
        print("separate z, order in z is", order_z, "order in-plane is", order)
        '''
        Here we cannot pre-allocate reshaped_data using np.zeros, because the new number of slices might be different
        from original number of slices. 
        For example, assuming that separate resampling is along z axis, and assuming that 
        original_shape = [4, 150, 156, 30] and new_shape (with channels) = [4, 128, 128, 64], 
        we do in-plane resampling from [150, 156] to [128, 128]. But the resulting image in each channel
        will have the shape [128, 128, 30]. Now, we should re-sample across slices to go from 30 slices to 64 slices.
        '''
        reshaped_data = []             # list of 3D volumes over channels, will covert to np.array at the end.

        n_slices_orig = orig_shape[axis]
        n_slices_new = new_shape[axis]
        if axis == 0:
            new_shape_2d = np.array([yn, zn])
        elif axis == 1:
            new_shape_2d = np.array([xn, zn])
        else:
            new_shape_2d = np.array([xn, yn])

        for ci in range(c):
            '''
            Assuming that separate resampling is over z axis, we make reshaped slices and reshaped_channel as lists
            and at the end convert them into numpy arrays.
            reshaped_slices: [xn, yn, z] 
            reshaped_channel: [xn, yn, zn]
            '''
            reshaped_slices = []        # assuming resampling along z, shape: [xn, yn, z]
            for s in range(n_slices_orig):
                if axis == 0:
                    reshaped_slice = resize_fn(data[ci, s, :, :], new_shape_2d, order, **kwargs).astype(data_dtype)
                elif axis == 1:
                    reshaped_slice = resize_fn(data[ci, :, s, :], new_shape_2d, order, **kwargs).astype(data_dtype)
                else:
                    reshaped_slice = resize_fn(data[ci, :, :, s], new_shape_2d, order, **kwargs).astype(data_dtype)
                reshaped_slices.append(reshaped_slice)

            reshaped_slices = np.stack(reshaped_slices, axis)        # [xn, yn, z]

            if n_slices_orig == n_slices_new:
                reshaped_channel = reshaped_slices.astype(data_dtype)

            else:
                '''
                Assuming axis is 2, here z is not equal to zn --> we should re-sample across slices:
                Here, reshaped_slices has shape [xn, yn, z] --> we should re-sample across slices to go to shape
                [xn, yn, zn]. 
                Since we don't know which axis we are separately resampling (and hence we don't know which
                two dimensions are already resampled in 2D in the previous step and which across-slice dimension 
                is remaining to be resampled), we call the temporary dimensions of reshaped_slices [xt, yt, zt].
                Assuming axis is 2, xt = xn and yt = yn, but zt = z (instead of zn) at this point.
                So we have to interpolate between slices to go from zt slices to zn slices.
                '''
                xt, yt, zt = reshaped_slices.shape
                x_scale, y_scale, z_scale = xt / xn, yt / yn, zt / zn
                map_x, map_y, map_z = np.mgrid[:xn, :yn, :zn]
                map_x = x_scale * (map_x + 0.5) - 0.5
                map_y = y_scale * (map_y + 0.5) - 0.5
                map_z = z_scale * (map_z + 0.5) - 0.5
                coord_map = np.array([map_x, map_y, map_z])
                if not is_seg or order_z == 0:
                    '''
                    if data is not segmentation, or if data is segmentation but the oder of interpolation between
                     slices is 0, we don't need one-hot encoding before interpolation between slices
                    '''
                    reshaped_channel = map_coordinates(reshaped_slices,
                                                       coord_map,
                                                       order=order_z,
                                                       mode='nearest').astype(data_dtype)
                else:
                    '''
                    if data is segmentation and the order of interpolation is more than 0, we need one-hot encoding
                    to prevent change of labels
                    '''
                    reshaped_channel = np.zeros(new_shape, dtype=data_dtype)                        # [xn, yn, zn]
                    unique_labels = np.unique(reshaped_slices)
                    for l in unique_labels:
                        reshaped_slices_onehot = (reshaped_slices == l).astype(float)               # [xn, yn, z]
                        reshaped_channel_onehot = np.round(map_coordinates(reshaped_slices_onehot,  # [xn, yn, zn]
                                                                           coord_map,
                                                                           order=order_z,
                                                                           mode='nearest'))
                        reshaped_channel[reshaped_channel_onehot > 0.5] = l                         # [xn, yn, zn]

            reshaped_data.append(reshaped_channel)

        reshaped_data = np.stack(reshaped_data, axis=0)         # [c, xn, yn, zn]
        return reshaped_data

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def get_lowres_axis(voxel_spacing):
    """
    This function finds which axis (x, y, or z) has the highest spacing between voxels.
    :param voxel_spacing: (tuple, list, or numpy.array) spacing between voxels. e.g. (0.7, 0.7, 3)
    :return: (int or None) the axis with the highest pixel spacing, e.g. 2. If two or more axes have large similar
        spacing --> function returns None.
    Notes:
    ------
    If voxel_spacing is (0.24, 1.25, 1.25), the function returns None because more than one  axis has equally-large
    voxel spacing.
    Beware that we cannot use voxel_spacing.argmax() in our code because it only returns one largest and omits
    multiple maxima. That's why we use np.where to find both more than one maxima.
    If voxel_spacing is (1.25, 1.25, 1.25), the function returns None.
    Explaining the indexing here:
        voxel_spacing = np.array((.7, .7, 3))
        axis = np.where(voxel_spacing == max(voxel_spacing)) --> axis = (array([2]),)
        axis = np.where(voxel_spacing == max(voxel_spacing))[0] --> axis = array([2])
        axis = np.where(voxel_spacing == max(voxel_spacing))[0][0] --> axis = 2
    """
    voxel_spacing = np.array(voxel_spacing)
    axis = np.where(voxel_spacing == max(voxel_spacing))[0]               # find which axes have highest spacing.
    return axis[0] if len(axis) <= 1 else None

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def imshow(img, voxsize=(1,1,1), coords=('L','A','S')):
    """
        Description:
            This function shows 2D/3D MRI slice/volume:
            - 2D: image is shown.
            - 3D: the volume mid-slices in axial, coronal and sagittal planes are shown.
        Inputs:
            - img: 2D/3D numpy array representing an MRI slice/volume/series
                    OR
            - voxsize: voxel size; 3-component tuple or list; default voxsize=(1,1,1)
            - coords: img coordinate system; 3-component character tuple or list; default coords=('L','I','A')
            In case img is a torch batch, voxsize and coords should describe the voxsize and coordinate system of
            each image in the batch.
        Output:
            None (plots img instead)
        Further info:
            You can use load_nifti (in dipy package) to obtain coords of an image, by setting return_coords=True.
            Example:
                from dipy.io.image import load_nifti
                img, affine, voxsize, coords = load_nifti('brainmask.mgz', return_voxsize=True, return_coords=True)
            To learn more about voxel coordinate systems, see:
                http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm
    """
    kwargs = dict(cmap='gray', origin='lower')
    dim = img.ndim
    assert dim in (2, 3), f'image shape: {img.shape}; imshow can only show 2D or 3D images.'

    if dim == 2:
        plt.imshow(img.T, **kwargs)

    elif dim == 3:
        img, voxsize = correct_coordinates(img, voxsize, coords)
        midaxial = img.shape[2] // 2
        midcoronal = img.shape[1] // 2
        midsagittal = img.shape[0] // 2
        axial_aspect_ratio = voxsize[1] / voxsize[0]
        coronal_aspect_ratio = voxsize[2] / voxsize[0]
        sagittal_aspect_ratio = voxsize[2] / voxsize[1]
        # print(f'''
        # axial aspect ratio: {axial_aspect_ratio}
        # coronal aspect ratio: {coronal_aspect_ratio}
        # sagittal aspect ratio: {sagittal_aspect_ratio}
        # ''')

        axial = plt.subplot(2, 3, 1)
        plt.imshow(img[:, :, midaxial, ...].T, **kwargs)
        axial.set_aspect(axial_aspect_ratio)

        coronal = plt.subplot(2, 3, 2)
        plt.imshow(img[:, midcoronal, :, ...].T, **kwargs)
        coronal.set_aspect(coronal_aspect_ratio)

        sagittal = plt.subplot(2, 3, 3)
        plt.imshow(img[midsagittal, :, :, ...].T, **kwargs)
        sagittal.set_aspect(sagittal_aspect_ratio)

    plt.show()


def correct_coordinates(img, voxsize, coords):
    """
    This function re-orients a 3D MRI volume or 4D MRI series  into the standard radiology coordinate system:
    ('L','A','S').
    If img is a 4D MRI series, the last component of the 4D MRI series should represent series volumes
    (e.g. Z / Y / X / seriesVolume).
    If the image coordinate system was not identified when you called the function and you received this error:
        'coords not identified: please go to MRIslicer.imshow and define the coordinate system :)
    you should revise this sub-function!
    Please do the following steps:
    1. Find the image coordinate system by using these commands:
    from dipy.io.image import load_nifti
    img, affine, voxsize, coords = load_nifti("brainmask.mgz", return_voxsize=True, return_coords=True)
    2. Add the new coords by following these steps:
      a. Compared coords with the standard radiology system of coordinates: ('L','A','S').
      b. Use np.swapaxes and np.flip to transform img coordinates to standard radiology system.
    For more info about voxel coordinate systems, see:
    http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm
    """
    dim = img.ndim

    if coords == ('L', 'A', 'S'):
        return img, voxsize
    if coords == ('R', 'A', 'S'):
        img = np.flip(img, 0)
        return img, voxsize
    if coords == ('L', 'I', 'A'):
        img = np.swapaxes(img, 1, 2)
        img = np.flip(img, 2)
        voxsize = (voxsize[0], voxsize[2], voxsize[1])
        return img, voxsize
    if coords == ('L', 'S', 'A'):
        img = np.swapaxes(img, 1, 2)
        voxsize = (voxsize[0], voxsize[2], voxsize[1])
        return img, voxsize
    if coords == ('L', 'P', 'S'):
        img = np.flip(img, 1)
        return img, voxsize
    raise Exception('coords not identified: please go to imshow function and define the coordinate system.')

def correct_orientation(nifti):
    """
    This function re-orients the MRI volume into the standard radiology system, i.e. ('L','A','S') = LAS+,
    and also corrects the affine transform for the volume (from the MRI volume space to the scanner space).
    Input:
        - nifti: MRI volume NIfTI file
    Outputs:
        - corrected nifti: corrected MRI volume in the standard radiology coordinate system: ('L','A','S') = LAS+.
    Notes:
    ------
    nib.io_orientation compares the orientation of nifti with RAS+ system. So if nifti is already in
    RAS+ system, the return from nib.io_orientation(nifti.affine) will be:
    [[0, 1],
     [1, 1],
     [2, 1]]
    If nifti is in LAS+ system, the return would be:
    [[0, -1],           # -1 means that the first axis is flipped compared to RAS+ system.
     [1, 1],
     [2, 1]]
    If nifti is in PIL+ system, the return would be:
    [[1, -1],           # P is the 2nd axis in RAS+ hence 1 (not 0), and is also flipped hence -1.
     [2, -1],           # I is the 3rd axis in RAS+ hence 2, and is also flipped hence -1.
     [0, -1]]           # L is the 1st axis in RAS+ hence 0, and is also flipped hence -1.
    Because we want to save images in LAS+ orientation rather than RAS+, in the code below we find axis 0 and
    negate the 2nd colum, hence going from RAS+ to LAS+. For instance, for PIL+, the orientation will be:
    [[1, -1],
     [2, -1],
     [0, -1]]
    This is PIL+ compared to RAS+. To compare it to LAS+, we should change it to:
    [[1, -1],
     [2, -1],
     [0, 1]]
    That is what this part of the code does:
    orientation[orientation[:, 0] == 0, 1] = - orientation[orientation[:, 0] == 0, 1]
    Another inefficient way of implementing this function is:
    ################################################################################
    original_orientation = nib.io_orientation(nifti.affine)
    target_orientation = nib.axcodes2ornt(('L', 'A', 'S'))
    orientation_transform = nib.ornt_transform(original_orientation, target_orientation)
    return nifti.as_reoriented(orientation_transform)
    ################################################################################
    """
    orientation = nib.io_orientation(nifti.affine)
    orientation[orientation[:, 0] == 0, 1] = - orientation[orientation[:, 0] == 0, 1]
    return nifti.as_reoriented(orientation)

# -------------------------------------------------- Code Testing ------------------------------------------------------

if __name__ == "__main__":

    # Example of image resampling:
    path = "C:\\Users\shasa\Desktop\Prostate Cancer Research\Prostate MRI\data\input\\raw/ProstateDx-01-0001.nii.gz"
    img, affine, voxsize, coords = load_nifti(path, return_voxsize=True, return_coords=True)
    print(coords)
    print(voxsize)
    #img = np.flipud(img)
    #img = np.rot90(img, k=2, axes=(0,1))
    #imshow(img, voxsize)

    #save_nifti("C:\\Users\shasa\Desktop\Prostate Cancer Research\Prostate MRI\data\preprocess\\resampled/YG_WT1RE8L4KBT1_0004.nii.gz", img, affine)

    #print('image shape: ', img.shape)
    #print('voxel size: ', voxsize)
    #print('coords: ', coords)



    #path = "C:\\Users\shasa\Desktop\Prostate Cancer Research\Prostate MRI\data\preprocess\segmentations\YG_0GGJKCPWLG8P.nii.gz"
    #img, affine, voxsize, coords = load_nifti(path, return_voxsize=True, return_coords=True)
    #imshow(img, voxsize)

    #print('image shape: ', img.shape)
    #print('voxel size: ', voxsize)
    #print('coords: ', coords)

    #img = img[None]

    #target_spacing = (0.5, 0.5, 1)
    #z_anisotropy_threshold = 3      # Sajid: you can play with this. I suggest trying 2 and 3.

    #print(voxsize[0])
    #print(target_spacing[0])
    #img2, seg2 = resample_patient(data=img,
    #                              seg=None,
    #                              original_spacing=voxsize,
    #                              target_spacing=target_spacing,
    #                              force_separate_z=None,
    #                              separate_z_anisotropy_threshold=z_anisotropy_threshold)

    #print('image2 shape', img2.shape)
    #imshow(img2[0])

    #x_ratio = target_spacing[0] / voxsize[0]
    #y_ratio = target_spacing[1] / voxsize[1]
    #z_ratio = target_spacing[2] / voxsize[2]

    #S_matrix = np.array([[x_ratio, 0, 0, 0],
    #                     [0, y_ratio, 0, 0],
    #                     [0, 0, z_ratio, 0],
    #                     [0, 0, 0, 1]])

    #affine_tx = affine @ S_matrix




    # Example of segmentation resampling:
    #path = "C:\\Users\shasa\Desktop\segmentations\YG_0GGJKCPWLG8P.nii.gz"
    #seg, affine, voxsize, coords = load_nifti(path, return_voxsize=True, return_coords=True)
    #imshow(seg, voxsize)

    #print('segmentation shape:', seg.shape)
    #print('voxel size', voxsize)
    #seg = seg[None]

    #target_spacing = (0.5, 0.5, 1)
    #z_anisotropy_threshold = 3  # Sajid: you can play with this. I suggest trying 2 and 3.

    #img2, seg2 = resample_patient(data=None,
    #                              seg=seg,
    #                              original_spacing=voxsize,
    #                              target_spacing=target_spacing,
    #                              force_separate_z=None,
    #                              separate_z_anisotropy_threshold=z_anisotropy_threshold)

    #x_ratio = target_spacing[0] / voxsize[0]
    #y_ratio = target_spacing[1] / voxsize[1]
    #z_ratio = target_spacing[2] / voxsize[2]

    #S_matrix = np.array([[x_ratio, 0, 0, 0],
    #                     [0, y_ratio, 0, 0],
    #                     [0, 0, z_ratio, 0],
    #                     [0, 0, 0, 1]])

    #affine_tx = affine @ S_matrix

    #print('segmentation2 shape', seg2.shape)
    #imshow(seg2[0])