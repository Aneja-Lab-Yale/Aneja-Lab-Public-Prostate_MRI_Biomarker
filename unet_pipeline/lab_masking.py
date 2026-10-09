"""Lab GitHub masking helpers, copied from Aneja-Lab-Public-Prostate_MRI_Biomarker.

Source:
  preprocess/Dicom_Tools.py  — dicom_apply_mask, crop_2D_mask, zero_pad_3D, crop_3D
  mri_preprocessing.py       — apply_mask_to_images order

Do not "fix" the exclusive crop (x_min:x_max drops the last voxel). EPE-Net was
trained on arrays built this way. Kernel is a 6x6x6 cube, not the paper's 10 px + KDE.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from skimage.morphology import dilation


def nifti_read(input_path: str):
    return sitk.ReadImage(str(input_path), sitk.sitkFloat32)


def nifti_write(image, output_dir: str, output_name: str, output_pixel_type: str = "Uint16") -> None:
    os.makedirs(output_dir, exist_ok=True)
    cast_type = sitk.sitkInt16 if output_pixel_type == "Uint16" else sitk.sitkFloat32
    sitk.WriteImage(sitk.Cast(image, cast_type), os.path.join(output_dir, output_name))


def save_dicom_as_npy(image, path: str, patient: str) -> np.ndarray:
    array = sitk.GetArrayFromImage(image)
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, patient + ".npy"), array)
    return array


def dicom_apply_mask(mask, image, output_pixel_type: str = "Uint16"):
    """Grow the mask with a 6x6x6 cube, then punch the MRI (outside → 0)."""
    _ = output_pixel_type
    kernel = np.ones((6, 6, 6), np.uint16)
    img1 = sitk.GetArrayFromImage(mask)
    img2 = sitk.GetArrayFromImage(image)
    img1[img1 == 2] = 1
    try:
        img1 = dilation(img1, footprint=kernel)
    except TypeError:
        img1 = dilation(img1, selem=kernel)
    img3 = img2 * img1
    dcm3 = sitk.GetImageFromArray(img3)
    dcm3 = sitk.Cast(dcm3, sitk.sitkInt16)
    dcm3.SetSpacing(image.GetSpacing())
    dcm3.SetOrigin(image.GetOrigin())
    dcm3.SetDirection(image.GetDirection())
    return dcm3


def crop_2d_mask(image):
    """Tight crop around nonzero voxels. Exclusive max, same as the GitHub function."""
    array = sitk.GetArrayFromImage(image)
    x = np.where(array != 0)
    if x[0].size == 0:
        raise RuntimeError("mask punch-out is empty; cannot crop")
    x_min, x_max = int(min(x[0])), int(max(x[0]))
    y_min, y_max = int(min(x[1])), int(max(x[1]))
    z_min, z_max = int(min(x[2])), int(max(x[2]))
    output_arr = array[x_min:x_max, y_min:y_max, z_min:z_max]
    image_out = sitk.GetImageFromArray(output_arr)
    image_out.SetSpacing(image.GetSpacing())
    image_out.SetOrigin(image.GetOrigin())
    image_out.SetDirection(image.GetDirection())
    return sitk.Cast(image_out, sitk.sitkInt16), output_arr.shape


def zero_pad_3d(image, size, mask=None):
    """Center-pad a SimpleITK image to Size = [x, y, z] in the GitHub convention."""
    _ = mask
    array = sitk.GetArrayFromImage(image)
    x = array.shape[1]
    y = array.shape[2]
    z = array.shape[0]
    new_size = [size[1], size[0], size[2]]
    old_size = [y, x, z]
    filt = sitk.ConstantPadImageFilter()
    upper_bound = [
        (new_size[0] - old_size[0]) // 2,
        (new_size[1] - old_size[1]) // 2,
        (new_size[2] - old_size[2]) // 2,
    ]
    lower_bound = list(upper_bound)
    if (2 * upper_bound[0] + old_size[0]) == (new_size[1] - 1):
        upper_bound[0] += 1
    if (2 * upper_bound[1] + old_size[1]) == (new_size[0] - 1):
        upper_bound[1] += 1
    if (2 * upper_bound[2] + old_size[2]) == (new_size[2] - 1):
        upper_bound[2] += 1
    filt.SetPadUpperBound(upper_bound)
    filt.SetPadLowerBound(lower_bound)
    padded_img = filt.Execute(image)
    return sitk.Cast(padded_img, sitk.sitkInt16)


def crop_3d(image, size, center1):
    """Fallback when the dilated crop is larger than the target pad (GitHub 500-cube path)."""
    array = sitk.GetArrayFromImage(image)
    x = array.shape[1]
    y = array.shape[2]
    z = array.shape[0]
    output_arr = np.zeros((size[2], size[0], size[1]))
    center = (np.array([size[2], size[0], size[1]]) / 2).astype(int)
    output_arr[
        max(0, center[0] - center1[0]) : min(size[2], center[0] - center1[0] + z),
        max(center[1] - center1[1], 0) : min(center[1] - center1[1] + x, size[0]),
        max(0, center[2] - center1[2]) : min(size[1], center[2] - center1[2] + y),
    ] = array[
        max(0, -center[0] + center1[0]) : min(z, center[0] + center1[0]),
        max(center1[1] - center[1], 0) : min(center[1] + center1[1], x),
        max(0, -center[2] + center1[2]) : min(y, center[2] + center1[2]),
    ]
    image_out = sitk.GetImageFromArray(output_arr)
    image_out.SetSpacing(image.GetSpacing())
    image_out.SetOrigin(image.GetOrigin())
    image_out.SetDirection(image.GetDirection())
    return sitk.Cast(image_out, sitk.sitkInt16)


def resample_mask_to_ref(mask_path: str | Path, ref_path: str | Path, out_path: str | Path) -> Path:
    """Nearest-neighbor resample a binary mask onto another MRI grid."""
    mask = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
    ref = sitk.ReadImage(str(ref_path))
    same = (
        mask.GetSize() == ref.GetSize()
        and mask.GetSpacing() == ref.GetSpacing()
        and mask.GetOrigin() == ref.GetOrigin()
        and mask.GetDirection() == ref.GetDirection()
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if same:
        sitk.WriteImage(mask, str(out_path))
        return out_path
    filt = sitk.ResampleImageFilter()
    filt.SetReferenceImage(ref)
    filt.SetInterpolator(sitk.sitkNearestNeighbor)
    filt.SetDefaultPixelValue(0)
    sitk.WriteImage(filt.Execute(mask), str(out_path))
    return out_path


def apply_mask_crop_pad(image_path: str | Path, mask_path: str | Path, zero_pad: list[int]):
    """GitHub apply_mask_to_images for one patient: dilate, crop, pad (or 500-cube fallback)."""
    image = nifti_read(str(image_path))
    mask = nifti_read(str(mask_path))
    if image.GetSize() != mask.GetSize():
        raise RuntimeError(
            f"image/mask grid mismatch: image {image.GetSize()} vs mask {mask.GetSize()}"
        )
    punched = dicom_apply_mask(mask, image)
    cropped, crop_shape = crop_2d_mask(punched)
    used_fallback = False
    try:
        padded = zero_pad_3d(cropped, [zero_pad[0], zero_pad[1], zero_pad[2]], mask=None)
    except Exception:
        used_fallback = True
        padded_big = zero_pad_3d(cropped, [500, 500, 500], mask=None)
        padded = crop_3d(padded_big, [zero_pad[0], zero_pad[1], zero_pad[2]], [250, 250, 250])
    return {
        "padded": padded,
        "crop_shape": tuple(int(x) for x in crop_shape),
        "used_fallback": used_fallback,
    }
