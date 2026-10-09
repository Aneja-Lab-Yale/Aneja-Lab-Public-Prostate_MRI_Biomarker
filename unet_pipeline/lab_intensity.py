"""Room 1 cooking: reorient → N4 → Nyul → resample 0.5×0.5×1.0.

Nyul uses the lab landmark file:
  data/surgery/split_train_test/trained_model_MRI_3d_t2.npz
N4 and Nyul transform match Aneja-Lab-Public-Prostate_MRI_Biomarker.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.interpolate import interp1d

TARGET_SPACING = (0.5, 0.5, 1.0)
DEFAULT_NYUL = Path(__file__).resolve().parent / "landmarks" / "trained_model_MRI_3d_t2.npz"


def load_mean_landmarks(path: str | Path | None = None) -> np.ndarray:
    path = Path(path or DEFAULT_NYUL)
    raw = np.load(path, allow_pickle=True)["trainedModel"]
    model = raw[0]
    return np.asarray(model["meanLandmarks"], dtype=np.float64)


def reorient_las(image):
    try:
        return sitk.DICOMOrient(image, "LAS")
    except Exception:
        filt = sitk.DICOMOrientImageFilter()
        filt.SetDesiredCoordinateOrientation("LAS")
        return filt.Execute(image)


def _histo_normalize_against_self(image):
    data = sitk.GetArrayFromImage(image).astype(np.float32)
    data -= np.min(data)
    histo, bins = np.histogram(data.flatten(), 10)
    hist = histo.astype(np.float64)
    s = hist.sum()
    if s > 0:
        hist = hist / s
    below = np.where(hist < 5e-3)[0]
    if below.size:
        cap = float(bins[int(below.min())])
        data = np.clip(data, 0, cap)
    out = sitk.GetImageFromArray(data)
    out.SetSpacing(image.GetSpacing())
    out.SetOrigin(image.GetOrigin())
    out.SetDirection(image.GetDirection())
    return out


def n4_bias_correct(image):
    """Lab dicom_bias_correct (SimpleITK N4 + Otsu mask)."""
    image = _histo_normalize_against_self(image)
    input_image = sitk.Cast(image, sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    out = corrector.Execute(input_image, mask)
    out.SetSpacing(image.GetSpacing())
    out.SetOrigin(image.GetOrigin())
    out.SetDirection(image.GetDirection())
    return out


def _get_cdf(hist):
    aux = np.cumsum(hist)
    aux = aux / aux[-1] * 100
    return aux


def _get_percentile(cdf, bins, perc):
    b = len(bins[cdf <= perc])
    if b >= len(bins):
        b = len(bins) - 1
    return bins[b] + ((bins[1] - bins[0]) / 2)


def _landmarks_for_image(image, nbins=1024, p_low=1, p_high=99, num_points=10):
    data = sitk.GetArrayFromImage(image)
    stats = sitk.StatisticsImageFilter()
    stats.Execute(image)
    mean = stats.GetMean()
    histo, bins = np.histogram(data.flatten(), nbins)
    cdf_original = _get_cdf(histo)
    histo = histo.astype(np.float64)
    histo[bins[:-1] < mean] = 0.0
    cdf_trunc = _get_cdf(histo)
    perc = [x for x in range(0, 100, 100 // num_points)][1:]
    return (
        [_get_percentile(cdf_original, bins[:-1], p_low)]
        + [_get_percentile(cdf_trunc, bins[:-1], x) for x in perc]
        + [_get_percentile(cdf_original, bins[:-1], p_high)]
    )


def nyul_transform(image, mean_landmarks: np.ndarray):
    """Lab Nyul transform() using saved meanLandmarks."""
    data = sitk.GetArrayFromImage(image)
    landmarks = np.array(_landmarks_for_image(image), dtype=np.float64)
    mapping = interp1d(landmarks, mean_landmarks, fill_value="extrapolate")
    mapped = mapping(data.ravel()).reshape(data.shape)
    mapped = mapped.astype(np.float32)
    mapped -= np.min(mapped)
    out = sitk.GetImageFromArray(mapped)
    out.SetSpacing(image.GetSpacing())
    out.SetOrigin(image.GetOrigin())
    out.SetDirection(image.GetDirection())
    return out


def resample_spacing(image, spacing=TARGET_SPACING):
    old = np.array(image.GetSpacing(), dtype=np.float64)
    size = np.array(image.GetSize(), dtype=np.float64)
    new_spacing = np.array(spacing, dtype=np.float64)
    new_size = [int(round(size[i] * old[i] / new_spacing[i])) for i in range(3)]
    return sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        image.GetOrigin(),
        tuple(float(x) for x in new_spacing),
        image.GetDirection(),
        0.0,
        sitk.sitkFloat32,
    )


def prepare_resampled(
    raw_path: str | Path,
    out_dir: str | Path,
    nyul_path: str | Path | None = None,
) -> dict:
    """Write Room 1 intermediates and return the resampled NIfTI path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = Path(raw_path)
    image = sitk.ReadImage(str(raw_path), sitk.sitkFloat32)
    image = reorient_las(image)
    sitk.WriteImage(image, str(out_dir / "reoriented.nii.gz"))
    print("=== Room 1: N4 bias correct (slow) ===", flush=True)
    image = n4_bias_correct(image)
    sitk.WriteImage(image, str(out_dir / "n4.nii.gz"))
    print("=== Room 1: Nyul (lab 3D T2 landmarks) ===", flush=True)
    landmarks = load_mean_landmarks(nyul_path)
    image = nyul_transform(image, landmarks)
    sitk.WriteImage(image, str(out_dir / "nyul.nii.gz"))
    print("=== Room 1: resample 0.5 x 0.5 x 1.0 mm ===", flush=True)
    image = resample_spacing(image)
    resampled_path = out_dir / "resampled.nii.gz"
    sitk.WriteImage(image, str(resampled_path))
    spacing = [round(float(x), 4) for x in image.GetSpacing()]
    size = [int(x) for x in image.GetSize()]
    return {
        "resampled": str(resampled_path),
        "spacing": spacing,
        "size_xyz": size,
        "nyul": str(Path(nyul_path or DEFAULT_NYUL)),
    }
