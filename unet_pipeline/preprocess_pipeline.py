#!/usr/bin/env python3
"""Organ pipeline: optional Room 1 → 3D U-Net → cook → lab classifier.

--organ prostate  resampled 3D T2 _0004, pad 512, cube 60×200×200, EPE-Net
--organ sv        segment axial T2 _0000 (pad 400); cook 3D T2 _0004 +
                  SV mask on that grid to 50×200×200; SVI-Net

Lab seminal_vesicles_3d cubes were built from _0004 + segmentations_3d SV
masks, not from axial T2. --from-raw adds N4 + Nyul + resample first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

import lab_masking as lab

_HERE = Path(__file__).resolve().parent
ORGANS = {
    "prostate": {
        "zero_pad": [200, 200, 60],
        "infer_pad": (512, 512, 128),
        "suffix": "_0004",
        "unet": str(Path.home() / "toy_prostate/runs/unet3d_n820_512/best_model.pt"),
        "clf": str(Path.home() / "toy_prostate/models/EPE_NET_final.pt"),
        "clf_name": "EPE-Net",
        "label": "epe",
        "nyul": str(_HERE / "landmarks/trained_model_MRI_3d_t2.npz"),
    },
    "sv": {
        "zero_pad": [200, 200, 50],
        "infer_pad": (400, 400, 128),
        "suffix": "_0000",
        "unet": str(Path.home() / "toy_prostate/runs/unet3d_sv_n820_400/best_model.pt"),
        "clf": str(Path.home() / "toy_prostate/models/SVI_NET_final.pt"),
        "clf_name": "SVI-Net",
        "label": "svi",
        "nyul": str(_HERE / "landmarks/trained_model_MRI_t2.npz"),
    },
}
PROSTATE_ZERO_PAD = ORGANS["prostate"]["zero_pad"]
PROSTATE_INFER_PAD = ORGANS["prostate"]["infer_pad"]
DEFAULT_UNET = ORGANS["prostate"]["unet"]
DEFAULT_EPE = ORGANS["prostate"]["clf"]


def center_pad(vol: np.ndarray, target: tuple[int, int, int], fill: float) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Pad (or center-crop) to target. Returns padded volume and origin offsets to unpad."""
    out = np.full(target, fill, dtype=vol.dtype)
    src = []
    dst = []
    origins = []
    for s, d in zip(vol.shape, target):
        if s == d:
            src.append((0, s))
            dst.append((0, d))
            origins.append(0)
        elif s > d:
            extra = s - d
            s0 = extra // 2
            src.append((s0, s0 + d))
            dst.append((0, d))
            origins.append(s0)
        else:
            extra = d - s
            d0 = extra // 2
            src.append((0, s))
            dst.append((d0, d0 + s))
            origins.append(-d0)
    sl_s = tuple(slice(a, b) for a, b in src)
    sl_d = tuple(slice(a, b) for a, b in dst)
    out[sl_d] = vol[sl_s]
    return out, tuple(origins)


def unpad(vol: np.ndarray, src_shape: tuple[int, int, int], origins: tuple[int, int, int]) -> np.ndarray:
    """Undo center_pad. origins[i] < 0 means we padded that axis."""
    slices = []
    for size, origin, padded in zip(src_shape, origins, vol.shape):
        if origin < 0:
            start = -origin
            slices.append(slice(start, start + size))
        elif origin > 0:
            slices.append(slice(0, size))
        else:
            slices.append(slice(0, min(size, padded)))
    cropped = vol[tuple(slices)]
    if cropped.shape != src_shape:
        raise RuntimeError(f"unpad got {cropped.shape}, expected {src_shape}")
    return cropped


def find_cook_image(pid: str, image_path: Path | None, cook_image: str, organ: str) -> Path | None:
    """SVI-Net cubes were cooked from 3D T2 (_0004). Prefer that when present."""
    if cook_image:
        return Path(cook_image)
    if image_path is None:
        return None
    if organ != "sv":
        return image_path
    candidates = [
        image_path.parent / f"{pid}_0004.nii.gz",
        Path.home() / "toy_prostate" / "images" / f"{pid}_0004.nii.gz",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return image_path


def patient_id_from_path(path: Path) -> str:
    name = path.name
    for suffix in ("_0004.nii.gz", "_0000.nii.gz", ".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def infer_pad_for(image_path: Path, base: tuple[int, int, int] = PROSTATE_INFER_PAD) -> tuple[int, int, int]:
    src = nib.load(str(image_path))
    src_shape = tuple(int(x) for x in src.shape[:3])
    return tuple(max(p, s) for p, s in zip(base, src_shape))


def write_mask_nifti(mask: np.ndarray, ref_path: Path, out_path: Path) -> None:
    ref = nib.load(str(ref_path))
    out = nib.Nifti1Image(mask.astype(np.uint8), ref.affine, ref.header)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(out_path))


def _device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    print("WARNING: no GPU; inference will be slow", flush=True)
    return torch.device("cpu")


def load_unet(ckpt_path: Path, device=None) -> dict:
    """Load the 3D U-Net once. Reuse the returned bundle for every patient."""
    import torch
    from monai.transforms import AsDiscrete, Compose, EnsureTyped, NormalizeIntensityd

    import train_3d_unet as t

    device = device or _device()
    ckpt = torch.load(str(ckpt_path), map_location=device)
    t.PATCH = tuple(ckpt["patch"])
    model = t.build_model(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tf = Compose(
        [
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys="image"),
        ]
    )
    return {
        "model": model,
        "device": device,
        "ckpt": ckpt,
        "patch": list(t.PATCH),
        "tf": tf,
        "to_pred": AsDiscrete(argmax=True, to_onehot=2),
        "ckpt_path": str(ckpt_path),
    }


def load_classifier(ckpt_path: Path, device=None):
    """Load lab EPE-Net or SVI-Net (same BoxNet pickle)."""
    import torch

    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from models.EPE_Net_final import BoxNet
    import models.SVI_Net_final  # noqa: F401  pickle may look for this module

    # Lab torch.save(model) pickled BoxNet as __main__.BoxNet
    import __main__ as _main

    _main.BoxNet = BoxNet

    device = device or _device()
    try:
        obj = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(str(ckpt_path), map_location=device)
    if isinstance(obj, torch.nn.Module):
        model = obj
    elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        model = BoxNet()
        model.load_state_dict(obj["model"])
    elif isinstance(obj, dict) and "state_dict" in obj:
        model = BoxNet()
        model.load_state_dict(obj["state_dict"])
    elif isinstance(obj, dict):
        model = BoxNet()
        model.load_state_dict(obj)
    else:
        raise TypeError(f"unrecognized classifier checkpoint type: {type(obj)}")
    model.to(device)
    model.eval()
    return model, device


load_epe = load_classifier


def apply_mask(image_path: Path, mask_path: Path, out_dir: Path, patient_id: str, zero_pad: list[int]) -> dict:
    """GitHub cook: dilate 6x6x6, crop, pad to EPE-Net size."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result = lab.apply_mask_crop_pad(image_path, mask_path, zero_pad)
    array = lab.save_dicom_as_npy(result["padded"], str(out_dir), "array")
    array_1ch = np.expand_dims(array, axis=0)
    np.save(out_dir / "array_1ch.npy", array_1ch)
    lab.nifti_write(result["padded"], str(out_dir), "array.nii.gz")
    expected_zyx = (zero_pad[2], zero_pad[0], zero_pad[1])
    if tuple(array.shape) != expected_zyx:
        raise RuntimeError(f"array shape {array.shape} != {expected_zyx}")
    return {
        "patient_id": patient_id,
        "image": str(image_path),
        "mask": str(mask_path),
        "array_shape": list(array.shape),
        "array_1ch_shape": list(array_1ch.shape),
        "crop_shape": list(result["crop_shape"]),
        "used_fallback": bool(result["used_fallback"]),
        "zero_pad": list(zero_pad),
        "nonzero_voxels": int((array != 0).sum()),
    }


def segment(
    image_path: Path,
    ckpt_path: Path,
    out_mask: Path,
    infer_pad: tuple[int, int, int] | None = None,
    unet: dict | None = None,
) -> dict:
    """Pad to the training canvas, run the 3D U-Net, unpad the mask to the MRI grid."""
    import torch

    import train_3d_unet as t

    if unet is None:
        unet = load_unet(Path(ckpt_path))
    image_path = Path(image_path)
    src = nib.load(str(image_path))
    vol = np.asanyarray(src.dataobj).astype(np.float32)
    if vol.ndim != 3:
        raise RuntimeError(f"expected 3D MRI, got {vol.shape}")
    src_shape = tuple(int(x) for x in vol.shape)
    if infer_pad is None:
        infer_pad = tuple(max(p, s) for p, s in zip(PROSTATE_INFER_PAD, src_shape))  # caller should pass organ pad
    padded, origins = center_pad(vol, infer_pad, 0.0)
    t.PATCH = tuple(unet["patch"])
    device = unet["device"]
    batch = unet["tf"]({"image": padded[None].astype(np.float32)})
    images = batch["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = t.sliding_window(unet["model"], images)
    pred = unet["to_pred"](logits[0])[1].detach().cpu().numpy()
    mask = (unpad(pred, src_shape, origins) > 0.5).astype(np.uint8)
    if int(mask.sum()) == 0:
        raise RuntimeError(f"U-Net mask is empty for {image_path.name}")
    write_mask_nifti(mask, image_path, out_mask)
    ckpt = unet["ckpt"]
    return {
        "ckpt": unet["ckpt_path"],
        "ckpt_epoch": int(ckpt.get("epoch", -1)),
        "patch": list(unet["patch"]),
        "src_shape": list(src_shape),
        "infer_pad": list(infer_pad),
        "mask_voxels": int(mask.sum()),
        "device": str(device),
    }


def cube_to_batch(arr: np.ndarray) -> np.ndarray:
    """Make (1, 1, 60, 200, 200) for Conv3d."""
    x = np.asarray(arr)
    if x.ndim == 3:
        x = x[None, None]
    elif x.ndim == 4:
        x = x[None]
    elif x.ndim == 5:
        pass
    else:
        raise RuntimeError(f"EPE cube ndim {x.ndim}, shape {x.shape}")
    if tuple(x.shape[1:]) not in {(1, 60, 200, 200), (1, 50, 200, 200)}:
        if x.shape[-3:] == (60, 200, 200) or x.shape[-3:] == (50, 200, 200):
            pass
        else:
            raise RuntimeError(f"EPE cube shape {x.shape}; expected (1,1,60,200,200)")
    return x.astype(np.float32)


def classify(
    array_path: Path,
    clf_model,
    device,
    out_dir: Path | None = None,
    label: str = "epe",
    clf_name: str = "EPE-Net",
) -> dict:
    """Run EPE-Net or SVI-Net. Same sigmoid + 0.5 rule as lab predict.py."""
    import torch
    from torch.nn import Sigmoid

    arr = np.load(str(array_path))
    batch = cube_to_batch(arr)
    x = torch.from_numpy(batch).to(device)
    act = Sigmoid()
    with torch.no_grad():
        logit = clf_model(x)
        prob = act(logit)
    logit_v = float(logit.detach().cpu().reshape(-1)[0])
    prob_v = float(prob.detach().cpu().reshape(-1)[0])
    pred = int(prob_v >= 0.5)
    rec = {
        "array": str(array_path),
        "array_shape": list(arr.shape),
        "batch_shape": list(batch.shape),
        "classifier": clf_name,
        f"{label}_logit": logit_v,
        f"{label}_prob": prob_v,
        f"{label}_pred": pred,
        "threshold": 0.5,
        "meaning": f"1 = {label.upper()} predicted present, 0 = absent",
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{label}.json").write_text(json.dumps(rec, indent=2))
    return rec


def compare_arrays(ours: np.ndarray, lab_array: np.ndarray) -> dict:
    a = np.squeeze(ours)
    b = np.squeeze(lab_array)
    if a.shape != b.shape:
        return {"match": False, "ours_shape": list(a.shape), "lab_shape": list(b.shape)}
    same = bool(np.array_equal(a, b))
    max_abs = float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))
    return {
        "match": same,
        "shape": list(a.shape),
        "max_abs_diff": max_abs,
        "ours_nonzero": int((a != 0).sum()),
        "lab_nonzero": int((b != 0).sum()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Organ pipeline: 3D U-Net, cook, EPE-Net or SVI-Net")
    p.add_argument("--mode", choices=("apply-mask", "segment", "classify", "full"), default="full")
    p.add_argument("--image", help="Resampled NIfTI (_0004 prostate, _0000 SV segment)")
    p.add_argument(
        "--cook-image",
        default="",
        help="MRI to punch/crop for the classifier cube. SV defaults to sibling _0004 if present.",
    )
    p.add_argument("--mask", help="Existing mask. Required for apply-mask if not already in out-dir.")
    p.add_argument("--array", help="Existing array_1ch.npy for --mode classify")
    p.add_argument("--ckpt", default="", help="3D U-Net checkpoint (default depends on --organ)")
    p.add_argument("--epe-ckpt", default="", help="EPE-Net or SVI-Net .pt (default depends on --organ)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--patient-id", default="")
    p.add_argument("--organ", default="prostate", choices=("prostate", "sv"))
    p.add_argument("--compare-npy", help="Optional lab prostates_3d or seminal_vesicles_3d npy")
    p.add_argument("--no-classify", action="store_true", help="Stop after the cube")
    p.add_argument(
        "--from-raw",
        action="store_true",
        help="Run N4 + Nyul + resample before the U-Net",
    )
    p.add_argument(
        "--nyul-npz",
        default="",
        help="Lab Nyul landmark file (default: 3d_t2 for prostate, t2 for SV)",
    )
    return p.parse_args()


def run_one(args: argparse.Namespace, unet=None, epe=None, epe_device=None) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = Path(args.image) if args.image else None
    pid = args.patient_id
    if not pid and image_path is not None:
        pid = patient_id_from_path(image_path)
    if not pid:
        pid = out_dir.name
    cfg = ORGANS[args.organ]
    zero_pad = list(cfg["zero_pad"])
    unet_ckpt = Path(args.ckpt or cfg["unet"])
    clf_ckpt = Path(args.epe_ckpt or cfg["clf"])
    nyul_npz = args.nyul_npz or cfg["nyul"]

    meta: dict = {
        "patient_id": pid,
        "organ": args.organ,
        "mode": args.mode,
        "from_raw": bool(getattr(args, "from_raw", False)),
        "note": "Room 2 = U-Net + cook + classifier. Room 1 runs only with --from-raw.",
    }
    if image_path is not None:
        if not image_path.exists():
            raise SystemExit(f"missing image: {image_path}")
        if getattr(args, "from_raw", False):
            import lab_intensity as intensity

            print(f"=== Room 1 cook raw {pid} ===", flush=True)
            meta["room1"] = intensity.prepare_resampled(image_path, out_dir, nyul_npz)
            image_path = Path(meta["room1"]["resampled"])
            print(f"  resampled {meta['room1']['size_xyz']} spacing={meta['room1']['spacing']}", flush=True)
        meta["image"] = str(image_path)
        resampled_out = out_dir / "resampled.nii.gz"
        if not resampled_out.exists():
            nib.save(nib.load(str(image_path)), str(resampled_out))

    mask_path = Path(args.mask) if args.mask else out_dir / "mask.nii.gz"

    if args.mode in ("segment", "full"):
        if image_path is None:
            raise SystemExit("--image is required for segment/full")
        print(f"=== segment {pid} with 3D U-Net ({args.organ}) ===", flush=True)
        infer_pad = infer_pad_for(image_path, cfg["infer_pad"])
        if unet is None:
            unet = load_unet(unet_ckpt)
        meta["segment"] = segment(image_path, unet_ckpt, mask_path, infer_pad, unet=unet)
        print(f"  wrote {mask_path}  voxels={meta['segment']['mask_voxels']}", flush=True)

    if args.mode in ("apply-mask", "full"):
        if image_path is None:
            raise SystemExit("--image is required for apply-mask/full")
        if not mask_path.exists():
            raise SystemExit(f"missing mask: {mask_path}")
        cook_arg = getattr(args, "cook_image", "")
        if args.mode == "full" or cook_arg:
            cook_image = find_cook_image(pid, image_path, cook_arg, args.organ)
        else:
            cook_image = image_path
        cook_mask = mask_path
        if cook_image is None:
            raise SystemExit("--image is required for apply-mask/full")
        if cook_image.resolve() != image_path.resolve():
            cook_mask = out_dir / "mask_on_cook_grid.nii.gz"
            print(f"=== resample SV mask onto {cook_image.name} ===", flush=True)
            lab.resample_mask_to_ref(mask_path, cook_image, cook_mask)
            meta["cook_resample"] = {
                "segment_image": str(image_path),
                "cook_image": str(cook_image),
                "mask_on_cook_grid": str(cook_mask),
            }
        print(f"=== apply-mask {pid} (6x6x6 dilate, pad {zero_pad}) ===", flush=True)
        meta["apply_mask"] = apply_mask(cook_image, cook_mask, out_dir, pid, zero_pad)
        print(
            f"  array {meta['apply_mask']['array_shape']}  "
            f"crop={meta['apply_mask']['crop_shape']}  "
            f"fallback={meta['apply_mask']['used_fallback']}",
            flush=True,
        )

    if args.compare_npy and (out_dir / "array.npy").exists():
        meta["compare"] = compare_arrays(np.load(out_dir / "array.npy"), np.load(args.compare_npy))
        print(f"  vs lab npy: {meta['compare']}", flush=True)

    do_cls = args.mode in ("classify", "full") and not args.no_classify
    if do_cls:
        array_path = Path(args.array) if args.array else out_dir / "array_1ch.npy"
        if not array_path.exists() and (out_dir / "array.npy").exists():
            array_path = out_dir / "array.npy"
        if not array_path.exists():
            raise SystemExit(f"missing cooked cube: {array_path}")
        print(f"=== classify {pid} with {cfg['clf_name']} ===", flush=True)
        if epe is None:
            epe, epe_device = load_classifier(clf_ckpt)
        elif epe_device is None:
            epe_device = next(epe.parameters()).device
        meta["classify"] = classify(
            array_path, epe, epe_device, out_dir, label=cfg["label"], clf_name=cfg["clf_name"]
        )
        tag = cfg["label"]
        print(
            f"  {tag.upper()} prob={meta['classify'][f'{tag}_prob']:.4f}  "
            f"pred={meta['classify'][f'{tag}_pred']}",
            flush=True,
        )

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"WROTE {out_dir}", flush=True)
    return meta


def main() -> None:
    args = parse_args()
    run_one(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
