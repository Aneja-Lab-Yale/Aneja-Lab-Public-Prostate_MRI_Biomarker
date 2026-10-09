#!/usr/bin/env python3
"""Train a 3D U-Net on the toy prostate set.

Input:  resampled 3D T2 MRI  (~/toy_prostate/images/<ID>_0004.nii.gz)
Target: binary prostate mask (~/toy_prostate/labels/<ID>.nii.gz)
Split:  from a CSV under --data-root (or an absolute --split-file). Test is scored once, at the end.

A Tesla T4 cannot hold a full 512 x 512 x 128 volume through this U-Net, so training uses
random 3D crops (--patch). Validation and test use sliding-window inference over the whole scan.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import DataLoader, Dataset, PersistentDataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
)
from torch.cuda.amp import GradScaler, autocast


PATCH = (96, 96, 32)
NUM_SAMPLES = 4
CHANNELS = (16, 32, 64, 128, 256)
STRIDES = (2, 2, 2, 2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(
    data_root: Path, split_file: str = "split.csv", image_suffix: str = "_0004"
) -> dict[str, list[dict]]:
    split_path = Path(split_file)
    if not split_path.is_absolute():
        split_path = data_root / split_file
    buckets = {"train": [], "val": [], "test": []}
    with split_path.open() as f:
        for row in csv.DictReader(f):
            pid = row["patient_id"]
            image = data_root / "images" / f"{pid}{image_suffix}.nii.gz"
            label = data_root / "labels" / f"{pid}.nii.gz"
            if not image.exists() or not label.exists():
                raise FileNotFoundError(f"Missing files for {pid}: {image} / {label}")
            buckets[row["split"]].append(
                {
                    "image": str(image),
                    "label": str(label),
                    "patient_id": pid,
                    "image_path": str(image),
                }
            )
    for name, items in buckets.items():
        if not items:
            raise RuntimeError(f"Split '{name}' is empty")
    return buckets


def train_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=PATCH,
                pos=2,
                neg=1,
                num_samples=NUM_SAMPLES,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(0, 1)),
            RandGaussianNoised(keys="image", prob=0.15, std=0.05),
        ]
    )


def infer_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def build_model(device: torch.device) -> UNet:
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=CHANNELS,
        strides=STRIDES,
        num_res_units=2,
    ).to(device)


def run_epoch(
    model, loader, loss_fn, optimizer, scaler, device, train: bool, accum_steps: int = 1
) -> float:
    model.train(train)
    running = 0.0
    n_steps = 0
    if train:
        optimizer.zero_grad(set_to_none=True)
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for i, batch in enumerate(loader):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with autocast(enabled=True):
                logits = model(images)
                loss = loss_fn(logits, labels)
                loss_to_scale = loss / accum_steps if train else loss
            if train:
                scaler.scale(loss_to_scale).backward()
                if (i + 1) % accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            running += float(loss.item())
            n_steps += 1
        if train and n_steps % accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
    return running / max(n_steps, 1)


def _dice_from_logits(logits, labels, dice_metric) -> float:
    to_pred = AsDiscrete(argmax=True, to_onehot=2)
    to_lab = AsDiscrete(to_onehot=2)
    preds = [to_pred(x) for x in decollate_batch(logits)]
    labs = [to_lab(x) for x in decollate_batch(labels)]
    dice_metric(y_pred=preds, y=labs)
    agg = dice_metric.aggregate()
    dice_metric.reset()
    return float(agg.item())


def sliding_window(model, images):
    return sliding_window_inference(
        inputs=images,
        roi_size=PATCH,
        sw_batch_size=2,
        predictor=model,
        overlap=0.25,
        mode="gaussian",
    )


def score_loader(model, loader, device, dice_metric) -> float:
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with autocast(enabled=True):
                logits = sliding_window(model, images)
            scores.append(_dice_from_logits(logits, labels, dice_metric))
    return float(np.mean(scores)) if scores else 0.0


def save_pred_nifti(pred_mask: np.ndarray, ref_path: str, out_path: Path) -> None:
    ref = nib.load(ref_path)
    out = nib.Nifti1Image(pred_mask.astype(np.uint8), ref.affine, ref.header)
    nib.save(out, str(out_path))


def score_cases(model, loader, device, dice_metric, pred_dir: Path | None) -> list[dict]:
    """Per-patient Dice on a full-volume loader (batch size 1)."""
    model.eval()
    rows = []
    to_pred = AsDiscrete(argmax=True, to_onehot=2)
    to_lab = AsDiscrete(to_onehot=2)
    with torch.no_grad():
        for batch in loader:
            pid = batch["patient_id"][0]
            image_path = batch["image_path"][0]
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            t0 = time.time()
            with autocast(enabled=True):
                logits = sliding_window(model, images)
            infer_s = time.time() - t0
            pred_oh = to_pred(logits[0])
            lab_oh = to_lab(labels[0])
            dice_metric(y_pred=[pred_oh], y=[lab_oh])
            dice = float(dice_metric.aggregate().item())
            dice_metric.reset()
            if pred_dir is not None:
                mask = pred_oh[1].detach().cpu().numpy()
                save_pred_nifti(mask, str(image_path), pred_dir / f"{pid}_pred.nii.gz")
            rows.append(
                {
                    "patient_id": pid,
                    "dice": round(dice, 4),
                    "infer_seconds": round(infer_s, 2),
                }
            )
            print(f"  {pid}  Dice={dice:.4f}  {infer_s:.1f}s", flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Toy 3D U-Net prostate segmentation")
    p.add_argument("--data-root", default=os.path.expanduser("~/toy_prostate"))
    p.add_argument("--split-file", default="split.csv", help="CSV path or name under data-root")
    p.add_argument(
        "--image-suffix",
        default="_0004",
        help="Image filename suffix before .nii.gz (_0004 prostate, _0000 SV)",
    )
    p.add_argument("--out-dir", default=os.path.expanduser("~/toy_prostate/runs/unet3d"))
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument(
        "--accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch = batch-size * accum-steps)",
    )
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument(
        "--cache-dir",
        default=os.path.expanduser("~/toy_prostate/cache_unet3d"),
        help="PersistentDataset cache so 100 volumes do not fill RAM",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Load NIfTI each epoch (needed for n=820; full-volume cache will not fit on scratch)",
    )
    p.add_argument(
        "--patch",
        nargs=3,
        type=int,
        default=[96, 96, 32],
        metavar=("H", "W", "D"),
        help="3D crop size (must be divisible by 16)",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Crops per volume from RandCropByPosNegLabeld",
    )
    p.add_argument("--smoke", action="store_true", help="2-step sanity check, then exit")
    return p.parse_args()


def main() -> None:
    global PATCH, NUM_SAMPLES
    args = parse_args()
    PATCH = tuple(args.patch)
    NUM_SAMPLES = args.num_samples
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "test_preds"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    splits = load_split(data_root, args.split_file, image_suffix=args.image_suffix)
    print("=== 3D U-Net ===", flush=True)
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"split: train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}",
        flush=True,
    )
    print(
        f"patch={PATCH}  num_samples={NUM_SAMPLES}  channels={CHANNELS}  "
        f"batch={args.batch_size}  accum={args.accum_steps}  "
        f"effective_batch={args.batch_size * args.accum_steps}  "
        f"image_suffix={args.image_suffix}",
        flush=True,
    )
    print("dice: foreground only (include_background=False)", flush=True)

    if args.no_cache:
        print("cache: off (Dataset, load from disk each epoch)", flush=True)
        train_ds = Dataset(data=splits["train"], transform=train_transforms())
        val_ds = Dataset(data=splits["val"], transform=infer_transforms())
    else:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"cache: {cache_dir}", flush=True)
        train_ds = PersistentDataset(
            data=splits["train"], transform=train_transforms(), cache_dir=str(cache_dir / "train")
        )
        val_ds = PersistentDataset(
            data=splits["val"], transform=infer_transforms(), cache_dir=str(cache_dir / "val")
        )
    test_ds = Dataset(data=splits["test"], transform=infer_transforms())

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    model = build_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params/1e6:.2f}M", flush=True)

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    if args.smoke:
        print("=== smoke test: one train batch + one val volume ===", flush=True)
        batch = next(iter(train_loader))
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        print(f"train batch image {tuple(images.shape)}  label {tuple(labels.shape)}", flush=True)
        with autocast(enabled=True):
            logits = model(images)
            loss = loss_fn(logits, labels)
        print(f"forward ok  logits {tuple(logits.shape)}  loss={float(loss):.4f}", flush=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print("backward ok", flush=True)
        val_dice = score_loader(model, val_loader, device, dice_metric)
        print(f"untrained val Dice={val_dice:.4f} (should be near 0)", flush=True)
        print("SMOKE TEST PASSED", flush=True)
        return

    history = []
    best_dice = -1.0
    best_epoch = 0
    stall = 0
    ckpt_path = out_dir / "best_model.pt"
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, True, args.accum_steps
        )
        val_dice = score_loader(model, val_loader, device, dice_metric)
        dt = time.time() - t0
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_dice": round(val_dice, 4),
                "seconds": round(dt, 1),
            }
        )
        star = ""
        if val_dice > best_dice + 1e-4:
            best_dice = val_dice
            best_epoch = epoch
            stall = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "val_dice": val_dice,
                    "args": vars(args),
                    "patch": PATCH,
                    "channels": CHANNELS,
                    "strides": STRIDES,
                },
                ckpt_path,
            )
            star = "  *best*"
        else:
            stall += 1
        print(
            f"epoch {epoch:03d}/{args.epochs}  loss={train_loss:.4f}  "
            f"val Dice={val_dice:.4f}  {dt:.0f}s{star}",
            flush=True,
        )
        if stall >= args.patience:
            print(
                f"early stop at epoch {epoch} (best {best_epoch}, Dice {best_dice:.4f})",
                flush=True,
            )
            break

    with (out_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)

    print(f"\n=== test set (best epoch {best_epoch}, val Dice {best_dice:.4f}) ===", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_rows = score_cases(model, test_loader, device, dice_metric, pred_dir)
    mean_dice = float(np.mean([r["dice"] for r in test_rows]))
    summary = {
        "best_epoch": best_epoch,
        "best_val_dice": round(best_dice, 4),
        "test_mean_dice": round(mean_dice, 4),
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "params_million": round(n_params / 1e6, 2),
        "patch": list(PATCH),
        "train_minutes": round((time.time() - t_start) / 60, 1),
        "test_cases": test_rows,
    }
    with (out_dir / "test_metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (out_dir / "test_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "dice", "infer_seconds"])
        w.writeheader()
        w.writerows(test_rows)
    print(f"\nmean test Dice={mean_dice:.4f}", flush=True)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
