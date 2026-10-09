# 3D U-Net pipeline (prostate EPE and SV SVI)

This folder is the **U-Net path**: your 3D U-Nets draw the organ, then the lab cook and EPE-Net / SVI-Net classify. It does **not** replace the public biomarker repo’s nnU-Net path.

This folder lives on the **`3D-UNet-pipeline` branch** of the public biomarker repo. It does not change `main` (nnU-Net path). Clone, then:

```bash
git clone https://github.com/Aneja-Lab-Yale/Aneja-Lab-Public-Prostate_MRI_Biomarker.git
cd Aneja-Lab-Public-Prostate_MRI_Biomarker
git checkout 3D-UNet-pipeline
cd unet_pipeline
```

GPU recommended (a T4 is enough).

## What it does

| Organ | Draw | Cook cube | Classifier |
|---|---|---|---|
| Prostate | 3D T2 `_0004`, pad 512 | 6×6×6 dilate → **60×200×200** | EPE-Net |
| SV | Axial T2 `_0000`, pad 400 | Mask onto `_0004`, then 6×6×6 → **50×200×200** | SVI-Net |

Dilation is GitHub **(6, 6, 6)** (~2–3 voxels), not the paper’s 10 pixels + KDE. Optional `--from-raw` runs N4 + Nyul + resample 0.5×0.5×1.0 mm first.

## Files in this folder

```text
preprocess_pipeline.py    # CLI: segment → cook → classify
lab_masking.py            # 6×6×6 dilate / crop / pad
lab_intensity.py          # Room 1 (only for --from-raw)
train_3d_unet.py          # U-Net architecture (loaded at inference)
models/                   # EPE-Net / SVI-Net BoxNet class (not the .pt weights)
```

Weights are **not** in git. Get them from S3 or a copy on disk.

## Python environment

```bash
conda create -n prostate python=3.9 -y
conda activate prostate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install monai nibabel numpy SimpleITK scikit-image scipy
```

Match CUDA to the machine. `nvidia-smi` should work.

## Weights

**3D U-Nets** (already trained):

| Organ | File |
|---|---|
| Prostate | `runs/unet3d_n820_512/best_model.pt` |
| SV | `runs/unet3d_sv_n820_400/best_model.pt` |

**Lab classifiers** (do not retrain):

```text
s3://aneja-lab-prostate-mri/Final_model_weights/Prostate_Model_Weights/models/EPE_NET_final.pt
s3://aneja-lab-prostate-mri/Final_model_weights/Prostate_Model_Weights/models/SVI_NET_final.pt
```

**Nyul** (only if `--from-raw`):

```text
s3://aneja-lab-prostate-mri/data/surgery/split_train_test/trained_model_MRI_3d_t2.npz
s3://aneja-lab-prostate-mri/data/surgery/split_train_test/trained_model_MRI_t2.npz
```

Need AWS access (`AWS_PROFILE=yale-transfer`, region `us-east-2`), or someone copies the files onto the box.

Defaults look under `~/toy_prostate/`. Either put weights there or pass `--ckpt` and `--epe-ckpt`.

Example layout:

```text
~/toy_prostate/
  runs/
    unet3d_n820_512/best_model.pt
    unet3d_sv_n820_400/best_model.pt
  models/
    EPE_NET_final.pt
    SVI_NET_final.pt
```

## Run it

From `unet_pipeline/`, with `conda activate prostate`:

**Prostate** (already-resampled 3D T2 `_0004`):

```bash
python preprocess_pipeline.py --mode full --organ prostate \
  --image /path/to/ID_0004.nii.gz \
  --ckpt ~/toy_prostate/runs/unet3d_n820_512/best_model.pt \
  --epe-ckpt ~/toy_prostate/models/EPE_NET_final.pt \
  --out-dir ./output/ID
```

**SV** (axial T2 `_0000` to draw; 3D T2 `_0004` for the SVI cube — put `_0004` next to `_0000` or in `~/toy_prostate/images/`, or pass `--cook-image`):

```bash
python preprocess_pipeline.py --mode full --organ sv \
  --image /path/to/ID_0000.nii.gz \
  --ckpt ~/toy_prostate/runs/unet3d_sv_n820_400/best_model.pt \
  --epe-ckpt ~/toy_prostate/models/SVI_NET_final.pt \
  --out-dir ./output/ID
```

**Raw scan** (N4 + Nyul + resample first): add `--from-raw` and `--nyul-npz` pointing at the landmark file.

## Outputs in `--out-dir`

- `mask.nii.gz` — organ outline
- `array.npy` / `array_1ch.npy` — cooked cube
- `epe.json` or `svi.json` — probability and 0/1
- `meta.json` — log of the steps

## Typical mistakes

- Run from **`unet_pipeline/`** so Python finds `train_3d_unet` and `models`.
- No GPU → it still runs, but slowly.
- SV `--mode full` without a `_0004` nearby → cube is cooked from axial T2, which is **not** what SVI-Net trained on.
- Identified clinical CSVs are not needed and should not be copied here.
