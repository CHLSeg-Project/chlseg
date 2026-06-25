```markdown
# CHLSeg: Cascaded Hierarchical Low-level Detail Recovery for Semantic Segmentation
> (Frequency-Spatial Detail Branch + Multi-scale Convolutional Cross-Attention).

This repository is built on
[MMSegmentation v1.2.2](https://github.com/open-mmlab/mmsegmentation).

<div align="center">
  <img src="https://raw.githubusercontent.com/CHLSeg-Project/chlseg/main/assets/architecture.png" width="90%" alt="CHLSeg architecture">
</div>

-------------------------------------------------------------------------------

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Software Environment](#software-environment)
- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference & Visualisation](#inference--visualisation)
- [Config Reference](#config-reference)
- [Project Structure](#project-structure)
- [Citation](#citation)

-------------------------------------------------------------------------------

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU**       | 1 × NVIDIA GPU, **8 GB** VRAM  | 1 × RTX 3090 / A100 (24/40 GB) |
| **CPU**       | 8 cores                        | 16+ cores                      |
| **RAM**       | 32 GB                          | 64 GB                          |
| **Disk**      | 50 GB (ADE20K dataset + checkpoints) | SSD preferred           |
| **CUDA**      | 11.3+                          | 12.1+                          |

> **Note:** The selective-scan CUDA kernel (`kernel/selective_scan`) is
> **optional** and only needed for SegMAN / VSSM ablation experiments.
> Standard CHLSeg training on MSCAN / MiT backbones does **not** require
> compiling any custom CUDA extensions.

## Software Environment

| Item              | Version / Notes                                                      |
|-------------------|----------------------------------------------------------------------|
| **OS**            | Ubuntu 20.04 / 22.04 LTS (recommended); Windows 10/11 also supported |
| **Python**        | 3.9                                                                  |
| **PyTorch**       | ≥ 1.12.0 (1.13.x or 2.x recommended)                                 |
| **torchvision**   | matching PyTorch version                                             |
| **CUDA Toolkit**  | 11.3                                                                 |
| **mmcv**          | ≥ 2.0.0rc4                                                           |
| **mmengine**      | ≥ 0.7.0                                                              |
| **MMSegmentation**| 1.2.2 (this repository, installed in editable mode)                  |

-------------------------------------------------------------------------------

## Environment Setup
### 1. Create a conda / venv environment (recommended)

```bash
conda create -n chlseg python=3.10 -y
conda activate chlseg
```

### 2. Install PyTorch

Choose the command matching your CUDA version from
[pytorch.org](https://pytorch.org/get-started/locally/).

Example (CUDA 12.1):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install mmcv + mmengine (OpenMMLab ecosystem)

```bash
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
pip install mmengine
```

> Replace `cu121/torch2.1` with the string matching your actual CUDA / PyTorch
> version.  See the [mmcv install guide](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)
> for the full matrix.

### 4. Install MMSegmentation + remaining dependencies

```bash
pip install -r requirements.txt
pip install -e .            # install this project as an editable package
```

### 5. (Optional) Compile the selective-scan CUDA kernel

Only needed if you plan to run the SegMAN backbone / VSSM ablation experiments:

```bash
cd kernel/selective_scan
pip install -e .
cd ../..
```
Training environment: All experiments were conducted on Ubuntu 20.04 / 22.04 LTS.
We strongly recommend training on a Linux distribution (Ubuntu) for the best
compatibility and performance. Windows 10/11 is also supported but may require
additional configuration.
-------------------------------------------------------------------------------

## Dataset Preparation

CHLSeg is primarily evaluated on **ADE20K (150 classes)**. For completeness, we
also provide download instructions for **Cityscapes** and **COCO-Stuff 164K**,
which may be used for additional experiments or ablation studies.

### ADE20K

```bash
mkdir -p data/ade20k
cd data/ade20k

# Download from MIT ADE20K official site
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip ADEChallengeData2016.zip
rm ADEChallengeData2016.zip
cd ../..
```

**Expected directory layout**

```text
data/ade20k/
├── images/
│   ├── training/        # 20210 images
│   └── validation/      #  2000 images
└── annotations/
    ├── training/        # 20210 masks (png, class ids 0-150)
    └── validation/      #  2000 masks
```

### Cityscapes

> **Note:** Cityscapes requires registration. Download `leftImg8bit_trainvaltest.zip`
> and `gtFine_trainvaltest.zip` from the [official website](https://www.cityscapes-dataset.com/downloads/).

```bash
mkdir -p data/cityscapes
cd data/cityscapes

# Place the downloaded zip files here and unzip
unzip leftImg8bit_trainvaltest.zip
unzip gtFine_trainvaltest.zip
cd ../..
```

**Expected directory layout**

```text
data/cityscapes/
├── leftImg8bit/
│   ├── train/
│   ├── val/
│   └── test/
└── gtFine/
    ├── train/
    ├── val/
    └── test/
```

For usage with MMSegmentation, you may need to create the `labelTrainIds` maps
or use the provided Cityscapes scripts in `tools/dataset_converters/`.

### COCO-Stuff 164K

COCO-Stuff 164K includes 164k images with stuff annotations. The required
pixel‑level masks (PNG) are provided in the `stuffthingmaps` archive.

```bash
mkdir -p data/coco_stuff164k
cd data/coco_stuff164k

# Download images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

# Download pixel-level annotations (stuff + thing maps)
wget http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip

unzip train2017.zip
unzip val2017.zip
unzip stuffthingmaps_trainval2017.zip
rm train2017.zip val2017.zip stuffthingmaps_trainval2017.zip
cd ../..
```

After extraction, the directory should look like:

```text
data/coco_stuff164k/
├── images/
│   ├── train2017/
│   └── val2017/
└── annotations/
    ├── train2017/
    └── val2017/
```

-------------------------------------------------------------------------------

## Training

All training commands use MMSegmentation's `tools/train.py` launcher:

```bash
python tools/train.py <config_path> [--work-dir <dir>] [--resume]
```

### Single-GPU training

```bash
# CHLSeg-T (MSCAN-T backbone, our primary model)
python tools/train.py \
    configs/chlseg/mscan-t_1xb16-adamw-160k_ade20k-512x512_chlseg.py \
    --work-dir work_dirs/chlseg_tiny

# CHLSeg-S (MSCAN-S backbone, larger variant)
python tools/train.py \
    configs/chlseg/mscan-s_1xb16-adamw-160k_ade20k-512x512_chlseg_small.py \
    --work-dir work_dirs/chlseg_small

# CHLSeg on MiT-B0 (SegFormer encoder, cross-backbone experiment)
python tools/train.py \
    configs/chlseg/mit-b0_1xb2-adamw-160k_ade20k-512x512_chlseg.py \
    --work-dir work_dirs/chlseg_mitb0
```

### Multi-GPU training (DistributedDataParallel)

```bash
# 4-GPU CHLSeg-T training
bash tools/dist_train.sh \
    configs/chlseg/mscan-t_1xb16-adamw-160k_ade20k-512x512_chlseg.py \
    4 --work-dir work_dirs/chlseg_tiny
```

### Resume from checkpoint

```bash
python tools/train.py \
    configs/chlseg/mscan-t_1xb16-adamw-160k_ade20k-512x512_chlseg.py \
    --resume --work-dir work_dirs/chlseg_tiny
```

### Monitoring

Training logs and metrics are written to the `work_dirs/<run>/` directory:
- `vis_data/`  — TensorBoard event files
- `*.log.json` — structured log (loss, IoU per iteration)
- `iter_*.pth` — saved checkpoints

Launch TensorBoard:
```bash
tensorboard --logdir work_dirs
```

### Key hyper-parameters

All CHLSeg configs share these settings:
- **Iterations:** 160,000 (ADE20K standard)
- **Optimizer:** AdamW, lr=6e-5, weight_decay=0.01, betas=(0.9, 0.999)
- **Schedule:** Linear warmup (1,500 iters) → PolyLR (power=1.0)
- **Loss weights:** λ_CE=1.0, λ_PAT=0.15, λ_SOAH=0.1, λ_Cluster=0.1
- **Input size:** 512×512 (random crop + resize during training)
- **Batch size:** 16 (MSCAN-T/S), 2 (MiT-B0, due to transformer memory)

-------------------------------------------------------------------------------

## Inference & Visualisation

### Demo: colourised overlays with ground truth

The repository includes `tools/demo.py`, a standalone script that runs CHLSeg
inference on a folder of images and generates per-image colourised overlays for
both the **prediction** and the **ground-truth mask**. The original image is
optionally blended under the segmentation mask for cleaner visualisation.

**Features**
- Works without any additional configuration files.
- Uses the official **ADE20K 150-class palette** for vivid, consistent colours.
- Automatically reads ground-truth masks (single‑channel PNGs) from a separate
  directory.
- Supports overlay blending with adjustable transparency.
- Outputs three files per input: original image copy, CHLSeg prediction overlay,
  and ground-truth overlay.

**Quick start**

Place your input images, ground-truth masks, model checkpoint, and config in
the default directories (override with command‑line arguments if needed):

```
images/          # Original images (e.g. .jpg)
masks/           # Ground-truth masks (PNG, class indices)
configs/         # MMSegmentation config file (.py)
weights/         # Trained checkpoint (.pth)
```

Run the demo:

```bash
python tools/demo.py
```

Results are saved to `figures/` by default.

**Command‑line arguments**

| Argument   | Default    | Description                                      |
|------------|------------|--------------------------------------------------|
| `--images` | `images`   | Directory containing original images             |
| `--masks`  | `masks`    | Directory containing ground‑truth PNG masks      |
| `--output` | `figures`  | Output directory for colourised results          |
| `--device` | `cuda:0`   | Inference device                                 |
| `--ext`    | `.jpg`     | Image file extension                             |
| `--overlay`| `True`     | Blend original image under the segmentation mask |
| `--alpha`  | `0.5`      | Transparency of the segmentation layer (0‑1)     |

> **Note:** The default model configuration and checkpoint are hard‑coded in
> `tools/demo.py` (the CHLSeg‑T model). Edit the `models_cfg` list inside the
> script to match your own config and weight paths.

**Example**

```bash
python tools/demo.py \
    --images my_dataset/images/validation \
    --masks my_dataset/annotations/validation \
    --output demo_output \
    --device cuda:0 \
    --overlay \
    --alpha 0.4
```

After execution, the output folder contains:
- `<stem>_image.png` – original image (for reference)
- `<stem>_gt.png` – ground-truth mask colourised and blended with the original
- `<stem>_chlseg.png` – CHLSeg prediction colourised and blended with the original

If `--overlay` is set to `False`, the masks are drawn on a black background.

-------------------------------------------------------------------------------

## Config Reference

The `configs/cclnet/` directory contains ready‑to‑use training configurations:

| Config file                         | Backbone | Decoder | Params (M) |
|-------------------------------------|----------|---------|------------|
| `mscan-t_1xb16-..._chlseg.py`       | MSCAN‑T | CHLSeg  | 8.2~8.4     |
| `mscan-s_1xb16-..._chlseg_small.py` | MSCAN‑S | CHLSeg  | 26.9~27.7   |
| `mit-b0_1xb2-..._chlseg.py`         | MiT‑B0  | CHLSeg  | 7.2~7.8     |

The configs are compatible with MMSegmentation’s standard `tools/train.py` and
`tools/test.py` launchers. Hyper‑parameters are set for 160k‑iteration training
on ADE20K with 512×512 crops.

-------------------------------------------------------------------------------

## Project Structure

```
CCASeg-main/
├── configs/
│   └── chlnet/                    # CHLSeg configs
├── mmseg/                         # MMSegmentation core (editable install)
├── tools/
│   ├── train.py                   # Training launcher
│   ├── test.py                    # Single‑image inference
│   ├── batch_inference.py         # Batch inference with colourised output
│   └── demo.py                    # Demo script for overlays (see Inference section)
├── kernel/
│   └── selective_scan/            # Optional CUDA kernel for VSSM backbone
├── figures/
│   └── batch_spec_example.json    # Example batch spec
├── requirements.txt
└── setup.py
```

-------------------------------------------------------------------------------