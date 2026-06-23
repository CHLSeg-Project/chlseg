"""
Generate IEEE-paper qualitative comparison figures for semantic segmentation.

Produces a multi-row figure where each row is:
    (a) Image  (b) Model₁  …  (e) Modelₙ  (f) Ground Truth

Each prediction is palette-colourised, a red bounding-box highlights the
region of largest joint error, and a zoomed-in crop is inset at a corner.

Usage (JSON spec):
    python tools/generate_qualitative_figure.py \
        --spec figures/qual_spec.json \
        --output figures/qualitative_comparison.pdf

JSON spec format:
{
  "rows": [
    {
      "image":    "data/ade20k/images/validation/ADE_val_00000001.jpg",
      "mask":     "data/ade20k/annotations/validation/ADE_val_00000001.png",
      "roi":      null
    }
  ],
  "models": [
    {"config": "configs/segformer/… .py",   "checkpoint": "weights/segformer.pth",   "label": "SegFormer"},
    {"config": "configs/cclnet/… .py",      "checkpoint": "weights/ccaseg.pth",      "label": "CCASeg"},
    {"config": "configs/cclnet/… _chlseg.py","checkpoint": "weights/chlseg.pth",      "label": "CHLSeg (Ours)"}
  ],
  "zoom": {
    "size": [220, 220],
    "position": "bottom_right",
    "border": 2
  },
  "figure": {
    "dpi": 600,
    "fontsize": 14,
    "title_gap": 60,
    "col_width": 350
  }
}

If "roi" is null for a row, the script automatically selects the connected
component with the largest combined error across all models.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

# --example flag is handled BEFORE any heavy imports, so users can see the
# spec format even without a full mmseg / OpenCV environment.
if '--example' in sys.argv:
    example = {
        "rows": [
            {"image": "data/ade20k/images/validation/ADE_val_00000001.jpg",
             "mask": "data/ade20k/annotations/validation/ADE_val_00000001.png",
             "roi": None},
            {"image": "data/ade20k/images/validation/ADE_val_00000002.jpg",
             "mask": "data/ade20k/annotations/validation/ADE_val_00000002.png",
             "roi": [180, 120, 280, 230]},
            {"image": "data/ade20k/images/validation/ADE_val_00000003.jpg",
             "mask": "data/ade20k/annotations/validation/ADE_val_00000003.png",
             "roi": None},
        ],
        "models": [
            {"config": "configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py",
             "checkpoint": "weights/segformer_mit-b0_512x512_160k_ade20k.pth",
             "label": "SegFormer"},
            {"config": "configs/cclnet/mscan-t_1xb16-adamw-160k_ade20k-512x512_chlseg.py",
             "checkpoint": "weights/chlseg_tiny_ade20k.pth",
             "label": "CHLSeg-T (Ours)"},
        ],
        "zoom": {"size": [220, 220], "position": "bottom_right", "border": 2},
        "figure": {"dpi": 600, "fontsize": 14, "title_gap": 60, "col_width": 350},
    }
    print(json.dumps(example, indent=2))
    sys.exit(0)

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from tqdm import tqdm

from mmseg.apis import init_model, inference_model
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules

# Register all MMSeg modules (including CHLSegHead_tiny, CHLSegHead_small,
# and other custom decode heads) before any model is built.
register_all_modules()

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Palette / colour utilities
# ---------------------------------------------------------------------------
ADE20K_PALETTE = np.array([
    [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
    [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
    [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
    [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
    [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
    [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
    [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
    [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
    [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
    [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
    [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
    [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
    [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
    [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
    [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
    [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
    [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
    [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
    [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
    [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
    [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
    [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
    [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
    [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
    [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
    [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
    [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
    [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
    [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
    [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
    [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
    [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
    [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
    [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
    [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
    [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
    [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
    [102, 255, 0], [92, 0, 255]
], dtype=np.uint8)

# 0-th entry duplicated so palette[0] is defined (unlabeled / background)
ADE20K_PALETTE = np.vstack([[[0, 0, 0]], ADE20K_PALETTE])


# ---------------------------------------------------------------------------
# Colourise a class-index mask
# ---------------------------------------------------------------------------
def colorize(mask: np.ndarray, palette: np.ndarray = ADE20K_PALETTE) -> np.ndarray:
    """Map a H×W class-index array to H×W×3 RGB via *palette*."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in np.unique(mask):
        if cls_id < len(palette):
            color[mask == cls_id] = palette[cls_id]
        else:
            # fallback to grey for out-of-range classes
            color[mask == cls_id] = [128, 128, 128]
    return color


# ---------------------------------------------------------------------------
# Region of interest — auto-selection
# ---------------------------------------------------------------------------
def auto_roi(gt: np.ndarray, preds: List[np.ndarray],
             min_size: int = 64) -> Tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) of the largest connected error component.

    The error map is the pixel-wise OR of all per-model error maps.
    """
    error = np.zeros_like(gt, dtype=np.uint8)
    for pred in preds:
        error = np.bitwise_or(error, (pred != gt).astype(np.uint8))
    if error.sum() == 0:
        # all models perfect — fall back to centre crop
        h, w = gt.shape
        cx, cy = w // 2, h // 2
        s = min(h, w) // 3
        return (cx - s, cy - s, cx + s, cy + s)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(error)
    if num_labels < 2:
        h, w = gt.shape
        cx, cy = w // 2, h // 2
        s = min(h, w) // 3
        return (cx - s, cy - s, cx + s, cy + s)
    # skip label 0 (background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + np.argmax(areas)
    x = int(stats[largest_idx, cv2.CC_STAT_LEFT])
    y = int(stats[largest_idx, cv2.CC_STAT_TOP])
    w_box = int(stats[largest_idx, cv2.CC_STAT_WIDTH])
    h_box = int(stats[largest_idx, cv2.CC_STAT_HEIGHT])
    # ensure minimum size for visibility
    if w_box < min_size:
        cx = x + w_box // 2
        x = max(0, cx - min_size // 2)
        w_box = min_size
    if h_box < min_size:
        cy = y + h_box // 2
        y = max(0, cy - min_size // 2)
        h_box = min_size
    return (x, y, x + w_box, y + h_box)


# ---------------------------------------------------------------------------
# Add zoom box + red rectangle to a single image
# ---------------------------------------------------------------------------
def add_zoom_box(image: np.ndarray,
                 roi: Tuple[int, int, int, int],
                 zoom_size: Tuple[int, int] = (220, 220),
                 position: str = 'bottom_right',
                 border: int = 2) -> np.ndarray:
    """Draw a red ROI rectangle on *image*, crop the ROI, resize it, and
    paste it in-place at *position*, also bordered in red.

    Returns a new array (the input is not mutated).
    """
    x1, y1, x2, y2 = roi
    vis = image.copy()

    # 1) red box on full image
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), thickness=border)

    # 2) zoomed crop
    crop = vis[y1:y2, x1:x2].copy()
    zh, zw = zoom_size
    zoom = cv2.resize(crop, (zw, zh), interpolation=cv2.INTER_NEAREST)

    # 3) place zoom
    h, w = vis.shape[:2]
    margin = 10
    positions = {
        'bottom_right': (w - zw - margin, h - zh - margin),
        'bottom_left':  (margin, h - zh - margin),
        'top_right':    (w - zw - margin, margin),
        'top_left':     (margin, margin),
    }
    sx, sy = positions.get(position, positions['bottom_right'])
    sx = max(0, min(sx, w - zw))
    sy = max(0, min(sy, h - zh))

    vis[sy:sy + zh, sx:sx + zw] = zoom
    cv2.rectangle(vis, (sx, sy), (sx + zw, sy + zh), (255, 0, 0),
                  thickness=border)

    # 4) thin line connecting ROI to zoom
    roi_bottom = (x2, y2)
    zoom_corner = (sx, sy + zh) if 'bottom' in position else (sx, sy)
    cv2.line(vis, roi_bottom, zoom_corner, (255, 0, 0), thickness=1)

    return vis


# ---------------------------------------------------------------------------
# Build a single row:  image | pred_1 | … | pred_N | GT
# ---------------------------------------------------------------------------
def build_row(img_path: str,
              gt_path: str,
              models: List,
              roi: Optional[Tuple[int, int, int, int]],
              zoom_size: Tuple[int, int],
              zoom_position: str,
              zoom_border: int,
              device: str,
              row_idx: int) -> np.ndarray:
    """Returns a H_row × W_row × 3 uint8 row image."""

    # --- load image & GT ---------------------------------------------------
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(f"Cannot read mask: {gt_path}")

    # --- inference ----------------------------------------------------------
    preds_rgb = []
    preds_idx = []
    for m in models:
        result = inference_model(m['model'], img_path)
        pred = result.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.int64)
        preds_idx.append(pred)
        preds_rgb.append(colorize(pred))

    gt_rgb = colorize(gt)

    # --- auto ROI if needed ------------------------------------------------
    if roi is None:
        roi = auto_roi(gt, preds_idx)
        print(f"  Row {row_idx}: auto ROI = {roi}")

    # --- add zoom boxes ----------------------------------------------------
    img_disp = img_rgb.copy()
    for p in preds_rgb:
        # don't draw zoom on original image — only on predictions and GT
        pass

    for i in range(len(preds_rgb)):
        preds_rgb[i] = add_zoom_box(preds_rgb[i], roi, zoom_size,
                                    zoom_position, zoom_border)
    gt_rgb = add_zoom_box(gt_rgb, roi, zoom_size, zoom_position, zoom_border)

    # --- horizontal stack:  image | pred_1 | ... | pred_N | GT -------------
    panels = [img_rgb] + preds_rgb + [gt_rgb]
    row = np.hstack(panels)
    return row


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Generate IEEE qualitative comparison figure')
    parser.add_argument('--spec', required=True,
                        help='Path to JSON specification file')
    parser.add_argument('--output', default='figures/qualitative_comparison.pdf',
                        help='Output PDF path')
    parser.add_argument('--device', default='cuda:0',
                        help='Inference device')
    args = parser.parse_args()

    # --- load spec ----------------------------------------------------------
    with open(args.spec, 'r', encoding='utf-8') as f:
        spec = json.load(f)

    rows_cfg = spec['rows']        # list of {image, mask, roi?}
    models_cfg = spec['models']     # list of {config, checkpoint, label}
    zoom_cfg = spec.get('zoom', {})
    fig_cfg = spec.get('figure', {})

    zoom_size = tuple(zoom_cfg.get('size', [220, 220]))
    zoom_position = zoom_cfg.get('position', 'bottom_right')
    zoom_border = zoom_cfg.get('border', 2)
    dpi = fig_cfg.get('dpi', 600)
    fontsize = fig_cfg.get('fontsize', 14)
    title_gap = fig_cfg.get('title_gap', 60)
    col_width = fig_cfg.get('col_width', 350)

    n_cols = 1 + len(models_cfg) + 1  # image + N models + GT

    # --- register & load models --------------------------------------------
    print(f"Loading {len(models_cfg)} model(s)…")
    loaded_models = []
    for mc in tqdm(models_cfg, desc="Models"):
        cfg_path = str(PROJECT_ROOT / mc['config']) if not Path(mc['config']).is_absolute() else mc['config']
        ckpt_path = str(PROJECT_ROOT / mc['checkpoint']) if not Path(mc['checkpoint']).is_absolute() else mc['checkpoint']
        model = init_model(cfg_path, ckpt_path, device=args.device)
        loaded_models.append({'model': model, 'label': mc['label']})

    # --- build rows --------------------------------------------------------
    all_rows = []
    for i, row_cfg in enumerate(tqdm(rows_cfg, desc="Rows")):
        img_path = str(PROJECT_ROOT / row_cfg['image']) if not Path(row_cfg['image']).is_absolute() else row_cfg['image']
        gt_path = str(PROJECT_ROOT / row_cfg['mask']) if not Path(row_cfg['mask']).is_absolute() else row_cfg['mask']
        roi = tuple(row_cfg['roi']) if row_cfg.get('roi') is not None else None

        row_img = build_row(
            img_path=img_path,
            gt_path=gt_path,
            models=loaded_models,
            roi=roi,
            zoom_size=zoom_size,
            zoom_position=zoom_position,
            zoom_border=zoom_border,
            device=args.device,
            row_idx=i + 1,
        )
        all_rows.append(row_img)

    # --- uniform column width ----------------------------------------------
    # Resize each panel in each row to the same width, keeping aspect ratio.
    panel_w = col_width
    uniform_rows = []
    for row in all_rows:
        h, total_w = row.shape[:2]
        raw_panel_w = total_w // n_cols
        panels = []
        for j in range(n_cols):
            panel = row[:, j * raw_panel_w:(j + 1) * raw_panel_w]
            if panel.shape[1] != panel_w:
                scale = panel_w / panel.shape[1]
                new_h = int(panel.shape[0] * scale)
                panel = cv2.resize(panel, (panel_w, new_h),
                                   interpolation=cv2.INTER_LINEAR)
            panels.append(panel)
        # pad all panels to same height
        max_h = max(p.shape[0] for p in panels)
        padded = []
        for p in panels:
            if p.shape[0] < max_h:
                pad = np.zeros((max_h - p.shape[0], panel_w, 3), dtype=np.uint8)
                p = np.vstack([p, pad])
            padded.append(p)
        uniform_rows.append(np.hstack(padded))

    # --- vertical stack ----------------------------------------------------
    canvas = np.vstack(uniform_rows)

    # --- matplotlib wrapper ------------------------------------------------
    fig_h = canvas.shape[0] / dpi * 1.0 + title_gap / dpi
    fig_w = canvas.shape[1] / dpi * 1.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(canvas)
    ax.axis('off')

    # sub-figure labels: (a) Image, (b..) models, (last) Ground Truth
    letter_base = ord('a')
    col_labels = [f"({chr(letter_base)}) Image"]
    for m in loaded_models:
        letter_base += 1
        col_labels.append(f"({chr(letter_base)}) {m['label']}")
    letter_base += 1
    col_labels.append(f"({chr(letter_base)}) Ground Truth")

    for j, label in enumerate(col_labels):
        x_center = (j + 0.5) * canvas.shape[1] / len(col_labels)
        y_text = canvas.shape[0] + title_gap * 0.55
        plt.text(x_center, y_text, label, ha='center', va='center',
                 fontsize=fontsize, fontweight='normal')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    print(f"\nSaved → {args.output}")
    plt.close(fig)


if __name__ == '__main__':
    main()
