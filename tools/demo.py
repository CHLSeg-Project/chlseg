"""
Colorize CHLSeg model predictions and ground-truth masks, and save them individually.
Optionally blend the original image as a background under the segmentation overlay (enabled by default).

For each input image, the script generates:
  - a copy of the original image
  - a colorized CHLSeg prediction (optionally blended with the original)
  - a colorized ground-truth label (optionally blended with the original)

Default directory layout:
  images/          -- original images
  masks/           -- grayscale ground-truth labels (png)
  weights/         -- model checkpoint
  configs/         -- MMSegmentation config file
  figures/         -- output directory
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import sys
from tqdm import tqdm
from PIL import Image

# Make sure mmseg is installed and custom modules are registered
try:
    from mmseg.apis import init_model, inference_model
    from mmseg.utils import register_all_modules
    register_all_modules()
except ImportError:
    print("Please install MMSegmentation and its dependencies, or ensure the environment is properly configured.")
    sys.exit(1)

# ------------------------------------------------------------
# Official full ADE20K palette (150 classes, no offset)
# ------------------------------------------------------------
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


def colorize(mask: np.ndarray, palette: np.ndarray = ADE20K_PALETTE) -> np.ndarray:
    """Map a H×W class-index mask to an RGB image. Ignore label 255 (shown as black)."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    max_cls = len(palette) - 1
    for cls_id in np.unique(mask):
        if cls_id == 255:                     # ignore label
            color[mask == cls_id] = [0, 0, 0]
        elif cls_id <= max_cls:
            color[mask == cls_id] = palette[cls_id]
        else:
            color[mask == cls_id] = [128, 128, 128]   # out-of-range warning (gray)
    return color


def overlay_mask_on_image(image_rgb: np.ndarray, mask_colored: np.ndarray,
                          alpha: float = 0.5) -> np.ndarray:
    """
    Blend the colorized segmentation mask over the original RGB image.
    image_rgb: original RGB image (H, W, 3)
    mask_colored: palette-colored segmentation map (H, W, 3)
    alpha: transparency of the mask layer, in [0, 1] (default 0.5)
    """
    if image_rgb.shape[:2] != mask_colored.shape[:2]:
        mask_colored = cv2.resize(mask_colored,
                                  (image_rgb.shape[1], image_rgb.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(image_rgb, 1 - alpha, mask_colored, alpha, 0)


def main():
    parser = argparse.ArgumentParser(description="Colorize segmentation results and save them individually")
    parser.add_argument('--images', default='images', help='Directory containing original images')
    parser.add_argument('--masks', default='masks', help='Directory containing ground-truth label masks')
    parser.add_argument('--output', default='figures', help='Output directory for colorized results')
    parser.add_argument('--device', default='cuda:0', help='Inference device')
    parser.add_argument('--ext', default='.jpg', help='Image file extension')
    parser.add_argument('--overlay', action='store_true', default=True,
                        help='Blend original image under the segmentation mask (enabled by default)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Transparency of the segmentation overlay (0-1), default 0.5')
    args = parser.parse_args()

    images_dir = Path(args.images)
    masks_dir = Path(args.masks)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===========================
    # Model list (CHLSeg checkpoint)
    # ===========================
    models_cfg = [
        #  Update the config and checkpoint paths to match your setup
        {
            "config": "configs/chlseg/mscan-s_1xb16-adamw-160k_ade20k-512x512_chlseg_tiny.py",
            "checkpoint": "weights/chlseg.pth",
            "label": "chlseg"
        }
    ]

    # Load models
    print("Loading model(s)...")
    models = []
    for m_cfg in models_cfg:
        config_path = Path(m_cfg["config"])
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        ckpt_path = Path(m_cfg["checkpoint"])
        if not ckpt_path.is_absolute():
            ckpt_path = Path.cwd() / ckpt_path
        print(f"  Loading {m_cfg['label']} : {config_path.name}")
        model = init_model(str(config_path), str(ckpt_path), device=args.device)
        models.append({"model": model, "label": m_cfg["label"]})

    # Iterate over all images
    image_files = sorted(images_dir.glob(f"*{args.ext}"))
    if not image_files:
        print(f"No {args.ext} files found in {images_dir}")
        return

    print(f"\nProcessing {len(image_files)} images...")
    for img_path in tqdm(image_files):
        gt_path = masks_dir / f"{img_path.stem}.png"
        if not gt_path.exists():
            print(f"  ⚠ Ground truth not found: {gt_path}, skipping {img_path.name}")
            continue

        # Read original image (RGB)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  Could not read image: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Save a copy of the original image
        out_img = output_dir / f"{img_path.stem}_image.png"
        cv2.imwrite(str(out_img), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

        # Read ground-truth label with PIL, ensure single-channel class indices
        gt_pil = Image.open(gt_path)
        if gt_pil.mode in ('RGB', 'RGBA'):
            gt_pil = gt_pil.convert('L')        # force grayscale
        gt_mask = np.array(gt_pil).astype(np.int64)

        # Colorize ground truth
        gt_colored = colorize(gt_mask)
        if args.overlay:
            gt_colored = overlay_mask_on_image(img_rgb, gt_colored, args.alpha)
        out_gt = output_dir / f"{img_path.stem}_gt.png"
        cv2.imwrite(str(out_gt), cv2.cvtColor(gt_colored, cv2.COLOR_RGB2BGR))

        # Run inference with CHLSeg model and save results
        for m in models:
            result = inference_model(m["model"], str(img_path))
            pred_mask = result.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.int64)

            pred_colored = colorize(pred_mask)
            if args.overlay:
                pred_colored = overlay_mask_on_image(img_rgb, pred_colored, args.alpha)
            out_pred = output_dir / f"{img_path.stem}_{m['label']}.png"
            cv2.imwrite(str(out_pred), cv2.cvtColor(pred_colored, cv2.COLOR_RGB2BGR))

    print(f"\nDone! All results saved to {output_dir}")


if __name__ == "__main__":
    main()