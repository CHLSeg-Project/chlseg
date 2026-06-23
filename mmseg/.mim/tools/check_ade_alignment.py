import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def collect_split(root, split, max_files=None, num_classes=150):
    pattern = f'annotations/{split}/*.png'
    files = sorted(Path(root).glob(pattern))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f'No files matched {Path(root) / pattern}')

    class_ids = set()
    total_pixels = 0
    valid_pixels = 0
    zero_only_files = 0
    rgb_files = 0
    max_values = []

    for file in files:
        arr = np.array(Image.open(file))
        if arr.ndim == 3:
            rgb_files += 1
            # PIL keeps RGB order; ADE semantic ids are stored in R.
            sem = arr[..., 0]
            max_values.append(tuple(int(arr[..., i].max()) for i in range(3)))
        else:
            sem = np.squeeze(arr)
            max_values.append(int(sem.max()))

        zero_only_files += int(sem.max() == 0)
        total_pixels += sem.size
        valid_pixels += int(((sem >= 1) & (sem <= num_classes)).sum())
        class_ids.update(np.unique(sem).tolist())

    valid_classes = sorted(c for c in class_ids if 1 <= c <= num_classes)
    return {
        'split': split,
        'files': len(files),
        'rgb_files': rgb_files,
        'zero_only_files': zero_only_files,
        'valid_pixel_ratio': valid_pixels / max(total_pixels, 1),
        'valid_classes': len(valid_classes),
        'class_range': (
            int(min(class_ids)) if class_ids else None,
            int(max(class_ids)) if class_ids else None),
        'max_values_head': max_values[:5],
    }


def main():
    parser = argparse.ArgumentParser(
        description='Check ADEChallengeData2016 semantic label alignment.')
    parser.add_argument(
        '--data-root',
        default='data/ade/ADEChallengeData2016',
        help='Path to ADEChallengeData2016.')
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Optionally scan only the first N files of each split.')
    parser.add_argument('--num-classes', type=int, default=150)
    args = parser.parse_args()

    root = Path(args.data_root)
    for split in ('training', 'validation'):
        stats = collect_split(
            root, split, max_files=args.max_files,
            num_classes=args.num_classes)
        print(f'[{split}]')
        for key, value in stats.items():
            if key != 'split':
                print(f'{key}: {value}')
        print()

    print('Expected: labels may be RGB, but semantic ids must be in channel R,')
    print('with class ids 1..150 before reduce_zero_label and 0 as ignore.')
    print('Note: the training transform reads PNGs through mmcv, which returns')
    print('BGR order in this environment, so it uses semantic_channel=2.')


if __name__ == '__main__':
    main()
