import warnings

import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms import BaseTransform

from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadADE20KRGBAnnotations(BaseTransform):
    """Load RGB ADE annotations as single-channel ADE150 semantic labels.

    This follows MMSeg's ``LoadAnnotations`` flow: read the PNG with
    ``mmcv.imfrombytes(..., flag='unchanged')`` and then apply
    ``reduce_zero_label``. The only extra step is needed for RGB ADE PNGs:
    ``mmcv`` returns BGR channel order in this environment, while the original
    ADE semantic id is stored in the PNG's R channel, so the default semantic
    channel index is 2.
    """

    def __init__(self,
                 reduce_zero_label=None,
                 backend_args=None,
                 imdecode_backend='pillow',
                 num_classes=150,
                 min_valid_ratio=0.01,
                 semantic_channel=2):
        self.reduce_zero_label = reduce_zero_label
        self.backend_args = backend_args
        self.imdecode_backend = imdecode_backend
        self.num_classes = num_classes
        self.min_valid_ratio = min_valid_ratio
        self.semantic_channel = semantic_channel
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')

    def transform(self, results):
        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).astype(np.uint16)

        if gt_semantic_seg.ndim == 3:
            gt_semantic_seg = gt_semantic_seg[
                ..., self.semantic_channel].astype(np.uint16)
        else:
            gt_semantic_seg = np.squeeze(gt_semantic_seg).astype(np.uint16)

        valid_ratio = np.logical_and(
            gt_semantic_seg >= 1,
            gt_semantic_seg <= self.num_classes).mean()
        if gt_semantic_seg.max() > 0 and valid_ratio < self.min_valid_ratio:
            warnings.warn(
                'ADE annotation alignment check failed for '
                f'{results["seg_map_path"]}: only {valid_ratio:.4f} pixels '
                f'are in [1, {self.num_classes}]. Please verify that the '
                f'annotation PNG stores semantic ids in channel '
                f'{self.semantic_channel} after mmcv decoding.')

        # Ignore classes outside ADE150 instead of mapping them to a wrong id.
        invalid = gt_semantic_seg > self.num_classes
        gt_semantic_seg[invalid] = 0

        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'

        gt_semantic_seg = gt_semantic_seg.astype(np.uint8)
        if self.reduce_zero_label:
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255

        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'reduce_zero_label={self.reduce_zero_label}, '
                f"imdecode_backend='{self.imdecode_backend}', "
                f'num_classes={self.num_classes}, '
                f'min_valid_ratio={self.min_valid_ratio}, '
                f'semantic_channel={self.semantic_channel}, '
                f'backend_args={self.backend_args})')
