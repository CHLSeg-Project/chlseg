import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from ..builder import MODELS
from ..losses import accuracy
from ..utils import resize
from .ccaseg_tiny import (InjectionMultiSum, Inject, LKCA2, LKCA3,
                          PyramidPoolAgg)
from .decode_head import BaseDecodeHead


class HaarWavelet2D(nn.Module):

    @staticmethod
    def dwt(x):
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh

    @staticmethod
    def idwt(ll, lh, hl, hh):
        x00 = (ll + lh + hl + hh) * 0.5
        x01 = (ll - lh + hl - hh) * 0.5
        x10 = (ll + lh - hl - hh) * 0.5
        x11 = (ll - lh - hl + hh) * 0.5
        b, c, h, w = ll.shape
        out = ll.new_zeros((b, c, h * 2, w * 2))
        out[:, :, 0::2, 0::2] = x00
        out[:, :, 0::2, 1::2] = x01
        out[:, :, 1::2, 0::2] = x10
        out[:, :, 1::2, 1::2] = x11
        return out


class L2SpectrumDecompositionAttention(nn.Module):
    """WaveSeg-style L2 frequency refinement with residual output."""

    def __init__(self, channels, norm_cfg, act_cfg=dict(type='GELU')):
        super().__init__()
        self.wavelet = HaarWavelet2D()
        self.low_branch = ConvModule(
            channels, channels, 3, padding=1, groups=channels,
            norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.low_proj = ConvModule(
            channels, channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.high_pre = ConvModule(
            channels * 3, channels * 3, 1, norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.high_scan = nn.Conv1d(
            channels * 3, channels * 3, kernel_size=7, padding=3,
            groups=channels * 3)
        self.rebalance = ConvModule(
            channels, channels, 1, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        residual = x
        ll, lh, hl, hh = self.wavelet.dwt(x)
        low = self.low_proj(self.low_branch(ll) + ll)
        high = self.high_pre(torch.cat([lh, hl, hh], dim=1))
        b, c, h, w = high.shape
        high = self.high_scan(high.flatten(2)).view(b, c, h, w)
        lh, hl, hh = torch.chunk(high, 3, dim=1)
        x = self.wavelet.idwt(low, lh, hl, hh)
        return self.rebalance(x) + residual


class HighFrequencyConfidence(nn.Module):
    """Sigmoid high-frequency confidence, avoiding channel competition."""

    def __init__(self, in_channels, out_channels, norm_cfg,
                 act_cfg=dict(type='GELU')):
        super().__init__()
        self.wavelet = HaarWavelet2D()
        self.lh_conv = ConvModule(
            in_channels, out_channels, (1, 3), padding=(0, 1),
            norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.hl_conv = ConvModule(
            in_channels, out_channels, (3, 1), padding=(1, 0),
            norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.hh_conv = ConvModule(
            in_channels, out_channels, 3, padding=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.fuse = nn.Sequential(
            ConvModule(
                out_channels * 3, out_channels, 1, norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Sigmoid())

    def forward(self, x, target_size):
        x = resize(x, size=target_size, mode='bilinear', align_corners=False)
        _, lh, hl, hh = self.wavelet.dwt(x)
        # Bias-free smoothness gate: uniform regions (water, blanket, ...)
        # have ~0 high-freq magnitude, so the sigmoid bias of self.fuse no
        # longer fabricates spurious "edges" there.
        hf_mag = torch.cat([lh.abs(), hl.abs(), hh.abs()],
                           dim=1).mean(dim=1, keepdim=True)
        hf_mag = hf_mag / (hf_mag.amax(dim=(2, 3), keepdim=True) + 1e-6)
        x_h = torch.cat(
            [self.lh_conv(lh), self.hl_conv(hl), self.hh_conv(hh)], dim=1)
        x_h = self.fuse(x_h) * hf_mag
        x_h = resize(x_h, size=target_size, mode='bilinear',
                     align_corners=False)
        return x_h


class ScaleAwareChannelAttentionLite(nn.Module):
    """COPNet SCA-lite: cross-scale channel calibration for detail branch."""

    def __init__(self, c2_channels, c3_channels, c4_channels, norm_cfg,
                 reduction=4):
        super().__init__()
        hidden = max(c2_channels // reduction, 16)
        self.proj3 = ConvModule(
            c3_channels, c2_channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.proj4 = ConvModule(
            c4_channels, c2_channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(c2_channels * 3, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, c2_channels, 1),
            nn.Sigmoid())
        self.context_proj = ConvModule(
            c2_channels * 3, c2_channels, 1, norm_cfg=norm_cfg,
            act_cfg=dict(type='GELU'))

    def forward(self, c2, c3, c4):
        size = c2.shape[2:]
        c3 = resize(self.proj3(c3), size=size, mode='bilinear',
                    align_corners=False)
        c4 = resize(self.proj4(c4), size=size, mode='bilinear',
                    align_corners=False)
        pooled = torch.cat([
            F.adaptive_avg_pool2d(c2, 1),
            F.adaptive_avg_pool2d(c3, 1),
            F.adaptive_avg_pool2d(c4, 1)
        ], dim=1)
        gate = self.channel_gate(pooled)
        context = self.context_proj(torch.cat([c2, c3, c4], dim=1))
        return c2 * (1 + 0.5 * gate) + 0.1 * context


class OverlapPatchCrossScaleLite(nn.Module):
    """COPNet PCA-lite using cheap overlapping local windows across scales."""

    def __init__(self, c2_channels, c3_channels, c4_channels, norm_cfg,
                 act_cfg=dict(type='GELU')):
        super().__init__()
        self.proj3 = ConvModule(
            c3_channels, c2_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.proj4 = ConvModule(
            c4_channels, c2_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.local3 = nn.Conv2d(
            c2_channels, c2_channels, 3, padding=1, groups=c2_channels)
        self.local5 = nn.Conv2d(
            c2_channels, c2_channels, 5, padding=2, groups=c2_channels)
        self.attn = nn.Sequential(
            ConvModule(
                c2_channels * 3, c2_channels, 1, norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            nn.Conv2d(c2_channels, c2_channels, 1),
            nn.Sigmoid())
        self.delta = ConvModule(
            c2_channels * 3, c2_channels, 3, padding=1, norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, c2, c3, c4):
        size = c2.shape[2:]
        c3 = resize(self.proj3(c3), size=size, mode='bilinear',
                    align_corners=False)
        c4 = resize(self.proj4(c4), size=size, mode='bilinear',
                    align_corners=False)
        local = self.local3(c2) + self.local5(c2)
        feat = torch.cat([local, c3, c4], dim=1)
        return self.attn(feat) * self.delta(feat)


class SmallObjectAuxHead(nn.Module):

    def __init__(self, channels, norm_cfg, act_cfg=dict(type='GELU')):
        super().__init__()
        self.block = nn.Sequential(
            ConvModule(
                channels, channels, 3, padding=1, groups=channels,
                norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(channels, channels, 1, norm_cfg=norm_cfg,
                       act_cfg=act_cfg),
            nn.Conv2d(channels, 2, kernel_size=1))

    def forward(self, x):
        return self.block(x)


class ClusterHead(nn.Module):
    """Hierarchical semantic cluster supervision head.

    Shared spatial backbone (DWConv3x3 -> Conv1x1) followed by parallel
    1x1 classifiers that predict coarse-to-fine semantic groupings.
    The supervision anchors the middle-layer feature on macro-category
    structure before fine-grained 150-way classification.
    """

    def __init__(self, in_channels, cluster_channels=(5, 4, 2), norm_cfg=None):
        super().__init__()
        self.shared = nn.Sequential(
            ConvModule(
                in_channels, in_channels, 3, padding=1, groups=in_channels,
                norm_cfg=norm_cfg, act_cfg=dict(type='GELU')),
            ConvModule(
                in_channels, in_channels, 1,
                norm_cfg=norm_cfg, act_cfg=dict(type='GELU')),
        )
        self.heads = nn.ModuleList([
            nn.Conv2d(in_channels, c, 1) for c in cluster_channels
        ])

    def forward(self, x):
        feat = self.shared(x)
        return [head(feat) for head in self.heads]


# ---------------------------------------------------------------------------
# ADE20K hierarchical cluster label mappings (150 classes -> coarse groups)
# ---------------------------------------------------------------------------
# 2-class:  stuff(0) vs thing(1)
_AD20K_STUFF = {
    0, 2, 3, 5, 6, 8, 9, 11, 13, 16, 18, 21, 25, 26, 29, 32, 34, 38, 42,
    46, 48, 52, 53, 54, 56, 59, 60, 61, 63, 68, 72, 77, 84, 86, 91, 94, 95,
    96, 101, 104, 105, 106, 109, 113, 121, 128, 136, 140, 144,
}

# 5-class macro categories
_AD20K_CLUSTER_5_STRUCTURAL = {
    0, 1, 3, 5, 6, 8, 11, 14, 18, 25, 32, 38, 42, 48, 52, 53, 54, 59, 61,
    63, 77, 84, 86, 91, 94, 95, 96, 101, 104, 105, 106, 121, 123, 136, 140,
    144,
}
_AD20K_CLUSTER_5_NATURE = {
    2, 4, 9, 13, 16, 17, 21, 26, 29, 34, 46, 60, 66, 68, 72, 113, 128,
}
_AD20K_CLUSTER_5_FURNITURE = {
    7, 10, 15, 19, 22, 23, 24, 27, 28, 30, 31, 33, 35, 37, 39, 40, 44, 45,
    47, 49, 50, 51, 55, 57, 62, 64, 65, 69, 70, 71, 73, 75, 78, 79, 81, 88,
    97, 99, 107, 108, 110, 114, 117, 118, 124, 129, 131, 137, 145, 146,
}
_AD20K_CLUSTER_5_OBJECTS = {
    36, 41, 43, 56, 58, 67, 74, 82, 85, 89, 92, 93, 98, 100, 109, 111, 112,
    115, 119, 120, 125, 130, 134, 135, 138, 139, 141, 142, 143, 147, 148,
    149,
}
_AD20K_CLUSTER_5_BEINGS_VEHICLES = {
    12, 20, 76, 80, 83, 87, 90, 102, 103, 116, 122, 126, 127, 132, 133,
}

# 4-class: merge nature+structural -> environment, beings+vehicles -> animate
_AD20K_CLUSTER_4_ENV = _AD20K_CLUSTER_5_STRUCTURAL | _AD20K_CLUSTER_5_NATURE
_AD20K_CLUSTER_4_FURNITURE = _AD20K_CLUSTER_5_FURNITURE
_AD20K_CLUSTER_4_OBJECTS = _AD20K_CLUSTER_5_OBJECTS
_AD20K_CLUSTER_4_ANIMATE = _AD20K_CLUSTER_5_BEINGS_VEHICLES


def _build_cluster_map(cluster_sets, num_classes=150):
    """Convert a list of category sets into a [num_classes] LongTensor."""
    mapping = torch.zeros(num_classes, dtype=torch.long)
    for cluster_id, cls_set in enumerate(cluster_sets):
        for cls_idx in cls_set:
            mapping[cls_idx] = cluster_id
    return mapping


# Pre-built cluster label mappers (registered as buffers at init time)
_CLUSTER_5_MAP = _build_cluster_map([
    _AD20K_CLUSTER_5_STRUCTURAL,
    _AD20K_CLUSTER_5_NATURE,
    _AD20K_CLUSTER_5_FURNITURE,
    _AD20K_CLUSTER_5_OBJECTS,
    _AD20K_CLUSTER_5_BEINGS_VEHICLES,
])
_CLUSTER_4_MAP = _build_cluster_map([
    _AD20K_CLUSTER_4_ENV,
    _AD20K_CLUSTER_4_FURNITURE,
    _AD20K_CLUSTER_4_OBJECTS,
    _AD20K_CLUSTER_4_ANIMATE,
])
_CLUSTER_2_MAP = _build_cluster_map([
    _AD20K_STUFF,
    set(range(150)) - _AD20K_STUFF,
])


@MODELS.register_module()
class CHLSegHead_tiny(BaseDecodeHead):
    """Fixed 4-Block CCASeg-T (per paper) + WaveSeg(L2) + COPNet-lite +
    small-object aux head.

    Structure:
      * CCA backbone follows the corrected ``CCASegHead_tiny``:
        4 CCA Blocks (c4/c3/c2 + c1-pooled), each with N LKCA refines
        (N = num_lkca_repeats, default 3; set to 4 for ablation).
        Full 4-input SFI (``multi*_5/_3/_4``) and 4-output prediction-side
        SFI (``predict1_2/2_3/3_4``).
      * The wavelet / COP-lite detail branch is preserved exactly as in
        the original ``CCAWaveCOPLiteHead_small`` and is plugged in
        between Block 3 (output ``new2``) and Block 4 so that the c1
        branch can also benefit from the refined ``new2``.
    """

    def __init__(self,
                 feature_strides,
                 small_object_classes=(12, 20, 22, 36, 43, 47, 58, 61, 66,
                                       71, 74, 76, 80, 83, 87, 89, 90, 93,
                                       98),
                 aux_loss_weight=0.1,
                 pat_loss_weight=0.15,
                 pat_temperature=2.0,
                 pat_eps=0.1,
                 max_refine_scale=0.15,
                 ablate_pat=False,
                 ablate_so_aux=False,
                 ablate_detail=False,
                 num_lkca_repeats=3,
                 cluster_loss_weight=0.1,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.small_object_classes = tuple(small_object_classes)
        self.aux_loss_weight = aux_loss_weight
        self.pat_loss_weight = pat_loss_weight
        self.pat_temperature = pat_temperature
        self.pat_eps = pat_eps
        self.max_refine_scale = max_refine_scale
        self.ablate_pat = ablate_pat
        self.ablate_so_aux = ablate_so_aux
        self.ablate_detail = ablate_detail
        self.num_lkca_repeats = num_lkca_repeats
        self.cluster_loss_weight = cluster_loss_weight

        c1, c2, c3, c4 = self.in_channels
        embedding_dim = kwargs['decoder_params']['embed_dim']

        # ---------- detail branch (wavelet + COP-lite + small-obj aux) ----------
        if not self.ablate_detail:
            self.hf_conf = HighFrequencyConfidence(c1, c2, norm_cfg=self.norm_cfg)
            self.l2_sda = L2SpectrumDecompositionAttention(c2, self.norm_cfg)
            self.sca_lite = ScaleAwareChannelAttentionLite(
                c2, c3, c4, norm_cfg=self.norm_cfg)
            self.pca_lite = OverlapPatchCrossScaleLite(
                c2, c3, c4, norm_cfg=self.norm_cfg)
            self.detail_fuse = ConvModule(
                c2 * 3, c2, 1, norm_cfg=self.norm_cfg, act_cfg=dict(type='GELU'))
            self.refine_scale = nn.Parameter(torch.tensor(-2.0))
        # small_head sources from `detail`, so it requires the detail branch.
        if not self.ablate_so_aux and not self.ablate_detail:
            self.small_head = SmallObjectAuxHead(c2, norm_cfg=self.norm_cfg)
        # hierarchical cluster head — anchors detail feature on macro-category
        # structure (coarse-to-fine: 2→4→5 classes) before 150-way classification.
        if self.cluster_loss_weight > 0 and not self.ablate_detail:
            self.cluster_head = ClusterHead(
                c2, cluster_channels=(5, 4, 2), norm_cfg=self.norm_cfg)
            self.register_buffer('_cluster_map_2', _CLUSTER_2_MAP, persistent=False)
            self.register_buffer('_cluster_map_4', _CLUSTER_4_MAP, persistent=False)
            self.register_buffer('_cluster_map_5', _CLUSTER_5_MAP, persistent=False)
            # cluster maps are registered buffers — accessed fresh each
            # forward (not cached in a tuple) so DDP replicas always see
            # the device-correct tensors.

        # ---------- CCA backbone (full 4-Block CCASeg-T) ----------
        # F2-Pooling Module
        self.down2_1 = PyramidPoolAgg(stride=2, dim=c2)
        self.down2_2 = PyramidPoolAgg(stride=4, dim=c2)
        # F3-Pooling Module
        self.down3_1 = PyramidPoolAgg(stride=2, dim=c3)
        # F1-Pooling Module (c1 stride=4 → pooled to H/8, H/16, H/32)
        self.down1_1 = PyramidPoolAgg(stride=2, dim=c1)
        self.down1_2 = PyramidPoolAgg(stride=4, dim=c1)
        self.down1_3 = PyramidPoolAgg(stride=8, dim=c1)

        # 4 CCA Blocks, each with 3 LKCA refines
        self.attnc4 = LKCA2(channels1=c4, channels2=c4)
        self.attnc3 = LKCA3(channels1=c3, channels2=c4)
        self.attnc2 = LKCA3(channels1=c2, channels2=c4)
        self.attnc1 = LKCA3(channels1=c1, channels2=c4)

        self.attnc4_1 = LKCA2(channels1=c4, channels2=c4)
        self.attnc3_1 = LKCA3(channels1=c3, channels2=c4)
        self.attnc2_1 = LKCA3(channels1=c2, channels2=c4)
        self.attnc1_1 = LKCA3(channels1=c1, channels2=c4)

        self.attnc4_2 = LKCA2(channels1=c4, channels2=c4)
        self.attnc3_2 = LKCA3(channels1=c3, channels2=c4)
        self.attnc2_2 = LKCA3(channels1=c2, channels2=c4)
        self.attnc1_2 = LKCA3(channels1=c1, channels2=c4)

        # ---- ablation: 4th LKCA repeat (only when num_lkca_repeats >= 4) ----
        if self.num_lkca_repeats >= 4:
            self.attnc4_3 = LKCA2(channels1=c4, channels2=c4)
            self.attnc3_3 = LKCA3(channels1=c3, channels2=c4)
            self.attnc2_3 = LKCA3(channels1=c2, channels2=c4)
            self.attnc1_3 = LKCA3(channels1=c1, channels2=c4)

        # Injection Block (Residual = high feature)
        self.inject2 = InjectionMultiSum(dim=c2)
        self.inject3 = InjectionMultiSum(dim=c3)
        self.inject4 = InjectionMultiSum(dim=c4)
        self.inject1 = InjectionMultiSum(dim=c1)

        # SFI first I step (c1_pool → c2_pool) for Blocks 1-3
        self.multi8_5 = Inject(dim1=c1, dim2=c2)
        self.multi16_5 = Inject(dim1=c1, dim2=c2)
        self.multi32_5 = Inject(dim1=c1, dim2=c2)
        # SFI second & third I steps for Blocks 1-3
        self.multi8_3 = Inject(dim1=c2, dim2=c3)
        self.multi8_4 = Inject(dim1=c3, dim2=c4)
        self.multi16_3 = Inject(dim1=c2, dim2=c3)
        self.multi16_4 = Inject(dim1=c3, dim2=c4)
        self.multi32_3 = Inject(dim1=c2, dim2=c3)
        self.multi32_4 = Inject(dim1=c3, dim2=c4)
        # SFI for Block 4 (target = c1_pool)
        self.multi64_5 = Inject(dim1=c1, dim2=c2)
        self.multi64_3 = Inject(dim1=c2, dim2=c3)
        self.multi64_4 = Inject(dim1=c3, dim2=c4)

        # Prediction-side SFI integrates {O_1, O_2, O_3, O_4}
        self.predict1_2 = Inject(dim1=c1, dim2=c2)
        self.predict2_3 = Inject(dim1=c2, dim2=c3)
        self.predict3_4 = Inject(dim1=c3, dim2=c4)

        self.linear_fuse = ConvModule(
            in_channels=c4, out_channels=embedding_dim, kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True))
        self.linear_pred = nn.Conv2d(
            embedding_dim, self.num_classes, kernel_size=1)

    def _forward_logits(self, inputs):
        c1, c2, c3, c4 = self._transform_inputs(inputs)
        _, _, h2, w2 = c2.shape

        # ---- pooling ----
        c2_1 = self.down2_1(c2)
        _, _, h2_1, w2_1 = c2_1.shape
        c2_2 = self.down2_2(c2)
        c3_1 = self.down3_1(c3)

        c1_64 = self.down1_1(c1)
        c1_32 = self.down1_2(c1)
        c1_16 = self.down1_3(c1)

        # ---- Block 1: target = c4 (16x16) ----
        key_4 = self.multi8_5(c1_16, c2_2)
        key_4 = self.multi8_3(key_4, c3_1)
        key_4 = self.multi8_4(key_4, c4)
        _c4 = self.attnc4(c4, key_4)
        _c4 = self.attnc4_1(_c4, key_4)
        _c4 = self.attnc4_2(_c4, key_4)
        if self.num_lkca_repeats >= 4:
            _c4 = self.attnc4_3(_c4, key_4)
        new4 = self.inject4(c4, _c4)

        # ---- Block 2: target = c3 (32x32) ----
        new4_up32 = resize(new4, size=(h2_1, w2_1), mode='bilinear',
                           align_corners=False)
        key_3 = self.multi16_5(c1_32, c2_1)
        key_3 = self.multi16_3(key_3, c3)
        key_3 = self.multi16_4(key_3, new4_up32)
        _c3 = self.attnc3(c3, key_3)
        _c3 = self.attnc3_1(_c3, key_3)
        _c3 = self.attnc3_2(_c3, key_3)
        if self.num_lkca_repeats >= 4:
            _c3 = self.attnc3_3(_c3, key_3)
        new3 = self.inject3(c3, _c3)

        # ---- Block 3: target = c2 (64x64) ----
        new3_up64 = resize(new3, size=(h2, w2), mode='bilinear',
                           align_corners=False)
        new4_up64 = resize(new4, size=(h2, w2), mode='bilinear',
                           align_corners=False)
        key_2 = self.multi32_5(c1_64, c2)
        key_2 = self.multi32_3(key_2, new3_up64)
        key_2 = self.multi32_4(key_2, new4_up64)
        _c2 = self.attnc2(c2, key_2)
        _c2 = self.attnc2_1(_c2, key_2)
        _c2 = self.attnc2_2(_c2, key_2)
        if self.num_lkca_repeats >= 4:
            _c2 = self.attnc2_3(_c2, key_2)
        new2 = self.inject2(c2, _c2)

        # ---- detail branch: wavelet + COP-lite refines new2 ----
        if self.ablate_detail:
            small_logits = None
            self._cluster_logits = None
        else:
            hf_conf = self.hf_conf(c1, target_size=(h2, w2))
            detail = self.l2_sda(c2 * (1 + 0.1 * hf_conf))
            detail = self.sca_lite(detail, new3, new4)
            patch_delta = self.pca_lite(detail, new3, new4)
            detail_delta = self.detail_fuse(torch.cat(
                [detail - new2, patch_delta, hf_conf * patch_delta], dim=1))
            refine_scale = self.max_refine_scale * torch.sigmoid(self.refine_scale)
            new2 = new2 + refine_scale * hf_conf * detail_delta
            small_logits = self.small_head(detail) if not self.ablate_so_aux else None
            # cluster head sees the same detail feature as SOAH
            self._cluster_logits = (
                self.cluster_head(detail)
                if hasattr(self, 'cluster_head') else None)

        # ---- Block 4: target = c1_pool (64x64), uses refined new2 ----
        key_1 = self.multi64_5(c1_64, new2)
        key_1 = self.multi64_3(key_1, new3_up64)
        key_1 = self.multi64_4(key_1, new4_up64)
        _c1 = self.attnc1(c1_64, key_1)
        _c1 = self.attnc1_1(_c1, key_1)
        _c1 = self.attnc1_2(_c1, key_1)
        if self.num_lkca_repeats >= 4:
            _c1 = self.attnc1_3(_c1, key_1)
        new1 = self.inject1(c1_64, _c1)

        # ---- Prediction-side SFI: 4-output integration ----
        pred1 = self.predict1_2(new1, new2)
        pred2 = self.predict2_3(pred1, new3)
        pred3 = self.predict3_4(pred2, new4)

        seg_logits = self.linear_fuse(pred3)
        seg_logits = self.dropout(seg_logits)
        seg_logits = self.linear_pred(seg_logits)
        return seg_logits, small_logits

    def forward(self, inputs):
        seg_logits, _ = self._forward_logits(inputs)
        return seg_logits

    def loss(self, inputs, batch_data_samples, train_cfg=None):
        seg_logits, small_logits = self._forward_logits(inputs)
        return self.loss_by_feat((seg_logits, small_logits), batch_data_samples)

    def _small_object_mask(self, seg_label):
        small_mask = torch.zeros_like(seg_label, dtype=torch.bool)
        for cls_id in self.small_object_classes:
            small_mask |= seg_label == int(cls_id)
        valid_mask = seg_label != self.ignore_index
        return small_mask, valid_mask

    def _pat_weight(self, seg_logits, seg_label, valid_mask):
        """PAT pixel-wise adaptive weight (Do et al., PR-Letters 2024).

        alpha_{i,j} = 1 / exp((p_{i,j} - 1 + eps) / T), where p_{i,j} is the
        softmax prob of the ground-truth class at that pixel. Low-confidence
        pixels get up-weighted; high-confidence pixels keep a small but
        non-zero weight (unlike Focal), and we normalise by per-class mask
        size so large head-class regions cannot dominate the gradient.
        """
        with torch.no_grad():
            prob = F.softmax(seg_logits, dim=1)
            gt = seg_label.clone()
            gt[~valid_mask] = 0
            p_gt = prob.gather(1, gt.unsqueeze(1)).squeeze(1)
            alpha = 1.0 / torch.exp(
                (p_gt - 1.0 + self.pat_eps) / self.pat_temperature)
            alpha = alpha * valid_mask.float()

            num_classes = seg_logits.shape[1]
            gt_flat = gt.view(-1)
            mask_flat = valid_mask.view(-1).float()
            class_size = torch.zeros(
                num_classes, device=seg_logits.device, dtype=alpha.dtype)
            class_size.scatter_add_(0, gt_flat, mask_flat)
            class_size = class_size.clamp_min(1.0)
            size_per_pixel = class_size.gather(0, gt_flat).view_as(alpha)
            alpha = alpha / size_per_pixel
            denom = alpha.sum().clamp_min(1e-6)
            alpha = alpha * (valid_mask.float().sum() / denom)
        return alpha

    def loss_by_feat(self, logits, batch_data_samples):
        seg_logits, small_logits = logits
        seg_label = self._stack_batch_gt(batch_data_samples).squeeze(1)
        seg_logits = resize(
            seg_logits, size=seg_label.shape[1:], mode='bilinear',
            align_corners=self.align_corners)
        if small_logits is not None:
            small_logits = resize(
                small_logits, size=seg_label.shape[1:], mode='bilinear',
                align_corners=self.align_corners)
        cluster_logits = getattr(self, '_cluster_logits', None)
        if cluster_logits is not None:
            cluster_logits = [
                resize(cl, size=seg_label.shape[1:], mode='bilinear',
                       align_corners=self.align_corners)
                for cl in cluster_logits
            ]

        small_mask, valid_mask = self._small_object_mask(seg_label)
        if valid_mask.sum() == 0:
            zero = seg_logits.sum() * 0
            return dict(
                loss_ce=zero,
                loss_pat=zero,
                loss_so_aux=zero,
                loss_cluster=zero,
                acc_seg=accuracy(
                    seg_logits, seg_label, ignore_index=self.ignore_index))

        per_pixel_loss = F.cross_entropy(
            seg_logits, seg_label, ignore_index=self.ignore_index,
            reduction='none')
        loss_main = per_pixel_loss[valid_mask].mean()

        if self.ablate_pat:
            loss_pat = seg_logits.sum() * 0
        else:
            alpha = self._pat_weight(seg_logits, seg_label, valid_mask)
            loss_pat = (alpha * per_pixel_loss).sum() / valid_mask.float().sum()
            loss_pat = self.pat_loss_weight * loss_pat

        if small_logits is None:
            loss_aux = seg_logits.sum() * 0
        else:
            small_label = torch.where(
                valid_mask, small_mask.long(),
                torch.full_like(seg_label, self.ignore_index))
            loss_aux = self.aux_loss_weight * F.cross_entropy(
                small_logits, small_label, ignore_index=self.ignore_index)

        # ---- hierarchical cluster loss ----
        if cluster_logits is not None:
            loss_cluster = seg_logits.sum() * 0
            # reference buffers directly — avoids stale-CPU-device bug
            # from a tuple captured in __init__ under DDP
            cluster_maps = (self._cluster_map_5, self._cluster_map_4,
                            self._cluster_map_2)
            n_clusters = [m.max().item() + 1 for m in cluster_maps]
            for cl_logits, cl_map, n_cl in zip(
                    cluster_logits, cluster_maps, n_clusters):
                cl_label = cl_map[seg_label.clamp(0, cl_map.shape[0] - 1)]
                cl_label[~valid_mask] = self.ignore_index
                loss_cluster = loss_cluster + F.cross_entropy(
                    cl_logits, cl_label, ignore_index=self.ignore_index)
            loss_cluster = self.cluster_loss_weight * loss_cluster
        else:
            loss_cluster = seg_logits.sum() * 0

        return dict(
            loss_ce=loss_main,
            loss_pat=loss_pat,
            loss_so_aux=loss_aux,
            loss_cluster=loss_cluster,
            acc_seg=accuracy(
                seg_logits, seg_label, ignore_index=self.ignore_index))
