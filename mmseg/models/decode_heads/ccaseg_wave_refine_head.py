import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import MODELS
from .ccaseg_small import (InjectionMultiSum, Inject, LKCA2, LKCA3,
                           PyramidPoolAgg)
from .decode_head import BaseDecodeHead
from ..utils import resize


class HaarWavelet2D(nn.Module):
    """Parameter-free Haar wavelet transform used by the L2 detail branch."""

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
    """WaveSeg-inspired SDA used only on the L2/C2 feature."""

    def __init__(self, channels, norm_cfg, act_cfg=dict(type='GELU')):
        super().__init__()
        self.wavelet = HaarWavelet2D()
        self.low_branch = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.low_proj = ConvModule(
            channels,
            channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.high_mixer = nn.Sequential(
            ConvModule(
                channels * 3,
                channels * 3,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            nn.Conv1d(
                channels * 3,
                channels * 3,
                kernel_size=7,
                padding=3,
                groups=channels * 3),
        )
        self.rebalance = ConvModule(
            channels, channels, kernel_size=1, norm_cfg=norm_cfg, act_cfg=None)
        self.local_perception = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.ffn = ConvModule(
            channels, channels, kernel_size=1, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        residual = x
        ll, lh, hl, hh = self.wavelet.dwt(x)

        low = self.low_proj(self.low_branch(ll) + ll)
        high = torch.cat([lh, hl, hh], dim=1)
        b, c, h, w = high.shape
        high = self.high_mixer[0](high)
        high = self.high_mixer[1](high.flatten(2)).view(b, c, h, w)
        high = torch.cumsum(high, dim=-1)
        lh, hl, hh = torch.chunk(high, 3, dim=1)

        x = self.wavelet.idwt(low, lh, hl, hh)
        x = self.rebalance(x) + residual
        x = self.local_perception(x) + x
        x = self.ffn(x) + x
        return x


class HighFrequencyPriorGuidance(nn.Module):
    """Learn high-frequency guidance from C1 and align it to C2."""

    def __init__(self, in_channels, out_channels, norm_cfg, act_cfg=dict(type='GELU')):
        super().__init__()
        self.wavelet = HaarWavelet2D()
        self.lh_conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=(1, 3),
            padding=(0, 1),
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.hl_conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=(3, 1),
            padding=(1, 0),
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.hh_conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.fuse = ConvModule(
            out_channels * 3,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x, target_size):
        x = resize(x, size=target_size, mode='bilinear', align_corners=False)
        _, lh, hl, hh = self.wavelet.dwt(x)
        x_h = torch.cat(
            [self.lh_conv(lh), self.hl_conv(hl), self.hh_conv(hh)], dim=1)
        x_h = self.fuse(x_h)
        x_h = resize(x_h, size=target_size, mode='bilinear', align_corners=False)
        return torch.softmax(x_h, dim=1)


@MODELS.register_module()
class CCAWaveRefineHead_small(BaseDecodeHead):
    """CCASeg-small main decoder plus WaveSeg L2 detail refinement."""

    def __init__(self, feature_strides, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.hpg = HighFrequencyPriorGuidance(
            c1_in_channels, c2_in_channels, norm_cfg=self.norm_cfg)
        self.l2_sda = L2SpectrumDecompositionAttention(
            c2_in_channels, norm_cfg=self.norm_cfg)
        self.hpg_scale = nn.Parameter(torch.tensor(0.1))
        self.refine_scale = nn.Parameter(torch.tensor(0.1))
        self.refine_gate = nn.Sequential(
            ConvModule(
                c2_in_channels * 3,
                c2_in_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=dict(type='GELU')),
            nn.Conv2d(c2_in_channels, c2_in_channels, kernel_size=1),
            nn.Sigmoid())

        self.down2_1 = PyramidPoolAgg(stride=2, dim=c2_in_channels)
        self.down2_2 = PyramidPoolAgg(stride=4, dim=c2_in_channels)
        self.down3_1 = PyramidPoolAgg(stride=2, dim=c3_in_channels)

        self.attnc4 = LKCA2(channels1=c4_in_channels, channels2=c4_in_channels)
        self.attnc3 = LKCA3(channels1=c3_in_channels, channels2=c4_in_channels)
        self.attnc2 = LKCA3(channels1=c2_in_channels, channels2=c4_in_channels)
        self.attnc4_1 = LKCA2(channels1=c4_in_channels, channels2=c4_in_channels)
        self.attnc3_1 = LKCA3(channels1=c3_in_channels, channels2=c4_in_channels)
        self.attnc2_1 = LKCA3(channels1=c2_in_channels, channels2=c4_in_channels)
        self.attnc4_2 = LKCA2(channels1=c4_in_channels, channels2=c4_in_channels)
        self.attnc3_2 = LKCA3(channels1=c3_in_channels, channels2=c4_in_channels)
        self.attnc2_2 = LKCA3(channels1=c2_in_channels, channels2=c4_in_channels)

        self.inject2 = InjectionMultiSum(dim=c2_in_channels)
        self.inject3 = InjectionMultiSum(dim=c3_in_channels)
        self.inject4 = InjectionMultiSum(dim=c4_in_channels)

        self.multi8_3 = Inject(dim1=c2_in_channels, dim2=c3_in_channels)
        self.multi8_4 = Inject(dim1=c3_in_channels, dim2=c4_in_channels)
        self.multi16_3 = Inject(dim1=c2_in_channels, dim2=c3_in_channels)
        self.multi16_4 = Inject(dim1=c3_in_channels, dim2=c4_in_channels)
        self.multi32_3 = Inject(dim1=c2_in_channels, dim2=c3_in_channels)
        self.multi32_4 = Inject(dim1=c3_in_channels, dim2=c4_in_channels)
        self.predict2_3 = Inject(dim1=c2_in_channels, dim2=c3_in_channels)
        self.predict3_4 = Inject(dim1=c3_in_channels, dim2=c4_in_channels)

        self.linear_fuse = ConvModule(
            in_channels=c4_in_channels,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True))
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        c1, c2, c3, c4 = self._transform_inputs(inputs)
        _, _, h2, w2 = c2.shape

        c2_1 = self.down2_1(c2)
        _, _, c2_1_h1, c2_1_w1 = c2_1.shape
        c2_2 = self.down2_2(c2)
        c3_1 = self.down3_1(c3)

        key_4 = self.multi8_3(c2_2, c3_1)
        key_4 = self.multi8_4(key_4, c4)
        _c4 = self.attnc4(c4, key_4)
        _c4 = self.attnc4_1(_c4, key_4)
        _c4 = self.attnc4_2(_c4, key_4)
        new4 = self.inject4(c4, _c4)

        new4_up32 = resize(
            new4, size=(c2_1_h1, c2_1_w1), mode='bilinear', align_corners=False)
        key_3 = self.multi16_3(c2_1, c3)
        key_3 = self.multi16_4(key_3, new4_up32)
        _c3 = self.attnc3(c3, key_3)
        _c3 = self.attnc3_1(_c3, key_3)
        _c3 = self.attnc3_2(_c3, key_3)
        new3 = self.inject3(c3, _c3)

        new3_up64 = resize(new3, size=(h2, w2), mode='bilinear', align_corners=False)
        new4_up64 = resize(new4, size=(h2, w2), mode='bilinear', align_corners=False)
        key_2 = self.multi32_3(c2, new3_up64)
        key_2 = self.multi32_4(key_2, new4_up64)
        _c2 = self.attnc2(c2, key_2)
        _c2 = self.attnc2_1(_c2, key_2)
        _c2 = self.attnc2_2(_c2, key_2)
        new2 = self.inject2(c2, _c2)

        hpg = self.hpg(c1, target_size=(h2, w2))
        c2_detail = self.l2_sda(c2 * (1 + self.hpg_scale * hpg))
        gate = self.refine_gate(torch.cat([new2, c2_detail, hpg], dim=1))
        new2 = new2 + self.refine_scale * gate * (c2_detail - new2)

        pred2 = self.predict2_3(new2, new3)
        pred3 = self.predict3_4(pred2, new4)
        x = self.linear_fuse(pred3)
        x = self.dropout(x)
        x = self.linear_pred(x)
        return x
