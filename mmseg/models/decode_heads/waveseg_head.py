import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
from ..utils import resize


class HaarWavelet2D(nn.Module):
    """Parameter-free Haar DWT/IDWT used in WaveSeg blocks."""

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


class RepSemanticBlock(nn.Module):
    """Lightweight low-frequency semantic branch (Rep-like)."""

    def __init__(self, channels, norm_cfg):
        super().__init__()
        self.conv_3x3 = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='GELU'))
        self.conv_1x1 = ConvModule(
            channels,
            channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.proj = ConvModule(
            channels,
            channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x):
        return self.proj(self.conv_3x3(x) + self.conv_1x1(x) + x)


class LinearGlobalMixer(nn.Module):
    """Linear-complexity global mixer as a practical Mamba substitute."""

    def __init__(self, channels, norm_cfg):
        super().__init__()
        self.pre = ConvModule(
            channels,
            channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='GELU'))
        self.dw_seq = nn.Conv1d(
            channels, channels, kernel_size=7, padding=3, groups=channels)
        self.out = ConvModule(
            channels,
            channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x):
        x = self.pre(x)
        b, c, h, w = x.shape
        seq = x.flatten(2)
        seq = self.dw_seq(seq)
        seq = torch.cumsum(seq, dim=-1)
        x = seq.view(b, c, h, w)
        return self.out(x)


class LocalPerceptionBlock(nn.Module):
    def __init__(self, channels, norm_cfg):
        super().__init__()
        self.dw = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='GELU'))
        self.pw = ConvModule(
            channels,
            channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x):
        return self.pw(self.dw(x))


class SpectrumDecompositionAttention(nn.Module):
    """SDA block: wavelet mixer + local perception."""

    def __init__(self, channels, norm_cfg):
        super().__init__()
        self.wavelet = HaarWavelet2D()
        self.low_branch = RepSemanticBlock(channels, norm_cfg=norm_cfg)
        self.high_branch = LinearGlobalMixer(channels * 3, norm_cfg=norm_cfg)
        self.rebalance = ConvModule(
            channels, channels, kernel_size=1, norm_cfg=norm_cfg, act_cfg=None)
        self.lpb = LocalPerceptionBlock(channels, norm_cfg=norm_cfg)
        self.ffn = ConvModule(
            channels, channels, kernel_size=1, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        ll, lh, hl, hh = self.wavelet.dwt(x)
        low = self.low_branch(ll)
        high = self.high_branch(torch.cat([lh, hl, hh], dim=1))
        lh_h, hl_h, hh_h = torch.chunk(high, 3, dim=1)
        y = self.wavelet.idwt(low, lh_h, hl_h, hh_h)
        y = self.rebalance(y) + x
        y = self.lpb(y) + y
        y = self.ffn(y) + y
        return y


class HighFrequencyPriorGuidance(nn.Module):
    """HPG from shallow detail-rich feature (practical replacement of raw RGB prior)."""

    def __init__(self, in_channels, out_channels, norm_cfg):
        super().__init__()
        self.wavelet = HaarWavelet2D()
        self.lh_conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=(1, 3),
            padding=(0, 1),
            norm_cfg=norm_cfg,
            act_cfg=dict(type='GELU'))
        self.hl_conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=(3, 1),
            padding=(1, 0),
            norm_cfg=norm_cfg,
            act_cfg=dict(type='GELU'))
        self.hh_conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='GELU'))
        self.fuse = ConvModule(
            out_channels * 3,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x, target_size):
        x = resize(
            x, size=target_size, mode='bilinear', align_corners=False)
        _, lh, hl, hh = self.wavelet.dwt(x)
        prior = torch.cat(
            [self.lh_conv(lh), self.hl_conv(hl), self.hh_conv(hh)], dim=1)
        prior = self.fuse(prior)
        prior = resize(
            prior, size=target_size, mode='bilinear', align_corners=False)
        return torch.softmax(prior, dim=1)


@MODELS.register_module()
class WaveSegHead(BaseDecodeHead):
    """WaveSeg-style decoder with HPG + SDA.

    Notes:
        The original paper learns high-frequency prior from RGB image.
        In this drop-in mmseg implementation, we use shallow feature (C1)
        as a practical substitute to keep interface compatibility.
    """

    def __init__(self, feature_strides=None, decoder_params=None, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.feature_strides = feature_strides
        self.decoder_params = decoder_params or {}
        assert len(self.in_channels) == 4, \
            'WaveSegHead expects 4 backbone stages (C1..C4).'

        c1, c2, c3, c4 = self.in_channels
        ch = self.channels
        self.interpolate_mode = 'bilinear'

        self.proj2 = ConvModule(
            c2, ch, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.proj3 = ConvModule(
            c3, ch, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.proj4 = ConvModule(
            c4, ch, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        self.sda_l2 = SpectrumDecompositionAttention(ch, norm_cfg=self.norm_cfg)
        self.hpg = HighFrequencyPriorGuidance(
            in_channels=c1, out_channels=ch, norm_cfg=self.norm_cfg)

        self.pre_fuse = ConvModule(
            ch * 3, ch, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.sda_fuse = SpectrumDecompositionAttention(ch, norm_cfg=self.norm_cfg)

        self.final_fuse = ConvModule(
            ch * 3, ch, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def forward(self, inputs):
        c1, c2, c3, c4 = self._transform_inputs(inputs)
        target_size = c2.shape[2:]

        f2 = self.sda_l2(self.proj2(c2))
        f3 = resize(
            self.proj3(c3),
            size=target_size,
            mode=self.interpolate_mode,
            align_corners=self.align_corners)
        f4 = resize(
            self.proj4(c4),
            size=target_size,
            mode=self.interpolate_mode,
            align_corners=self.align_corners)

        prior = self.hpg(c1, target_size=target_size)
        f2_en = f2 * prior + f2
        f3_en = f3 * prior + f3
        f4_en = f4 * prior + f4

        f_con = self.pre_fuse(torch.cat([f2_en, f3_en, f4_en], dim=1))
        f_attn = self.sda_fuse(f_con)

        fa2 = f_attn + f2_en
        fa3 = f_attn + f3_en
        fa4 = f_attn + f4_en
        out = self.final_fuse(torch.cat([fa2, fa3, fa4], dim=1))
        out = self.cls_seg(out)
        return out
