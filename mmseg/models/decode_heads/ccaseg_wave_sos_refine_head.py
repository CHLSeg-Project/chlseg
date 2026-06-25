import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from ..builder import MODELS
from ..losses import accuracy
from .ccaseg_small import (InjectionMultiSum, Inject, LKCA2, LKCA3,
                           PyramidPoolAgg)
from .decode_head import BaseDecodeHead
from ..utils import resize


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
    def __init__(self, channels, norm_cfg, act_cfg=dict(type='GELU')):
        super().__init__()
        self.wavelet = HaarWavelet2D()
        self.low_branch = ConvModule(
            channels, channels, 3, padding=1, groups=channels,
            norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.low_proj = ConvModule(
            channels, channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.high_pre = ConvModule(
            channels * 3, channels * 3, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.high_scan = nn.Conv1d(
            channels * 3, channels * 3, kernel_size=7, padding=3,
            groups=channels * 3)
        self.rebalance = ConvModule(
            channels, channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.local_perception = ConvModule(
            channels, channels, 3, padding=1, groups=channels,
            norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.ffn = ConvModule(
            channels, channels, 1, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        residual = x
        ll, lh, hl, hh = self.wavelet.dwt(x)
        low = self.low_proj(self.low_branch(ll) + ll)
        high = self.high_pre(torch.cat([lh, hl, hh], dim=1))
        b, c, h, w = high.shape
        high = self.high_scan(high.flatten(2)).view(b, c, h, w)
        high = torch.cumsum(high, dim=-1)
        lh, hl, hh = torch.chunk(high, 3, dim=1)
        x = self.wavelet.idwt(low, lh, hl, hh)
        x = self.rebalance(x) + residual
        x = self.local_perception(x) + x
        x = self.ffn(x) + x
        return x


class HighFrequencyPriorGuidance(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg, act_cfg=dict(type='GELU')):
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
        self.fuse = ConvModule(
            out_channels * 3, out_channels, 1, norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x, target_size):
        x = resize(x, size=target_size, mode='bilinear', align_corners=False)
        _, lh, hl, hh = self.wavelet.dwt(x)
        x_h = torch.cat(
            [self.lh_conv(lh), self.hl_conv(hl), self.hh_conv(hh)], dim=1)
        x_h = self.fuse(x_h)
        x_h = resize(x_h, size=target_size, mode='bilinear', align_corners=False)
        return torch.softmax(x_h, dim=1)


class SmallObjectAuxHead(nn.Module):
    """Lightweight 2-class top head: non-small vs small object."""

    def __init__(self, channels, norm_cfg, act_cfg=dict(type='GELU')):
        super().__init__()
        self.block = nn.Sequential(
            ConvModule(
                channels, channels, 3, padding=1, groups=channels,
                norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(channels, channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            nn.Conv2d(channels, 2, kernel_size=1))

    def forward(self, x):
        return self.block(x)


@MODELS.register_module()
class CCAWaveSOSRefineHead_small(BaseDecodeHead):
    """CCASeg-small + WaveSeg(L2) + SOSNet-style small-object training."""

    def __init__(self,
                 feature_strides,
                 small_object_classes=(12, 20, 22, 36, 43, 47, 58, 61, 66,
                                       71, 74, 76, 80, 83, 87, 89, 90, 93,
                                       98),
                 aux_loss_weight=0.2,
                 hier_loss_weight=0.1,
                 soem_ratio=0.2,
                 soem_threshold=0.5,
                 use_soem=True,
                 enable_small_gate=True,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.small_object_classes = tuple(small_object_classes)
        self.aux_loss_weight = aux_loss_weight
        self.hier_loss_weight = hier_loss_weight
        self.soem_ratio = soem_ratio
        self.soem_threshold = soem_threshold
        self.use_soem = use_soem
        self.enable_small_gate = enable_small_gate

        c1, c2, c3, c4 = self.in_channels
        embedding_dim = kwargs['decoder_params']['embed_dim']

        self.hpg = HighFrequencyPriorGuidance(c1, c2, norm_cfg=self.norm_cfg)
        self.l2_sda = L2SpectrumDecompositionAttention(c2, norm_cfg=self.norm_cfg)
        self.small_head = SmallObjectAuxHead(c2, norm_cfg=self.norm_cfg)
        self.hpg_scale = nn.Parameter(torch.tensor(0.1))
        self.refine_scale = nn.Parameter(torch.tensor(0.1))
        self.refine_gate = nn.Sequential(
            ConvModule(
                c2 * 3, c2, 3, padding=1, norm_cfg=self.norm_cfg,
                act_cfg=dict(type='GELU')),
            nn.Conv2d(c2, c2, kernel_size=1),
            nn.Sigmoid())

        self.down2_1 = PyramidPoolAgg(stride=2, dim=c2)
        self.down2_2 = PyramidPoolAgg(stride=4, dim=c2)
        self.down3_1 = PyramidPoolAgg(stride=2, dim=c3)

        self.attnc4 = LKCA2(channels1=c4, channels2=c4)
        self.attnc3 = LKCA3(channels1=c3, channels2=c4)
        self.attnc2 = LKCA3(channels1=c2, channels2=c4)
        self.attnc4_1 = LKCA2(channels1=c4, channels2=c4)
        self.attnc3_1 = LKCA3(channels1=c3, channels2=c4)
        self.attnc2_1 = LKCA3(channels1=c2, channels2=c4)
        self.attnc4_2 = LKCA2(channels1=c4, channels2=c4)
        self.attnc3_2 = LKCA3(channels1=c3, channels2=c4)
        self.attnc2_2 = LKCA3(channels1=c2, channels2=c4)

        self.inject2 = InjectionMultiSum(dim=c2)
        self.inject3 = InjectionMultiSum(dim=c3)
        self.inject4 = InjectionMultiSum(dim=c4)

        self.multi8_3 = Inject(dim1=c2, dim2=c3)
        self.multi8_4 = Inject(dim1=c3, dim2=c4)
        self.multi16_3 = Inject(dim1=c2, dim2=c3)
        self.multi16_4 = Inject(dim1=c3, dim2=c4)
        self.multi32_3 = Inject(dim1=c2, dim2=c3)
        self.multi32_4 = Inject(dim1=c3, dim2=c4)
        self.predict2_3 = Inject(dim1=c2, dim2=c3)
        self.predict3_4 = Inject(dim1=c3, dim2=c4)

        self.linear_fuse = ConvModule(
            in_channels=c4, out_channels=embedding_dim, kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True))
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def _forward_logits(self, inputs):
        c1, c2, c3, c4 = self._transform_inputs(inputs)
        _, _, h2, w2 = c2.shape

        c2_1 = self.down2_1(c2)
        _, _, h2_1, w2_1 = c2_1.shape
        c2_2 = self.down2_2(c2)
        c3_1 = self.down3_1(c3)

        key_4 = self.multi8_3(c2_2, c3_1)
        key_4 = self.multi8_4(key_4, c4)
        _c4 = self.attnc4(c4, key_4)
        _c4 = self.attnc4_1(_c4, key_4)
        _c4 = self.attnc4_2(_c4, key_4)
        new4 = self.inject4(c4, _c4)

        new4_up32 = resize(new4, size=(h2_1, w2_1), mode='bilinear', align_corners=False)
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
        small_logits = self.small_head(c2_detail)
        small_prob = torch.softmax(small_logits, dim=1)[:, 1:2]
        gate = self.refine_gate(torch.cat([new2, c2_detail, hpg], dim=1))
        if self.enable_small_gate:
            gate = gate * small_prob
        new2 = new2 + self.refine_scale * gate * (c2_detail - new2)

        pred2 = self.predict2_3(new2, new3)
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

    def _soem_loss(self, per_pixel_loss, small_mask, valid_mask):
        if not self.use_soem:
            return per_pixel_loss[valid_mask].mean()

        losses = []
        n_min = max(int(valid_mask.sum().item() * self.soem_ratio), 1)
        for mask in (small_mask & valid_mask, (~small_mask) & valid_mask):
            loss_part = per_pixel_loss[mask]
            if loss_part.numel() == 0:
                continue
            hard = loss_part[loss_part > self.soem_threshold]
            if hard.numel() < n_min:
                k = min(n_min, loss_part.numel())
                hard = loss_part.topk(k)[0]
            losses.append(hard)
        if not losses:
            return per_pixel_loss[valid_mask].mean()
        return torch.cat(losses).mean()

    def loss_by_feat(self, logits, batch_data_samples):
        seg_logits, small_logits = logits
        seg_label = self._stack_batch_gt(batch_data_samples).squeeze(1)
        seg_logits = resize(
            seg_logits, size=seg_label.shape[1:], mode='bilinear',
            align_corners=self.align_corners)
        small_logits = resize(
            small_logits, size=seg_label.shape[1:], mode='bilinear',
            align_corners=self.align_corners)

        small_mask, valid_mask = self._small_object_mask(seg_label)
        small_label = small_mask.long()
        small_label = torch.where(valid_mask, small_label,
                                  torch.full_like(small_label, self.ignore_index))

        loss_main = F.cross_entropy(
            seg_logits, seg_label, ignore_index=self.ignore_index,
            reduction='none')
        loss_main = self._soem_loss(loss_main, small_mask, valid_mask)

        loss_aux = F.cross_entropy(
            small_logits, small_label, ignore_index=self.ignore_index)

        seg_prob = torch.softmax(seg_logits, dim=1)
        small_prob_from_seg = torch.zeros(
            (seg_logits.shape[0], 2, seg_logits.shape[2], seg_logits.shape[3]),
            device=seg_logits.device, dtype=seg_logits.dtype)
        small_class_mask = seg_prob.new_zeros((self.num_classes,))
        for cls_id in self.small_object_classes:
            if 0 <= int(cls_id) < self.num_classes:
                small_class_mask[int(cls_id)] = 1
        small_prob = (seg_prob * small_class_mask.view(1, -1, 1, 1)).sum(dim=1)
        small_prob_from_seg[:, 1] = small_prob
        small_prob_from_seg[:, 0] = 1 - small_prob
        top_prob = torch.softmax(small_logits, dim=1)
        loss_hier_map = (top_prob - small_prob_from_seg.detach()).pow(2).sum(dim=1)
        loss_hier = loss_hier_map[valid_mask].mean()

        losses = dict()
        losses['loss_ce'] = loss_main
        losses['loss_so_aux'] = self.aux_loss_weight * loss_aux
        losses['loss_hier'] = self.hier_loss_weight * loss_hier
        losses['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        return losses
