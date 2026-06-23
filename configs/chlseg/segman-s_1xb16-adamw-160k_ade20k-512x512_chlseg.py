# CHLSegHead_small paired with SegMAN-S encoder on ADE20K 160k.
#
# Backbone widths follow the SegMAN paper: embed_dims=[64, 144, 288, 512],
# which makes feature_strides=[4, 8, 16, 32] identical to MSCAN-S. The decode
# head's in_channels just need to match those widths -- CHLSegHead_small
# derives every internal module size from self.in_channels at runtime, so the
# head needs no further changes.
#
# Pretrained backbone weights: set `backbone.pretrained` below to the path of
# SegMAN_Encoder_s.pth.tar on your training box (the SegMAN authors release
# this checkpoint at https://github.com/yunxiangfu2001/SegMAN).
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py',
    '../_base_/datasets/ade20k.py',
]

work_dir = 'work_dirs/chlseg_segman_s'

crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))

norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='SegMANEncoder_s',
        pretrained='/pretrained/segman_s_ade.pth',
        # Gradient checkpointing on the deep stage-3 (depth=10, embed=288)
        # which dominates activation memory. Recomputes activations in the
        # backward pass -- pure compute trade, no numerical change.
        use_checkpoint=[0, 0, 10, 0],
    ),
    decode_head=dict(
        type='CHLSegHead_small',
        in_channels=[64, 144, 288, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=512),
        small_object_classes=[
            12, 20, 22, 36, 43, 47, 58, 61, 66, 71,
            74, 76, 80, 83, 87, 89, 90, 93, 98,
        ],
        aux_loss_weight=0.1,
        pat_loss_weight=0.15,
        pat_temperature=2.0,
        pat_eps=0.1,
        max_refine_scale=0.15,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# Effective batch size = 8 * 2 = 16, mathematically equivalent to the
# original batch_size=16 single-step training. Lets us cut per-step memory
# nearly in half.
train_dataloader = dict(batch_size=8)

# AMP DISABLED. Earlier attempt used AmpOptimWrapper, and training NaN-ed
# during warmup around iter 1150-1500: loss_ce and loss_pat went NaN first
# (PAT's softmax/log path overflows under FP16), then so_aux followed once
# the encoder weights were corrupted. SegMAN's selective_scan and NATTEN are
# `@custom_fwd` FP32 internally, but the outer FP16 grads still poisoned the
# AdamW state with this PyTorch 2.0.0 + lr_mult=10 head config. Going back
# to FP32 + gradient checkpointing + grad accumulation -- mathematically
# identical to the original training, just slower and lower-memory.
#
# SegMAN paper uses AdamW lr=6e-5 with weight decay 0.01 and a 1500-step
# warmup; segnext-style paramwise rules (no decay on pos_block/norm, 10x lr
# for the head) give the head room to adapt to the new backbone.
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    accumulative_counts=2,
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.),
        }))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', power=1.0, begin=1500, end=160000, eta_min=0.0,
         by_epoch=False),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=10000)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=16000))
