_base_ = '../segnext/segnext_mscan-t_1xb16-adamw-160k_ade20k-512x512.py'

model = dict(
    decode_head=dict(
        _delete_=True,
        type='CHLSegHead_tiny',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=256,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        small_object_classes=[
            12, 20, 22, 36, 43, 47, 58, 61, 66, 71,
            74, 76, 80, 83, 87, 89, 90, 93, 98
        ],
        aux_loss_weight=0.1,
        pat_loss_weight=0.15,
        pat_temperature=2.0,
        pat_eps=0.1,
        max_refine_scale=0.15,
        # ---- ablation: 4× LKCA instead of default 3× ----
        num_lkca_repeats=4,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))


param_scheduler = [
    dict(begin=0, by_epoch=False, end=1500, start_factor=1e-06, type='LinearLR'),
    dict(begin=1500, by_epoch=False, end=160000, eta_min=0.0, power=1.0, type='PolyLR'),
]
train_cfg = dict(max_iters=160000, type='IterBasedTrainLoop', val_interval=10000)
default_hooks = dict(checkpoint=dict(by_epoch=False, interval=16000, type='CheckpointHook'))