# CHLSeg decoder on MiT-B0 backbone (SegFormer encoder + CHLSegHead_tiny)
# for ADE20K 160k iters.  The MiT-B0 backbone matches MSCAN-T channel widths
# ([32, 64, 160, 256]), so CHLSegHead_tiny can be plugged in directly.
_base_ = '../segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py'

work_dir = 'work_dirs/chlseg_mitb0'

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
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))
