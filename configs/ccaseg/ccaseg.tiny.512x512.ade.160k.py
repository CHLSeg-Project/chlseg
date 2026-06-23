_base_ = [
    '../../_base_/models/ccaseg-tiny.py', '../../_base_/datasets/ade20k.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_160k_adamw.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size, test_cfg=dict(size_divisor=32))
randomness = dict(seed=1037607852)
checkpoint= 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth'  # noqa


model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(num_classes=150))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=160000,
    val_interval=10000
)

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(data_root='data/ade/ADEChallengeData2016'))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(data_root='data/ade/ADEChallengeData2016'))
test_dataloader = val_dataloader

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')