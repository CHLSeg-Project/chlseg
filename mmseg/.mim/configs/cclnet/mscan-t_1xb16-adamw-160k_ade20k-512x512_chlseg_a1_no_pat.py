# A1: v0 minus PAT loss. All modules built as in v0; only loss_pat is zeroed.
_base_ = './mscan-t_1xb16-adamw-160k_ade20k-512x512_chlseg.py'

work_dir = 'work_dirs/chlseg_a1_no_pat'

model = dict(decode_head=dict(ablate_pat=True))
