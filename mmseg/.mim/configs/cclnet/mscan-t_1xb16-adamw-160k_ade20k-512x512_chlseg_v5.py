# v5 = v0 + PyTorch-only global context injection (MMSCopE proxy).
# The model delta is fully inside CHLSegHead_tiny (GlobalContextScan,
# beta-gated). Config mirrors v0 and only overrides work_dir so v5's
# checkpoints stay separate from v0's.
_base_ = './mscan-t_1xb16-adamw-160k_ade20k-512x512_chlseg.py'

work_dir = 'work_dirs/chlseg_v5'
