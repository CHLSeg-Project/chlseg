# A2: v0 minus small-object auxiliary head. small_head is not constructed,
# small_logits is None, loss_so_aux is zeroed. Detail branch (HF/L2-SDA/COP)
# is preserved -- new2 still benefits from the wavelet refinement.
_base_ = './mscan-t_1xb16-adamw-160k_ade20k-512x512_chlseg.py'

work_dir = 'work_dirs/chlseg_a2_no_aux'

model = dict(decode_head=dict(ablate_so_aux=True))
