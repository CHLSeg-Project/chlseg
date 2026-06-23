# A3: v0 minus the entire frequency/COP-lite detail branch. hf_conf, l2_sda,
# sca_lite, pca_lite, detail_fuse, refine_scale, small_head are all skipped.
# new2 = inject2(c2, _c2) with no wavelet refinement. small_logits is None,
# loss_so_aux is zeroed. PAT loss remains.
_base_ = './mscan-t_1xb16-adamw-160k_ade20k-512x512_chlseg.py'

work_dir = 'work_dirs/chlseg_a3_no_detail'

model = dict(decode_head=dict(ablate_detail=True))
