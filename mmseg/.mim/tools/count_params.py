"""CHLSeg-T vs CHLSeg-S: parameter & FLOPs estimation at 512x512."""
import sys

def c1x1p(c_in, c_out, bias=True):
    p = c_in * c_out
    if bias:
        p += c_out
    return p

def dwp(c, k, bias=True):
    p = c * (k if isinstance(k, int) else k[0]*k[1])
    if bias:
        p += c
    return p

def bnp(c):
    return 2 * c

def cmp(in_ch, out_ch, k, groups=1, bn=True):
    p = 0
    if isinstance(k, int):
        p += c1x1p(in_ch, out_ch) if k == 1 else dwp(out_ch, k) if groups > 1 else in_ch*out_ch*k*k
    else:
        p += in_ch * out_ch * k[0] * k[1]
    if bn:
        p += bnp(out_ch)
    return p

def inject_p(d1, d2):
    return cmp(d1, d2, 1) + cmp(d2, d2, 1)

def ims_p(d):
    return cmp(d, d, 1) + cmp(d, d, 1)

def lkca_p(c1, c2, ks):
    """ks = [k0, [_, k1], [_, k2], [_, k3]]"""
    p = bnp(c1) + bnp(c2)
    p += c1x1p(c1, c1) + c1x1p(c2, c1)  # proj1, proj2
    p += dwp(c1, ks[0]) * 2  # conv1, conv2 (DW)
    # strip branches: 3 branches x 4 DWconvs each (2 Q + 2 K)
    for kk in [ks[1][1], ks[2][1], ks[3][1]]:
        p += dwp(c1, kk) * 4  # 1xk for Q+K (2) + for the other
    for kk in [ks[1][1], ks[2][1], ks[3][1]]:
        p += dwp(c1, kk) * 4  # kx1 for Q+K (2) + for the other
    p += c1x1p(c1, c1) * 2  # conv3, conv3b
    p += bnp(c1) + c1x1p(c1, c1) + dwp(c1, 3) + c1x1p(c1, c1)  # MLP
    return p

# ---------- FLOPs ----------
def c1x1f(c_in, c_out, h, w):
    return 2.0 * c_in * c_out * h * w

def dwf(c, k, h, w):
    return 2.0 * c * (k * k if isinstance(k, int) else k[0] * k[1]) * h * w

def lkca_f(c1, c2, ks, hw):
    h, w = hw
    f = c1x1f(c1, c1, h, w) + c1x1f(c2, c1, h, w)  # proj1,2
    f += dwf(c1, ks[0], h, w) * 2                    # conv1,2
    for kk in [ks[1][1], ks[2][1], ks[3][1]]:
        f += dwf(c1, kk, h, w) * 8                   # all strip convs
    f += c1x1f(c1, c1, h, w) * 2                     # conv3,3b
    f += c1x1f(c1, c1, h, w) + dwf(c1, 3, h, w) + c1x1f(c1, c1, h, w)  # MLP
    return f

def estimate(label, c1, c2, c3, c4, ed, bb_p, bb_f):
    # ---- Params ----
    total = 0
    for d in [c2, c2, c3, c1, c1, c1]:
        total += cmp(d, d, 1, bn=False)

    total += cmp(c1, c2, (1, 3)) + cmp(c1, c2, (3, 1)) + cmp(c1, c2, 3)  # HFP dw
    total += cmp(c2 * 3, c2, 1) + c1x1p(c2, c2)  # HFP fuse

    total += cmp(c2, c2, 3, c2) + cmp(c2, c2, 1)  # WSDA low
    total += cmp(c2 * 3, c2 * 3, 1)  # WSDA high_pre
    total += dwp(c2 * 3, 7)  # WSDA scan
    total += cmp(c2, c2, 1)  # WSDA rebalance

    hid = max(c2 // 4, 16)
    total += cmp(c3, c2, 1) + cmp(c4, c2, 1)  # CS-CA proj
    total += c1x1p(c2 * 3, hid) + c1x1p(hid, c2) + cmp(c2 * 3, c2, 1)

    total += cmp(c3, c2, 1) + cmp(c4, c2, 1)  # OP-CS proj
    total += dwp(c2, 3) + dwp(c2, 5)  # OP-CS local
    total += cmp(c2 * 3, c2, 1) + c1x1p(c2, c2) + cmp(c2 * 3, c2, 3)

    total += cmp(c2 * 3, c2, 1)  # detail_fuse
    total += 1  # refine_scale

    total += cmp(c2, c2, 3, c2) + cmp(c2, c2, 1) + c1x1p(c2, 2)  # small_head
    total += cmp(c2, c2, 3, c2) + cmp(c2, c2, 1) + c1x1p(c2, 5) + c1x1p(c2, 4) + c1x1p(c2, 2)

    total += (inject_p(c1, c2) + inject_p(c2, c3) + inject_p(c3, c4)) * 4  # CCA
    total += inject_p(c1, c2) + inject_p(c2, c3) + inject_p(c3, c4)  # pred
    total += ims_p(c1) + ims_p(c2) + ims_p(c3) + ims_p(c4)

    lk2 = lkca_p(c4, c4, [3, [1, 5], [1, 9], [1, 13]])
    lk3_ks = [5, [1, 7], [1, 11], [1, 21]]
    lk = (lk2 + lkca_p(c3, c4, lk3_ks) + lkca_p(c2, c4, lk3_ks) + lkca_p(c1, c4, lk3_ks)) * 3
    total += lk
    total += cmp(c4, ed, 1) + c1x1p(ed, 150)

    # ---- FLOPs ----
    f = 0.0
    f += lkca_f(c4, c4, [3, [1, 5], [1, 9], [1, 13]], (16, 16)) * 3
    f += lkca_f(c3, c4, lk3_ks, (32, 32)) * 3
    f += lkca_f(c2, c4, lk3_ks, (64, 64)) * 3
    f += lkca_f(c1, c4, lk3_ks, (64, 64)) * 3
    f += dwf(c2, 3, 64, 64) + dwf(c2 * 3, 7, 1, 4096)  # WSDA
    f += c1x1f(c2, c2, 64, 64) * 15  # detail + SFI (approximate sum)
    f += c1x1f(c3, c3, 32, 32) * 6
    f += c1x1f(c4, c4, 16, 16) * 6
    f += c1x1f(c4, ed, 16, 16) + c1x1f(ed, 150, 16, 16)

    dec_g = f / 1e9
    tot_p = total + bb_p
    tot_g = dec_g + bb_f
    print(f"{label:12s}: decoder {total/1e6:5.1f}M params + backbone {bb_p/1e6:.1f}M = {tot_p/1e6:.1f}M")
    print(f"{'':12s}  decoder {dec_g:5.2f} GFLOPs + backbone {bb_f:.1f}G = {tot_g:.1f} GFLOPs")
    print()

print("=" * 60)
print("  CHLSeg Parameter & FLOPs (@ 512x512)")
print("=" * 60)
estimate("CHLSeg-T",  32, 64,  160, 256, 256, 4.6e6, 3.4)
estimate("CHLSeg-S",  64, 128, 320, 512, 512, 13.9e6, 9.3)
