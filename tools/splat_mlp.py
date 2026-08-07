"""
NeuralGS-style chunk fitting for a 3DGS .ply.

Reuses BatchedLinear (batch dim = chunks; --num-chunks 1 for fast architecture
search), the Morton spatial sort, and the plyfile reader (which mmaps the file).
Adds NeRF positional encoding, per-attribute [-1,1] normalization, configurable
architecture, and a compressed-size projection.

A centered Morton region is cached to a single .npy (--cache) so that sweeping
parameters never re-reads / re-sorts the full 5.96 GB file.

ROCm torch via uv (no pyproject needed):
  uv run --with torch --with numpy --with plyfile \
      --index https://download.pytorch.org/whl/rocm7.2 \
      tools/splat_mlp.py poland.ply --cache poland_mid.npy \
      --pe-freqs 6 --hidden 64 --layers 3 --activation relu --chunk-size 8192

Iterate on:
  A) chunking/size : --chunk-size --num-chunks --offset --no-morton
  B) PE            : --pe-freqs --no-identity
  C) weight format : --weight-bytes {1,2,4}
  D) size/layers   : --hidden --layers
  E) activation    : --activation --out-act {tanh,none}
"""
import argparse
import math
import os

import numpy as np
import torch
from plyfile import PlyData
from torch import nn


# ---------------------------------------------------------------- model --------
class BatchedLinear(nn.Module):                       # unchanged
    def __init__(self, num_chunks, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_chunks, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(num_chunks, 1, out_dim))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x):                             # x: (num_chunks, chunk_size, in_dim)
        return torch.bmm(x, self.weight) + self.bias


class PositionalEncoding(nn.Module):
    """NeRF PE on chunk-local [0,1]^3. Weight-free; operates on (...,3)."""
    def __init__(self, freqs, identity=True):
        super().__init__()
        self.freqs, self.identity = freqs, identity
        self.register_buffer("bands", math.pi * (2.0 ** torch.arange(freqs)))

    @property
    def out_dim(self):
        return (3 if self.identity else 0) + 2 * 3 * self.freqs

    def forward(self, x):                             # x in [0,1]
        xb = x[..., None] * self.bands               # (...,3,L)
        pe = torch.cat([xb.sin(), xb.cos()], -1).reshape(*x.shape[:-1], 2 * 3 * self.freqs)
        return torch.cat([x, pe], -1) if self.identity else pe


_ACTS = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "silu": nn.SiLU,
         "tanh": nn.Tanh, "gelu": nn.GELU, "hardswish": nn.Hardswish, "elu": nn.ELU}


class Mlp(nn.Module):
    """PE -> (Linear -> act) x `layers` -> Linear -> out_act."""
    def __init__(self, num_chunks, out_dim, pe_freqs, hidden, layers,
                 activation, out_act, identity=True):
        super().__init__()
        self.pe = PositionalEncoding(pe_freqs, identity)
        dims = [self.pe.out_dim] + [hidden] * layers + [out_dim]
        mods = []
        for i in range(len(dims) - 1):
            mods.append(BatchedLinear(num_chunks, dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                mods.append(_ACTS[activation]())
        self.net = nn.Sequential(*mods)
        self.out_act = nn.Tanh() if out_act == "tanh" else nn.Identity()

    def forward(self, x):                             # x: (C,N,3) chunk-local [0,1]
        return self.out_act(self.net(self.pe(x)))


def per_chunk_params(pe_out_dim, hidden, layers, out_dim):
    dims = [pe_out_dim] + [hidden] * layers + [out_dim]
    return sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))


def quat_to_rotmat(q):
    """q: (...,4) in (x,y,z,w) -> R (...,3,3) with columns = rotatedBasis(q,i).
    Matches shaders/splats/splats.slang. Assumes |q|=1 (normalize first)."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    c0 = torch.stack([1 - 2*(y*y+z*z), 2*(x*y+z*w),   2*(x*z-y*w)], -1)
    c1 = torch.stack([2*(x*y-z*w),     1 - 2*(x*x+z*z), 2*(y*z+x*w)], -1)
    c2 = torch.stack([2*(x*z+y*w),     2*(y*z-x*w),     1 - 2*(x*x+y*y)], -1)
    return torch.stack([c0, c1, c2], -1)


def shape_sigma(q_xyzw, scale_log):
    """Σ = R(q) diag(s^2) R(q)^T with s = exp(scale_log) -- the exact 3D Gaussian
    covariance that computeCov2D projects to 2D. Invariant to q/-q, 180-degree
    flips, and equal-scale axis swaps, so it is the correct 'same shape' quantity."""
    qn = q_xyzw / (q_xyzw.norm(dim=-1, keepdim=True) + 1e-8)
    R = quat_to_rotmat(qn)
    s2 = torch.exp(2.0 * scale_log.clamp(-20.0, 8.0))   # guard overflow; covers real range
    return R @ torch.diag_embed(s2) @ R.transpose(-1, -2)


# ---------------------------------------------------------------- morton -------
def _split_by_3(x):
    x = x & np.uint64(0x1fffff)
    x = (x | (x << 32)) & np.uint64(0x1f00000000ffff)
    x = (x | (x << 16)) & np.uint64(0x1f0000ff0000ff)
    x = (x | (x << 8)) & np.uint64(0x100f00f00f00f00f)
    x = (x | (x << 4)) & np.uint64(0x10c30c30c30c30c3)
    x = (x | (x << 2)) & np.uint64(0x1249249249249249)
    return x


def morton3d(xyz):
    q = np.clip(xyz * ((1 << 21) - 1), 0, (1 << 21) - 1).astype(np.uint64)
    return _split_by_3(q[:, 0]) | (_split_by_3(q[:, 1]) << 1) | (_split_by_3(q[:, 2]) << 2)


# ---------------------------------------------------------------- data ---------
ATTRS = ("f_dc_0", "f_dc_1", "f_dc_2", "opacity",
         "scale_0", "scale_1", "scale_2",
         "rot_0", "rot_1", "rot_2", "rot_3")


def prepare_region(ply, attrs, region_size, cache):
    """Centered Morton window of `region_size` points; cached as one .npy.

    Stored layout: float32 array (region_size, 3 + len(attrs)) = [xyz | attr].
    Memory-careful: Morton-sorts from xyz only, then gathers attributes just for
    the selected window (so the 106M-row attribute blob is never materialized)."""
    want_cols = 3 + len(attrs)
    if cache and os.path.exists(cache):
        reg = np.load(cache, mmap_mode="r")
        if reg.shape[0] >= region_size and reg.shape[1] == want_cols:
            print(f"[cache] loaded {cache} {reg.shape}", flush=True)
            return np.array(reg[:region_size])
        print(f"[cache] {cache} stale ({reg.shape}); re-extracting", flush=True)

    print(f"[ply] reading {ply} and Morton-sorting a {region_size:,}-pt region ...", flush=True)
    v = PlyData.read(ply)["vertex"]
    n = v.count
    xyz = np.stack([np.asarray(v[c]) for c in "xyz"], 1).astype(np.float32)
    mn, mx = xyz.min(0), xyz.max(0)
    order = np.argsort(morton3d((xyz - mn) / np.maximum(mx - mn, 1e-6)), kind="stable")
    off = max(0, (n - region_size) // 2)
    win = order[off:off + region_size]
    xyz_win = xyz[win]
    attr_win = np.stack([np.asarray(v[c])[win] for c in attrs], 1).astype(np.float32)
    del xyz, order
    reg = np.concatenate([xyz_win, attr_win], axis=1).astype(np.float32)
    if cache:
        np.save(cache, reg)
        print(f"[cache] saved {cache} {reg.shape}", flush=True)
    return reg


# ---------------------------------------------------------------- main ---------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("--cache", default=None, help="path to centered-region .npy")
    ap.add_argument("--region-size", type=int, default=262144)
    ap.add_argument("--attrs", default=",".join(ATTRS))
    ap.add_argument("--chunk-size", type=int, default=8192)
    ap.add_argument("--num-chunks", type=int, default=1)
    ap.add_argument("--offset", type=int, default=-1, help="-1 = centered in region")
    ap.add_argument("--no-morton", action="store_true", help="random (load order) instead of morton")
    ap.add_argument("--pe-freqs", type=int, default=6)
    ap.add_argument("--no-identity", action="store_true")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--activation", default="relu", choices=list(_ACTS))
    ap.add_argument("--out-act", default="tanh", choices=["tanh", "none"])
    ap.add_argument("--shape-loss", default="aligned", choices=["aligned", "mse"],
                    help="aligned = sign-aligned quat MSE (q/-q invariant) + opacity clamp; "
                         "mse = raw per-component quat MSE. Both report Σ shape PSNR.")
    ap.add_argument("--opacity-clamp", type=float, default=8.0,
                    help="clamp logit-opacity target to +/-this (sigmoid-saturated beyond ~6)")
    ap.add_argument("--w-fdc", dest="w_fdc", type=float, default=1.0)
    ap.add_argument("--w-opa", dest="w_opa", type=float, default=1.0)
    ap.add_argument("--w-shape", dest="w_shape", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight-bytes", type=int, default=2, choices=(1, 2, 4))
    ap.add_argument("--pos-bytes", type=int, default=12, help="12=f32 now, 6=future u16")
    ap.add_argument("--report-every", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-train", action="store_true", help="prep cache + print stats only")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    attrs = [a.strip() for a in args.attrs.split(",") if a.strip()]
    out_dim = len(attrs)

    reg = prepare_region(args.ply, attrs, args.region_size, args.cache)
    xyz = np.ascontiguousarray(reg[:, :3]).astype(np.float32)
    attr = np.ascontiguousarray(reg[:, 3:]).astype(np.float32)

    # ---- attribute groups (fixed PLY layout) ----
    FDC = [i for i, a in enumerate(attrs) if a.startswith("f_dc")]
    OPA = [i for i, a in enumerate(attrs) if a == "opacity"]
    SCL = [i for i, a in enumerate(attrs) if a.startswith("scale")]
    ROT = [i for i, a in enumerate(attrs) if a.startswith("rot")]
    full = len(FDC) == 3 and len(OPA) == 1 and len(SCL) == 3 and len(ROT) == 4
    sign_align = (args.shape_loss == "aligned") and full
    C = float(args.opacity_clamp)
    print(f"[region] {reg.shape[0]:,} pts; quat sign_align={sign_align}; "
          f"opacity(logit) clamped to +/-{C}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP not available (use --device cpu)")
    if args.no_train:
        return

    # spatial blocking inside the cached region; centered `need` window
    if args.no_morton:
        order = np.arange(len(xyz))
    else:
        mn, mx = xyz.min(0), xyz.max(0)
        order = np.argsort(morton3d((xyz - mn) / np.maximum(mx - mn, 1e-6)), kind="stable")
    xyz = xyz[order]
    attr = attr[order]
    need = args.num_chunks * args.chunk_size
    off = max(0, min((len(xyz) - need) // 2 if args.offset < 0 else args.offset, len(xyz) - need))
    xyz_c = xyz[off:off + need].reshape(args.num_chunks, args.chunk_size, 3)
    attr_c = attr[off:off + need].reshape(args.num_chunks, args.chunk_size, out_dim)

    # per-chunk local normalization -> PE bands map to the chunk's extent
    cmin, cmax = xyz_c.min(1, keepdims=True), xyz_c.max(1, keepdims=True)
    pos = torch.from_numpy((xyz_c - cmin) / np.maximum(cmax - cmin, 1e-6)).to(args.device)

    model = Mlp(args.num_chunks, out_dim, args.pe_freqs, args.hidden, args.layers,
                args.activation, args.out_act, identity=not args.no_identity).to(args.device)
    pcp = per_chunk_params(model.pe.out_dim, args.hidden, args.layers, out_dim)
    print(f"[model] pe_out_dim={model.pe.out_dim} params/chunk={pcp:,} "
          f"act={args.activation} out_act={args.out_act}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)

    # ---- per-channel chunk min/max (normalize now, un-normalize at eval) ----
    amn = attr_c.reshape(-1, out_dim).min(0)
    amx = attr_c.reshape(-1, out_dim).max(0)
    arng = np.maximum(amx - amn, 1e-6)
    dev = args.device
    amn_t = torch.from_numpy(amn.astype(np.float32)).to(dev)
    arng_t = torch.from_numpy(arng.astype(np.float32)).to(dev)
    rot_perm = [ROT[1], ROT[2], ROT[3], ROT[0]]               # (w,x,y,z) -> (x,y,z,w)

    # normalized targets [-1,1]; opacity clamped to +/-C first (sigmoid-saturated beyond)
    tgt_np = (2.0 * (attr_c - amn) / arng - 1.0).astype(np.float32)
    tgt_np[..., OPA[0]] = (np.clip(attr_c[..., OPA[0]], -C, C) / C).astype(np.float32)
    tgt = torch.from_numpy(tgt_np).to(dev)
    ATTR_GRP = FDC + OPA + SCL                                 # fdc+opa+scale: plain MSE

    # raw targets for the shape (Σ) eval
    Sig_t = shape_sigma(torch.from_numpy(attr_c[..., rot_perm].astype(np.float32)).to(dev),
                        torch.from_numpy(attr_c[..., SCL].astype(np.float32)).to(dev))
    fdc_raw_t = torch.from_numpy(attr_c[..., FDC].astype(np.float32)).to(dev)
    alpha_t = torch.sigmoid(torch.from_numpy(attr_c[..., OPA[0]].astype(np.float32)).to(dev))

    def quat_loss(qp, qt):
        # q and -q encode the same rotation: flip qp into qt's hemisphere, then MSE.
        s = torch.sign((qp * qt).sum(-1, keepdim=True))
        s = torch.where(s == 0, torch.ones_like(s), s)
        return (((s * qp) - qt) ** 2).mean()

    for it in range(args.iters):
        opt.zero_grad()
        pr = model(pos)
        l_attr = ((pr[..., ATTR_GRP] - tgt[..., ATTR_GRP]) ** 2).mean()
        l_rot = quat_loss(pr[..., ROT], tgt[..., ROT]) if sign_align \
            else ((pr[..., ROT] - tgt[..., ROT]) ** 2).mean()
        loss = l_attr + args.w_shape * l_rot
        loss.backward(); opt.step(); sched.step()
        if it % args.report_every == 0 or it == args.iters - 1:
            print(f"  it={it:5d} attr={l_attr.item():.5f} rot={l_rot.item():.5f} "
                  f"tot={loss.item():.5f}", flush=True)

    model.eval()
    with torch.no_grad():
        pr = model(pos)
        # un-normalize quat(w,x,y,z) + scale, build Σ (the renderer's shape quantity)
        q_wxyz = 0.5 * (pr[..., ROT] + 1) * arng_t[ROT] + amn_t[ROT]
        s_log = 0.5 * (pr[..., SCL] + 1) * arng_t[SCL] + amn_t[SCL]
        Sig_p = shape_sigma(q_wxyz[..., [1, 2, 3, 0]], s_log)
        # global size-weighted relative Σ error: mean(||Σp-Σt||^2) / mean(||Σt||^2)
        shape_rel = ((Sig_p - Sig_t) ** 2).mean().item() / ((Sig_t ** 2).mean().item() + 1e-9)
        # f_dc color PSNR (denormalized)
        fdc_p = 0.5 * (pr[..., FDC] + 1) * arng_t[FDC] + amn_t[FDC]
        dmse = ((fdc_p - fdc_raw_t) ** 2).mean().item()
        dr = float(arng[FDC].max())
        # opacity alpha PSNR (opacity is trained in clamp/+/-C space in both modes)
        opa_logit_p = C * pr[..., OPA[0]].clamp(-1, 1)
        amse = ((torch.sigmoid(opa_logit_p) - alpha_t) ** 2).mean().item()
        print(f"\nf_dc color  MSE={dmse:.6f}  PSNR="
              f"{10 * math.log10(dr * dr / max(dmse, 1e-12)):.2f} dB")
        print(f"opacity     alpha-MSE={amse:.6f}  PSNR="
              f"{10 * math.log10(1.0 / max(amse, 1e-12)):.2f} dB")
        print(f"shape(Σ)    rel-MSE={shape_rel:.6f}  PSNR="
              f"{-10 * math.log10(max(shape_rel, 1e-12)):.2f} dB")
        print(f"[quat sign_align={sign_align}]  (lower rel-Σ-MSE == better splat shape)")
        # decompose: pred-rotation/true-scale  vs  true-rotation/pred-scale
        tq_xyzw = torch.from_numpy(attr_c[..., rot_perm].astype(np.float32)).to(dev)
        ts_log = torch.from_numpy(attr_c[..., SCL].astype(np.float32)).to(dev)
        sig_t_sq = (Sig_t ** 2).mean().item() + 1e-9
        rel_rot = ((shape_sigma(q_wxyz[..., [1, 2, 3, 0]], ts_log) - Sig_t) ** 2).mean().item() / sig_t_sq
        rel_scl = ((shape_sigma(tq_xyzw, s_log) - Sig_t) ** 2).mean().item() / sig_t_sq
        print(f"shape decomp: rotation-only rel-Σ={rel_rot:.3f}  scale-only rel-Σ={rel_scl:.3f}"
              f"  (full {shape_rel:.3f})")

    attr_bpp = pcp * args.weight_bytes / args.chunk_size
    total_bpp = attr_bpp + args.pos_bytes
    print("\n=== size projection (per chunk) ===")
    print(f"  params/chunk  : {pcp:,} x {args.weight_bytes} B")
    print(f"  attr bytes/pt : {attr_bpp:.3f}  (orig {out_dim*4} -> "
          f"{(out_dim*4)/max(attr_bpp,1e-9):.0f}x on attributes)")
    print(f"  + positions   : {args.pos_bytes} B/pt")
    print(f"  -> total      : {total_bpp:.3f} B/pt  (orig 56 -> "
          f"{56/total_bpp:.2f}x, {100*(1-total_bpp/56):.1f}% reduction)")


if __name__ == "__main__":
    main()
