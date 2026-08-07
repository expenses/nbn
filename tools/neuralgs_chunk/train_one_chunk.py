#!/usr/bin/env python3
"""
Prototype: fit ONE chunk of a 3DGS .ply with a single tiny MLP.

This is the architecture-search tool described in PLAN.md. It defines exactly the
model a Vulkan cooperative-vector / cooperative-matrix shader will later evaluate
(PE -> Linear/act x N -> Linear/tanh), trains it on one chunk, and reports
per-attribute error plus the projected compressed size so you can see whether a
config hits the ~5.6 bytes/point budget for poland.ply.

It intentionally does NOT use the BatchedLinear/bmm system from splat_mlp.py:
one chunk, one nn.Linear MLP, fast to iterate on.

Usage (from this directory, inside the uv project):
    uv run train_one_chunk.py ../../poland.ply \
        --chunk-size 16384 --pe-freqs 6 --hidden 48 --layers 3 \
        --activation relu --pos-bits 12 --weight-bytes 2 --iters 30000
"""
import argparse
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

# --- attribute layout for the poland.ply schema --------------------------------
POS_NAMES = ("x", "y", "z")
ATTR_NAMES = (
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
)
OUT_DIM = len(ATTR_NAMES)


# --- fast .ply loader (np.fromfile on a structured dtype) -----------------------
def read_ply_header(path):
    props, n = [], 0
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if line.startswith(b"element vertex"):
                n = int(line.split()[-1])
            elif line.startswith(b"property"):
                # property float name
                parts = line.split()
                ptype, pname = parts[-2].decode(), parts[-1].decode()
                props.append((pname, "<f4"))
            elif line.strip() == b"end_header":
                break
        offset = f.tell()
    return n, offset, props


def stream_attr_minmax(path, offset, n, dtype, block=2_000_000):
    """Single streaming pass over the binary blob -> per-attribute min/max."""
    rec = np.dtype([("pad", "<f4", 3), ("attr", "<f4", OUT_DIM)])
    mn = np.full(OUT_DIM, np.inf, np.float64)
    mx = np.full(OUT_DIM, -np.inf, np.float64)
    with open(path, "rb") as f:
        f.seek(offset)
        read = 0
        while read < n:
            m = min(block, n - read)
            a = np.fromfile(f, dtype=rec, count=m)["attr"]
            mn = np.minimum(mn, a.min(0))
            mx = np.maximum(mx, a.max(0))
            read += m
    return mn, mx


def load_sample(path, offset, n, dtype, stride):
    """Read a uniform strided sample (streaming -> bounded RAM for the 5.96 GB file)."""
    out = np.empty((n + stride - 1) // stride, dtype=dtype)
    i, read, K = 0, 0, 1 << 16  # process stride*K records at a time
    with open(path, "rb") as f:
        f.seek(offset)
        while read < n:
            want = min(stride * K, n - read)
            blob = np.fromfile(f, dtype=dtype, count=want)
            step = blob[::stride]
            out[i:i + len(step)] = step
            i += len(step)
            read += want
    return out[:i]


# --- Morton code (same as splat_mlp.py) ----------------------------------------
def _split_by_3(x):
    x = x & np.uint64(0x1FFFFF)
    x = (x | (x << 32)) & np.uint64(0x1F00000000FFFF)
    x = (x | (x << 16)) & np.uint64(0x1F0000FF0000FF)
    x = (x | (x << 8)) & np.uint64(0x100F00F00F00F00F)
    x = (x | (x << 4)) & np.uint64(0x10C30C30C30C30C3)
    x = (x | (x << 2)) & np.uint64(0x1249249249249249)
    return x


def morton_key(xyz01):
    q = np.clip(xyz01 * ((1 << 21) - 1), 0, (1 << 21) - 1).astype(np.uint64)
    return _split_by_3(q[:, 0]) | (_split_by_3(q[:, 1]) << 1) | (_split_by_3(q[:, 2]) << 2)


# --- model ---------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, freqs, identity=True):
        super().__init__()
        self.freqs = freqs
        self.identity = identity
        # 2^0 .. 2^{L-1} * pi
        self.register_buffer("bands", math.pi * (2.0 ** torch.arange(freqs)))

    @property
    def out_dim(self):
        return (3 if self.identity else 0) + 2 * 3 * self.freqs

    def forward(self, x):  # x: (..., 3) in [0,1]
        xb = x[..., None] * self.bands  # (...,3,L)
        pe = torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)  # (...,3,2L)
        pe = pe.reshape(*x.shape[:-1], 3 * 2 * self.freqs)
        if self.identity:
            pe = torch.cat([x, pe], dim=-1)
        return pe


def make_activation(name):
    return {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "silu": nn.Silu,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "hardswish": nn.Hardswish,
    }[name]


class TinyMLP(nn.Module):
    """PE -> (Linear -> act) x `layers` -> Linear -> out_act. Exactly what a
    cooperative-vector shader evaluates."""

    def __init__(self, pe_freqs, hidden, layers, activation, out_act="tanh",
                 identity=True, pad16=False):
        super().__init__()
        self.pe = PositionalEncoding(pe_freqs, identity=identity)
        in_dim = self.pe.out_dim
        if pad16:
            in_dim = ((in_dim + 15) // 16) * 16
        self.pad = pad16
        self.pad_in = (in_dim - self.pe.out_dim)
        act = make_activation(activation)
        mods = []
        d = in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), act()]
            d = hidden
        self.body = nn.Sequential(*mods)
        self.head = nn.Linear(d, OUT_DIM)
        self.out_act = nn.Tanh() if out_act == "tanh" else nn.Identity()

    def forward(self, x):  # x: (N,3) chunk-local [0,1]
        pe = self.pe(x)
        if self.pad:
            pe = nn.functional.pad(pe, (0, self.pad_in))
        return self.out_act(self.head(self.body(pe)))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# --- size math -----------------------------------------------------------------
def projected_bytes_per_point(num_params, weight_bytes, chunk_size, pos_bits):
    pos_bytes = 3 * pos_bits / 8.0
    weight_bpp = num_params * weight_bytes / chunk_size
    return pos_bytes + weight_bpp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("--chunk-size", type=int, default=16384)
    ap.add_argument("--chunk-index", type=int, default=0)
    ap.add_argument("--sample-every", type=int, default=128,
                    help="load 1-in-N verts of the file (RAM bound for arch search)")
    ap.add_argument("--pe-freqs", type=int, default=6)
    ap.add_argument("--no-identity", action="store_true", help="drop the raw xyz from PE")
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--activation", default="relu",
                    choices=["relu", "leaky_relu", "silu", "tanh", "gelu", "hardswish"])
    ap.add_argument("--out-act", default="tanh", choices=["tanh", "none"])
    ap.add_argument("--pad16", action="store_true", help="zero-pad PE input to mult of 16")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--pos-bits", type=int, default=12, help="for size projection")
    ap.add_argument("--weight-bytes", type=int, default=2, help="1=int8, 2=fp16, 4=fp32")
    ap.add_argument("--quantise-pos", action="store_true",
                    help="also emulate int`pos-bits` position quantisation at train time")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if dev.type == "cpu":
        print("WARNING: CUDA unavailable, running on CPU.", file=sys.stderr)

    # ---- load -----------------------------------------------------------------
    n, offset, props = read_ply_header(args.ply)
    dtype = np.dtype(props)
    assert all(p in [n for n, _ in props] for p in POS_NAMES + ATTR_NAMES), \
        "unexpected .ply schema"

    print(f"[ply] {n:,} verts; sampling 1-in-{args.sample_every} ...", flush=True)
    arr = load_sample(args.ply, offset, n, dtype, args.sample_every)
    n_sample = arr.shape[0]
    print(f"[ply] sample = {n_sample:,} verts", flush=True)

    xyz = np.stack([arr[c] for c in POS_NAMES], axis=1).astype(np.float64)  # (N,3)
    attr = np.stack([arr[c] for c in ATTR_NAMES], axis=1).astype(np.float32)  # (N,11)

    # global per-attribute min/max (from this sample; production: stream full file)
    attr_min, attr_max = attr.min(0), attr.max(0)
    rng = np.maximum(attr_max - attr_min, 1e-6)
    attr_norm = 2.0 * (attr - attr_min) / rng - 1.0  # -> [-1,1]

    # ---- spatial blocking on the sample: normalise -> morton -> blocks ---------
    xyz_min, xyz_max = xyz.min(0), xyz.max(0)
    xyz01 = (xyz - xyz_min) / np.maximum(xyz_max - xyz_min, 1e-6)
    order = np.argsort(morton_key(xyz01), kind="stable")
    xyz01, attr_norm = xyz01[order], attr_norm[order]

    n_blocks = n_sample // args.chunk_size
    if args.chunk_index >= n_blocks:
        raise SystemExit(f"chunk_index {args.chunk_index} >= n_blocks {n_blocks}")
    s = args.chunk_index * args.chunk_size
    e = s + args.chunk_size
    blk_pos = xyz01[s:e].astype(np.float32)        # chunk-local-ish [0,1] (global scene)
    blk_tgt = attr_norm[s:e].astype(np.float32)

    # re-normalise positions to THIS block's bbox so PE octaves span the block extent
    bmin, bmax = blk_pos.min(0), blk_pos.max(0)
    blk_pos = (blk_pos - bmin) / np.maximum(bmax - bmin, 1e-6)

    if args.quantise_pos:
        levels = (1 << args.pos_bits) - 1
        blk_pos = np.round(blk_pos * levels) / levels

    pos = torch.from_numpy(blk_pos).to(dev)
    tgt = torch.from_numpy(blk_tgt).to(dev)

    # ---- model + optim --------------------------------------------------------
    model = TinyMLP(args.pe_freqs, args.hidden, args.layers, args.activation,
                    args.out_act, identity=not args.no_identity, pad16=args.pad16).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)
    print(f"[model] PE out_dim={model.pe.out_dim} params={model.num_params():,}", flush=True)

    best = float("inf")
    pbar = tqdm(range(args.iters))
    for it in pbar:
        opt.zero_grad()
        pred = model(pos)
        loss = ((pred - tgt) ** 2).mean()
        loss.backward()
        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        sched.step()
        if loss.item() < best:
            best = loss.item()
        if it % 500 == 0:
            pbar.set_postfix(loss=f"{loss.item():.5f}", best=f"{best:.5f}")

    # ---- eval: per-attribute MSE + color PSNR for f_dc ------------------------
    model.eval()
    with torch.no_grad():
        pred = model(pos)
        se = ((pred - tgt) ** 2).mean(0).cpu().numpy()   # per-attr MSE (normalised space)
    print("\n=== per-attribute MSE (normalised [-1,1]) ===")
    for name, v in zip(ATTR_NAMES, se):
        print(f"  {name:9s} {v:.6f}")
    # f_dc PSNR in raw color units: denorm, treat as linear RGB-ish
    dc_pred = 0.5 * (pred[:, :3].cpu().numpy() + 1) * (attr_max[:3] - attr_min[:3]) + attr_min[:3]
    dc_tgt = attr[s:e, :3]
    dc_mse = ((dc_pred - dc_tgt) ** 2).mean()
    dc_psnr = 10 * math.log10((dc_tgt.max() - dc_tgt.min()) ** 2 / max(dc_mse, 1e-12)) \
        if dc_mse > 0 else float("inf")
    print(f"\nf_dc color MSE={dc_mse:.6f}  approx dynamic-range PSNR={dc_psnr:.2f} dB")

    # ---- projected size -------------------------------------------------------
    bpp = projected_bytes_per_point(model.num_params(), args.weight_bytes,
                                    args.chunk_size, args.pos_bits)
    total_n = n  # full file, no pruning
    total_mb = bpp * total_n / 1e6
    ratio = 56.0 / bpp
    print("\n=== size projection (whole file, no pruning) ===")
    print(f"  params/block      : {model.num_params():,}")
    print(f"  position bits/pt  : 3 x {args.pos_bits} = {3*args.pos_bits}")
    print(f"  weight bytes/param: {args.weight_bytes}  (block {args.chunk_size:,})")
    print(f"  -> bytes/point    : {bpp:.3f}   (budget 5.6 for 90%)")
    print(f"  -> projected total: {total_mb:.1f} MB   ({ratio:.1f}x smaller, "
          f"{100*(1-1/ratio):.1f}% reduction)")


if __name__ == "__main__":
    main()
