# NeuralGS-style chunk compression for `poland.ply` — plan

Goal: fit one tiny MLP per chunk of a 3DGS `.ply`, decode each chunk later with a
Vulkan **cooperative vector** / **cooperative matrix** shader, and reach **90% file
size reduction**. No camera/render training, no pruning (all points kept).

## 0. The numbers that drive everything

```
poland.ply : 106,447,647 verts × 56 B  = 5.96 GB
predict    : f_dc(3) + opacity(1) + scale(3) + rot(4) = 11 floats   (xyz is stored, not predicted)
target 90% : <= 596 MB total           -> per-point budget = 5.6 bytes
```

The **dominant cost is positions**, not MLP weights. With 106M points, full-precision
xyz alone is `12 B/pt = 1.27 GB` — already 2× the whole budget. So position
quantization + large chunks (to amortise the one MLP) are the levers that decide
whether 90% is reachable. MLP width/depth is a secondary knob.

Sanity configs (per-point bytes; `pe L=6 -> in 39`, hidden 48, 3 hidden layers, out 11,
MLP ≈ 7163 params):

| block size | positions          | MLP wts (fp16) | total B/pt | reduction |
|-----------:|--------------------|----------------:|-----------:|----------:|
|     16,384 | int16×3 = 6.0      | 0.87           | 6.87       | 87.7 %    |
|     16,384 | **int12×3 = 4.5**  | 0.87           | **5.37**   | **90.4 %**|
|     32,768 | int12×3 = 4.5      | 0.44           | **4.94**   | **91.2 %**|
|     32,768 | int16×3 = 6.0      | 0.22 (int8 w)  | 6.22       | 88.9 %    |

**Takeaway:** ~90% is very achievable with **large spatial blocks (16k–64k)**, **bit-packed
chunk-local positions (int12–int16)**, and a **small PE MLP with fp16 weights**.

## 1. Chunking: use SPATIAL blocks, not attribute clusters, for this file

The paper clusters by **attribute similarity** (k-means) so each MLP sees smooth targets.
That is the right call for their scenes, but for *this file* it backfires on the budget:

- attribute clusters are **scattered in space** → each cluster's bbox ≈ the whole scene →
  you can't quantise positions to int12/int16 locally. You're forced back to ~6–12 B/pt
  for positions, which blows the 5.6 B/pt budget on 106M points.
- **spatial blocks** (Morton/KD-tree/BVH-region) have a tiny local bbox → int12 packed
  positions give sub-millimetre resolution and ~4.5 B/pt. That is what makes 90% reachable.

Fitting difficulty from spatial blocks (neighbours with different attributes) is then
handled by **positional encoding + enough capacity** (and optionally a light attribute
sub-clustering *within* a spatial block). So:

- **Primary: spatial blocking** (Morton or KD-tree) into ~16k–64k-point blocks.
- Each block stores: 3-float origin + per-axis scale (for int→float dequant), bit-packed
  quantised positions, and one MLP.
- Optional refinement: within a block, run 2–4 attribute mini-clusters, each its own MLP,
  if a single MLP can't reach the target MSE. (Revisit only if fitting is the bottleneck.)

Renderer note: spatial blocks also map naturally to tile/region streaming, which is a
bonus vs. attribute clusters (which would scatter a screen region across many MLPs).

## 2. Positional encoding (this is the big missing piece)

Without PE the MLP has spectral bias and can't fit the high-frequency attribute variation
of neighbouring Gaussians — the #1 reason the current script underfits.

- Use classic NeRF PE on **chunk-local normalised** xyz ∈ [0,1]³:
  `γ(x) = [x, sin(2⁰πx), cos(2⁰πx), …, sin(2^{L-1}πx), cos(2^{L-1}πx)]` → dim `3 + 6L`.
  `L = 6..10` is a good starting range; pick L so `block_extent / 2^{L-1}` ≈ desired res.
- PE is **weight-free** → costs nothing in storage and is exactly what you want for
  cooperative vector: compute PE in the shader (a handful of sin/cos, once per point),
  then feed the vector to the first matmul. (This is also *why* the paper picks PE over a
  hash grid: a hash grid would need storing the grid, killing the compression ratio.)

## 3. Activation: pick one your Vulkan path exposes

Cooperative **vector** (`VK_NV_cooperative_vector` / the KHR effort) fuses bias+activation
and exposes a fixed enum (ReLU / GELU / tanh / sigmoid / SiLU(swish) / hardswish / leakyReLU
are typically available — **verify against your device's supported set**). Cooperative
**matrix** does only the matmul, so you write the activation yourself; there, transcendentals
(tanh, sigmoid, SiLU) are expensive and piecewise-linear is cheap.

Recommendation: **ReLU on hidden layers** (cheapest, universally available, trains fine for
these tiny regressors) and a **`tanh` on the final layer** to bound outputs to (−1,1) — which
matches the per-attribute min-max normalisation below (this is why the paper uses tanh). If
you stay purely on the fused cooperative-vector path, SiLU is also fine.

## 4. Target normalisation (matches the paper, improves conditioning)

Normalise each of the 11 attributes to **[−1,1]** with **global** min/max across the file
(shared by all blocks → 22 floats = 88 B total, negligible). Fit MSE on normalised targets;
decode does `attr = 0.5*(pred+1)*(max-min)+min`. Notes:

- `opacity`, `scale` are already in logit/log space in 3DGS — just min-max them.
- `rot` quaternion: fit the 4 components (normalised), **renormalise to unit length at decode**.

Importance-weighting (paper) needs rendered-pixel contribution → skipped per your "no camera"
constraint.

## 5. Replace the `BatchedLinear` system with a single MLP + two-phase training

The `BatchedLinear` bmm module couples architecture definition to batched training, which
makes architecture search painful and hides the actual model you'll ship to Vulkan. Fix:

- **Define one MLP** with stock `nn.Linear` layers (PE → Linear×N → out). *This is exactly
  the model the shader evaluates*, nothing more.
- **Phase 1 — architecture search:** load **one block**, train that single MLP, measure
  per-attribute MSE + projected bpp. Iterate fast on `L`, hidden width, #layers, activation.
  `train_one_chunk.py` does this.
- **Phase 2 — production:** train every block. Two clean options that keep the model
  definition identical to Phase 1:
  - **(a) embarrassingly-parallel per-block jobs** — each block is an independent tiny
    training; trivially matches the shader 1:1, parallelise across GPUs/streams. This is
    what the paper means by "fit clusters in parallel".
  - **(b) `torch.func.vmap` over the single MLP** — vectorise the same module across blocks
    for throughput without rewriting it (clean replacement for the hand-rolled bmm). Use only
    if per-block throughput becomes the bottleneck.

Reserve the batched form purely as a training-throughput optimisation layered *on top* of the
single-MLP definition — never as the definition itself.

## 6. Architecture starting point

```
input  : PE(xyz_local)  -> 3 + 6L   (L=6 -> 39; pad to 48 if you want 16-aligned matmuls)
hidden : 48 or 64, 3 hidden Linear layers, ReLU
output : Linear -> 11, then tanh
```
~7k–12k params → 14–24 KB/block at fp16. With 32k blocks that's ≤0.7 B/pt amortised, leaving
the budget almost entirely for positions. Cooperative-vector alignment: keep `hidden` a
multiple of 16/32 and consider zero-padding the PE input to a multiple of 16.

## 7. Export / decode format (Vulkan)

Per block, dump:
- PE config (`L`, include-identity flag), per-axis position quant (origin, scale, bits),
- per-layer `(W, b)` as raw **fp16** (or int8 + scale for an extra ~2×),
- global per-attribute min/max (shared, stored once).

Decode shader (one block = one cooperative-vector MLP eval): `unpack pos -> local float ->
PE -> Linear/act × N -> Linear/tanh -> denorm`. Positions unpack + PE are pure functions;
only the Linears use the stored weights — i.e. storage = MLP weights only, which is the whole
point.

## 8. Suggested workflow / milestones

1. `train_one_chunk.py` on a Morton block: confirm PE + ReLU+tanh + per-attr norm converges
   to low MSE on all 11 attributes. Sweep `L`, width, #layers, block size.
2. Add position quantisation emulation to the eval (round targets' positions to int12/int16,
   re-measure MSE) so the reported number reflects the *decoded* quality, not float inputs.
3. Add weight quantisation emulation (fp16 / int8) to the eval.
4. Pick the config that meets your MSE bar **and** the 5.6 B/pt budget; then write the
   Phase-2 trainer (per-block or vmap) and the `.ply -> chunk pack` exporter.
5. Write the Vulkan cooperative-vector decode shader for one block; validate output vs PyTorch.
6. Scale to the whole file; measure end-to-end size + a decode-time benchmark.

## 9. NixOS + uv setup

nix-ld is already enabled on this machine (`NIX_LD`, `NIX_LD_LIBRARY_PATH` set), and
`uv` + CPython 3.11/3.13 are cached. So: manage deps with a local uv project
(`tools/neuralgs_chunk/pyproject.toml`) and install **PyPI torch (CUDA wheel)** into it.
The only thing the torch CUDA wheel needs beyond nix-ld's default is `libstdc++` and the
NVIDIA driver libs. Add a small python devShell to `flake.nix`:

```nix
devShells.python = with pkgs; mkShell {
  nativeBuildInputs = [ python3 uv ];
  NIX_LD_LIBRARY_PATH = lib.makeLibraryPath [
    stdenv.cc.cc.lib zlib                      # libstdc++ etc. for PyPI wheels
    "/run/opengl-driver/lib"                   # libcuda.so / driver libs (your GPU box)
  ];
  # NIX_LD itself is already provided system-wide by programs.nix-ld.enable
};
```

Then:

```
cd tools/neuralgs_chunk
uv sync                       # installs torch (CUDA), numpy, tqdm
uv run train_one_chunk.py ../../poland.ply --chunk-size 16384 --pe-freqs 6 --hidden 48
```

(If the PyPI CUDA wheel still complains about a loader, the bulletproof fallback is a
`buildFHSEnv` shell instead of plain `mkShell` — heavier, but prebuilt wheels "just work".)
