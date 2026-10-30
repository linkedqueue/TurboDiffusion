# H3 FL2VA with TurboDiffusion W8A8

The H3 model and inference implementation are included in this repository.
The default `fullopt-dbcache` profile performs **four DMD denoising updates**
(`dit.steps: 5` means five sigma points) with conservative F8/B4 caching:
threshold 0.20, one warmup, at most one consecutive hit, and TaylorSeer disabled.
`fullopt` uses the same settings without DBCache. Both profiles retain the
precomputed AdaLN table.

## Implementation

- `turbodiffusion/h3/dit/model.py`: H3 Omni-DiT and token refiner.
- `turbodiffusion/h3/dit/processor.py`: four-update DMD sampling and AdaLN reuse.
- `turbodiffusion/h3/dit/turbo_int8.py`: replacement of 200 main-block projections
  with the existing `turbodiffusion.ops.core.Int8Linear`.
- `turbodiffusion/h3/dit/dbcache.py`, `parallel.py`, `offload.py`: conservative
  caching, Ulysses/SP8, quantized/asynchronous communication and block offload.
- `turbodiffusion/h3/textencoder/`: tensor-parallel Qwen3-VL layer-50 encoding.
- `turbodiffusion/h3/stages.py`, `vae.py`: image encoding, sampling, tiled
  video/audio decoding and MP4 export. The video VAE uses the same native
  `Int8Linear` for 144 attention/FFN projections in its 36 ViT decoder blocks.
- `turbodiffusion/inference/export_h3_w8a8.py`: full-model W8A8 export.

No separate H3 inference checkout is required. W8A8 uses TurboDiffusion's
existing CUDA quantization/GEMM kernels, with 128x128 weight blocks; it does
not introduce another quantization backend. Keyframe encoding remains FP32,
video decoder block projections use W8A8, and audio VAE decoding is FP32.
Video decoder I/O projections, convolutions and other unquantized tensors
retain BF16 (normalization follows the public VAE implementation).
SpargeAttn is a separately installed attention dependency.

## Installation

Use Python 3.10 or newer and the repository's CUDA build environment:

```bash
pip install -e '.[h3]' --no-build-isolation
```

Local validation used Python 3.10, PyTorch `2.7.0+cu128`, CUDA 12.8,
Triton 3.3.0, Transformers 4.57.3 and Diffusers 0.38.0. The H3 extra pins
Diffusers to that tested version.

Install `ffmpeg` on the host for video/audio export. The profiles use 8 GPU
ranks for DiT/VAE and 4 for text/keyframe encoding. CPU RAM must hold each
rank's offloaded weights; these are not single-GPU profiles.
They do not use Wan's rCM schedule or SLA.

### SpargeAttn Compatibility

**The default profiles currently require an H3-compatible SpargeAttn build
whose public release is pending.** Installing public `thu-ml/SpargeAttn` alone
does not provide the interface used by the local H3 validation:

```python
from sparge_attn import spas_sage2_attn_topk_cuda
```

The required API accepts `scale`, `smooth_k`, `topk`, `pre_sink_length` and
`post_sink_length`, together with the matching CUDA implementation. The sink
regions preserve dense query/key block rows and columns for text, audio and
first-frame conditioning tokens. The public main-branch interface
`spas_sage_attn.spas_sage2_attn_meansim_topk_cuda` is not a verified drop-in
replacement; changing only the import can change this behavior.

The dependency request is tracked in
[PR #144](https://github.com/thu-ml/TurboDiffusion/pull/144#issuecomment-6518616283).
Once upstream publishes a compatible build, its installation command and
commit/tag must be recorded and validated here. There is no silent fallback:
the SpargeAttn path raises an import error if the required API is missing.
For a dense-attention reference, explicitly set `dit.attn_backend: sdpa` in
a custom config; it is not the optimized profile and has different performance.

### NCCL on RTX 5090

For RTX 5090 hosts, build NCCL 2.26.2 from public source and use that library
in the H3 inference shell. Do not copy a library from another machine into
PyTorch's `site-packages`. The official `v2.26.2-1` tag resolves to
`f44ac759fee12ecb3cc6891e9e739a000f66fd70`.

Install Git, Make and a CUDA-compatible C++ compiler first. The example uses
CUDA 12.8 with a CUDA 12.8 PyTorch build; adjust the toolkit path for your
environment. Compiling `sm_120` requires CUDA 12.8 or newer. This build targets
5090 GPUs only; mixed-architecture hosts need additional `NVCC_GENCODE` entries.

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export NCCL_SRC="$HOME/.local/src/nccl-2.26.2"
mkdir -p "$(dirname "$NCCL_SRC")"
git clone --branch v2.26.2-1 --depth 1 \
  https://github.com/NVIDIA/nccl.git "$NCCL_SRC"
test "$(git -C "$NCCL_SRC" rev-parse HEAD)" = \
  f44ac759fee12ecb3cc6891e9e739a000f66fd70
make -C "$NCCL_SRC" -j"${NCCL_BUILD_JOBS:-8}" src.build \
  CUDA_HOME="$CUDA_HOME" \
  NVCC_GENCODE="-gencode=arch=compute_120,code=sm_120"
```

After a successful build, set these variables in the same shell used to launch
H3. They are inherited by all `torchrun` workers. No root installation or
replacement of the Python environment's files is needed.

```bash
export NCCL_HOME="$HOME/.local/src/nccl-2.26.2/build"
export LD_LIBRARY_PATH="$NCCL_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$NCCL_HOME/lib/libnccl.so.2${LD_PRELOAD:+:$LD_PRELOAD}"

python - <<'PY'
import os
from pathlib import Path
import torch

expected = (Path(os.environ["NCCL_HOME"]) / "lib/libnccl.so.2").resolve(strict=True)
version = torch.cuda.nccl.version()
loaded = {
    Path(line.split(maxsplit=5)[5].strip()).resolve()
    for line in Path("/proc/self/maps").read_text().splitlines()
    if len(line.split(maxsplit=5)) == 6 and "libnccl.so" in line.split(maxsplit=5)[5]
}
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("NCCL:", version, "loaded:", sorted(map(str, loaded)))
if version != (2, 26, 2) or loaded != {expected}:
    raise SystemExit("NCCL version/path mismatch; check the runtime environment before inference")
PY
```

The local H3 validation environment reports NCCL 2.26.2 but contains a binary
that differs from the official wheel. The source-build procedure above makes
the intended dependency explicit; it has **not yet been validated with a full
H3 multi-GPU run**, and does not establish parity with that local binary.
Validate the default INT8/asynchronous all-to-all path on the target host
before collecting benchmarks.

If DiT all-to-all fails, restart inference with `H3_USE_NCCL_A2A=0` to select
the existing all-gather fallback. This changes communication cost and does
not eliminate NCCL or affect the VAE's own communication implementation.

## Download

The full BF16 four-update DMD checkpoint and matching AdaLN table are at
[liran12345/h3-turbo-diffusion](https://huggingface.co/liran12345/h3-turbo-diffusion).
Original small FP32 tensors are preserved. The transformer has 34 shards
and is approximately 61.73 GiB.

The text encoder, processor/tokenizer and video/audio VAEs come from
[MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3).
Download their complete directories, including VAE Python modules and `source/`
weights, but not the original many-step transformer:

```bash
hf download liran12345/h3-turbo-diffusion \
  --local-dir /models/h3-turbo-diffusion
hf download MiniMaxAI/MiniMax-H3 \
  --include 'FL2VA/text_encoder/*' 'FL2VA/processor/*' 'FL2VA/tokenizer/*' \
            'FL2VA/video_vae/*' 'FL2VA/audio_vae/*' \
  --local-dir /models/MiniMax-H3
```

The VAEs use the public Python modules supplied with those weights.
`--trust-model-code` explicitly permits loading that code; review its source
before enabling it. The code is loaded from the local model directory,
not a private checkout. Third-party attribution is in `turbodiffusion/h3/NOTICE`.
Model weights remain subject to the
[MiniMax H3 Community License Agreement](https://huggingface.co/liran12345/h3-turbo-diffusion/blob/main/LICENSE),
not this repository's code license.

## Export W8A8

Run from the TurboDiffusion checkout after building its CUDA extension:

```bash
CUDA_VISIBLE_DEVICES=0 python -m turbodiffusion.inference.export_h3_w8a8 \
  --model-path /models/h3-turbo-diffusion/transformer \
  --output /models/h3-turbo-diffusion/model_w8a8.safetensors
```

The exporter wraps only main-block QKV, attention output and MLP projections.
It does not modify the source checkpoint or the CUDA kernels.
Reuse `adaln/adaln_i2va_s5_v12_a3.safetensors` **and its companion `.json`**
unchanged for `i2va / steps=5 / video_shift=12 / audio_shift=3`.
W8A8 export alone does not require re-baking AdaLN.

The video VAE is quantized once at decode startup from the original public
weights using the unchanged native `Int8Linear.from_linear`. No separate VAE
export or quantized VAE checkpoint is required. The source weights are not
modified. VAE quantization does not require re-baking DiT AdaLN either.

## Run

```bash
bash scripts/inference_h3_i2va.sh \
  --model-root /models/MiniMax-H3/FL2VA \
  --dit-model-path /models/h3-turbo-diffusion/transformer \
  --w8a8-path /models/h3-turbo-diffusion/model_w8a8.safetensors \
  --adaln-path /models/h3-turbo-diffusion/adaln/adaln_i2va_s5_v12_a3.safetensors \
  --trust-model-code \
  --image /path/to/first-frame.png \
  --prompt 'Video description and soundscape...' \
  --output-dir output/h3_demo
```

Conservative DBCache is enabled by default (`--profile fullopt-dbcache`).
Add `--profile fullopt` to disable DBCache for the no-cache reference.
Both profiles default to native W8A8 video decoding; use `--vae-precision bf16`
for the unquantized video decoder reference.
Use separate output directories and identical inputs/seed/duration to compare
profiles. The output with audio is `<output-dir>/t2va.mp4`; `run.yaml` records
the configuration, and conditioning/latent artifacts remain alongside it.

To reuse conditioning, pass both `--prompt-embeds` and `--visual-cond`, plus
the original `--seed`, `--duration-seconds` and `--aspect-ratio`.
`--skip-encode` instead uses conditioning files already in the output directory.
`--dry-run` validates paths, writes the configuration and prints the bundled
stage commands without starting GPUs or loading model Python code.

Relative paths in the shipped config resolve against this repository;
relative paths in custom configs resolve against their config directory.
CLI path overrides resolve against the working directory.
Older configs referencing a separate inference checkout or a quantized video
VAE checkpoint should be replaced with `configs/h3_i2va_w8a8.yaml`.

To isolate VAE quality, reuse the exact same `latents.pt` and `visual_cond.pt`
in two separate output directories, and decode with
`vae.video_decoder_precision: bf16` and `w8a8`, respectively.
The native W8A8 VAE path differs from earlier local latency experiments.
Do not infer end-to-end speed or quality parity from the W8A8 DiT benchmark;
this repository does not claim Wan2.2's speedup for H3.

## Validation Scope

Local validation includes native W8A8 CUDA tests, full-model four-update DiT
sampling with reused conditioning, and video/audio decoding with both BF16 and
native W8A8 video VAEs. A same-latent VAE comparison was generated for the first
benchmark case. These checks used the local SpargeAttn/NCCL environment described
above, not a fresh public-only installation.

The latest offload-buffer sink propagation fix has a regression test; the
full-model video run has not been repeated after that fix. The bundled
text/keyframe encoding path and the default DBCache profile also still need
full-model end-to-end validation. Unit tests do not establish quality or
latency parity for those paths.
