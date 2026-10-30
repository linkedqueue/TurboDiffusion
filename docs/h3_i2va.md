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

H3 uses the public [thu-ml/SpargeAttn](https://github.com/thu-ml/SpargeAttn)
custom block-mask API, not a private H3-specific fork:

```python
from spas_sage_attn import block_sparse_sage2_attn_cuda
```

The pinned public revision is `ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a`.
Build it with the CUDA toolkit matching PyTorch. On RTX 5090, use CUDA 12.8
or newer and expose only the intended build GPU architecture:

```bash
git clone https://github.com/thu-ml/SpargeAttn.git
git -C SpargeAttn checkout ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a
CUDA_VISIBLE_DEVICES=0 env -u TORCH_CUDA_ARCH_LIST \
  pip install ./SpargeAttn --no-build-isolation --no-deps
```

This revision's explicit `TORCH_CUDA_ARCH_LIST` allowlist predates SM120,
so the command lets its build script detect the visible GPU instead.
No SpargeAttn source patch is required for this adapter.

`turbodiffusion/h3/attention.py` reuses the repository's pooled top-k block
selection, then makes prefix/suffix query rows and key columns dense for
text, audio and first-frame conditioning tokens. Blocks are Q128/K64
(Q64/K128 on SM90), with sinks rounded up to block boundaries. The mask is
passed as `mask_id`; `pvthreshd=1e6` disables additional PV-based skipping.
The public top-k convenience function is not called because it does not
accept H3's prefix/suffix sink lengths. The shared pooling helper explicitly
zero-fills masked tail elements for sequences not divisible by the block size.

This preserves the block-mask policy, not bitwise equivalence with the earlier
local backend: quantization and accumulation inside the public kernel differ.
Query sequences shorter than 128 tokens use SDPA because the public kernel
does not support them. A missing public dependency raises an actionable error,
not a silent fallback. For a fully dense reference, explicitly set
`dit.attn_backend: sdpa`; it has different performance.

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

Local validation includes native W8A8 CUDA tests and actual public SpargeAttn
CUDA comparisons against the same block mask expanded for SDPA, including
partial blocks and conditioning sinks. Run these after building the pinned
public dependency:

```bash
H3_TEST_PUBLIC_SPARGE=1 python -m unittest discover -s tests -p 'test_h3*.py' -v
```

The first five benchmark cases have been re-sampled with the offload-buffer
sink propagation fix and conservative DBCache enabled. Each case compares the
earlier local attention backend with the pinned public backend, holding the
checkpoint, AdaLN table, conditioning, seed and native W8A8 settings fixed.
All ten outputs were decoded with the native W8A8 video VAE and FP32 audio VAE.
The five side-by-side MP4s retain all source frames and both audio tracks.
Both backends had identical cache decisions: one hit out of four updates for
cases 1 and 2, no hits for cases 3 through 5. Each timing follows a full warmup.
Case 5 was restarted in a fresh worker group after a multi-shape compilation
cache limit was observed; its earlier eager-communication draft is excluded.
These shared RTX 5090 measurements are not a controlled latency benchmark.

| Case | Duration | Earlier backend DiT | Public SpargeAttn DiT | Cache hits (both) |
| --- | --- | --- | --- | --- |
| 1 | 5 s | 9.56 s | 9.91 s | 1/4 |
| 2 | 5 s | 9.82 s | 9.77 s | 1/4 |
| 3 | 10 s | 22.66 s | 24.34 s | 0/4 |
| 4 | 8 s | 19.13 s | 19.75 s | 0/4 |
| 5 | 5 s | 12.30 s | 13.04 s | 0/4 |

Timings exclude model loading, conditioning encoding and VAE decoding, and use
the maximum synchronized sampling wall time over eight ranks. The public
backend took about 4.5% longer in aggregate in this run. Visual spot-checks
showed similar sharpness, with motion/framing differences especially in cases
4 and 5. Decoded audio waveforms also differ, especially in cases 3 and 4;
waveform similarity is not a perceptual quality score. The audio VAE is
unchanged, and differences are already present in the sampled audio latents.
These comparisons do not establish lossless equivalence or a quality ranking.
Generated media and local benchmark data are not included in the repository.

A same-latent BF16/native-W8A8 video VAE comparison was also generated for the
first benchmark case. The validation environment retains the local NCCL
binary described above; it is not a fresh public-only installation. The
bundled text/keyframe encoding path and source-built NCCL still need full-model
end-to-end validation. Unit tests do not establish quality or latency parity
for those paths.
