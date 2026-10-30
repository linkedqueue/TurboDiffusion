# H3 FL2VA with TurboDiffusion W8A8

This integration runs the existing H3 image-to-video-with-audio pipeline using
TurboDiffusion's native 128x128-block W8A8 Linear. The default `fullopt`
profile performs **four DMD denoising updates** (`dit.steps: 5` is the number
of sigma points) without DBCache. `fullopt-dbcache` keeps all the same
settings and enables a conservative F8/B4 cache (threshold 0.20, one warmup,
at most one consecutive hit). It does not use Wan's rCM schedule or SLA.

The model, Qwen3-VL text encoder, keyframe encoder, DiT sampling, VAE, offload,
AdaLN table, Ulysses/SP8 and cache implementation come from an **external**
`h3-fast-infer` checkout containing the H3 Turbo W8A8 integration.
TurboDiffusion provides the INT8 CUDA ops; SpargeAttn (DiT attention) and
Beacon (video VAE INT8) are separate dependencies. Install those in the same
environment as H3; do not add the Wan-specific libraries as H3 requirements.
See `h3-fast-infer/docs/local_turbo_w8a8.md` for the export and five-case
quality/latency comparison. The required H3 DiT export contains 200 quantized
main-block projections, and uses an unchanged AdaLN table made from the same
DMD source weights for `i2va / steps=5 / video_shift=12 / audio_shift=3`.

## Download the Model

The released checkpoint is
[liran12345/h3-turbo-diffusion](https://huggingface.co/liran12345/h3-turbo-diffusion).
It contains the **full BF16 four-update DMD transformer**, with the original
small FP32 tensors preserved, and its matching AdaLN table. This is the
**full-parameter DMD checkpoint**; export W8A8 locally from these weights.

Using the Hugging Face CLI:

```bash
hf download liran12345/h3-turbo-diffusion \
  --local-dir /models/h3-turbo-diffusion
```

The download includes `transformer/` (34 shards, approximately 61.73 GiB),
`adaln/adaln_i2va_s5_v12_a3.safetensors` and its companion `.json`, reference
configs and a checksum manifest. Text encoder, tokenizer and video/audio VAEs
are **not included**; obtain the original FL2VA components separately from
[MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3).
The examples below use `/models/MiniMax-H3/FL2VA` as that component root.
These weights use the
[MiniMax H3 Community License Agreement](https://huggingface.co/liran12345/h3-turbo-diffusion/blob/main/LICENSE),
not TurboDiffusion's code license; review its conditions before use.

## Prepare W8A8 Weights

The TurboDiffusion H3 entry point requires a local W8A8 export. After building
the TurboDiffusion CUDA extension, use the exporter in the external H3 checkout:

```bash
cd /path/to/h3-fast-infer
PYTHONPATH=/path/to/TurboDiffusion:. CUDA_VISIBLE_DEVICES=0 \
  python example/export_dit_turbo_int8.py \
  -i /path/to/TurboDiffusion/configs/h3_i2va_w8a8.yaml \
  --base-model-dir /models/h3-turbo-diffusion/transformer \
  -o /models/h3-turbo-diffusion/model_w8a8.safetensors
```

The exporter quantizes the full DMD weights directly. Reuse the downloaded AdaLN table
and its `.json` unchanged for `i2va / steps=5 / video_shift=12 / audio_shift=3`;
**W8A8 export alone does not require re-baking AdaLN**. Do not reuse this table
with different source weights or a different schedule.

The default profile also requires a separately exported Beacon INT8 video VAE.
In the same H3 checkout and environment:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
  python example/export_vae_int8.py \
  --vae-dir /models/MiniMax-H3/FL2VA/video_vae \
  -o /models/h3/video_vae_int8.safetensors
```

## Run

Run from the TurboDiffusion checkout after preparing the weights above.
The source checkout defaults to a sibling `../h3-fast-infer`, or set
`--h3-root`/`H3_ROOT`. Edit `configs/h3_i2va_w8a8.yaml` for your local
checkpoints, or supply paths on the command line:

```bash
bash scripts/inference_h3_i2va.sh \
  --h3-root ../h3-fast-infer \
  --model-root /models/MiniMax-H3/FL2VA \
  --dit-model-path /models/h3-turbo-diffusion/transformer \
  --w8a8-path /models/h3-turbo-diffusion/model_w8a8.safetensors \
  --adaln-path /models/h3-turbo-diffusion/adaln/adaln_i2va_s5_v12_a3.safetensors \
  --vae-int8-path /models/h3/video_vae_int8.safetensors \
  --image /path/to/first-frame.png \
  --prompt 'Video description and soundscape...' \
  --output-dir output/h3_demo
```

Use `--profile fullopt-dbcache` to enable conservative DBCache. To compare
profiles fairly, use separate output directories, identical prompt/image/
seed/duration, and `--profile fullopt` for the no-cache run. The default is
8 GPU ranks for DiT/VAE and 4 for Qwen TP/first-frame encode. The resulting
video with audio is `<output-dir>/t2va.mp4`; `run.yaml` records the effective
configuration. The original H3 stage artifacts remain alongside it.

To reuse existing H3 encodings without loading Qwen or re-encoding the image,
pass **both** `--prompt-embeds` and `--visual-cond`, plus the original
`--seed`, `--duration-seconds` and `--aspect-ratio`. These files are copied
into the output directory. `--skip-encode` instead uses files already there.
`--dry-run` writes the effective config and prints the stage commands without
starting GPUs. For the verified local H3 config, `--config
/path/to/h3-fast-infer/config/local_h3_i2va_dmd4_turbo.yaml` works instead
of passing the four checkpoint paths individually. Relative checkpoint paths
in H3 configs resolve against the H3 checkout; those in this repository's
template resolve against this repository.

Neither the DMD checkpoint nor the AdaLN table is converted in place. This
integration does not claim Wan2.2's end-to-end speedup: H3 has its own model,
schedule and 8-GPU deployment envelope.
