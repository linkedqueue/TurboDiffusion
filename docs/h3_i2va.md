# H3 FL2VA with TurboDiffusion W8A8

This adapter runs the existing H3 image-to-video-with-audio pipeline using
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

## Run

The source checkout defaults to a sibling `../h3-fast-infer`, or set
`--h3-root`/`H3_ROOT`. Edit `configs/h3_i2va_w8a8.yaml` for your local
checkpoints, or supply paths on the command line:

```bash
bash scripts/inference_h3_i2va.sh \
  --h3-root ../h3-fast-infer \
  --model-root /path/to/MiniMax-H3/FL2VA \
  --dit-model-path /path/to/i2v_dmd2-step4000/transformer \
  --w8a8-path /path/to/model_compact.safetensors \
  --adaln-path /path/to/adaln_i2va_s5_v12_a3.safetensors \
  --vae-int8-path /path/to/video_vae/model_int8.safetensors \
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

Prepare weights using the external H3 exporter after building the
TurboDiffusion CUDA extension:

```bash
cd /path/to/h3-fast-infer
PYTHONPATH=/path/to/TurboDiffusion:. CUDA_VISIBLE_DEVICES=0 \
  python example/export_dit_turbo_int8.py \
  -i config/local_h3_i2va_dmd4_turbo.yaml \
  -o /path/to/model_compact.safetensors
```

Neither the DMD checkpoint nor the AdaLN table is converted in place. This
adapter does not claim Wan2.2's end-to-end speedup: H3 has its own model,
schedule and 8-GPU deployment envelope.
