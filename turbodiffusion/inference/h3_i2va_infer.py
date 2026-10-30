#!/usr/bin/env python3
"""Run the full H3 DMD model with TurboDiffusion W8A8 and optional conservative DBCache.

H3 owns the model, conditioning, sampling and VAE; this entry point only
orchestrates its existing stages and makes this repository's CUDA ops importable.
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/h3_i2va_w8a8.yaml"
CONSERVATIVE_CACHE = {
    "enabled": True,
    "probe_only": False,
    "Fn_compute_blocks": 8,
    "Bn_compute_blocks": 4,
    "max_warmup_steps": 1,
    "max_continuous_cached_steps": 1,
    "residual_diff_threshold": 0.20,
    "enable_taylorseer": False,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h3-root", type=Path, default=Path(
        os.environ.get("H3_ROOT", ROOT.parent / "h3-fast-infer")
    ), help="h3-fast-infer source checkout with the Turbo W8A8 integration")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--model-root", type=Path, help="upstream FL2VA shared text/video/audio encoders")
    ap.add_argument("--dit-model-path", type=Path, help="full BF16 DMD transformer checkpoint directory")
    ap.add_argument("--w8a8-path", type=Path, help="exported H3 TurboDiffusion INT8 safetensors")
    ap.add_argument("--adaln-path", type=Path, help="matching i2va / 5-point AdaLN bake")
    ap.add_argument("--vae-int8-path", type=Path, help="Beacon video VAE decoder checkpoint")
    ap.add_argument("--profile", choices=("fullopt", "fullopt-dbcache"), default="fullopt")
    ap.add_argument("--image", type=Path, help="first frame; required when encoding")
    ap.add_argument("--prompt", help="H3 prompt; required when encoding")
    ap.add_argument("--prompt-embeds", type=Path, help="reuse an existing encoded prompt")
    ap.add_argument("--visual-cond", type=Path, help="reuse an existing encoded keyframe")
    ap.add_argument("--skip-encode", action="store_true", help="use inputs already in output-dir")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "output/h3_i2va")
    ap.add_argument("--seed", type=int, help="override config seed")
    ap.add_argument("--duration-seconds", type=float, help="override config duration")
    ap.add_argument("--resolution", choices=("360p", "480p", "540p", "720p", "768p"))
    ap.add_argument("--short-edge", type=int)
    ap.add_argument("--aspect-ratio", help="width:height; default follows the first image")
    ap.add_argument("--nproc", type=int, default=8, help="DiT / VAE GPU ranks")
    ap.add_argument("--encode-nproc", type=int, default=4, help="Qwen TP / encode GPU ranks")
    ap.add_argument("--master-port", type=int, default=14770)
    ap.add_argument("--dry-run", action="store_true", help="write effective config and print commands")
    return ap.parse_args(argv)


def _config_root(config_path: Path, h3_root: Path) -> Path:
    if config_path.is_relative_to(h3_root):
        return h3_root
    if config_path.is_relative_to(ROOT):
        return ROOT
    return config_path.parent


def effective_config(args: argparse.Namespace) -> dict:
    config_path = args.config.resolve()
    h3_root = args.h3_root.resolve()
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"expected a mapping in {config_path}")
    base = _config_root(config_path, h3_root)

    def absolute(value: str | Path, relative_to: Path = base) -> str:
        path = Path(value).expanduser()
        return str((path if path.is_absolute() else relative_to / path).resolve())

    cfg["model_root"] = absolute(
        args.model_root or cfg["model_root"], Path.cwd() if args.model_root else base
    )
    dit = cfg.setdefault("dit", {})
    vae = cfg.setdefault("vae", {})
    for section, key, override in (
        (dit, "model_path", args.dit_model_path),
        (dit, "turbo_int8_path", args.w8a8_path),
        (dit, "adaln_bake_path", args.adaln_path),
        (vae, "int8_path", args.vae_int8_path),
    ):
        section[key] = absolute(
            override or section[key], Path.cwd() if override else base
        )
    if dit.get("int8_path") or cfg.get("beacon_params"):
        raise ValueError("H3 DiT must use Turbo W8A8 alone; Beacon is only for the VAE")
    if int(dit.get("steps", 0)) != 5 or (float(dit.get("video_shift", 0)),
                                          float(dit.get("audio_shift", 0))) != (12.0, 3.0):
        raise ValueError("the shipped AdaLN bake requires steps=5, video_shift=12, audio_shift=3")
    dit["dbcache"] = (
        dict(CONSERVATIVE_CACHE) if args.profile == "fullopt-dbcache" else {"enabled": False}
    )

    text = cfg.setdefault("text_encoder", {})
    text.setdefault("engine", "tp")
    text.setdefault("tp_size", args.encode_nproc)
    text["backend"] = "file"
    text["embeds_path"] = str(args.output_dir.resolve() / "prompt_embeds.pt")
    sample = cfg.setdefault("sample", {})
    sample["task"] = "i2va"
    for key, value in (
        ("seed", args.seed), ("duration_seconds", args.duration_seconds),
        ("short_edge", args.short_edge), ("resolution", args.resolution),
        ("aspect_ratio", args.aspect_ratio), ("prompt", args.prompt),
    ):
        if value is not None:
            sample[key] = value
    cfg["output_dir"] = str(args.output_dir.resolve())
    return cfg


def validate_inputs(args: argparse.Namespace, cfg: dict) -> None:
    h3_root = args.h3_root.resolve()
    for script in ("encode_infer.py", "dit_i2va_infer.py", "vae_decode_infer.py"):
        if not (h3_root / "example" / script).is_file():
            raise FileNotFoundError(f"missing H3 stage: {h3_root / 'example' / script}")
    for path in (
        Path(cfg["model_root"]),
        Path(cfg["dit"]["model_path"]) / "config.json",
        Path(cfg["dit"]["turbo_int8_path"]),
        Path(cfg["dit"]["adaln_bake_path"]),
        Path(cfg["dit"]["adaln_bake_path"]).with_suffix(".json"),
        Path(cfg["vae"]["int8_path"]),
    ):
        if not path.exists():
            raise FileNotFoundError(f"missing H3 model artifact: {path}")
    if args.nproc != 8 or args.encode_nproc != 4:
        raise ValueError("this profile is validated for 8 DiT/VAE ranks and 4 encode ranks")
    if args.skip_encode or args.prompt_embeds or args.visual_cond:
        if bool(args.prompt_embeds) != bool(args.visual_cond):
            raise ValueError("--prompt-embeds and --visual-cond must be given together")
        for path in (args.prompt_embeds, args.visual_cond):
            if path is not None and not path.is_file():
                raise FileNotFoundError(path)
        if not args.prompt_embeds:
            for name in ("prompt_embeds.pt", "visual_cond.pt"):
                if not (args.output_dir / name).is_file():
                    raise FileNotFoundError(args.output_dir / name)
    else:
        if not args.image or not args.image.is_file():
            raise ValueError("pass an existing --image for first-frame encoding")
        if not (args.prompt or cfg["sample"].get("prompt")):
            raise ValueError("pass --prompt for first-frame encoding")
    if (args.output_dir / "latents.pt").exists() or (args.output_dir / "t2va.mp4").exists():
        raise FileExistsError(f"output already contains a sample: {args.output_dir}")


def stage_commands(args: argparse.Namespace, cfg_path: Path, cfg: dict) -> list[list[str]]:
    h3_root = args.h3_root.resolve()

    def stage(script: str, nproc: int, port: int) -> list[str]:
        return [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={nproc}", "--master_addr=127.0.0.1",
            "--nnodes=1", "--node_rank=0", f"--master_port={port}",
            str(h3_root / "example" / script), "-i", str(cfg_path),
        ]

    commands = []
    if not (args.skip_encode or args.prompt_embeds):
        cmd = stage("encode_infer.py", args.encode_nproc, args.master_port - 1)
        cmd += [
            "--task", "fl2va", "--images", str(args.image.resolve()),
            "--prompt", str(args.prompt or cfg["sample"]["prompt"]),
            "--out-dir", str(args.output_dir.resolve()),
            "--resolution", str(cfg["sample"].get("resolution", "768p")),
        ]
        if args.aspect_ratio:
            cmd += ["--aspect-ratio", args.aspect_ratio]
        commands.append(cmd)
    commands.append(stage("dit_i2va_infer.py", args.nproc, args.master_port))
    commands.append(stage("vae_decode_infer.py", args.nproc, args.master_port + 1))
    return commands


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    cfg = effective_config(args)
    validate_inputs(args, cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = args.output_dir / "run.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    if args.prompt_embeds:
        for src, name in (
            (args.prompt_embeds, "prompt_embeds.pt"),
            (args.visual_cond, "visual_cond.pt"),
        ):
            dst = args.output_dir / name
            if src.resolve() != dst:
                shutil.copy2(src, dst)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(ROOT), str(args.h3_root.resolve()), env.get("PYTHONPATH", "")
    )))
    env.setdefault("OMP_NUM_THREADS", "8")
    for command in stage_commands(args, cfg_path, cfg):
        print("[h3]", shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
    if not args.dry_run:
        output = args.output_dir / "t2va.mp4"
        if not output.is_file():
            raise RuntimeError(f"H3 decode did not produce {output}")
        print(f"[h3] ready -> {output}", flush=True)


if __name__ == "__main__":
    main()
