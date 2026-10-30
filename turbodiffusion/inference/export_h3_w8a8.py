"""Export full H3 DMD weights using the existing TurboDiffusion W8A8 operators."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from safetensors.torch import save_file

    from turbodiffusion.h3.dit.config import H3DiTConfig
    from turbodiffusion.h3.dit.model import MiniMaxH3DiTModel
    from turbodiffusion.h3.dit.turbo_int8 import FORMAT, apply_turbo_int8, is_mlp_alias_key
    from turbodiffusion.h3.dit.weight_loader import load_fl2va_streaming

    arch = H3DiTConfig.from_json(str(args.model_path / "config.json"))
    with torch.device("meta"):
        model = MiniMaxH3DiTModel(arch, attn_backend="sdpa")
    model.to_empty(device="cpu")
    for parameter in model.parameters():
        parameter.data = torch.empty(parameter.shape, dtype=torch.bfloat16)
    for buffer in model.buffers():
        if buffer.is_floating_point():
            buffer.data = torch.empty(buffer.shape, dtype=torch.bfloat16)
    loaded, missing = load_fl2va_streaming(model, args.model_path)
    if missing:
        raise RuntimeError(f"incomplete source checkpoint: {missing} tensors missing")
    projections = apply_turbo_int8(model, quantize=True)
    if projections != 4 * arch.num_layers:
        raise RuntimeError(f"unexpected number of quantized projections: {projections}")
    for block in model.blocks:
        block.adaln_proj = None
    model.final_layer.adaln_proj = None
    state = {
        key: value.detach().cpu().contiguous().clone()
        for key, value in model.state_dict().items() if not is_mlp_alias_key(key)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(args.output), metadata={"format": FORMAT})
    args.output.with_suffix(".json").write_text(json.dumps({
        "format": FORMAT, "projections": projections, "adaln": "separate",
    }, indent=2))
    print(f"Exported {loaded} source tensors / {projections} W8A8 projections to {args.output}")


if __name__ == "__main__":
    main()
