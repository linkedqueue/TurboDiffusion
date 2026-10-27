"""CPU-only coverage for the external H3 W8A8 pipeline adapter."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from turbodiffusion.inference import h3_i2va_infer as infer


class H3InferenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.h3 = self.root / "h3"
        (self.h3 / "example").mkdir(parents=True)
        for name in ("encode_infer.py", "dit_i2va_infer.py", "vae_decode_infer.py"):
            (self.h3 / "example" / name).touch()
        self.assets = self.root / "assets"
        (self.assets / "shared").mkdir(parents=True)
        (self.assets / "transformer").mkdir()
        (self.assets / "transformer" / "config.json").touch()
        for name in ("model.safetensors", "adaln.safetensors", "adaln.json", "vae.safetensors"):
            (self.assets / name).touch()
        self.image = self.root / "first.png"
        self.image.touch()

    def args(self, *extra):
        return infer.parse_args([
            "--h3-root", str(self.h3),
            "--model-root", str(self.assets / "shared"),
            "--dit-model-path", str(self.assets / "transformer"),
            "--w8a8-path", str(self.assets / "model.safetensors"),
            "--adaln-path", str(self.assets / "adaln.safetensors"),
            "--vae-int8-path", str(self.assets / "vae.safetensors"),
            "--output-dir", str(self.root / "output"),
            *extra,
        ])

    def test_profiles_and_full_pipeline(self):
        args = self.args("--image", str(self.image), "--prompt", "a test", "--seed", "73")
        config = infer.effective_config(args)
        infer.validate_inputs(args, config)
        self.assertEqual(config["dit"]["steps"], 5)
        self.assertEqual(config["dit"]["turbo_int8_path"], str(self.assets / "model.safetensors"))
        self.assertEqual(config["sample"]["seed"], 73)
        self.assertEqual(config["dit"]["dbcache"], {"enabled": False})
        commands = infer.stage_commands(args, self.root / "run.yaml", config)
        self.assertEqual(len(commands), 3)
        self.assertIn("encode_infer.py", " ".join(commands[0]))
        self.assertIn("--task fl2va", " ".join(commands[0]))
        self.assertIn("--nproc_per_node=4", commands[0])
        self.assertIn("--nproc_per_node=8", commands[1])
        self.assertIn("--master_port=14771", commands[2])

        args = self.args("--image", str(self.image), "--prompt", "a test",
                         "--profile", "fullopt-dbcache")
        self.assertEqual(infer.effective_config(args)["dit"]["dbcache"], infer.CONSERVATIVE_CACHE)

    def test_prepared_inputs_and_config_origin(self):
        config_dir = self.h3 / "config"
        config_dir.mkdir()
        source = yaml.safe_load(infer.DEFAULT_CONFIG.read_text())
        source["model_root"] = "../assets/shared"
        source["dit"]["model_path"] = "../assets/transformer"
        config_path = config_dir / "sample.yaml"
        config_path.write_text(yaml.safe_dump(source))
        prompt = self.root / "prompt.pt"
        visual = self.root / "visual.pt"
        prompt.touch()
        visual.touch()
        args = infer.parse_args([
            "--h3-root", str(self.h3), "--config", str(config_path),
            "--output-dir", str(self.root / "output"),
            "--prompt-embeds", str(prompt), "--visual-cond", str(visual),
            "--w8a8-path", str(self.assets / "model.safetensors"),
            "--adaln-path", str(self.assets / "adaln.safetensors"),
            "--vae-int8-path", str(self.assets / "vae.safetensors"),
        ])
        config = infer.effective_config(args)
        self.assertEqual(config["model_root"], str((self.h3 / "../assets/shared").resolve()))
        self.assertEqual(config["dit"]["model_path"], str((self.h3 / "../assets/transformer").resolve()))
        infer.validate_inputs(args, config)
        self.assertEqual(len(infer.stage_commands(args, self.root / "run.yaml", config)), 2)

    def test_preflight_rejects_partial_inputs_and_wrong_schedule(self):
        prompt = self.root / "prompt.pt"
        prompt.touch()
        args = self.args("--prompt-embeds", str(prompt))
        with self.assertRaisesRegex(ValueError, "together"):
            infer.validate_inputs(args, infer.effective_config(args))

        args = self.args("--image", str(self.image), "--prompt", "test")
        # Schedule validation happens before the multi-GPU stages start.
        source = yaml.safe_load(infer.DEFAULT_CONFIG.read_text())
        source["dit"]["steps"] = 51
        path = self.root / "wrong.yaml"
        path.write_text(yaml.safe_dump(source))
        args.config = path
        with self.assertRaisesRegex(ValueError, "AdaLN bake"):
            infer.effective_config(args)

    def test_dry_run_writes_config_without_launching(self):
        args = self.args("--image", str(self.image), "--prompt", "a test",
                         "--profile", "fullopt-dbcache", "--dry-run")
        with patch.object(infer, "parse_args", return_value=args), patch.object(
            infer.subprocess, "run"
        ) as run:
            infer.main()
        run.assert_not_called()
        cfg = yaml.safe_load((self.root / "output" / "run.yaml").read_text())
        self.assertTrue(cfg["dit"]["dbcache"]["enabled"])


if __name__ == "__main__":
    unittest.main()
