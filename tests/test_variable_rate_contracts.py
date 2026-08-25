import ast
import io
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class VariableRatePacketTest(unittest.TestCase):
    def test_packet_round_trip_carries_quality_and_counts_header(self):
        from codec_packet import CodecPacket, read_codec_packet, write_codec_packet

        packet = CodecPacket(
            strings=[[b"y-stream"], [b"z-stream"]],
            shape=(3, 5),
            quality_index=4,
        )
        buffer = io.BytesIO()
        written = write_codec_packet(buffer, packet)

        self.assertEqual(written, len(buffer.getvalue()))
        self.assertGreater(written, len(b"y-stream") + len(b"z-stream"))
        buffer.seek(0)
        self.assertEqual(read_codec_packet(buffer), packet)

    def test_packet_rejects_invalid_quality(self):
        from codec_packet import CodecPacket

        with self.assertRaisesRegex(ValueError, "quality_index"):
            CodecPacket(strings=[[b"y"], [b"z"]], shape=(1, 1), quality_index=6)


class VariableRateStructureTest(unittest.TestCase):
    def test_latent_codec_requires_quality_for_variable_rate_paths(self):
        source_path = ROOT / "models" / "latent_codec.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        codec_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "LatentCodec"
        )
        methods = {
            node.name: node
            for node in codec_class.body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertEqual(
            [argument.arg for argument in methods["forward"].args.args],
            ["self", "latent", "quality_index"],
        )
        self.assertEqual(
            [argument.arg for argument in methods["compress"].args.args],
            ["self", "latent", "quality_index"],
        )

        source = source_path.read_text(encoding="utf-8")
        self.assertIn("class QualityController", source)
        self.assertIn("self.quality_controller.encoder_scale", source)
        self.assertIn("self.quality_controller.decoder_scale", source)

    def test_multirate_plain_samples_one_ddp_quality_per_step(self):
        source = (ROOT / "train_nogan_ddp.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        sample_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "sample_shared_quality_index"
        )
        sample_source = ast.get_source_segment(source, sample_function)
        self.assertIn("dist.broadcast", sample_source)
        self.assertIn("src=0", sample_source)

        do_train = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "do_train"
        )
        train_source = ast.get_source_segment(source, do_train)
        sample_position = train_source.index("sample_shared_quality_index(")
        forward_position = train_source.index("model(x, quality_index)")
        step_position = train_source.index("opt.step()")
        self.assertLess(sample_position, forward_position)
        self.assertLess(forward_position, step_position)

    def test_multirate_config_freezes_codec_vae_and_text_projection(self):
        config_path = ROOT / "configs" / "multirate_plain_dit.yaml"
        config = config_path.read_text(encoding="utf-8")
        self.assertIn("freeze_codec: true", config.lower())
        self.assertIn("train_vae_lora: false", config.lower())
        self.assertIn("train_text_projection: false", config.lower())
        self.assertIn("variable_rate: true", config.lower())
        self.assertIn("per_gpu_batch_size: 8", config.lower())

        contract = json.loads(
            (ROOT / "docs" / "baseline_contract.json").read_text(encoding="utf-8")
        )
        self.assertTrue(contract["freeze_codec"])
        self.assertFalse(contract["train_vae_decoder_lora"])
        self.assertFalse(contract["train_text_projection"])

    def test_launcher_supports_one_or_two_gpus_with_batch_overrides(self):
        launcher = (
            ROOT / "scripts" / "run_variable_rate_train.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("num_gpus < 1 || num_gpus > 2", launcher)
        self.assertIn("--nproc_per_node=\"${num_gpus}\"", launcher)
        self.assertIn("per_gpu_batch_size=${PER_GPU_BATCH_SIZE:-8}", launcher)
        self.assertIn("--per_gpu_batch_size", launcher)
        self.assertIn("--accumulation_steps", launcher)

        training_source = (ROOT / "train_nogan_ddp.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('backend = "nccl"', training_source)
        self.assertNotIn(
            'torch.cuda.device_count() > 1 else "gloo"', training_source
        )

    def test_lic_configs_define_automatic_two_phase_schedule(self):
        pilot = (ROOT / "configs" / "variable_rate_lic_pilot.yaml").read_text(
            encoding="utf-8"
        )
        main = (ROOT / "configs" / "variable_rate_lic_main.yaml").read_text(
            encoding="utf-8"
        )
        for config in (pilot, main):
            self.assertIn("phase_a:", config)
            self.assertIn("phase_b:", config)
            self.assertIn("latent_mse_weight: 1.0", config)
            self.assertIn("pixel_l1_weight: 1.0", config)
            self.assertIn("lpips_weight: 1.0", config)
            self.assertIn("dists_weight: 1.0", config)
            self.assertNotIn("image_mse_weight", config)
        self.assertEqual(pilot.count("steps: 2500"), 2)
        self.assertEqual(main.count("steps: 50000"), 2)

    def test_lic_phase_a_skips_decode_and_phase_b_drops_latent_loss(self):
        source_path = ROOT / "train_variable_rate_lic_ddp.py"
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        stage_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "VariableRateLICStage"
        )
        forward = next(
            node
            for node in stage_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        forward_source = ast.get_source_segment(source, forward)
        self.assertIn("if decode_image:", forward_source)
        self.assertIn("self.vae.decode(", forward_source)

        train_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "train"
        )
        train_source = ast.get_source_segment(source, train_function)
        phase_a_branch = train_source.index("if current_phase == PHASE_A:")
        phase_b_l1 = train_source.index("pixel_l1 = F.l1_loss", phase_a_branch)
        next_loss = train_source.index("loss = rate + current_lambda", phase_b_l1)
        phase_b_source = train_source[phase_b_l1:next_loss]
        self.assertIn('phase_config["pixel_l1_weight"]', phase_b_source)
        self.assertIn('phase_config["lpips_weight"]', phase_b_source)
        self.assertIn('phase_config["dists_weight"]', phase_b_source)
        self.assertNotIn("latent_mse", phase_b_source)

    def test_lic_checkpoint_records_phase_for_safe_resume(self):
        source = (ROOT / "train_variable_rate_lic_ddp.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"training_phase": training_phase(', source)
        self.assertIn('"phase_step": phase_step(', source)
        self.assertIn('"phase_a_steps": phase_a_steps', source)
        self.assertIn('"phase_b_steps": phase_b_steps', source)
        self.assertIn('checkpoint_dir / "phase_a_complete.pt"', source)
        self.assertIn(
            '"phase A complete; automatically starting phase B"', source
        )


if __name__ == "__main__":
    unittest.main()
