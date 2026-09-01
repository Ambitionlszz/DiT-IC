import ast
import io
import json
import re
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

    def test_inference_only_falls_back_when_packet_magic_is_absent(self):
        source = (ROOT / "compress.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "decompress_one_image"
        )
        function_source = ast.get_source_segment(source, function)
        self.assertIn("magic == PACKET_MAGIC", function_source)
        self.assertNotIn("except ValueError", function_source)


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
        self.assertIn("self.encoder_scale_endpoints", source)
        self.assertIn("self.decoder_scale_endpoints", source)
        self.assertIn("def _interpolate(self, endpoints, quality_index)", source)
        self.assertIn("torch.log(minimum)", source)
        self.assertIn("torch.log(maximum) - torch.log(minimum)", source)
        self.assertNotIn("encoder_scale_table", source)
        self.assertNotIn("decoder_scale_table", source)

    def test_latent_codec_uses_complete_dcvc_fm_quantization_control(self):
        source_path = ROOT / "models" / "latent_codec.py"
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        codec_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "LatentCodec"
        )
        methods = {
            node.name: ast.get_source_segment(source, node)
            for node in codec_class.body
            if isinstance(node, ast.FunctionDef)
        }

        prior_source = methods["_first_prior_parameters"]
        self.assertIn("M * 2 + (2 if self.variable_rate else 0)", source)
        self.assertIn("params[:, :2, :, :]", prior_source)
        self.assertIn("torch.sigmoid(quantization_control) * 1.5 + 0.5", prior_source)
        self.assertIn(
            "scales, means = params[:, 2:, :, :].chunk(2, 1)", prior_source
        )

        forward_source = methods["forward"]
        self.assertIn("y_quant = y * w_e", forward_source)
        self.assertEqual(forward_source.count("self.forward_with_mask(y_quant"), 4)
        self.assertIn("y_hat = y_hat * w_d", forward_source)

        compress_source = methods["compress"]
        self.assertIn("y_quant = y * w_e", compress_source)
        self.assertEqual(
            compress_source.count("self.compress_group_with_mask("), 4
        )
        self.assertEqual(compress_source.count(", y_quant,"), 4)

        decompress_source = methods["decompress"]
        self.assertIn("_, w_d, means_0_supp", decompress_source)
        self.assertIn("y_hat = y_hat * w_d", decompress_source)

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

    def test_multirate_config_freezes_codec_and_trains_decoder_dit_components(self):
        config_path = ROOT / "configs" / "multirate_plain_dit.yaml"
        config = config_path.read_text(encoding="utf-8")
        self.assertIn("freeze_codec: true", config.lower())
        self.assertIn("vae_lora: true", config.lower())
        self.assertIn("train_vae_lora: true", config.lower())
        self.assertIn("decoder_lora_only: true", config.lower())
        self.assertIn("lora_rank_vae: 32", config.lower())
        self.assertIn("train_text_projection: true", config.lower())
        self.assertIn("dit_lora: true", config.lower())
        self.assertIn("variable_rate: true", config.lower())
        self.assertIn("per_gpu_batch_size: 8", config.lower())

        smoke = (
            ROOT / "configs" / "multirate_plain_dit_smoke.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("decoder_lora_only: true", smoke.lower())
        self.assertIn("lora_rank_vae: 32", smoke.lower())

        contract = json.loads(
            (ROOT / "docs" / "baseline_contract.json").read_text(encoding="utf-8")
        )
        self.assertTrue(contract["freeze_codec"])
        self.assertFalse(contract["train_vae_decoder_lora"])
        self.assertTrue(contract["train_text_projection"])
        self.assertEqual(contract["dit_lora_initialization"], "fresh")

    def test_launcher_supports_one_or_two_gpus_with_batch_overrides(self):
        launcher = (
            ROOT / "scripts" / "run_variable_rate_train.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("num_gpus < 1 || num_gpus > 2", launcher)
        self.assertIn("--nproc_per_node=\"${num_gpus}\"", launcher)
        self.assertNotIn("default_per_gpu_batch_size=16", launcher)
        self.assertEqual(launcher.count("default_per_gpu_batch_size=8"), 12)
        self.assertIn(
            "per_gpu_batch_size=${PER_GPU_BATCH_SIZE:-${default_per_gpu_batch_size}}",
            launcher,
        )
        self.assertIn("--per_gpu_batch_size", launcher)
        self.assertIn("--accumulation_steps", launcher)
        self.assertIn("plain-smoke)", launcher)
        self.assertIn("lic-adv-phase2)", launcher)
        self.assertIn("plain-adv)", launcher)
        self.assertIn("e2e-variable-nogan)", launcher)
        self.assertIn("e2e-variable-gan)", launcher)

        training_source = (ROOT / "train_nogan_ddp.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("'--codec_path'", training_source)
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
            self.assertIn("codec_initialization: scratch", config)
            self.assertNotIn("source_checkpoint", config)
            self.assertNotIn("vae_lora", config)
            self.assertIn("phase_a:", config)
            self.assertIn("phase_b:", config)
            self.assertIn("latent_mse_weight: 1.0", config)
            self.assertIn("pixel_mse_weight: 1.0", config)
            self.assertIn("lpips_weight: 1.0", config)
            self.assertIn("dists_weight: 0.2", config)
            self.assertIn("per_gpu_batch_size: 8", config)
            self.assertEqual(
                config.count(
                    "lambda_values: [16.0, 8.0, 4.0, 2.0, 1.0, 0.5]"
                ),
                2,
            )
            self.assertNotIn("lambda_min", config)
            self.assertNotIn("lambda_max", config)
            self.assertNotIn("image_mse_weight", config)
        self.assertEqual(pilot.count("steps: 2500"), 2)
        self.assertIn("steps: 50000", main)
        self.assertIn("steps: 100000", main)
        self.assertIn("exp_name: variable_rate_lic_v5_pilot", pilot)
        self.assertIn("exp_name: variable_rate_lic_v5_main", main)

        smoke = (ROOT / "configs" / "variable_rate_lic_smoke.yaml").read_text(
            encoding="utf-8"
        )
        self.assertEqual(smoke.count("steps: 20"), 2)
        self.assertIn("max_steps: 40", smoke)
        self.assertIn("per_gpu_batch_size: 8", smoke)
        self.assertIn("codec_initialization: scratch", smoke)
        self.assertNotIn("source_checkpoint", smoke)
        self.assertIn("exp_name: variable_rate_lic_v5_smoke", smoke)

        plain_smoke = (
            ROOT / "configs" / "multirate_plain_dit_smoke.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("max_steps: 20", plain_smoke)
        self.assertIn("per_gpu_batch_size: 8", plain_smoke)
        self.assertIn("freeze_codec: true", plain_smoke)
        self.assertIn("train_text_projection: true", plain_smoke)

        training_source = (ROOT / "train_variable_rate_lic_ddp.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('phase_config["lambda_values"]', training_source)
        self.assertNotIn("log_interpolated_lambda", training_source)
        self.assertIn(
            "loss = rate_distortion_loss(distortion, rate, current_lambda)",
            training_source,
        )
        self.assertIn("return distortion + rate_lambda * rate", training_source)

    def test_main_can_require_an_explicit_calibration_override(self):
        source = (ROOT / "train_variable_rate_lic_ddp.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'parser.add_argument("--calibration_confirmed", action="store_true")',
            source,
        )
        self.assertIn(
            "calibration_confirmed=args.calibration_confirmed", source
        )

    def test_v5_then_dit_launcher_validates_and_chains_checkpoint(self):
        source = (ROOT / "scripts" / "run_v5_then_dit.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('checkpoint.get("schema_version", -1)) != 8', source)
        self.assertIn('checkpoint.get("stage") != "variable_quality_codec"', source)
        self.assertIn(
            '"dcvc_fm_endpoint_interp_qe_qd_we_wd_v2"', source
        )
        self.assertIn('"pixel_perceptual_rd"', source)
        self.assertIn("--resume_checkpoint", source)
        self.assertIn("source_checkpoint_sha256", source)
        self.assertIn('expected_steps = int(config["train"]["max_steps"])', source)
        self.assertIn('checkpoint.get("stage") != "multirate_reconstruction"', source)
        self.assertIn('bash scripts/run_variable_rate_train.sh plain', source)
        self.assertIn('--codec_path "${lic_checkpoint}"', source)
        self.assertIn('Plain DiT exited without a complete checkpoint', source)
        self.assertIn("evaluate_lic_for_handoff", source)
        self.assertIn("lic_handoff.json", source)
        self.assertIn('lic_checkpoint="${best_checkpoint}"', source)
        self.assertNotIn("GATE_MIN_BPP_SPAN", source)
        self.assertNotIn("select-lic", source)
        self.assertNotIn("both real-rANS gates", source)

    def test_v5_launcher_scans_each_candidate_before_using_global_step(self):
        source = (ROOT / "scripts" / "run_v5_then_dit.sh").read_text(
            encoding="utf-8"
        )
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", source, re.DOTALL)
        checkpoint_blocks = [block for block in blocks if "global_step" in block]
        self.assertEqual(len(checkpoint_blocks), 1)
        tree = ast.parse(checkpoint_blocks[0])
        loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)]
        assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "global_step"
                for target in node.targets
            )
        ]
        self.assertTrue(
            any(
                any(node is assignment for node in ast.walk(loop))
                for loop in loops
                for assignment in assignments
            )
        )

    def test_two_p0_launcher_diagnoses_runs_sequentially_and_bundles_evidence(self):
        source = (
            ROOT / "scripts" / "run_two_p0_then_bundle.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('if [[ "${DIAGNOSE_ONLY:-0}" == "1" ]]', source)
        self.assertIn("[t]rain_nogan_ddp.py", source)
        self.assertLess(
            source.index('"V5 P0" "${v5_repo}"'),
            source.index('"E2E P0" "${e2e_repo}"'),
        )
        self.assertIn('e2e_gpu=${2:-$1}', source)
        self.assertIn('bash "${pipeline}" "${run_gpu}"', source)
        self.assertIn('"E2E P0" "${e2e_repo}"', source)
        self.assertIn('&\n  v5_pipeline_pid=$!', source)
        self.assertIn('&\n  e2e_pipeline_pid=$!', source)
        self.assertIn('wait "${v5_pipeline_pid}"', source)
        self.assertIn('wait "${e2e_pipeline_pid}"', source)
        self.assertIn('report.get("accepted") is not True', source)
        self.assertIn('get("byte_identical") is True', source)
        self.assertIn('sha256sum "${repo}/checkpoints/${p0_exp}', source)
        self.assertIn('tar -C "${bundle_parent}" -czf', source)
        self.assertNotIn('/rec"', source)

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
        phase_b_mse = train_source.index("pixel_mse = F.mse_loss", phase_a_branch)
        next_loss = train_source.index(
            "loss = rate_distortion_loss(", phase_b_mse
        )
        phase_b_source = train_source[phase_b_mse:next_loss]
        self.assertIn('phase_config["pixel_mse_weight"]', phase_b_source)
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
        self.assertIn('CHECKPOINT_SCHEMA_VERSION = 8', source)
        self.assertIn(
            'VARIABLE_RATE_MECHANISM = "dcvc_fm_endpoint_interp_qe_qd_we_wd_v2"',
            source,
        )
        self.assertIn('"variable_rate_mechanism": VARIABLE_RATE_MECHANISM', source)
        self.assertIn('"objective_contract": objective_contract(config)', source)
        self.assertIn('"dists_input_range": "minus_one_to_one"', source)
        self.assertIn(
            'resume.get("initialization_contract") != initialization_contract(config)',
            source,
        )
        self.assertIn('"model": {"state_dict_codec": codec_state}', source)
        self.assertNotIn("load_source_checkpoint", source)
        self.assertIn("evaluate_with_codec_state(", source)
        self.assertIn(
            "optimizer, entropy_optimizer, scheduler = build_training_state(",
            source,
        )
        self.assertIn(
            "ema_codec = deepcopy(model.module.codec.state_dict())", source
        )
        self.assertIn(
            '"automatically starting phase B"', source
        )
        self.assertIn('"loss_form": "distortion_plus_rate_lambda"', source)
        self.assertIn("validation_quality_scales=%s", source)
        self.assertIn("validation_bpp_span=%.8f", source)

    def test_adversarial_phase_b_matches_official_dit_ic_contract(self):
        source = (ROOT / "train_variable_rate_lic_ddp.py").read_text(
            encoding="utf-8"
        )
        stage1_config = (
            ROOT / "configs" / "variable_rate_lic_adv_phase_b.yaml"
        ).read_text(encoding="utf-8")
        stage2_config = (
            ROOT / "configs" / "multirate_plain_dit_adv.yaml"
        ).read_text(encoding="utf-8")

        self.assertIn("class PhaseBAdversarialLoss", source)
        self.assertIn("NLayerDiscriminator", source)
        self.assertIn("hinge_d_loss", source)
        self.assertIn("generator_loss = -torch.mean(logits_fake)", source)
        self.assertIn("torch.norm(reconstruction_grads)", source)
        self.assertIn("torch.norm(generator_grads) + 1e-4", source)
        self.assertIn("ADVERSARIAL_CHECKPOINT_SCHEMA_VERSION = 9", source)
        self.assertIn("is_phase_a_handoff", source)
        self.assertIn('int(resume.get("global_step", -1)) == phase_a_steps', source)
        self.assertIn('"phase_a_handoff_sha256"', source)

        for config in (stage1_config, stage2_config):
            self.assertIn("disc_start: 0", config)
            self.assertIn("disc_weight: 0.5", config)
            self.assertIn("disc_factor: 1.0", config)
            self.assertIn("disc_num_layers: 3", config)
            self.assertIn("disc_in_channels: 3", config)
            self.assertIn("disc_loss: hinge", config)
        self.assertIn("enabled: true", stage1_config)
        self.assertIn("freeze_codec: true", stage2_config)
        self.assertIn("max_steps: 100000", stage2_config)

    def test_adversarial_pipeline_chains_phase_b_stage2_and_evaluation(self):
        source = (
            ROOT / "scripts" / "run_adv_phase2_then_stage2.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("validate_phase_a_checkpoint", source)
        self.assertIn("phase_a_complete.pt", source)
        self.assertIn("lic-adv-phase2", source)
        self.assertIn("plain-adv", source)
        self.assertIn("phase_a_handoff_sha256", source)
        self.assertIn("source_checkpoint_sha256", source)
        self.assertIn("run_stage1_eval", source)
        self.assertIn("run_variable_rate_eval.sh", source)
        self.assertIn("--resume_checkpoint", source)
        self.assertNotIn("variable_rate_gate.py", source)

    def test_end_to_end_variable_rate_nogan_then_gan_contract(self):
        nogan_config = (
            ROOT / "configs" / "multirate_e2e_variable_nogan.yaml"
        ).read_text(encoding="utf-8")
        gan_config = (
            ROOT / "configs" / "multirate_e2e_variable_gan.yaml"
        ).read_text(encoding="utf-8")
        training_source = (ROOT / "train_nogan_ddp.py").read_text(
            encoding="utf-8"
        )
        model_source = (ROOT / "models" / "DiT_IC.py").read_text(
            encoding="utf-8"
        )
        pipeline_source = (
            ROOT / "scripts" / "run_e2e_variable_nogan_then_gan.sh"
        ).read_text(encoding="utf-8")

        for config in (nogan_config, gan_config):
            self.assertIn("freeze_codec: false", config)
            self.assertIn("rate_weights: [16.0, 8.0, 4.0, 2.0, 1.0, 0.5]", config)
            self.assertIn("train_text_projection: true", config)
            self.assertIn("decoder_lora_only: true", config)
        self.assertIn("e2e_phase: nogan", nogan_config)
        self.assertIn("max_steps: 200000", nogan_config)
        self.assertIn("disc_weight: 0.0", nogan_config)
        self.assertIn("e2e_phase: gan", gan_config)
        self.assertIn("max_steps: 50000", gan_config)
        self.assertIn("disc_start: 0", gan_config)
        self.assertIn("disc_weight: 0.5", gan_config)
        self.assertIn("end_to_end_variable_rate_contract", training_source)
        self.assertIn("quality_rate_weight", training_source)
        self.assertIn('"e2e_variable_rate_contract"', training_source)
        self.assertIn('model.vae.load_state_dict(ema["state_dict_vae"], strict=False)', training_source)
        self.assertIn("assert_end_to_end_variable_rate_contract", model_source)
        self.assertIn("e2e-variable-nogan", pipeline_source)
        self.assertIn("e2e-variable-gan", pipeline_source)
        self.assertIn("last_ckpt.pt", pipeline_source)
        self.assertIn("source_checkpoint_sha256", pipeline_source)
        self.assertIn("run_variable_rate_eval.sh", pipeline_source)

    def test_phase1_warmup_e2e_schedule_and_pipeline_contract(self):
        warmup = (
            ROOT / "configs" / "multirate_e2e_latent_warmup.yaml"
        ).read_text(encoding="utf-8")
        nogan = (
            ROOT / "configs" / "multirate_e2e_warmup_nogan.yaml"
        ).read_text(encoding="utf-8")
        gan = (
            ROOT / "configs" / "multirate_e2e_warmup_gan.yaml"
        ).read_text(encoding="utf-8")
        lic_training = (ROOT / "train_variable_rate_lic_ddp.py").read_text(
            encoding="utf-8"
        )
        e2e_training = (ROOT / "train_nogan_ddp.py").read_text(
            encoding="utf-8"
        )
        model_source = (ROOT / "models" / "DiT_IC.py").read_text(
            encoding="utf-8"
        )
        pipeline = (
            ROOT / "scripts" / "run_e2e_warmup_nogan_then_gan.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("stop_after_phase_a: true", warmup)
        self.assertIn("steps: 50000", warmup)
        self.assertIn("max_steps: 50000", warmup)
        self.assertIn("training_mode: phase1_warmup_e2e", warmup)
        self.assertIn("max_steps: 150000", nogan)
        self.assertIn("freeze_codec: false", nogan)
        self.assertIn("freeze_quality_endpoints: true", nogan)
        self.assertIn("max_steps: 50000", gan)
        self.assertIn("freeze_codec: false", gan)
        self.assertIn("freeze_quality_endpoints: true", gan)
        self.assertIn("stop_after_phase_a", lic_training)
        self.assertIn("set_quality_endpoints_trainable", e2e_training)
        self.assertIn("require_frozen_quality_endpoints", model_source)
        self.assertIn("eval_variable_rate_lic_direct.py", pipeline)
        self.assertNotIn("variable_rate_gate.py", pipeline)
        self.assertIn("continuing to NoGAN", pipeline)
        self.assertIn("continuing to GAN", pipeline)
        self.assertIn("e2e-warmup-nogan", pipeline)
        self.assertIn("e2e-warmup-gan", pipeline)

    def test_joint_multirate_loss_matches_baseline_lambda_convention(self):
        source = (ROOT / "train_variable_rate_lic_ddp.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "rate_distortion_loss"
        )
        function_source = ast.get_source_segment(source, function)
        self.assertIn("return distortion + rate_lambda * rate", function_source)
        self.assertNotIn("distortion / rate_lambda", function_source)

    def test_lic_dists_training_matches_baseline_minus_one_to_one_range(self):
        losses_source = (ROOT / "models" / "losses.py").read_text(
            encoding="utf-8"
        )
        training_source = (ROOT / "train_variable_rate_lic_ddp.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "self.dists(\n                    inputs.contiguous(),", losses_source
        )
        self.assertIn(
            "dists_loss = self.dists(reference, reconstruction).mean()",
            training_source,
        )
        self.assertNotIn("def _dists_inputs", losses_source)

    def test_direct_lic_evaluator_round_trips_serialized_rans_packet(self):
        source = (
            ROOT / "scripts" / "eval_variable_rate_lic_direct.py"
        ).read_text(encoding="utf-8")
        self.assertIn("stage.codec.compress(clean_latent, quality_index)", source)
        self.assertIn("write_codec_packet(file, packet)", source)
        self.assertIn("stored_packet = read_codec_packet(file)", source)
        self.assertIn("stage.codec.decompress(stored_packet)", source)
        self.assertIn("stage.vae.decode(", source)
        self.assertIn(
            "load_config(args.config, calibration_confirmed=True)", source
        )
        self.assertNotIn("models.DiT_IC", source)
        self.assertNotIn("source_checkpoint", source)

    def test_plain_rejects_a_lic_checkpoint_without_phase_b_training(self):
        source = (ROOT / "models" / "DiT_IC.py").read_text(encoding="utf-8")
        self.assertIn('checkpoint_stage == "variable_quality_codec"', source)
        self.assertIn(
            'int(checkpoint.get("schema_version", 0)) not in (8, 9)', source
        )
        self.assertIn(
            '!= "dcvc_fm_endpoint_interp_qe_qd_we_wd_v2"', source
        )
        self.assertIn("global_step <= phase_a_steps", source)
        self.assertIn(
            "multi-rate Plain requires a LIC checkpoint trained ", source
        )
        self.assertIn("keeping fresh DiT LoRA", source)
        self.assertIn('self.codec.load_state_dict(state["state_dict_codec"]', source)

    def test_multirate_plain_selects_best_checkpoint_with_ema_weights(self):
        source = (ROOT / "train_nogan_ddp.py").read_text(encoding="utf-8")
        self.assertIn("test_loss = evaluate_multirate_ema(", source)
        self.assertIn('ema["state_dict_DiT"]', source)
        self.assertIn('ema["state_dict_text_proj"]', source)
        self.assertIn("finally:", source)
        self.assertIn("model.DiT.load_state_dict(live_dit", source)
        self.assertIn("model.text_proj.load_state_dict(live_text_proj", source)

    def test_multirate_plain_keeps_frozen_codec_exact_in_ema(self):
        source = (ROOT / "train_nogan_ddp.py").read_text(encoding="utf-8")
        self.assertIn('exact_parts=("state_dict_codec",)', source)
        self.assertIn("ema_codec_sha256", source)
        self.assertIn("codec_sha256 != ema_codec_sha256", source)
        self.assertIn(
            'VARIABLE_RATE_MECHANISM = "dcvc_fm_endpoint_interp_qe_qd_we_wd_v2"',
            source,
        )
        model_source = (ROOT / "models" / "DiT_IC.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'int(checkpoint.get("schema_version", 0)) != 3',
            model_source,
        )
        self.assertIn(
            'strict=checkpoint_stage == "multirate_reconstruction"',
            model_source,
        )


if __name__ == "__main__":
    unittest.main()
