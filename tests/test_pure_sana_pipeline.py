import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PureSanaStructureTest(unittest.TestCase):
    def test_only_auxed_is_restored(self):
        removed_paths = [
            ROOT / "ELIC" / "base.py",
            ROOT / "ELIC" / "ckbd.py",
            ROOT / "ELIC" / "elic_official.py",
            ROOT / "models" / "foundation_models.py",
            ROOT / "models" / "compression_meanflow.py",
            ROOT / "scripts" / "measure_codec_factorization.py",
            ROOT / "scripts" / "evaluate_correction_ownership.py",
        ]
        self.assertFalse(any(path.exists() for path in removed_paths))

        core_files = [
            ROOT / "models" / "DiT_IC.py",
            ROOT / "models" / "latent_codec.py",
            ROOT / "models" / "scheduler.py",
            ROOT / "train_ddp.py",
            ROOT / "train_nogan_ddp.py",
            ROOT / "compress.py",
        ]
        banned_symbols = (
            "LatentConditionAlignment",
            "use_variance_guide",
            "use_self_dist",
            "state_dict_prompter",
        )
        source = "\n".join(path.read_text(encoding="utf-8") for path in core_files)
        for symbol in banned_symbols:
            self.assertNotIn(symbol, source)
        self.assertIn("use_aux_encoder", source)
        self.assertIn("use_aux_decoder", source)
        self.assertIn("AuxDecoder", source)

    def test_latent_codec_has_only_auxed_reconstruction_outputs(self):
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
            ["self", "latent", "aux_features"],
        )
        self.assertEqual(
            [argument.arg for argument in methods["compress"].args.args],
            ["self", "latent", "aux_features"],
        )

        codec_source = source_path.read_text(encoding="utf-8")
        self.assertIn("self.pre1", codec_source)
        self.assertIn("self.pre2", codec_source)
        self.assertNotIn('"scale":', codec_source)
        self.assertNotIn('"prompt":', codec_source)
        self.assertIn('"res":', codec_source)

    def test_scheduler_is_fixed_one_step_flow(self):
        source = (ROOT / "models" / "scheduler.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        scheduler_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "OneStepFlowScheduler"
        )
        step = next(
            node
            for node in scheduler_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "step"
        )
        self.assertEqual(
            [argument.arg for argument in step.args.args],
            ["self", "model_output", "sample"],
        )
        self.assertIn("return sample - sigma_t * model_output", source)

    def test_top_level_forward_returns_image_and_rate(self):
        source_path = ROOT / "models" / "DiT_IC.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        codec_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Codec"
        )
        forward = next(
            node
            for node in codec_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        returns = [node for node in ast.walk(forward) if isinstance(node, ast.Return)]
        self.assertEqual(len(returns), 1)
        self.assertIsInstance(returns[0].value, ast.Tuple)
        self.assertEqual(len(returns[0].value.elts), 2)

        sana_decode = next(
            node
            for node in codec_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "_decode_with_sana"
        )
        dit_calls = [
            node
            for node in ast.walk(sana_decode)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr == "DiT"
        ]
        self.assertEqual(len(dit_calls), 1)

    def test_frozen_codec_stays_in_eval_mode(self):
        source = (ROOT / "models" / "DiT_IC.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        codec_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Codec"
        )
        train_method = next(
            node
            for node in codec_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "train"
        )
        train_source = ast.get_source_segment(source, train_method)
        self.assertIn("self.codec.eval()", train_source)
        self.assertIn("self.aux_encoder.eval()", train_source)

    def test_aux_encoder_is_frozen_and_uses_unit_range_input(self):
        source = (ROOT / "models" / "DiT_IC.py").read_text(encoding="utf-8")
        train_source = (
            ROOT / "train_nogan_ddp.py"
        ).read_text(encoding="utf-8")
        self.assertIn("load_elic_analysis_encoder(elic_path)", source)
        self.assertIn("self.aux_encoder((image + 1) / 2)", source)
        self.assertIn("parameter.requires_grad", source)
        self.assertIn("model.module.assert_auxed_contract()", train_source)

    def test_entropy_aux_step_matches_official_nogan_semantics(self):
        source = (ROOT / "train_nogan_ddp.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        do_train = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "do_train"
        )
        train_source = ast.get_source_segment(source, do_train)

        main_backward = train_source.index("loss.backward()")
        main_clip = train_source.index("clip_grad_norm_(")
        global_clip_parameters = train_source.index(
            "model.parameters()",
            main_clip,
        )
        main_step = train_source.index("opt.step()")
        entropy_objective = train_source.index(
            "entropy_loss = (",
            main_step,
        )
        entropy_zero = train_source.index(
            "opt_entropy.zero_grad(set_to_none=True)",
            entropy_objective,
        )
        entropy_backward = train_source.index(
            "entropy_loss.backward()",
            entropy_zero,
        )
        entropy_step = train_source.index("opt_entropy.step()")
        entropy_log = train_source.index(
            "running_entropy_loss +="
        )
        next_train_step = train_source.index("train_steps += 1")

        self.assertLess(main_backward, main_clip)
        self.assertLess(main_clip, global_clip_parameters)
        self.assertLess(global_clip_parameters, main_step)
        self.assertLess(main_step, entropy_objective)
        self.assertLess(entropy_objective, entropy_zero)
        self.assertLess(entropy_zero, entropy_backward)
        self.assertLess(entropy_backward, entropy_step)
        self.assertLess(main_step, entropy_step)
        self.assertLess(entropy_step, entropy_log)
        self.assertNotIn(
            "opt_entropy.zero_grad",
            train_source[entropy_step:next_train_step],
        )
        self.assertIn("Aux:", train_source)
        self.assertNotIn("Entropy(post-step)", train_source)
        self.assertIn(
            "Entropy updates: official DiT-IC NoGAN order ",
            source,
        )
        self.assertIn(
            "Main-aligned entropy updates require "
            "train.accumulation_steps=1",
            train_source,
        )

    def test_main_and_entropy_optimizer_parameters_are_disjoint_and_complete(self):
        source = (ROOT / "train_nogan_ddp.py").read_text(encoding="utf-8")
        self.assertIn(
            "if main_parameter_ids & entropy_parameter_ids:",
            source,
        )
        self.assertIn(
            "if optimizer_parameter_ids != trainable_parameter_ids:",
            source,
        )
        self.assertIn(
            "Optimizer parameter partition does not match trainable model",
            source,
        )
        self.assertNotIn(
            "ENTROPY_BOTTLENECK_MAIN_OPTIMIZER_VERSION",
            source,
        )
        self.assertNotIn(
            "opt.state.pop(parameter, None)",
            source,
        )
        self.assertIn(
            "if not torch.isfinite(entropy_loss).item():",
            source,
        )

    def test_official_entropy_determinism_settings_are_active(self):
        train_source = (ROOT / "train_nogan_ddp.py").read_text(
            encoding="utf-8"
        )
        codec_source = (ROOT / "models" / "latent_codec.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("torch.backends.cudnn.benchmark = False", train_source)
        self.assertIn("torch.backends.cudnn.deterministic = True", train_source)
        self.assertIn(
            "torch.use_deterministic_algorithms(True, warn_only=True)",
            train_source,
        )
        self.assertIn(
            'os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"',
            train_source,
        )
        self.assertEqual(
            codec_source.count(
                "\n        torch.backends.cudnn.deterministic = True"
            ),
            2,
        )
        self.assertEqual(
            codec_source.count(
                "\n        torch.backends.cudnn.deterministic = False"
            ),
            2,
        )

    def test_legacy_entropy_optimizer_state_is_not_loaded_blindly(self):
        source = (ROOT / "train_nogan_ddp.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        do_train = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "do_train"
        )
        train_source = ast.get_source_segment(source, do_train)

        self.assertIn(
            "saved_entropy_version == ENTROPY_OPTIMIZER_VERSION",
            train_source,
        )
        self.assertIn(
            "saved_entropy_signature == entropy_parameter_signature",
            train_source,
        )
        self.assertIn(
            "opt_entropy.load_state_dict(entropy_state)",
            train_source,
        )
        self.assertIn(
            "Resetting legacy/incompatible entropy optimizer moments",
            train_source,
        )

    def test_resume_records_and_checks_entropy_state(self):
        source = (ROOT / "train_nogan_ddp.py").read_text(encoding="utf-8")
        self.assertIn('"entropy_bottleneck_loss": float(', source)
        self.assertIn(
            "Restored entropy-bottleneck loss before the first update",
            source,
        )
        self.assertIn(
            "Entropy-bottleneck state changed while restoring the checkpoint",
            source,
        )
        self.assertIn("'--resume_checkpoint'", source)


if __name__ == "__main__":
    unittest.main()
