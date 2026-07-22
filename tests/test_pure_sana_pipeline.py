import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PureSanaStructureTest(unittest.TestCase):
    def test_removed_implementations_are_absent(self):
        removed_paths = [
            ROOT / "ELIC" / "base.py",
            ROOT / "ELIC" / "ckbd.py",
            ROOT / "ELIC" / "elic_official.py",
            ROOT / "models" / "foundation_models.py",
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
            "use_aux_encoder",
            "use_aux_decoder",
            "AuxDecoder",
            "aux_codec",
            "state_dict_prompter",
        )
        source = "\n".join(path.read_text(encoding="utf-8") for path in core_files)
        for symbol in banned_symbols:
            self.assertNotIn(symbol, source)

    def test_latent_codec_has_one_reconstruction_output(self):
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
            ["self", "latent"],
        )
        self.assertEqual(
            [argument.arg for argument in methods["compress"].args.args],
            ["self", "latent"],
        )

        codec_source = source_path.read_text(encoding="utf-8")
        self.assertIn("self.pre1", codec_source)
        self.assertNotIn("self.pre2", codec_source)
        self.assertNotIn('"scale":', codec_source)
        self.assertNotIn('"prompt":', codec_source)
        self.assertNotIn('"res":', codec_source)

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

    def test_gs_fusion_initialization_preserves_shared_rng_stream(self):
        source = (ROOT / "models" / "DiT_IC.py").read_text(encoding="utf-8")
        self.assertIn("with torch.random.fork_rng(devices=[]):", source)

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


if __name__ == "__main__":
    unittest.main()
