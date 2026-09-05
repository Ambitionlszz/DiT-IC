import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class GanDiscriminatorContractTest(unittest.TestCase):
    def test_real_and_fake_batchnorm_stats_are_separate(self):
        path = ROOT / "models" / "losses.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        loss_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "LPIPSWithDiscriminator"
        )
        forward = next(
            node
            for node in loss_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        forward_source = ast.get_source_segment(source, forward)

        self.assertIn("logits_real = discriminator(real_input)", forward_source)
        self.assertIn("logits_fake = discriminator(fake_input)", forward_source)
        self.assertNotIn(
            "discriminator(discriminator_input).chunk(2)", forward_source
        )


if __name__ == "__main__":
    unittest.main()
