import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def method_source(path, class_name, method_name):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return ast.get_source_segment(source, method_node)


class GanDiscriminatorContractTest(unittest.TestCase):
    def test_shared_loss_keeps_real_and_fake_batchnorm_stats_separate(self):
        source = method_source(
            ROOT / "models" / "losses.py",
            "LPIPSWithDiscriminator",
            "forward",
        )

        self.assertIn("logits_real = discriminator(real_input)", source)
        self.assertIn("logits_fake = discriminator(fake_input)", source)
        self.assertNotIn("discriminator(discriminator_input).chunk(2)", source)

    def test_stage1_loss_keeps_real_and_fake_batchnorm_stats_separate(self):
        source = method_source(
            ROOT / "train_variable_rate_lic_ddp.py",
            "PhaseBAdversarialLoss",
            "discriminator_loss",
        )

        self.assertIn(
            "logits_real = discriminator(reference.contiguous().detach())",
            source,
        )
        self.assertIn(
            "logits_fake = discriminator(reconstruction.contiguous().detach())",
            source,
        )
        self.assertNotIn("torch.cat", source)


if __name__ == "__main__":
    unittest.main()
