"""Forward-AD smoke test for the diffusers SANA + PEFT stack."""

import unittest

try:
    import torch
    from diffusers import SanaTransformer2DModel
    from peft import LoraConfig, get_peft_model
    from models.compression_meanflow import (
        CompressionMeanFlow,
        SanaAdaLNConditioningEmbedding,
        _math_sdpa_context,
    )
    HAS_SANA_STACK = True
except ModuleNotFoundError:
    HAS_SANA_STACK = False


@unittest.skipUnless(
    HAS_SANA_STACK,
    "SANA/PyTorch/PEFT test dependencies unavailable",
)
class SanaMeanFlowJvpTest(unittest.TestCase):
    def test_tiny_sana_adaln_and_lora_support_meanflow_jvp(self):
        sana = SanaTransformer2DModel(
            in_channels=4,
            out_channels=4,
            num_attention_heads=2,
            attention_head_dim=4,
            num_layers=1,
            num_cross_attention_heads=2,
            cross_attention_head_dim=4,
            cross_attention_dim=8,
            caption_channels=8,
            mlp_ratio=1.0,
            dropout=0.0,
            sample_size=4,
            patch_size=1,
        )
        sana.time_embed.emb = SanaAdaLNConditioningEmbedding(
            sana.time_embed.emb,
            embedding_dim=8,
        )
        sana = get_peft_model(
            sana,
            LoraConfig(
                r=2,
                lora_alpha=2,
                target_modules=[
                    "to_q",
                    "to_k",
                    "to_v",
                    "to_out.0",
                ],
                bias="none",
                init_lora_weights="gaussian",
            ),
        )
        meanflow = CompressionMeanFlow(
            adaln_embedding_dim=8,
            frequency_embedding_size=8,
            norm_p=0.0,
        )
        with torch.no_grad():
            meanflow.adaln_interval_adapter.t_embedder.mlp[
                2
            ].weight.normal_(std=0.01)
            meanflow.adaln_interval_adapter.h_embedder.mlp[
                2
            ].weight.normal_(std=0.01)

        base_caption = torch.randn(1, 2, 8)
        native_timestep = torch.tensor([1.0])

        def velocity_fn(z, t, r, batch_indices=None):
            if batch_indices is None:
                caption = base_caption
                selected_timestep = native_timestep
            else:
                caption = base_caption.index_select(0, batch_indices)
                selected_timestep = native_timestep.index_select(
                    0,
                    batch_indices,
                )
            interval_embedding = meanflow.condition_adaln(t, r)
            packed_timestep = torch.cat(
                (
                    selected_timestep[:, None].to(interval_embedding),
                    interval_embedding,
                ),
                dim=1,
            )
            prediction = sana(
                z,
                encoder_hidden_states=caption,
                timestep=packed_timestep,
                return_dict=False,
            )[0]
            return 0.7 * prediction

        z = torch.randn(1, 4, 4, 4)
        direction = torch.randn_like(z)
        t = torch.tensor([0.8])
        r = torch.tensor([0.2])

        with _math_sdpa_context():
            _, jvp = torch.func.jvp(
                lambda z_value, t_value, r_value: velocity_fn(
                    z_value,
                    t_value,
                    r_value,
                ),
                (z, t, r),
                (
                    direction,
                    torch.ones_like(t),
                    torch.zeros_like(r),
                ),
            )
        self.assertTrue(torch.isfinite(jvp).all())

        eps = 1e-3
        with torch.no_grad():
            plus = velocity_fn(z + eps * direction, t + eps, r)
            minus = velocity_fn(z - eps * direction, t - eps, r)
        finite_difference = (plus - minus) / (2.0 * eps)
        torch.testing.assert_close(
            jvp,
            finite_difference,
            rtol=2e-2,
            atol=2e-2,
        )

        clean = torch.randn_like(z)
        compressed = torch.randn_like(z)
        loss, stats = meanflow.compute_loss(
            velocity_fn,
            clean,
            compressed,
            t=t,
            r=r,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(stats["correction_ratio"]))
        loss.backward()

        adapter_gradient = (
            meanflow.adaln_interval_adapter.t_embedder.mlp[2].weight.grad
        )
        self.assertIsNotNone(adapter_gradient)
        self.assertGreater(adapter_gradient.abs().sum().item(), 0.0)
        lora_gradients = [
            parameter.grad
            for name, parameter in sana.named_parameters()
            if "lora" in name and parameter.requires_grad
        ]
        self.assertTrue(
            any(
                gradient is not None
                and gradient.abs().sum() > 0
                for gradient in lora_gradients
            )
        )


if __name__ == "__main__":
    unittest.main()
