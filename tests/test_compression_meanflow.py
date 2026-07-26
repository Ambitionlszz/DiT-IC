import unittest
from unittest import mock

try:
    import torch
    import torch.nn as nn
    from models.compression_meanflow import (
        ADALN_CONDITIONING,
        ADALN_CONDITIONING_ID,
        CODEC_OBSERVABLE_MODE_ID,
        CODEC_OBSERVABLE_OBJECTIVE,
        MEANFLOW_MODE,
        MEANFLOW_MODE_ID,
        ORIGINAL_OBJECTIVE,
        AdaLNIntervalAdapter,
        CompressionMeanFlow,
        SanaAdaLNConditioningEmbedding,
    )
    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False


if HAS_TORCH:
    class ToyVelocity(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.7))

        def forward(self, z, t, r):
            shape = (z.shape[0],) + (1,) * (z.ndim - 1)
            return (
                self.scale * z
                + t.reshape(shape)
                - 0.2 * r.reshape(shape)
            )


@unittest.skipUnless(HAS_TORCH, "PyTorch unavailable")
class AdaLNConditioningTest(unittest.TestCase):
    def test_adapter_is_zero_initialized(self):
        adapter = AdaLNIntervalAdapter(
            hidden_size=8,
            frequency_embedding_size=4,
        )
        t = torch.tensor([0.25, 0.75])
        h = torch.tensor([0.1, 0.5])

        torch.testing.assert_close(
            adapter(t, h),
            torch.zeros(2, 8),
            rtol=0,
            atol=0,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in adapter.parameters()),
            224,
        )

    def test_official_size_adapter_parameter_count(self):
        adapter = AdaLNIntervalAdapter(
            hidden_size=1152,
            frequency_embedding_size=256,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in adapter.parameters()),
            3_248_640,
        )

    def test_sana_wrapper_adds_the_interval_condition(self):
        class BaseEmbedding(nn.Module):
            def forward(self, timestep, *args, **kwargs):
                return timestep[:, None].repeat(1, 4)

        wrapper = SanaAdaLNConditioningEmbedding(
            BaseEmbedding(),
            embedding_dim=4,
        )
        native_timestep = torch.tensor([[2.0], [3.0]])
        interval = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
            ]
        )

        combined = wrapper(
            torch.cat((native_timestep, interval), dim=1)
        )
        torch.testing.assert_close(
            combined,
            native_timestep.expand(-1, 4) + interval,
        )

    def test_only_adaln_5a_is_supported(self):
        module = CompressionMeanFlow(
            adaln_embedding_dim=8,
            frequency_embedding_size=4,
        )
        self.assertEqual(module.mode, MEANFLOW_MODE)
        self.assertEqual(module.conditioning_mode, ADALN_CONDITIONING)
        self.assertEqual(module.objective, ORIGINAL_OBJECTIVE)
        self.assertEqual(int(module.training_mode_id), MEANFLOW_MODE_ID)
        self.assertEqual(
            int(module.conditioning_mode_id),
            ADALN_CONDITIONING_ID,
        )
        with self.assertRaisesRegex(ValueError, "only.*adaln"):
            CompressionMeanFlow(conditioning_mode="text_token")
        with self.assertRaisesRegex(ValueError, "only MeanFlow 5A"):
            CompressionMeanFlow(mode="velocity_control")
        with self.assertRaisesRegex(ValueError, "objective must be"):
            CompressionMeanFlow(objective="unknown")

        co_meanflow = CompressionMeanFlow(
            objective=CODEC_OBSERVABLE_OBJECTIVE,
            endpoint_ratio=0.25,
        )
        self.assertEqual(
            int(co_meanflow.training_mode_id),
            CODEC_OBSERVABLE_MODE_ID,
        )
        with self.assertRaisesRegex(ValueError, "supported only"):
            CompressionMeanFlow(endpoint_ratio=0.25)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            CompressionMeanFlow(
                objective=CODEC_OBSERVABLE_OBJECTIVE,
                equal_time_ratio=0.8,
                endpoint_ratio=0.3,
            )

    def test_adaln_condition_jvp_matches_finite_difference(self):
        module = CompressionMeanFlow(
            adaln_embedding_dim=4,
            frequency_embedding_size=4,
        )
        with torch.no_grad():
            module.adaln_interval_adapter.t_embedder.mlp[
                2
            ].weight.normal_(std=0.01)
            module.adaln_interval_adapter.h_embedder.mlp[
                2
            ].weight.normal_(std=0.01)
        t = torch.tensor([0.8, 0.6])
        r = torch.tensor([0.2, 0.1])

        def condition_fn(t_value, r_value):
            return module.condition_adaln(t_value, r_value)

        _, jvp = torch.func.jvp(
            condition_fn,
            (t, r),
            (torch.ones_like(t), torch.zeros_like(r)),
        )
        eps = 1e-3
        finite_difference = (
            condition_fn(t + eps, r) - condition_fn(t - eps, r)
        ) / (2.0 * eps)
        torch.testing.assert_close(
            jvp,
            finite_difference,
            rtol=3e-3,
            atol=3e-3,
        )


@unittest.skipUnless(HAS_TORCH, "PyTorch unavailable")
class MeanFlowObjectiveTest(unittest.TestCase):
    def test_target_uses_the_meanflow_correction(self):
        module = CompressionMeanFlow(
            adaln_embedding_dim=4,
            frequency_embedding_size=4,
        )
        velocity = torch.tensor([[2.0, 4.0]])
        du_dt = torch.tensor([[3.0, 5.0]])
        t = torch.tensor([0.8])
        r = torch.tensor([0.3])
        target, correction = module.build_target(
            velocity,
            du_dt,
            t,
            r,
        )
        torch.testing.assert_close(correction, 0.5 * du_dt)
        torch.testing.assert_close(target, velocity - correction)

    def test_bridge_is_detached_but_predictor_trains(self):
        module = CompressionMeanFlow(
            adaln_embedding_dim=4,
            frequency_embedding_size=4,
            norm_p=0.0,
        )
        velocity_model = ToyVelocity()
        clean = torch.randn(2, 3, requires_grad=True)
        compressed = torch.randn(2, 3, requires_grad=True)
        t = torch.tensor([0.9, 0.7])
        r = torch.tensor([0.1, 0.2])

        loss, stats = module.compute_loss(
            lambda z, t_value, r_value, _indices: velocity_model(
                z,
                t_value,
                r_value,
            ),
            clean,
            compressed,
            t=t,
            r=r,
        )
        loss.backward()

        self.assertIsNone(clean.grad)
        self.assertIsNone(compressed.grad)
        self.assertIsNotNone(velocity_model.scale.grad)
        self.assertGreater(stats["correction_ratio"].item(), 0.0)

    def test_codec_observable_objective_uses_diagonal_tangent(self):
        module = CompressionMeanFlow(
            objective=CODEC_OBSERVABLE_OBJECTIVE,
            adaln_embedding_dim=4,
            frequency_embedding_size=4,
            norm_p=0.0,
        )
        optimized_model = ToyVelocity()
        reference_model = ToyVelocity()
        reference_model.load_state_dict(optimized_model.state_dict())
        clean = torch.randn(3, 4, requires_grad=True)
        compressed = torch.randn(3, 4, requires_grad=True)
        t = torch.tensor([0.9, 0.7, 0.5])
        r = torch.tensor([0.1, 0.2, 0.3])

        optimized_loss, stats = module.compute_loss(
            lambda z, t_value, r_value, _indices: optimized_model(
                z,
                t_value,
                r_value,
            ),
            clean,
            compressed,
            t=t,
            r=r,
        )
        optimized_loss.backward()

        clean_reference = clean.detach()
        compressed_reference = compressed.detach()
        velocity = compressed_reference - clean_reference
        z_t = (
            (1.0 - t[:, None]) * clean_reference
            + t[:, None] * compressed_reference
        )
        with torch.no_grad():
            observable_tangent = reference_model(z_t, t, t).detach()

        def full_velocity_fn(z, t_value, r_value):
            return reference_model(z, t_value, r_value)

        prediction, du_dt = torch.func.jvp(
            full_velocity_fn,
            (z_t, t, r),
            (
                observable_tangent,
                torch.ones_like(t),
                torch.zeros_like(r),
            ),
        )
        correction = (
            (t - r)[:, None] * du_dt
        ).detach()
        reference_loss = (
            (prediction + correction - velocity)
            .square()
            .sum(dim=1)
            .mean()
        )
        reference_loss.backward()

        torch.testing.assert_close(optimized_loss, reference_loss)
        torch.testing.assert_close(
            optimized_model.scale.grad,
            reference_model.scale.grad,
        )
        torch.testing.assert_close(
            stats["observable_tangent_mse"],
            (observable_tangent - velocity).square().mean(),
        )
        self.assertIsNone(clean.grad)
        self.assertIsNone(compressed.grad)

    def test_codec_observable_sampler_has_an_exact_endpoint_bucket(self):
        module = CompressionMeanFlow(
            objective=CODEC_OBSERVABLE_OBJECTIVE,
            adaln_embedding_dim=4,
            frequency_embedding_size=4,
            equal_time_ratio=0.5,
            endpoint_ratio=0.25,
        )
        first = torch.tensor([0.2, 0.3, 0.4, 0.5])
        second = torch.tensor([0.8, 0.7, 0.6, 0.9])
        buckets = torch.tensor([0.1, 0.6, 0.8, 0.9])

        with mock.patch.object(
            module,
            "_sample_time",
            side_effect=[first, second],
        ), mock.patch("torch.rand", return_value=buckets):
            t, r = module.sample_tr(
                4,
                torch.device("cpu"),
                torch.float32,
            )

        self.assertEqual(t[0].item(), r[0].item())
        self.assertEqual(t[1].item(), 1.0)
        self.assertEqual(r[1].item(), 0.0)
        torch.testing.assert_close(t[2:], torch.tensor([0.6, 0.9]))
        torch.testing.assert_close(r[2:], torch.tensor([0.4, 0.5]))

    def test_equal_time_samples_skip_jvp(self):
        module = CompressionMeanFlow(
            adaln_embedding_dim=4,
            frequency_embedding_size=4,
            norm_p=0.0,
        )
        velocity_model = ToyVelocity()
        clean = torch.randn(3, 4)
        compressed = torch.randn(3, 4)
        t = torch.tensor([0.9, 0.5, 0.2])

        with mock.patch(
            "torch.func.jvp",
            side_effect=AssertionError("JVP must be skipped"),
        ):
            loss, stats = module.compute_loss(
                lambda z, t_value, r_value, _indices: velocity_model(
                    z,
                    t_value,
                    r_value,
                ),
                clean,
                compressed,
                t=t,
                r=t.clone(),
            )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            stats["nonzero_interval_fraction"].item(),
            0.0,
        )
        self.assertEqual(stats["correction_ratio_all"].item(), 0.0)

    def test_mixed_batch_split_matches_full_jvp(self):
        module = CompressionMeanFlow(
            adaln_embedding_dim=4,
            frequency_embedding_size=4,
            norm_p=0.0,
        )
        optimized_model = ToyVelocity()
        reference_model = ToyVelocity()
        reference_model.load_state_dict(optimized_model.state_dict())
        clean = torch.randn(4, 3)
        compressed = torch.randn(4, 3)
        t = torch.tensor([0.9, 0.8, 0.5, 0.4])
        r = torch.tensor([0.9, 0.1, 0.5, 0.2])

        optimized_loss, stats = module.compute_loss(
            lambda z, t_value, r_value, _indices: optimized_model(
                z,
                t_value,
                r_value,
            ),
            clean,
            compressed,
            t=t,
            r=r,
        )
        optimized_loss.backward()

        velocity = compressed - clean
        z_t = (1.0 - t[:, None]) * clean + t[:, None] * compressed

        def full_velocity_fn(z, t_value, r_value):
            return reference_model(z, t_value, r_value)

        prediction, du_dt = torch.func.jvp(
            full_velocity_fn,
            (z_t, t, r),
            (velocity, torch.ones_like(t), torch.zeros_like(r)),
        )
        target, _ = module.build_target(velocity, du_dt, t, r)
        reference_loss = (
            (prediction - target).square().sum(dim=1).mean()
        )
        reference_loss.backward()

        torch.testing.assert_close(optimized_loss, reference_loss)
        torch.testing.assert_close(
            optimized_model.scale.grad,
            reference_model.scale.grad,
        )
        self.assertEqual(
            stats["nonzero_interval_fraction"].item(),
            0.5,
        )

    def test_adaln_adapter_receives_meanflow_gradient(self):
        module = CompressionMeanFlow(
            adaln_embedding_dim=4,
            frequency_embedding_size=4,
            norm_p=0.0,
        )
        clean = torch.randn(2, 4)
        compressed = torch.randn(2, 4)
        t = torch.tensor([0.9, 0.7])
        r = torch.tensor([0.1, 0.2])

        def velocity_fn(z, t_value, r_value, _indices):
            return z + module.condition_adaln(t_value, r_value)

        loss, _ = module.compute_loss(
            velocity_fn,
            clean,
            compressed,
            t=t,
            r=r,
        )
        loss.backward()
        gradient = (
            module.adaln_interval_adapter.t_embedder.mlp[2].weight.grad
        )
        self.assertIsNotNone(gradient)
        self.assertGreater(gradient.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
