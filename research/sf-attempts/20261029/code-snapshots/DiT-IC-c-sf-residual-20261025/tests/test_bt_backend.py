from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from diffusers import AutoencoderDC, SanaTransformer2DModel
from peft import LoraConfig, get_peft_model

from models.bt_self_flow.codec import BTLatentCodec
from models.bt_self_flow.conditioning import ContinuousDistortionConditioner, residual_update
from models.bt_self_flow.decoder import decode_latent
from models.bt_self_flow.model import BTCodec, BTCrossRateTeacher, cross_rate_loss, pad_image
from models.bt_self_flow.packet import BTImagePacket, RateControl
from models.bt_self_flow.scale import DistortionScale
from models.bt_self_flow.precision import install_stable_attention
from models.sana_self_flow import RepresentationProjector, quality_conditioned_sana_forward
from models.scheduler import OneStepFlowScheduler


torch.set_num_threads(1)


class TinyAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, 1).requires_grad_(False)
        self.decoder = nn.Conv2d(4, 3, 1)
        self.config = SimpleNamespace(scaling_factor=1.)

    def encode(self, image):
        return SimpleNamespace(latent=self.encoder(F.avg_pool2d(image, 32)))

    def decode(self, latent):
        return SimpleNamespace(sample=F.interpolate(self.decoder(latent), scale_factor=32, mode="nearest"))


def tiny_sana():
    return SanaTransformer2DModel(in_channels=4, out_channels=4, num_attention_heads=4,
                                 attention_head_dim=8, num_layers=4, num_cross_attention_heads=4,
                                 cross_attention_head_dim=8, cross_attention_dim=32,
                                 caption_channels=16, sample_size=8, patch_size=1, mlp_ratio=1.)


def tiny_model(backend="D", representation=False):
    # Exercise production BT methods with a real small diffusers/PEFT SANA.
    torch.manual_seed(903)
    model = BTCodec.__new__(BTCodec)
    nn.Module.__init__(model)
    model.codec = BTLatentCodec(ch_emd=4, channel=16, channel_out=4)
    model.vae = TinyAE()
    model.DiT = get_peft_model(tiny_sana().requires_grad_(False),
                               LoraConfig(r=2, lora_alpha=2, target_modules=["to_q", "to_v"], bias="none"))
    model.register_buffer("fixed_text_embed", torch.randn(1, 2, 12))
    model.text_proj = nn.Linear(12, 16)
    model.quality_conditioner = ContinuousDistortionConditioner(32, 4) if backend == "D" else None
    model.representation_projector = RepresentationProjector(32) if representation else None
    model.distortion_scale = DistortionScale()
    model.backend, model.checkpoint_blocks = backend, False
    model.freeze_codec, model.freeze_vae = False, False
    model.student_layer, model.teacher_layer = 0, 2
    model.prediction_calls, model.last_compress_stats = 0, {}
    model.sched = OneStepFlowScheduler(SimpleNamespace(sigmas=torch.tensor([.7]), timesteps=torch.tensor([700.])))
    model.codec.update(force=True)
    model.distortion_scale.calibrate(torch.tensor(1.), [torch.tensor([.1, .5, 1.])])
    install_stable_attention(sana=model.DiT.get_base_model())
    return model


class SanaAdapterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(903)

    def test_zero_adapter_matches_native_sana(self):
        sana = tiny_sana().eval()
        condition = ContinuousDistortionConditioner(32, 4)
        latent, text = torch.randn(1, 4, 8, 8), torch.randn(1, 2, 16)
        time = torch.tensor([700.])
        native = sana(latent, text, time, return_dict=False)[0]
        install_stable_attention(sana=sana)
        adapted, _ = quality_conditioned_sana_forward(sana, latent, text, time, torch.randn(1, 8, 8),
                                                      condition, continuous_conditioning=True)
        self.assertTrue(torch.allclose(native, adapted, atol=1e-6, rtol=1e-6))
        old = latent-.7*adapted
        self.assertTrue(torch.equal(old, residual_update(latent, adapted, torch.ones_like(latent), .7, 0)))

    def test_checkpointing_preserves_output_and_gradients(self):
        sana = tiny_sana().train()
        condition = ContinuousDistortionConditioner(32, 4)
        nn.init.normal_(condition.block_projection.weight, std=.005)
        latent, text = torch.randn(1, 4, 8, 8, requires_grad=True), torch.randn(1, 2, 16)
        time, log_rho = torch.tensor([700.]), torch.randn(1, 8, 8)
        results = []
        for checkpointed in (False, True):
            sana.zero_grad(set_to_none=True)
            condition.zero_grad(set_to_none=True)
            latent.grad = None
            output, _ = quality_conditioned_sana_forward(sana, latent, text, time, log_rho, condition,
                                                          continuous_conditioning=True, checkpoint_blocks=checkpointed)
            output.square().mean().backward()
            results.append((output.detach().clone(), latent.grad.clone(), condition.block_projection.weight.grad.clone()))
        for plain, checkpointed in zip(*results):
            self.assertTrue(torch.allclose(plain, checkpointed, atol=1e-6, rtol=1e-6))

    def test_dc_decoder_checkpointing_matches_native(self):
        vae = AutoencoderDC(latent_channels=4, encoder_block_out_channels=(16, 16),
                            decoder_block_out_channels=(16, 16), encoder_layers_per_block=(1, 1),
                            decoder_layers_per_block=(1, 1), encoder_qkv_multiscales=((), ()),
                            decoder_qkv_multiscales=((), ()), attention_head_dim=8)
        latent = torch.randn(1, 4, 4, 4, requires_grad=True)
        outputs = []
        for enabled in (False, True):
            vae.zero_grad(set_to_none=True)
            latent.grad = None
            output = decode_latent(vae, latent, enabled)
            output.square().mean().backward()
            outputs.append((output.detach().clone(), latent.grad.clone(), vae.decoder.conv_in.weight.grad.clone()))
        for plain, checkpointed in zip(*outputs):
            self.assertTrue(torch.allclose(plain, checkpointed, atol=1e-6, rtol=1e-6))

    def test_dc_stable_attention_preserves_fp32_output(self):
        vae = AutoencoderDC(latent_channels=4, encoder_block_out_channels=(16, 16),
                            decoder_block_out_channels=(16, 16), encoder_layers_per_block=(1, 1),
                            decoder_layers_per_block=(1, 1), attention_head_dim=8,
                            encoder_block_types=("ResBlock", "EfficientViTBlock"),
                            decoder_block_types=("ResBlock", "EfficientViTBlock"),
                            encoder_qkv_multiscales=((), (3,)), decoder_qkv_multiscales=((), (3,)))
        native = deepcopy(vae)
        install_stable_attention(vae=vae)
        for side in (2, 4):  # quadratic and linear attention paths
            latent = torch.randn(1, 4, side, side, requires_grad=True)
            expected = native.decode(latent).sample
            actual = vae.decode(latent).sample
            self.assertTrue(torch.allclose(expected, actual, atol=1e-6, rtol=1e-6))
            actual.square().mean().backward()
            self.assertTrue(torch.isfinite(latent.grad).all())


class EndpointAndTeacherTest(unittest.TestCase):
    def test_mixed_packet_and_small_image_padding(self):
        model = tiny_model("D").eval()
        image = torch.rand(1, 3, 5, 9)*2-1
        mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        mask[..., ::2, ::2] = True
        control = RateControl(.2, 1., mask)
        packet = model.compress(image, control)
        expected = model(pad_image(image), rate_control=control).image[..., :5, :9]
        def forbidden(*args, **kwargs):
            raise AssertionError("independent decoder used a sender feature")
        model.codec.router.forward = forbidden
        model.encode_clean_latent = forbidden
        actual = model.decompress(packet.to_bytes())
        self.assertEqual(actual.shape, image.shape)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-5, rtol=1e-5))
        self.assertTrue(packet.mask)

    def test_c_and_d_packet_decode_use_one_prediction_and_no_router(self):
        image = torch.rand(1, 3, 256, 256)*2-1
        for backend in ("C", "D"):
            model = tiny_model(backend).eval()
            packet = model.compress(image, RateControl(.3))
            self.assertEqual(model.prediction_calls, 0)
            direct = model(image, rate_control=RateControl(.3)).image
            def forbidden(*args, **kwargs):
                raise AssertionError("decoder called sender-only Router")
            model.codec.router.forward = forbidden
            before = model.prediction_calls
            decoded = model.decompress(BTImagePacket.from_bytes(packet.to_bytes()))
            self.assertEqual(model.prediction_calls-before, 1)
            self.assertTrue(torch.allclose(decoded, direct, atol=1e-5, rtol=1e-5))
            self.assertEqual(bool(packet.scales), backend == "D")

    def test_teacher_is_frozen_and_representation_gradient_reaches_frontend(self):
        model = tiny_model(representation=True).train()
        teacher = BTCrossRateTeacher(model.DiT, model.text_proj, model.quality_conditioner, model.teacher_layer)
        mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        mask[..., ::2, ::2] = True
        image = torch.rand(1, 3, 256, 256)*2-1
        result = model(image, rate_control=RateControl(.1, 1., mask), capture=True)
        with torch.no_grad():
            high = model.codec.quantize_analysis(result.source, RateControl(1.))
            info = model.distortion_scale(result.clean, high["mean"])
            target = teacher(high["mean"], model.fixed_text_embed, model.self_flow_timestep(1, "cpu"), info["log_rho"])
        # Force valid blocks to test the gradient route independently of the
        # random untrained codec's rate/quality ordering.
        loss, valid = cross_rate_loss(result.feature, target, torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 4, 4))
        loss.backward()
        for parameter in (model.codec.router.head[-1].weight, model.codec.g_a.pre1[0].weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in teacher.parameters()))
        self.assertTrue(all(p.grad is None for p in model.vae.encoder.parameters()))
        state = teacher.get_ema_state_dict()
        with torch.no_grad():
            model.quality_conditioner.block_projection.bias.add_(1)
        teacher.update_from_student(model.DiT, model.text_proj, model.quality_conditioner, 0.)
        self.assertTrue(torch.equal(teacher.quality_conditioner.block_projection.bias,
                                    model.quality_conditioner.block_projection.bias))
        empty, fraction = cross_rate_loss(result.feature, target, torch.ones(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))
        self.assertEqual(empty.item(), 0)
        self.assertEqual(fraction.item(), 0)


if __name__ == "__main__":
    unittest.main()
