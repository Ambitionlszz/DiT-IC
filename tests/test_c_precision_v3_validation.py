"""Actual-rate validation and calibration preserve the codec packet version."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from diffusers import SanaTransformer2DModel
from torch import nn
from torch.nn import functional as F

from models.bt_self_flow import validation
from models.bt_self_flow.codec import BTLatentCodec
from models.bt_self_flow.losses import ImageLoss
from models.bt_self_flow.model import BTCodec
from models.bt_self_flow.packet import BTImagePacket
from models.bt_self_flow.precision import install_stable_attention
from models.bt_self_flow.precision_conditioning import PrecisionGeneratorConditioner
from models.bt_self_flow.scale import DistortionScale
from models.scheduler import OneStepFlowScheduler


class TinyAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, 1)
        self.decoder = nn.Conv2d(4, 3, 1)
        self.config = SimpleNamespace(scaling_factor=1.)

    def encode(self, image):
        return SimpleNamespace(latent=self.encoder(F.avg_pool2d(image, 32)))

    def decode(self, latent):
        return SimpleNamespace(sample=F.interpolate(self.decoder(latent), scale_factor=32, mode="nearest"))


class TinyImages:
    fingerprint = "precision-validation-fixture"

    def __init__(self):
        generator = torch.Generator().manual_seed(1903)
        self.image = torch.rand(3, 256, 256, generator=generator) * 2 - 1

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.image


@pytest.fixture(params=[1, 3], ids=["legacy-v1", "precision-v3"])
def model(request):
    torch.manual_seed(903)
    model = BTCodec.__new__(BTCodec)
    nn.Module.__init__(model)
    model.codec = BTLatentCodec(ch_emd=4, channel=16, channel_out=4,
                                precision_conditioning=True, precision_consistent=request.param == 3)
    model.vae = TinyAE().requires_grad_(False)
    model.DiT = SanaTransformer2DModel(
        in_channels=4, out_channels=4, num_attention_heads=4, attention_head_dim=8,
        num_layers=4, num_cross_attention_heads=4, cross_attention_head_dim=8,
        cross_attention_dim=32, caption_channels=16, sample_size=8, patch_size=1, mlp_ratio=1.)
    model.register_buffer("fixed_text_embed", torch.randn(1, 2, 12))
    model.text_proj = nn.Linear(12, 16)
    model.quality_conditioner = PrecisionGeneratorConditioner(32, 4) if request.param == 3 else None
    model.representation_projector = None
    model.distortion_scale = DistortionScale()
    model.backend, model.checkpoint_blocks = "C", False
    model.freeze_codec, model.freeze_vae = False, True
    model.prediction_calls = 0
    model.sched = OneStepFlowScheduler(SimpleNamespace(sigmas=torch.tensor([.7]), timesteps=torch.tensor([700.])))
    install_stable_attention(sana=model.DiT)
    return model.eval()


def test_actual_rate_validation_decodes_the_codec_packet_version(model):
    dataset = TinyImages()
    image_loss = ImageLoss(mse_weight=4, lpips_weight=0, dists_weight=0)
    with patch.object(model, "decompress", wraps=model.decompress) as decode, \
         patch.object(model, "compress", side_effect=AssertionError("actual-rate validation must reuse its packet")):
        report = validation.validate(model, dataset, image_loss, qualities=(.4,),
                                     actual_rate=True, packet_samples=1)
    assert decode.call_count == 1
    packet = decode.call_args.args[0]
    assert packet.version == model.codec.packet_version
    metrics = report["0.4"]
    actual_bpp = 8 * len(packet.to_bytes()) / dataset.image.shape[-2:].numel()
    assert metrics["rate_actual"] == pytest.approx(actual_bpp)
    assert metrics["packet_bpp"] == pytest.approx(actual_bpp)
    assert metrics["packet_forward_max_error"] <= 2e-4


def test_calibration_packets_use_the_codec_packet_version(model, tmp_path):
    packets = []

    def record_packet(*args, **kwargs):
        packet = BTImagePacket(*args, **kwargs)
        packets.append(packet)
        return packet

    output = tmp_path / "calibration.json"
    with patch.object(validation, "BTImagePacket", side_effect=record_packet):
        report = validation.calibrate_interface(model, TinyImages(), output, samples=1, packet_samples=1)
    assert len(packets) == 6
    assert json.loads(output.read_text())["prefix_raw_codec_actual_bpp"] == report["prefix_raw_codec_actual_bpp"]
    for quality, packet in zip((0., .2, .4, .6, .8, 1.), packets):
        assert packet.version == model.codec.packet_version
        received = BTImagePacket.from_bytes(packet.to_bytes())
        assert model.codec.decompress(received)["mean"].shape == (1, 4, 8, 8)
        assert report["prefix_raw_codec_actual_bpp"][str(quality)] == pytest.approx(
            8 * len(packet.to_bytes()) / (256 * 256))
