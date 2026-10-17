"""Small actual GPU checks; full public-backbone 512 smoke is separate."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderDC
from diffusers.models.attention_processor import SanaMultiscaleLinearAttention

import train_bt_self_flow as trainer
from models.bt_self_flow.checkpoint import METHOD, SCHEMA, read_checkpoint, save_atomic
from models.bt_self_flow.decoder import decode_latent
from models.bt_self_flow.codec import BTLatentCodec
from models.bt_self_flow.losses import ImageLoss
from models.bt_self_flow.packet import BTImagePacket, RateControl
from models.bt_self_flow.precision import StableSanaCrossAttention, install_stable_attention
from test_bt_backend import tiny_model, tiny_sana
from test_bt_training import small_config


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is inaccessible")
class GPUPrecisionTest(unittest.TestCase):
    def test_large_cross_attention_queries_preserve_finite_forward_and_backward(self):
        torch.manual_seed(907)
        attention = tiny_sana().cuda().transformer_blocks[-1].attn2
        with torch.no_grad():
            attention.to_q.weight.copy_(torch.eye(32, device="cuda") * 512)
            attention.to_q.bias.zero_()
        states = torch.full((1, 8, 32), 512., device="cuda", requires_grad=True)
        text = torch.randn(1, 2, 32, device="cuda", requires_grad=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            overflowing_query = attention.to_q(states)
            native = attention(states, encoder_hidden_states=text)
        self.assertFalse(torch.isfinite(overflowing_query).all())
        self.assertFalse(torch.isfinite(native).all())
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            reference = attention(states, encoder_hidden_states=text)
        state_keys = tuple(attention.state_dict())
        attention.set_processor(StableSanaCrossAttention())
        with torch.autocast("cuda", dtype=torch.float16):
            actual = attention(states, encoder_hidden_states=text)
        self.assertEqual(tuple(attention.state_dict()), state_keys)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(actual, reference))
        actual.square().mean().backward()
        for gradient in [states.grad, text.grad, *[p.grad for p in attention.parameters()]]:
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())

    @torch.no_grad()
    def test_channels_last_analysis_has_bit_exact_coded_reconstruction(self):
        torch.manual_seed(907)
        codec = BTLatentCodec(ch_emd=4, channel=32, channel_out=4).cuda().eval()
        codec.update(force=True)
        clean = torch.randn(1, 4, 16, 16, device="cuda").contiguous(memory_format=torch.channels_last)
        for quality in (0., .4, 1.):
            control = RateControl(quality)
            direct = codec(clean, control, trace=True)
            encoded = codec.compress(clean, control, trace=True)
            packet = BTImagePacket("C", (512, 512), encoded["y_shape"], encoded["hyper_shape"],
                                   control.low_code, control.high_code, encoded["hyper_stream"],
                                   encoded["main_stream"], encoded["allocation"])
            decoded = codec.decompress(packet, trace=True)
            self.assertTrue(torch.equal(direct["mean"], encoded["mean"]))
            self.assertTrue(torch.equal(encoded["mean"], decoded["mean"]))
            for simulated, restored in zip(direct["trace"], decoded["trace"]):
                for field in ("symbols", "indexes", "restored"):
                    self.assertTrue(torch.equal(simulated[field], restored[field]), field)

    def test_fp16_zero_attention_and_dc_decoder_are_finite(self):
        sana = tiny_sana().cuda()
        install_stable_attention(sana=sana)
        states = torch.zeros(1, 8, 32, device="cuda", requires_grad=True)
        with torch.autocast("cuda", dtype=torch.float16):
            output = sana.transformer_blocks[0].attn1(states)
        output.float().square().mean().backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(states.grad).all())
        vae = AutoencoderDC(latent_channels=4, encoder_block_out_channels=(16, 16),
                            decoder_block_out_channels=(16, 16), encoder_layers_per_block=(1, 1),
                            decoder_layers_per_block=(1, 1), attention_head_dim=8,
                            encoder_block_types=("ResBlock", "EfficientViTBlock"),
                            decoder_block_types=("ResBlock", "EfficientViTBlock"),
                            encoder_qkv_multiscales=((), (3,)), decoder_qkv_multiscales=((), (3,))).cuda()
        install_stable_attention(vae=vae)
        with torch.no_grad():
            for module in vae.modules():
                if isinstance(module, SanaMultiscaleLinearAttention):
                    for projection in (module.to_q, module.to_k):
                        projection.weight.zero_()
                        if projection.bias is not None:
                            projection.bias.zero_()
        latent = torch.randn(1, 4, 4, 4, device="cuda", requires_grad=True)
        with torch.autocast("cuda", dtype=torch.float16):
            decoded = decode_latent(vae, latent, checkpoint_blocks=True)
        decoded.float().square().mean().backward()
        self.assertTrue(torch.isfinite(decoded).all())
        self.assertTrue(torch.isfinite(latent.grad).all())

    def test_production_trainer_amp_and_real_packet(self):
        native_bf16 = torch.cuda.get_device_capability(0)[0] >= 8 and torch.cuda.is_bf16_supported()
        report = {"device": torch.cuda.get_device_name(0), "precision": "bf16" if native_bf16 else "fp16+GradScaler",
                  "scope": "small real SANA/PEFT and rANS; full 512 public-backbone smoke is pending", "backends": {}}
        torch.cuda.reset_peak_memory_stats()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"images").mkdir()
            Image.fromarray(np.random.default_rng(903).integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(root/"images/one.png")
            for backend in ("C", "D"):
                config = small_config(root, "joint")
                config["model"]["backend"] = backend
                config["train"]["output"] = str(root/backend)
                source = tiny_model(backend).eval()
                source.vae.requires_grad_(False)
                source_config = deepcopy(config)
                source_config["train"].update(phase="backend", steps=1)
                source_file = root/f"{backend}.pt"
                save_atomic(source_file, {"method": METHOD, "schema": SCHEMA, "phase": "backend", "backend": backend,
                                          "step": 1, "config": source_config, "raw": source.export_state(), "assets": {"test": "mini"}})
                del source
                models = []
                def factory(cfg, device):
                    model = tiny_model(backend).to(device)
                    model.vae.requires_grad_(False)
                    models.append(model)
                    return model
                with patch.object(trainer, "make_model", side_effect=factory), \
                     patch.object(trainer, "asset_identity", return_value={"test": "mini"}), \
                     patch.object(trainer, "ImageLoss", side_effect=lambda **kw: ImageLoss(lpips_weight=0, dists_weight=0)), \
                     redirect_stdout(io.StringIO()):
                    trainer.train(config, SimpleNamespace(resume=None, init_checkpoint=str(source_file), max_updates=None),
                                  0, 1, torch.device("cuda:0"))
                saved = read_checkpoint(root/backend/"complete.pt")
                self.assertEqual(saved["step"], 2)
                self.assertEqual(saved["micro_seen"], 4)
                model = models.pop().eval()
                model.codec.update(force=True)
                with torch.no_grad():
                    image = torch.rand(1, 3, 256, 256, device="cuda")*2-1
                    direct = model(image, rate_control=RateControl(.3)).image
                    packet = model.compress(image, RateControl(.3))
                    before = model.prediction_calls
                    decoded = model.decompress(packet.to_bytes())
                error = (decoded-direct).abs().max().item()
                self.assertLessEqual(error, 2e-4)
                self.assertEqual(model.prediction_calls-before, 1)
                report["backends"][backend] = {"updates": saved["step"], "attempts": saved["attempts"],
                                                "consumed_microbatches": saved["micro_seen"],
                                                "forward_packet_max_error": error, "packet_bytes": len(packet.to_bytes()), "nfe": 1}
                del model, saved, image, direct, decoded
                torch.cuda.empty_cache()
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        if os.environ.get("BT_TEST_GPU_REPORT"):
            Path(os.environ["BT_TEST_GPU_REPORT"]).write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    unittest.main()
