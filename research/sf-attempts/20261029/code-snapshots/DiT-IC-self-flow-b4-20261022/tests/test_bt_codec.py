import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from models.bt_self_flow.codec import BTLatentCodec
from models.bt_self_flow.packet import (BTImagePacket, RateControl, fixed_side_bits,
                                        pack_codes, unpack_codes)
from models.bt_self_flow.router import ste_round
from models.bt_self_flow.scale import DistortionScale


torch.set_num_threads(1)


class PacketTest(unittest.TestCase):
    def test_all_code_widths_and_corrupt_packets(self):
        for bits in (1, 3, 4):
            values = [i % (1 << bits) for i in range(67)]
            self.assertEqual(unpack_codes(pack_codes(values, bits), bits, len(values)), values)
        packet = BTImagePacket("D", (256, 256), (4, 4), (1, 1), 1, 65535,
                               b"hyper", b"main", pack_codes([4]*16, 3),
                               pack_codes([7]*16, 4), pack_codes([0, 1]*8, 1))
        self.assertEqual(BTImagePacket.from_bytes(packet.to_bytes()), packet)
        self.assertEqual(len(packet.to_bytes()), sum(packet.byte_breakdown().values()))
        side = len(packet.to_bytes())-len(packet.hyper_stream)-len(packet.main_stream)
        self.assertEqual(8*side, fixed_side_bits(4, 4, "D", True))
        for bad in (packet.to_bytes()[:-1], packet.to_bytes()+b"x", b"bad"):
            with self.assertRaises(ValueError):
                BTImagePacket.from_bytes(bad)

    def test_canonical_control(self):
        control = RateControl(.1234567)
        recovered = RateControl(control.low_code/65535)
        self.assertEqual(control.value, recovered.value)
        self.assertTrue(torch.equal(control.tensor(1, 4, 4, "cpu"), recovered.tensor(1, 4, 4, "cpu")))


class QuantizationTest(unittest.TestCase):
    def test_step_gradient_survives_hard_forward(self):
        y = torch.tensor([.2, 1.8], requires_grad=True)
        delta = torch.tensor(.7, requires_grad=True)
        u = y/delta
        reconstructed = delta*ste_round(u)
        self.assertTrue(torch.equal(ste_round(u).detach(), u.detach().round()))
        reconstructed.sum().backward()
        self.assertTrue(torch.allclose(delta.grad, (u.detach().round()-u.detach()).sum()))
        self.assertTrue(torch.equal(y.grad, torch.ones_like(y)))

    def test_scale_quantization_and_stop_gradient(self):
        model = DistortionScale()
        clean = torch.randn(2, 4, 8, 8, requires_grad=True)
        coarse = clean.detach()+.2*torch.randn_like(clean)
        scales = model.measure(clean, coarse)
        model.calibrate(clean.detach().square().mean(), [scales])
        info = model(clean, coarse)
        decoded, log_rho = model.decode(info["codes"])
        self.assertTrue(torch.equal(decoded, info["scale"]))
        self.assertTrue(torch.equal(log_rho, info["log_rho"]))
        self.assertFalse(info["scale"].requires_grad)
        self.assertTrue((decoded > 0).all())


class RealCoderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(903)
        cls.codec = BTLatentCodec(ch_emd=4, channel=16, channel_out=4).eval()
        cls.codec.update(force=True)
        cls.clean = torch.randn(1, 4, 8, 8)

    def test_forward_rans_and_fresh_process_agree(self):
        mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        mask[..., ::2, ::2] = True
        for control in (RateControl(0), RateControl(.42), RateControl(1), RateControl(.1, 1, mask)):
            with torch.no_grad():
                forward = self.codec(self.clean, control, trace=True)
                encoded = self.codec.compress(self.clean, control, trace=True)
            packet = BTImagePacket("C", (256, 256), encoded["y_shape"], encoded["hyper_shape"],
                                   control.low_code, control.high_code, encoded["hyper_stream"],
                                   encoded["main_stream"], encoded["allocation"],
                                   mask=pack_codes(mask.flatten().tolist(), 1) if control.mixed else b"")
            decoded = self.codec.decompress(BTImagePacket.from_bytes(packet.to_bytes()), trace=True)
            self.assertTrue(torch.equal(encoded["mean"], decoded["mean"]))
            self.assertTrue(torch.allclose(forward["mean"], decoded["mean"], atol=1e-6, rtol=1e-6))
            for a, b, c in zip(forward["trace"], encoded["trace"], decoded["trace"]):
                self.assertTrue(torch.equal(a["symbols"], b["symbols"]))
                self.assertTrue(torch.equal(b["symbols"], c["symbols"]))
                self.assertTrue(torch.equal(b["indexes"], c["indexes"]))
                self.assertTrue(torch.equal(b["restored"], c["restored"]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(self.codec.state_dict(), root/"model.pt")
            (root/"image.btic").write_bytes(packet.to_bytes())
            script = """
import sys, torch
from pathlib import Path
from models.bt_self_flow.codec import BTLatentCodec
from models.bt_self_flow.packet import BTImagePacket
torch.set_num_threads(1)
p=Path(sys.argv[1]); m=BTLatentCodec(ch_emd=4,channel=16,channel_out=4).eval()
m.load_state_dict(torch.load(p/'model.pt', weights_only=True), strict=True)
torch.save(m.decompress(BTImagePacket.from_bytes((p/'image.btic').read_bytes()))['mean'],p/'decoded.pt')
"""
            subprocess.run([sys.executable, "-c", script, str(root)], check=True, timeout=120)
            self.assertTrue(torch.equal(decoded["mean"], torch.load(root/"decoded.pt", weights_only=True)))

    def test_final_latent_loss_reaches_router_and_analysis(self):
        self.codec.train().zero_grad(set_to_none=True)
        output = self.codec(self.clean, RateControl(.4))
        (output["mean"]-self.clean).square().mean().backward()
        for parameter in (self.codec.router.head[-1].weight, self.codec.g_a.pre1[0].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.codec.eval()


if __name__ == "__main__":
    unittest.main()
