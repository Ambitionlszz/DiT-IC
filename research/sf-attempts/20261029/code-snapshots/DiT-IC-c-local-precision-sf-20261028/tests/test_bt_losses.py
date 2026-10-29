import unittest
from unittest.mock import patch

import torch
from torch import nn

from models.bt_self_flow.losses import ImageLoss


class SmallDistance(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.tensor(.5))

    def forward(self, prediction, target):
        return self.weight*(prediction-target).square().mean((1, 2, 3))


class PerceptualDeviceTest(unittest.TestCase):
    def test_cpu_loss_on_cuda_host_keeps_device_and_image_gradient(self):
        # Exercise the real pyiqa wrapper/device logic without downloading VGG.
        # Its default automatic CUDA placement used to strand a device sentinel
        # when ImageLoss was subsequently moved back to CPU.
        with patch("pyiqa.models.inference_model.build_network", side_effect=lambda options: SmallDistance()), \
             patch("torch.cuda.is_available", return_value=True):
            objective = ImageLoss(lpips_weight=0).to("cpu").eval()
            prediction = torch.zeros(1, 3, 16, 16, requires_grad=True)
            target = torch.ones_like(prediction)*.5
            loss, metrics = objective(prediction, target)
            loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["dists"].device.type, "cpu")
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
