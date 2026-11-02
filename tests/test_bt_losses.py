import unittest
from types import SimpleNamespace
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


class StatefulDistance(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3), nn.Dropout(.5),
        )

    def forward(self, prediction, target):
        difference = self.features(prediction) - self.features(target)
        return difference.square().mean((1, 2, 3))


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

    def test_parent_train_keeps_frozen_metrics_deterministic_with_image_gradients(self):
        lpips, dists = StatefulDistance(), StatefulDistance()
        with patch("models.bt_self_flow.perceptual.StableLPIPS", return_value=lpips), \
             patch("pyiqa.create_metric", return_value=SimpleNamespace(net=dists)):
            objective = ImageLoss(mse_weight=0.)
        parent = nn.ModuleDict({"loss": objective})
        for mode in (False, True, False, True):
            parent.train(mode)
            self.assertEqual(objective.training, mode)
            self.assertTrue(all(not module.training for metric in (lpips, dists)
                                for module in metric.modules()))
        prediction = torch.zeros(1, 3, 4, 4, requires_grad=True)
        target = torch.ones_like(prediction)*.5
        buffers = {name: value.clone() for name, value in objective.named_buffers()}
        first, _ = objective(prediction, target)
        second, _ = objective(prediction, target)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        first.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad.abs().sum().item(), 0)
        self.assertTrue(all(not parameter.requires_grad and parameter.grad is None
                            for parameter in objective.parameters()))
        for name, value in objective.named_buffers():
            torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)

    def test_mse_only_loss_accepts_train_and_eval(self):
        objective = ImageLoss(lpips_weight=0., dists_weight=0.)
        self.assertIs(objective.train(), objective)
        self.assertTrue(objective.training)
        self.assertIs(objective.eval(), objective)
        self.assertFalse(objective.training)


if __name__ == "__main__":
    unittest.main()
