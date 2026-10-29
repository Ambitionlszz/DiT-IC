import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from models.bt_self_flow.losses import ImageLoss, rate_beta, reconstruction_objective
from models.bt_self_flow.packet import RateControl


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


class LatentAuxiliaryTest(unittest.TestCase):
    def result(self):
        return SimpleNamespace(
            coarse=torch.full((1, 2, 2, 2), 2., requires_grad=True),
            latent=torch.full((1, 2, 2, 2), 3., requires_grad=True),
            clean=torch.zeros(1, 2, 2, 2),
            image=torch.ones(1, 3, 2, 2, requires_grad=True),
            likelihoods={'main': torch.full((1, 2, 2, 2), .5, requires_grad=True)}, side_bits=8)

    def test_disabled_auxiliaries_have_no_loss_or_gradient_contribution(self):
        for base_weight, latent_weight in ((0., 0.), (.1, 0.), (0., .1), (.1, .1)):
            with self.subTest(base_weight=base_weight, latent_weight=latent_weight):
                result = self.result()
                objective = ImageLoss(lpips_weight=0, dists_weight=0,
                                      base_weight=base_weight, latent_weight=latent_weight)
                loss, values = reconstruction_objective(result, torch.zeros_like(result.image),
                                                        RateControl(.4), torch.tensor(2.), objective)
                expected = 2. + base_weight*1. + latent_weight*2.25 + rate_beta(RateControl(.4))*4.
                self.assertAlmostEqual(loss.item(), expected, places=5)
                loss.backward()
                self.assertEqual(result.coarse.grad is None, base_weight == 0)
                self.assertEqual(result.latent.grad is None, latent_weight == 0)
                self.assertEqual(values['base'].requires_grad, base_weight > 0)
                self.assertEqual(values['latent'].requires_grad, latent_weight > 0)
                self.assertGreater(result.image.grad.abs().sum().item(), 0)
                self.assertGreater(result.likelihoods['main'].grad.abs().sum().item(), 0)

    def test_image_reconstruction_still_trains_frontend_without_auxiliaries(self):
        result = self.result()
        result.image = result.latent[:, :1].expand(-1, 3, -1, -1)
        objective = ImageLoss(lpips_weight=0, dists_weight=0, base_weight=0, latent_weight=0)
        loss, _ = reconstruction_objective(result, torch.zeros_like(result.image), RateControl(),
                                            torch.tensor(2.), objective)
        loss.backward()
        self.assertGreater(result.latent.grad.abs().sum().item(), 0)
        self.assertIsNone(result.coarse.grad)

    def test_default_objective_matches_previous_formula(self):
        result = self.result()
        objective = ImageLoss(lpips_weight=0, dists_weight=0)
        loss, values = reconstruction_objective(result, torch.zeros_like(result.image), RateControl(),
                                                torch.tensor(2.), objective)
        previous = (2*values['mse'] + .1*values['latent'] + .1*values['base']
                    + rate_beta(RateControl())*values['rate_est'])
        self.assertTrue(torch.equal(loss, previous))

    def test_warmup_retains_latent_reconstruction_target(self):
        result = self.result()
        loss, values = reconstruction_objective(result, torch.zeros_like(result.image), RateControl(), torch.tensor(2.))
        self.assertAlmostEqual(loss.item(), (values['base']+rate_beta(RateControl())*values['rate_est']).item())
        loss.backward()
        self.assertGreater(result.coarse.grad.abs().sum().item(), 0)

    def test_rate_weight_changes_only_rate_loss_and_its_gradient(self):
        for image_objective in (None, ImageLoss(lpips_weight=0, dists_weight=0)):
            records = []
            for rate_weight in (1., 3.):
                result = self.result()
                loss, values = reconstruction_objective(result, torch.zeros_like(result.image), RateControl(.4),
                                                        torch.tensor(2.), image_objective, rate_weight=rate_weight)
                loss.backward()
                records.append((loss.detach(), values['rate_est'].detach(), result.coarse.grad.clone(),
                                result.likelihoods['main'].grad.clone()))
            self.assertTrue(torch.equal(records[0][1], records[1][1]))
            self.assertTrue(torch.equal(records[0][2], records[1][2]))
            self.assertTrue(torch.allclose(records[0][3]*3, records[1][3]))
            self.assertTrue(torch.allclose(records[1][0]-records[0][0],
                                           2*rate_beta(RateControl(.4))*records[0][1]))

    def test_invalid_auxiliary_weights_fail_before_loading_metrics(self):
        for weight in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                ImageLoss(base_weight=weight)
            with self.assertRaises(ValueError):
                ImageLoss(latent_weight=weight)


if __name__ == "__main__":
    unittest.main()
