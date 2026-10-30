from types import SimpleNamespace
import math

import pytest
import torch
from torch import nn

from models.bt_self_flow.precision import StableSanaCrossAttention


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('magnitude', [1., 1e12])
def test_cross_attention_backward_matches_float64_for_large_finite_queries(device, magnitude):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    attention = SimpleNamespace(heads=1, to_q=nn.Identity(), to_k=nn.Identity(),
                                to_v=nn.Identity(), norm_q=None, norm_k=None,
                                to_out=[nn.Identity(), nn.Identity()], rescale_output_factor=1.)
    query = torch.full((1, 1, 2), magnitude, device=device, requires_grad=True)
    context = torch.eye(2, device=device).unsqueeze(0).requires_grad_()
    weights = torch.tensor([1., -1.], device=device)
    result = StableSanaCrossAttention()(attention, query, context)
    gradients = torch.autograd.grad((result*weights).sum(), (query, context))
    query64 = query.detach().double().requires_grad_()
    context64 = context.detach().double().requires_grad_()
    reference = ((query64 @ context64.transpose(-1, -2))/math.sqrt(query.shape[-1])).softmax(-1) @ context64
    expected = torch.autograd.grad((reference*weights).sum(), (query64, context64))
    torch.testing.assert_close(result.double(), reference, rtol=1e-6, atol=1e-7)
    for actual, target in zip(gradients, expected):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.double(), target, rtol=1e-5, atol=1e-6)
