"""LPIPS with a finite feature-normalization backward at zero."""

import torch
from lpips import LPIPS
from lpips.lpips import spatial_average, upsample


def normalize_features(feature, eps=1e-10):
    # Keep LPIPS's norm + eps forward; vector_norm defines the zero derivative.
    norm = torch.linalg.vector_norm(feature, dim=1, keepdim=True)
    return feature / (norm + eps)


class StableLPIPS(LPIPS):
    """Inherit official initialization and state keys; replace only the norm."""

    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        if normalize:
            in0, in1 = 2 * in0 - 1, 2 * in1 - 1

        if self.version == "0.1":
            in0_input, in1_input = self.scaling_layer(in0), self.scaling_layer(in1)
        else:
            in0_input, in1_input = in0, in1
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        differences = [(normalize_features(outs0[index]) - normalize_features(outs1[index])).square()
                       for index in range(self.L)]

        if self.lpips:
            results = [self.lins[index](differences[index]) for index in range(self.L)]
        else:
            results = [difference.sum(dim=1, keepdim=True) for difference in differences]
        if self.spatial:
            results = [upsample(result, out_HW=in0.shape[2:]) for result in results]
        else:
            results = [spatial_average(result, keepdim=True) for result in results]

        value = 0
        for result in results:
            value = value + result
        return (value, results) if retPerLayer else value
