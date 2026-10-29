import torch.nn as nn
import torch.nn.functional as F


RELAY_MODES = ("decoder_self", "bitstream")


def _resample_channels(feature, channels):
    """Parameter-free channel resampling used by the capacity control."""
    if feature.shape[1] == channels:
        return feature

    batch, source_channels, height, width = feature.shape
    sequences = feature.permute(0, 2, 3, 1).reshape(-1, 1, source_channels)
    if source_channels > channels:
        sequences = F.adaptive_avg_pool1d(sequences, channels)
    else:
        sequences = F.interpolate(
            sequences,
            size=channels,
            mode="linear",
            align_corners=False,
        )
    return sequences.reshape(batch, height, width, channels).permute(0, 3, 1, 2)


class FeatureModulationHead(nn.Module):
    def __init__(self, source_channels, hidden_channels, target_channels):
        super().__init__()
        self.in_conv = nn.Conv2d(
            source_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.norm = nn.GroupNorm(8, hidden_channels)
        self.act = nn.SiLU()
        self.out_conv = nn.Conv2d(
            hidden_channels,
            2 * target_channels,
            kernel_size=1,
        )
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, source):
        return self.out_conv(self.act(self.norm(self.in_conv(source))))


class DecoderFeatureRelay(nn.Module):
    """Inject codec-grounded or decoder-self features through matched SFT heads."""

    def __init__(
        self,
        mode,
        stage_channels,
        stage_block_indices=(3, 2, 1),
        source_channels=320,
        hidden_channels=64,
    ):
        super().__init__()
        if mode not in RELAY_MODES:
            raise ValueError(
                f"decoder relay mode must be one of {RELAY_MODES}, got {mode!r}"
            )
        if len(stage_channels) != len(stage_block_indices):
            raise ValueError(
                "stage_channels and stage_block_indices must have equal length"
            )
        if hidden_channels % 8:
            raise ValueError("hidden_channels must be divisible by 8")

        self.mode = mode
        self.stage_block_indices = tuple(int(index) for index in stage_block_indices)
        self.source_channels = int(source_channels)
        self.hidden_channels = int(hidden_channels)
        self.heads = nn.ModuleDict(
            {
                str(index): FeatureModulationHead(
                    self.source_channels,
                    self.hidden_channels,
                    int(channels),
                )
                for index, channels in zip(
                    self.stage_block_indices,
                    stage_channels,
                )
            }
        )

    def forward(self, block_index, hidden_states, bitstream_feature=None):
        block_index = int(block_index)
        if block_index not in self.stage_block_indices:
            return hidden_states

        if self.mode == "bitstream":
            if bitstream_feature is None:
                raise ValueError("bitstream relay requires the codec bridge feature")
            source = bitstream_feature
        else:
            source = hidden_states

        source = _resample_channels(source, self.source_channels)
        source = F.interpolate(
            source,
            size=hidden_states.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        gamma, beta = self.heads[str(block_index)](source).chunk(2, dim=1)
        return hidden_states * (1.0 + gamma) + beta

    def contract(self):
        return {
            "mode": self.mode,
            "source": (
                "codec_g_s_preprojection_320ch"
                if self.mode == "bitstream"
                else "vae_decoder_self_feature"
            ),
            "stage_block_indices": list(self.stage_block_indices),
            "source_channels": self.source_channels,
            "hidden_channels": self.hidden_channels,
            "modulation": "zero_init_sft",
        }
