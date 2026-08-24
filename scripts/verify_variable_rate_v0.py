import argparse
from copy import deepcopy

from PIL import Image
import torch
from torchvision import transforms

from models.latent_codec import LatentCodec
from train_variable_rate_lic_ddp import (
    VariableRateLICStage,
    load_config,
    load_source_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(
        description="Verify identity-initialized q0..q5 against the M0 codec."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    _, source_state = load_source_checkpoint(
        config["model"]["source_checkpoint"],
        bool(config["model"].get("source_use_ema", False)),
    )

    variable_stage = VariableRateLICStage(config, source_state).to(device).eval()
    legacy_params = deepcopy(config["model"]["codec_param"])
    legacy_params["variable_rate"] = False
    legacy_params.pop("num_quality_levels", None)
    legacy_codec = LatentCodec(**legacy_params).to(device).eval()
    load_result = legacy_codec.load_state_dict(
        source_state["state_dict_codec"], strict=False
    )
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            f"legacy M0 load mismatch: missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )

    image = transforms.ToTensor()(Image.open(args.image).convert("RGB"))
    image = transforms.Normalize([0.5] * 3, [0.5] * 3)(image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        latent = variable_stage.encode_clean_latent(image)
        reference = legacy_codec(latent)
        for quality_index in range(6):
            candidate = variable_stage.codec(latent, quality_index)
            if not torch.equal(candidate["mean"], reference["mean"]):
                raise SystemExit(f"q{quality_index} forward reconstruction differs")
            for name in ("y", "z"):
                if not torch.equal(
                    candidate["likelihoods"][name],
                    reference["likelihoods"][name],
                ):
                    raise SystemExit(f"q{quality_index} {name} likelihood differs")

        legacy_codec.update(force=True)
        variable_stage.codec.update(force=True)
        reference_packet = legacy_codec.compress(latent)
        for quality_index in range(6):
            candidate_packet = variable_stage.codec.compress(
                latent, quality_index
            )
            if candidate_packet.strings != reference_packet["strings"]:
                raise SystemExit(f"q{quality_index} rANS payload differs")
            decoded = variable_stage.codec.decompress(candidate_packet)
            reference_decoded = legacy_codec.decompress(
                reference_packet["strings"], reference_packet["shape"]
            )
            if not torch.equal(decoded, reference_decoded):
                raise SystemExit(f"q{quality_index} decoded latent differs")

    print("V0 identity passed for forward, likelihoods, rANS payload, and decode")


if __name__ == "__main__":
    main()
