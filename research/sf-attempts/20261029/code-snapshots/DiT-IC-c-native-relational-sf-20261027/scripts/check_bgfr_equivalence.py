#!/usr/bin/env python3
import argparse
import gc
import hashlib
import random

import numpy as np
import torch
import yaml

from config_compat import normalize_pure_sana_config
from models.DiT_IC import Codec


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(params, mode, device, seed):
    set_seed(seed)
    model_params = dict(params)
    model_params["training"] = False
    model_params["decoder_relay_params"] = dict(
        model_params["decoder_relay_params"]
    )
    model_params["decoder_relay_params"]["mode"] = mode
    model = Codec(
        device=device,
        model_type="Codec",
        **model_params,
    )
    return model.to(device).eval()


def adapter_digest(model):
    digest = hashlib.sha256()
    for part_name, state_dict in model.get_state_dicts_for_save().items():
        if part_name in ("state_dict_codec", "state_dict_decoder_relay"):
            continue
        for key in sorted(state_dict):
            value = state_dict[key].detach().cpu().contiguous()
            digest.update(part_name.encode("utf-8"))
            digest.update(key.encode("utf-8"))
            digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bgfr_e2e.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quality", type=int, default=2)
    parser.add_argument("--seed", type=int, default=903)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = normalize_pure_sana_config(yaml.safe_load(file))
    params = config["model"]["params"]
    generator = torch.Generator(device="cpu").manual_seed(77)
    image = torch.randn(1, 3, 256, 256, generator=generator).to(args.device)

    print("Building B0...", flush=True)
    with torch.no_grad():
        baseline = build_model(params, "none", args.device, args.seed)
        baseline_digest = adapter_digest(baseline)
        baseline_image, baseline_rate = baseline(image, args.quality)
        baseline_image = baseline_image.cpu()
        baseline_rate = {
            key: value.cpu() for key, value in baseline_rate.items()
        }
    del baseline
    gc.collect()
    torch.cuda.empty_cache()

    print("Building B2...", flush=True)
    with torch.no_grad():
        bgfr = build_model(params, "bitstream", args.device, args.seed)
        bgfr_digest = adapter_digest(bgfr)
        bgfr_image, bgfr_rate = bgfr(image, args.quality)
        bgfr_image = bgfr_image.cpu()
        bgfr_rate = {key: value.cpu() for key, value in bgfr_rate.items()}

    report = {
        "adapter_digest_equal": baseline_digest == bgfr_digest,
        "output_bit_exact": torch.equal(baseline_image, bgfr_image),
        "output_max_abs": (baseline_image - bgfr_image).abs().max().item(),
        "likelihoods_bit_exact": all(
            torch.equal(baseline_rate[key], bgfr_rate[key])
            for key in baseline_rate
        ),
        "relay_parameters": sum(
            parameter.numel() for parameter in bgfr.decoder_relay.parameters()
        ),
        "relay_contract": bgfr.decoder_relay.contract(),
    }
    for key, value in report.items():
        print(f"{key}={value}", flush=True)
    if not all(
        report[key]
        for key in (
            "adapter_digest_equal",
            "output_bit_exact",
            "likelihoods_bit_exact",
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
