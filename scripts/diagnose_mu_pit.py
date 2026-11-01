"""Paired mu-condition interventions with the formal SANA/AE feature fixed."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from models.bt_self_flow.checkpoint import read_checkpoint, sha256_file
from models.bt_self_flow.data import ImageDataset
from models.bt_self_flow.experiment import asset_identity, make_model, MU_PIXEL_METHODS
from models.bt_self_flow.losses import ImageLoss
from models.bt_self_flow.packet import RateControl


@torch.no_grad()
def capture_packet_tail(model, packet):
    """Cache actual receiver inputs immediately before D; run SANA once."""
    saved = {}
    def capture(module, inputs):
        saved["features"], saved["mu"] = [value.detach().clone() for value in inputs]
    handle = model.pixel_adapter.register_forward_pre_hook(capture)
    before = model.prediction_calls
    try:
        image = model.decompress(packet)
    finally:
        handle.remove()
    if model.prediction_calls - before != 1 or not saved:
        raise AssertionError("mu PiT diagnosis requires one formal receiver prediction")
    return image, saved["features"], saved["mu"]


@torch.no_grad()
def decode_cached_tail(model, features, mu, original_hw, bypass=False):
    """Replay only D and the original RGB tail; no codec/SANA/AE upsampling."""
    base = model.vae.get_base_model() if hasattr(model.vae, "get_base_model") else model.vae
    value = features if bypass else model.pixel_adapter(features, mu, checkpoint_blocks=False)
    value = base.decoder.norm_out(value.movedim(1, -1)).movedim(-1, 1)
    image = base.decoder.conv_out(base.decoder.conv_act(value)).clamp(-1, 1)
    height, width = original_hw
    return image[..., :height, :width]


@torch.no_grad()
def diagnose(args):
    checkpoint = read_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    if config.get("decoder", {}).get("method") not in MU_PIXEL_METHODS:
        raise ValueError("diagnosis requires a mu-only PiT A/B checkpoint")
    if args.sana_path:
        config["model"]["sana_path"] = args.sana_path
    if args.text_path:
        config["model"]["text_path"] = args.text_path
    if checkpoint["assets"] != asset_identity(config):
        raise ValueError("checkpoint pretrained asset identity differs")
    device = torch.device(args.device)
    model = make_model(config, device).eval()
    model.load_bt_state(checkpoint[args.weights])
    model.requires_grad_(False)
    model.codec.update(force=True)
    dataset = ImageDataset(args.input, crop=args.crop, training=False, limit=args.limit)
    if len(dataset) < 2:
        raise ValueError("wrong-image intervention requires at least two images")
    loss = ImageLoss(**config["image_loss"]).to(device).eval()
    from pytorch_msssim import ms_ssim
    rows = []
    for quality in args.qualities:
        # All cropped images share the same grid; wrong mu is from another real
        # same-quality packet and never replaces the recipient's formal input.
        packets, received = [], []
        for index in range(len(dataset)):
            image = dataset[index].unsqueeze(0).to(device)
            packet = model.compress(image, RateControl(quality))
            packets.append(packet.to_bytes())
            received.append(model.codec.decompress(packet)["mean"])
        for index, data in enumerate(packets):
            target = dataset[index].unsqueeze(0).to(device)
            normal, features, mu = capture_packet_tail(model, data)
            before = model.prediction_calls
            wrong = (index + 1) % len(dataset)
            interventions = dict(normal=mu, zero_mu=torch.zeros_like(mu), wrong_mu=received[wrong],
                                 shifted_mu=mu.roll(1, dims=-1), bypass=mu)
            normal_d = None
            for name, condition in interventions.items():
                image = decode_cached_tail(model, features, condition, target.shape[-2:], bypass=name == "bypass")
                if name == "normal":
                    torch.testing.assert_close(image, normal, atol=2e-6, rtol=2e-6)
                distortion, metrics = loss(image, target)
                if name == "normal":
                    normal_d = float(distortion)
                rows.append(dict(quality=quality, image=str(dataset.paths[index]),
                                 wrong_image=str(dataset.paths[wrong]) if name == "wrong_mu" else None,
                                 intervention=name, actual_bpp=8 * len(data) / (args.crop ** 2),
                                 distortion=float(distortion), distortion_delta=float(distortion) - normal_d,
                                 output_rmse_01=float(((image - normal) / 2).square().mean().sqrt()),
                                 ms_ssim=float(ms_ssim((image + 1) / 2, (target + 1) / 2, data_range=1.)),
                                 **{key: float(value) for key, value in metrics.items()}))
            if model.prediction_calls != before:
                raise AssertionError("condition interventions must not rerun SANA")
    summary = {}
    for quality in args.qualities:
        summary[str(quality)] = {}
        for name in interventions:
            selected = [row for row in rows if row["quality"] == quality and row["intervention"] == name]
            keys = ("distortion", "distortion_delta", "output_rmse_01", "actual_bpp", "psnr", "ms_ssim", "lpips", "dists")
            summary[str(quality)][name] = {key: sum(row[key] for row in selected) / len(selected) for key in keys}
    return dict(checkpoint_sha256=sha256_file(args.checkpoint), weights=args.weights,
                decoder_method=config["decoder"]["method"], crop=args.crop,
                fixed_formal_sana_and_ae_features=True, summary=summary, rows=rows,
                interpretation="Paired condition dependence; output sensitivity alone does not establish useful conditioning.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", nargs="+", required=True, help="image directories")
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", choices=("raw", "ema"), default="ema")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--qualities", type=float, nargs="+", default=[.2, .6])
    parser.add_argument("--crop", type=int, default=512)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--sana-path")
    parser.add_argument("--text-path")
    args = parser.parse_args()
    if args.crop < 192 or args.limit < 2 or not args.qualities:
        parser.error("use crop >=192, limit >=2 and at least one quality")
    path = Path(args.output)
    if path.exists():
        parser.error("report already exists; choose another output")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    report = diagnose(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
