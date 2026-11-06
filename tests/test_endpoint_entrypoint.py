"""Real trainer handoff, continuation, and batch4/accum1 two-rank checks."""

from contextlib import redirect_stdout
from copy import deepcopy
from datetime import timedelta
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import train_bt_self_flow as trainer
from models.bt_self_flow.checkpoint import METHOD, SCHEMA, clone_state, read_checkpoint, save_atomic
from models.bt_self_flow.experiment import make_model
from models.bt_self_flow.model import BTCodec, pad_image
from models.bt_self_flow.packet import RateControl
from scripts.launch_endpoint_sf import ablation_configs
from test_endpoint_decoder import tiny_endpoint_model
from test_pixel_adapter import SMALL_PIXEL, activate_output
from test_precision_mu_pixel_adapter import precision_model
from test_spatial_training import check_state_equal


ASSETS = {"test": "real-tiny-SANA-native-C-LoRA-AE"}


def tiny_factory(config, device="cpu"):
    # Production settings are validated separately. A factory patch lets the
    # unchanged trainer exercise real tiny modules on one CPU process as well.
    torch.manual_seed(config["train"]["seed"])
    model = tiny_endpoint_model(sf=config["endpoint_self_flow"]["enabled"],
                                d=config["decoder"]["method"] != "native", device=device,
                                train_rgb_head=config["model"].get("train_rgb_head", False))
    model.endpoint_settings = dict(config["endpoint_self_flow"])
    return model


def setup_run(root, variant="c_d_sf", world=1, batch=1):
    root = Path(root)
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(19)
    for index in range(2):
        Image.fromarray(generator.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(images / f"{index}.png")
    config = ablation_configs()[variant]
    config["data"].update(train=[str(images)], valid=[str(images)], crop=256, valid_crop=256,
                          valid_limit=1, workers=0)
    config["train"].update(world_size=world, batch_size=batch, effective_batch=world * batch,
                           tensorboard=False, steps=2, eval_every=100, save_every=100,
                           full_decode_every=0, packet_samples=0, log_every=1, output=str(root / "run"))
    config["image_loss"] = dict(mse_weight=4., lpips_weight=0., dists_weight=0.)
    if config["decoder"]["method"] != "native":
        config["decoder"]["pixel_adapter"] = dict(SMALL_PIXEL, seed=903, checkpoint_blocks=True)
    pure = deepcopy(config)
    pure["decoder"] = {"method": "native"}
    pure["endpoint_self_flow"].update(enabled=False, target="adjacent_ema")
    model = tiny_factory(pure)
    state = clone_state(model.export_state(), cpu=True)
    source = root / "shared.pt"
    save_atomic(source, dict(method=METHOD, schema=SCHEMA, phase="backend_init", backend="shared", step=0,
                             raw=state, ema=state, config=pure, assets=ASSETS,
                             calibration={"rate_span_pass": True, "quality_span_pass": True}))
    return config, source


def trainer_args(source, resume=None, stop=None):
    return SimpleNamespace(init_checkpoint=str(source) if resume is None else None,
                           resume=str(resume) if resume is not None else None, max_updates=stop)


def test_production_factory_routes_four_variants_with_identical_initial_reconstruction():
    def tiny_base(self, *args, **kwargs):
        base = precision_model()
        del base.pixel_adapter
        self.__dict__.update(base.__dict__)

    models, common = [], None
    for name in ("c", "c_d", "c_sf", "c_d_sf"):
        config = ablation_configs()[name]
        if config["decoder"]["method"] != "native":
            config["decoder"]["pixel_adapter"] = dict(SMALL_PIXEL, seed=903, checkpoint_blocks=True)
        with patch.object(BTCodec, "__init__", tiny_base):
            model = make_model(config, torch.device("cpu")).eval()
        assert model.sf_enabled == config["endpoint_self_flow"]["enabled"]
        assert model.d_enabled == ("_d" in name)
        public = {part: values for part, values in model.export_state().items() if part != "pixel_adapter"}
        if common is not None:
            check_state_equal(public, common)
        common = clone_state(public)
        models.append(model)
    mu = torch.randn(1, 4, 8, 8)
    delta = torch.ones(1, 1, 4, 4)
    with torch.no_grad():
        outputs = [model.decode_received(mu, delta)["image"] for model in models]
    for output in outputs[1:]:
        torch.testing.assert_close(output, outputs[0], atol=2e-6, rtol=1e-6)


def test_real_endpoint_packet_uses_only_received_bits_and_preserves_c_format():
    model, receiver = tiny_endpoint_model().eval(), tiny_endpoint_model().eval()
    activate_output(model.pixel_adapter)
    receiver.load_bt_state(model.export_state())
    native = deepcopy(model)
    native.__class__ = BTCodec
    del native.pixel_adapter
    image = torch.rand(1, 3, 233, 377) * 2 - 1
    for quality in (0., .4, 1.):
        with torch.no_grad():
            packet = model.compress(image, RateControl(quality))
            assert packet.to_bytes() == native.compress(image, RateControl(quality)).to_bytes()
            expected = model(pad_image(image), RateControl(quality)).image[..., :233, :377]
            with patch.object(receiver, "encode_clean_latent", side_effect=AssertionError("sender encoder")), \
                 patch.object(receiver.codec.router, "forward", side_effect=AssertionError("sender router")):
                actual = receiver.decompress(packet.to_bytes())
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_real_endpoint_training_resumes_exactly_including_ema_and_rng(tmp_path):
    config, source = setup_run(tmp_path)
    with patch.object(trainer, "make_model", side_effect=tiny_factory), \
         patch.object(trainer, "asset_identity", return_value=ASSETS), redirect_stdout(io.StringIO()):
        trainer.train(config, trainer_args(source), 0, 1, torch.device("cpu"))
        config["train"]["output"] = str(tmp_path / "split")
        trainer.train(config, trainer_args(source, stop=1), 0, 1, torch.device("cpu"))
        paused = tmp_path / "split" / "paused.pt"
        trainer.train(config, trainer_args(source, resume=paused), 0, 1, torch.device("cpu"))
    full = read_checkpoint(tmp_path / "run" / "complete.pt")
    split = read_checkpoint(tmp_path / "split" / "complete.pt")
    for key in ("raw", "ema", "optimizer", "aux_optimizer", "scheduler", "scaler", "rng_by_rank"):
        check_state_equal(full[key], split[key])
    assert full["step"] == split["step"] == 2
    assert "pixel_adapter" in full["raw"] and "spatial_projector" not in full["raw"]
    assert full.get("teacher") is None  # deployment EMA is the only teacher history


def test_real_resume_weight_change_is_explicit_and_preserves_continuation_history(tmp_path):
    config, source = setup_run(tmp_path)
    config["endpoint_self_flow"]["coefficient"] = 1.
    with patch.object(trainer, "make_model", side_effect=tiny_factory), \
         patch.object(trainer, "asset_identity", return_value=ASSETS), redirect_stdout(io.StringIO()):
        trainer.train(config, trainer_args(source, stop=1), 0, 1, torch.device("cpu"))
        paused = tmp_path / "run" / "paused.pt"
        initial = read_checkpoint(paused)
        config["endpoint_self_flow"]["coefficient"] = .2
        config["train"]["output"] = str(tmp_path / "reweighted")
        args = trainer_args(source, resume=paused)
        with pytest.raises(ValueError, match="resume contract"):
            trainer.train(config, args, 0, 1, torch.device("cpu"))
        args.resume_sf_coefficient = .2
        trainer.train(config, args, 0, 1, torch.device("cpu"))
    saved = read_checkpoint(tmp_path / "reweighted" / "complete.pt")
    assert saved["step"] == saved["attempts"] == 2
    assert saved["micro_seen"] == initial["micro_seen"] + 1
    assert saved["scheduler"]["last_epoch"] == initial["scheduler"]["last_epoch"] + 1
    assert saved["config"]["endpoint_self_flow"]["coefficient"] == .2
    assert saved["objective_transitions"] == [{
        "field": "endpoint_self_flow.coefficient", "old": 1., "new": .2,
        "step": 1, "checkpoint": str(paused.resolve())}]
    check_state_equal(read_checkpoint(paused), initial)


@pytest.mark.parametrize("variant", ["c", "c_d", "c_d_gt_sf"])
def test_real_trainer_controls_do_not_allocate_ema_teacher(tmp_path, variant):
    config, source = setup_run(tmp_path, variant)
    with patch.object(trainer, "make_model", side_effect=tiny_factory), \
         patch.object(trainer, "asset_identity", return_value=ASSETS), \
         patch("models.bt_self_flow.endpoint_self_flow.EndpointEMATeacher",
               side_effect=AssertionError("unexpected EMA teacher allocation")), \
         redirect_stdout(io.StringIO()):
        trainer.train(config, trainer_args(source, stop=1), 0, 1, torch.device("cpu"))
    state = read_checkpoint(tmp_path / "run" / "paused.pt")
    assert state["step"] == 1 and "spatial_projector" not in state["raw"]
    assert ("pixel_adapter" in state["raw"]) == (variant != "c")


@pytest.mark.parametrize("mutation", ["step", "mixed", "projector", "rate", "quality", "backend"])
def test_direct_trainer_rejects_invalid_common_initialization(tmp_path, mutation):
    config, source = setup_run(tmp_path)
    checkpoint = read_checkpoint(source)
    if mutation == "step":
        checkpoint["step"] = 123
    elif mutation == "mixed":
        checkpoint["config"]["self_flow"]["mixed_sampling"] = True
    elif mutation == "projector":
        checkpoint["config"]["spatial_self_flow"] = {"enabled": False, "projector_enabled": True}
    elif mutation == "backend":
        checkpoint["config"]["model"]["backend"] = "D"
    else:
        checkpoint["calibration"][f"{mutation}_span_pass"] = False
    with pytest.raises(ValueError):
        trainer.initialize_from(tiny_factory(config), checkpoint, "joint")


def endpoint_ddp_worker(rank, directory):
    torch.set_num_threads(1)
    root = Path(directory)
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method=f"file://{root / 'rendezvous'}", timeout=timedelta(seconds=120))
    models = []

    def factory(config, device):
        model = tiny_factory(config, device)
        models.append(model)
        return model

    try:
        config = json.loads((root / "ddp.json").read_text())
        with patch.object(trainer, "make_model", side_effect=factory), \
             patch.object(trainer, "asset_identity", return_value=ASSETS), redirect_stdout(io.StringIO()):
            trainer.train(config, trainer_args(root / "shared.pt"), rank, 2, torch.device("cpu"))
        assert config["train"]["batch_size"] == 4 and config["train"]["accumulation_steps"] == 1
        for name, parameter in models[0].named_parameters():
            if parameter.requires_grad:
                reference = parameter.detach().clone()
                dist.broadcast(reference, src=0)
                assert torch.equal(reference, parameter), name
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_two_rank_real_trainer_batch4_accum1_remains_synchronized(tmp_path):
    config, _ = setup_run(tmp_path, world=2, batch=4)
    (tmp_path / "ddp.json").write_text(json.dumps(config))
    mp.spawn(endpoint_ddp_worker, args=(str(tmp_path),), nprocs=2, join=True)
