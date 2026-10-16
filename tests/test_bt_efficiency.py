"""Optimizations must preserve validation values and recoverable files."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from models.bt_self_flow.checkpoint import BestCheckpoint, METHOD, SCHEMA, evaluation_state, read_checkpoint, save_atomic
from models.bt_self_flow.losses import reconstruction_objective
from models.bt_self_flow.data import ImageDataset
from models.bt_self_flow.packet import RateControl
from models.bt_self_flow.validation import validate
from test_bt_training import make_tiny_stage


def test_reused_analysis_preserves_every_quality_metric():
    model = make_tiny_stage({"train": {"seed": 903}}, torch.device("cpu"))
    generator = torch.Generator().manual_seed(906)
    images = [torch.rand(3, 256, 256, generator=generator).mul(2).sub(1) for _ in range(2)]
    qualities = (0., .4, 1.)
    expected = {}
    with evaluation_state(model):
        for quality in qualities:
            totals = {}
            for image in images:
                control = RateControl(quality)
                result = model(image.unsqueeze(0), control)
                loss, metrics = reconstruction_objective(result, image.unsqueeze(0), control,
                                                        model.distortion_scale.s_data)
                for name, value in dict(metrics, loss=loss).items():
                    totals[name] = totals.get(name, 0.)+value.item()/len(images)
            expected[str(quality)] = totals
    with patch.object(model.codec, "analyze", wraps=model.codec.analyze) as analysis, \
         patch.object(model, "encode_clean_latent", wraps=model.encode_clean_latent) as encoder:
        actual = validate(model, images, qualities=qualities)
    assert analysis.call_count == encoder.call_count == len(images)
    for quality in expected:
        for name, value in expected[quality].items():
            assert actual[quality][name] == pytest.approx(value, rel=2e-7)


def test_failed_checkpoint_write_keeps_last_recoverable_state():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/"state.pt"
        checkpoint = {"method": METHOD, "schema": SCHEMA, "step": 17, "weight": torch.arange(128.)}
        save_atomic(path, checkpoint)

        def interrupted_save(value, target):
            target.write(b"incomplete tensor storage")
            raise OSError("injected interrupted disk write")

        with patch("models.bt_self_flow.checkpoint.torch.save", side_effect=interrupted_save):
            with pytest.raises(OSError, match="interrupted disk write"):
                save_atomic(path, dict(checkpoint, step=18))
        restored = read_checkpoint(path)
        assert restored["step"] == 17
        assert torch.equal(restored["weight"], checkpoint["weight"])
        assert not path.with_suffix(".pt.tmp").exists()


def test_best_copy_failure_preserves_previous_weights(tmp_path):
    qualities = ("0.0", "0.2", "0.4", "0.6", "0.8", "1.0")
    best = BestCheckpoint(tmp_path)
    for step in (1, 2):
        save_atomic(tmp_path/f"{step:07d}.pt", {"method": METHOD, "schema": SCHEMA, "step": step})
    assert best.consider({q: {"loss": 2.} for q in qualities}, 1)
    best.publish(tmp_path/"0000001.pt")
    assert best.consider({q: {"loss": 1.} for q in qualities}, 2)
    with patch("models.bt_self_flow.checkpoint.os.link", side_effect=OSError("no hardlinks")), \
         patch("models.bt_self_flow.checkpoint.shutil.copyfileobj", side_effect=OSError("disk failure")):
        with pytest.raises(OSError, match="disk failure"):
            best.publish(tmp_path/"0000002.pt")
    assert read_checkpoint(tmp_path/"best.pt")["step"] == 1
    assert not (tmp_path/"best.pt.tmp").exists()


def test_kodak_style_validation_keeps_full_image_and_counts_real_packet(tmp_path):
    import numpy as np
    from PIL import Image
    pixels = np.random.default_rng(903).integers(0, 256, (512, 768, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(tmp_path/"kodim01.png")
    dataset = ImageDataset([tmp_path], crop=0, training=False)
    assert tuple(dataset[0].shape) == (3, 512, 768)
    model = make_tiny_stage({"train": {"seed": 903}}, torch.device("cpu"))
    report = validate(model, dataset, qualities=(0.,), actual_rate=True)
    from models.bt_self_flow.packet import BTImagePacket
    from models.bt_self_flow.checkpoint import evaluation_state
    with evaluation_state(model):
        clean = model.encode_clean_latent(dataset[0].unsqueeze(0))
        encoded = model.codec.compress(clean, RateControl(0.))
        packet = BTImagePacket("C", (512, 768), encoded["y_shape"], encoded["hyper_shape"], 0, 0,
                              encoded["hyper_stream"], encoded["main_stream"], encoded["allocation"])
    assert report["0.0"]["rate_actual"] == pytest.approx(8*len(packet.to_bytes())/(512*768), rel=1e-7)
    assert report["0.0"]["loss"] == pytest.approx(report["0.0"]["base"]+8*report["0.0"]["rate_actual"], rel=1e-6)


def test_best_selection_does_not_mix_validation_datasets(tmp_path):
    import json
    qualities = ("0.0", "0.2", "0.4", "0.6", "0.8", "1.0")
    weights = tmp_path/"0000002.pt"
    save_atomic(weights, {"method": METHOD, "schema": SCHEMA, "step": 2})
    previous = BestCheckpoint(tmp_path, identity={"dataset": "DIV2K"})
    previous.consider({q: {"loss": 1.} for q in qualities}, 2)
    report = {q: {"loss": 1.} for q in qualities}
    report["_validation_identity"] = {"dataset": "DIV2K"}
    (tmp_path/"validation_0000002.json").write_text(json.dumps(report))
    current = BestCheckpoint(tmp_path, resume_step=2, restored=previous.state, identity={"dataset": "Kodak"})
    assert current.state is None
    assert current.consider({q: {"loss": 5.} for q in qualities}, 2)
