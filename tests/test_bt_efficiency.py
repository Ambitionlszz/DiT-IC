"""Optimizations must preserve validation values and recoverable files."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from models.bt_self_flow.checkpoint import METHOD, SCHEMA, evaluation_state, read_checkpoint, save_atomic
from models.bt_self_flow.losses import reconstruction_objective
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
