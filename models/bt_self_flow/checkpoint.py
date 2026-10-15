"""Separate BT experiment contract, deployment EMA and exact continuation state."""

from contextlib import contextmanager
import hashlib
from pathlib import Path
import random

import numpy as np
import torch


METHOD = "bt_self_flow_v1"
SCHEMA = 1


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for data in iter(lambda: source.read(8*1024*1024), b""):
            digest.update(data)
    return digest.hexdigest()


def clone_state(state, cpu=False):
    return {part: {name: value.detach().to("cpu" if cpu else value.device).clone()
                   for name, value in values.items()} for part, values in state.items()}


class DeploymentEMA:
    def __init__(self, model, decay=.999):
        self.decay = float(decay)
        self.state = clone_state(model.export_state())
        components = {"codec": model.codec, "distortion_scale": model.distortion_scale}
        if hasattr(model, "DiT"):
            components.update(vae=model.vae, dit=model.DiT, text=model.text_proj,
                              conditioner=model.quality_conditioner, projector=model.representation_projector)
        self.averaged = {(part, name) for part, module in components.items() if module is not None
                         for name, param in module.named_parameters() if param.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for part, values in model.export_state().items():
            for name, source in values.items():
                target = self.state[part][name]
                if target.shape != source.shape:
                    self.state[part][name] = source.detach().clone()
                elif (part, name) in self.averaged:
                    target.lerp_(source.detach(), 1-self.decay)
                else:
                    target.copy_(source.detach())


def capture_rng(device=None):
    numpy = np.random.get_state()
    return {"python": random.getstate(), "numpy": (numpy[0], numpy[1].tolist(), *numpy[2:]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device is not None and device.type == "cuda" else None}


def restore_rng(state):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu())


def save_atomic(path, checkpoint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def read_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("method") != METHOD or checkpoint.get("schema") != SCHEMA:
        raise ValueError("not a supported BT checkpoint; legacy checkpoints cannot be mixed")
    return checkpoint


@contextmanager
def evaluation_state(model, state=None):
    training = model.training
    saved = clone_state(model.export_state()) if state is not None else None
    try:
        if state is not None:
            model.load_bt_state(state)
        model.eval()
        model.codec.update(force=True)
        yield
    finally:
        if saved is not None:
            model.load_bt_state(saved)
        model.train(training)
