"""Separate BT experiment contract, deployment EMA and exact continuation state."""

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil

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
        if hasattr(model, "residual_head"):
            components["residual"] = model.residual_head
        self.averaged = {(part, name) for part, module in components.items() if module is not None
                         # state_dict includes legacy aliases (down/branch1,
                         # up/branch1). Every exported name must receive the
                         # same EMA; otherwise load_state_dict's later alias
                         # silently overwrites the averaged weight with raw.
                         for name, param in module.named_parameters(remove_duplicate=False) if param.requires_grad}

    @torch.no_grad()
    def update(self, model):
        grouped = {}
        for part, values in model.export_state().items():
            for name, source in values.items():
                target = self.state[part][name]
                if target.shape != source.shape:
                    self.state[part][name] = source.detach().clone()
                elif (part, name) in self.averaged:
                    if target.is_cuda:
                        targets, sources = grouped.setdefault((target.device, target.dtype), ([], []))
                        targets.append(target)
                        sources.append(source.detach())
                    else:
                        target.lerp_(source.detach(), 1-self.decay)
                else:
                    target.copy_(source.detach())
        for targets, sources in grouped.values():
            torch._foreach_lerp_(targets, sources, 1-self.decay)


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


def _replace_and_sync(temporary, target):
    temporary.replace(target)
    # File fsync alone does not persist the rename in its parent directory.
    # Do not announce a completed save before that directory is synchronized.
    descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_checkpoint_alias(source, target):
    """Publish identical recovery bytes without serializing the state again."""
    source, target = Path(source), Path(target)
    temporary = target.with_suffix(".pt.tmp")
    temporary.unlink(missing_ok=True)
    try:
        try:
            os.link(source, temporary)
        except OSError:
            with source.open("rb") as incoming, temporary.open("wb", buffering=8*1024*1024) as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=8*1024*1024)
                outgoing.flush()
                os.fsync(outgoing.fileno())
        _replace_and_sync(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def save_atomic(path, checkpoint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    try:
        # Buffered Python I/O avoids a tiny native write for each tensor/storage
        # on the experiment volume. Publish only a completely flushed file.
        with temporary.open("wb", buffering=8*1024*1024) as target:
            torch.save(checkpoint, target)
            target.flush()
            os.fsync(target.fileno())
        _replace_and_sync(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_checkpoint(path):
    # BT checkpoints always use torch.save's zip format. Map storages lazily
    # instead of reading multi-GB raw/EMA/optimizer copies up front; atomic
    # replacement keeps the mapped inode valid during subsequent saves.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if checkpoint.get("method") != METHOD or checkpoint.get("schema") != SCHEMA:
        raise ValueError("not a supported BT checkpoint; legacy checkpoints cannot be mixed")
    return checkpoint


def prune_checkpoints(directory, keep, best=None):
    """Bound long-run storage while retaining best and complete/paused states."""
    if keep is None:
        return
    if int(keep) < 1:
        raise ValueError("keep_last_checkpoints must be positive")
    snapshots = sorted((p for p in Path(directory).glob("*.pt") if p.stem.isdecimal()),
                       key=lambda p: int(p.stem), reverse=True)
    for path in snapshots[int(keep):]:
        if best is None or path.name != best["checkpoint"]:
            path.unlink()


class BestCheckpoint:
    """Select deployment EMA by mean validation RD loss within one phase."""

    def __init__(self, directory, resume_step=0, restored=None, identity=None):
        self.directory = Path(directory)
        self.identity = identity
        self.state = restored if restored is None or restored.get("validation_identity") == identity else None
        # Older checkpoints have no selection metadata. Only reports with an
        # available weight snapshot at/before the resume point are eligible.
        if self.state is None:
            for report in sorted(self.directory.glob("validation_*.json")):
                step = int(report.stem.split("_")[-1])
                weights = self.directory/f"{step:07d}.pt"
                if step <= resume_step and weights.is_file():
                    values = json.loads(report.read_text())
                    if values.get("_validation_identity") == identity:
                        self.consider(values, step)
        if self.state is not None:
            self.publish(self.directory/self.state["checkpoint"])

    def consider(self, report, step):
        qualities = ("0.0", "0.2", "0.4", "0.6", "0.8", "1.0")
        values = [float(report[quality]["loss"]) for quality in qualities]
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("nonfinite validation checkpoint selection loss")
        score = sum(values)/len(values)
        if self.state is not None and score >= self.state["score"]:
            return False
        self.state = dict(step=step, score=score, metric="mean_six_quality_validation_rd_loss",
                          weights="deployment_ema", checkpoint=f"{step:07d}.pt", validation_identity=self.identity)
        return True

    def publish(self, source):
        source = Path(source)
        if self.state is None or source.name != self.state["checkpoint"]:
            return
        target = self.directory/"best.pt"
        publish_checkpoint_alias(source, target)
        metadata = self.directory/"best.json"
        pending = metadata.with_suffix(".json.tmp")
        pending.write_text(json.dumps(self.state, indent=2)+"\n")
        pending.replace(metadata)


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
