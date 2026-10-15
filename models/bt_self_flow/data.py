"""Counter-based data: worker prefetch and model RNG never change the crops."""

import hashlib
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler


EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def counter_seed(seed, counter, stream=0):
    digest = hashlib.blake2b(f"{seed}:{stream}:{counter}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63)-1)


class ImageDataset(Dataset):
    def __init__(self, roots, crop=512, training=True, limit=None):
        paths = []
        for root in roots:
            directory = Path(root)
            if not directory.is_dir():
                raise ValueError(f"missing image directory: {root}")
            paths.extend(p for p in directory.rglob("*") if p.suffix.lower() in EXTENSIONS and p.is_file())
        self.paths = sorted(set(paths))
        if limit:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise ValueError("image dataset is empty")
        self.crop = crop
        self.training = training
        self.fingerprint = hashlib.sha256("\n".join(map(str, self.paths)).encode()).hexdigest()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, key):
        index, seed = key if isinstance(key, tuple) else (key, 0)
        rng = random.Random(seed)
        with Image.open(self.paths[index]) as source:
            image = source.convert("RGB")
        size = self.crop
        if size:
            w, h = image.size
            if min(w, h) < size:
                factor = size/min(w, h)
                image = image.resize((max(size, round(w*factor)), max(size, round(h*factor))), Image.Resampling.BICUBIC)
            w, h = image.size
            left = rng.randrange(w-size+1) if self.training else (w-size)//2
            top = rng.randrange(h-size+1) if self.training else (h-size)//2
            image = image.crop((left, top, left+size, top+size))
        if self.training:
            if rng.random() < .5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if rng.random() < .5:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        pixels = np.array(image, dtype=np.uint8, copy=True)
        return torch.from_numpy(pixels).permute(2, 0, 1).float().div(127.5).sub(1)


class CounterSampler(Sampler):
    def __init__(self, length, seed, rank=0, world_size=1, batch_size=1,
                 micro_batches_seen=0, stream=0):
        self.length, self.seed, self.rank = length, seed, rank
        self.world_size, self.batch_size = world_size, batch_size
        self.start, self.stream = micro_batches_seen, stream

    def __iter__(self):
        micro = self.start
        while True:
            for local in range(self.batch_size):
                counter = (micro*self.world_size+self.rank)*self.batch_size+local
                seed = counter_seed(self.seed, counter, self.stream)
                index = random.Random(seed).randrange(self.length)
                yield index, counter_seed(seed, 0, 91)
            micro += 1

    def __len__(self):
        return 2**60
