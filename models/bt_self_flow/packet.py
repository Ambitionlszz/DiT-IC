"""Versioned, independently decodable BT image packets; all bytes are billed."""

from dataclasses import dataclass
import io
import math
import struct


MAGIC = b"BTIC"
VERSION = 1
PRECISION_CONSISTENT_VERSION = 3
# magic/version/backend/mixed, original H/W, y H/W, hyper H/W, lambda low/high.
_HEADER = struct.Struct(">4sBBB6I2H")
_LENGTHS = struct.Struct(">5I")
BACKENDS = {"C": 0, "D": 1}
FIXED_HEADER_BYTES = _HEADER.size + _LENGTHS.size


def quality_code(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("lambda must be finite and in [0, 1]")
    return int(round(value * 65535))


def pack_codes(values, bits):
    """MSB-first codes, with zero padding only at the end of the payload."""
    if not 1 <= bits <= 8:
        raise ValueError("code width must be in [1, 8]")
    data = bytearray()
    buffer = count = 0
    for value in values:
        value = int(value)
        if not 0 <= value < (1 << bits):
            raise ValueError("code outside its bit width")
        buffer = (buffer << bits) | value
        count += bits
        while count >= 8:
            count -= 8
            data.append((buffer >> count) & 255)
            buffer &= (1 << count) - 1
    if count:
        data.append(buffer << (8 - count))
    return bytes(data)


def unpack_codes(payload, bits, count):
    if not 1 <= bits <= 8 or len(payload) != (bits * count + 7) // 8:
        raise ValueError("invalid packed-code length")
    values = []
    buffer = available = 0
    for byte in payload:
        buffer = (buffer << 8) | byte
        available += 8
        while available >= bits and len(values) < count:
            available -= bits
            values.append((buffer >> available) & ((1 << bits) - 1))
            buffer &= (1 << available) - 1
    if buffer:
        raise ValueError("nonzero packed-code padding")
    return values


@dataclass(frozen=True)
class RateControl:
    """Canonical packet-recoverable quality, optionally a binary mixture.

    The mask is a bool tensor [B, 1, Hy, Wy] in training and B=1 for packets.
    No arbitrary floating-point spatial quality map is accepted at decode.
    """

    low: float = 0.5
    high: float | None = None
    high_mask: object = None

    def __post_init__(self):
        quality_code(self.low)
        if (self.high is None) != (self.high_mask is None):
            raise ValueError("mixed control needs both high lambda and mask")
        if self.high is not None and quality_code(self.high) < quality_code(self.low):
            raise ValueError("high lambda must not be below low lambda")

    @property
    def low_code(self):
        return quality_code(self.low)

    @property
    def high_code(self):
        return self.low_code if self.high is None else quality_code(self.high)

    @property
    def value(self):
        return self.low_code / 65535

    @property
    def mixed(self):
        return self.high is not None

    def tensor(self, batch, height, width, device):
        import torch

        low = torch.full((batch, 1, height, width), self.value, device=device)
        if not self.mixed:
            return low
        mask = self.high_mask
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise TypeError("high_mask must be a bool tensor")
        if tuple(mask.shape) != tuple(low.shape):
            raise ValueError("high_mask must match the codec spatial grid")
        return torch.where(mask.to(device), self.high_code / 65535, low)


@dataclass(frozen=True)
class BTImagePacket:
    backend: str
    original_shape: tuple[int, int]
    y_shape: tuple[int, int]
    hyper_shape: tuple[int, int]
    low_code: int
    high_code: int
    hyper_stream: bytes
    main_stream: bytes
    allocation: bytes
    scales: bytes = b""
    mask: bytes = b""
    version: int = VERSION

    def __post_init__(self):
        if self.version not in (VERSION, PRECISION_CONSISTENT_VERSION):
            raise ValueError("unsupported BT packet version")
        if self.version == PRECISION_CONSISTENT_VERSION and (self.backend != "C" or self.mask):
            raise ValueError("precision-consistent packets require pure C and whole-image quality")
        if self.backend not in BACKENDS:
            raise ValueError("BT backend must be C or D (E deploys as D)")
        for shape in (self.original_shape, self.y_shape, self.hyper_shape):
            if len(shape) != 2 or any(int(v) != v or not 0 < v < 2**32 for v in shape):
                raise ValueError("packet shapes must contain two positive uint32 values")
        if not 0 <= self.low_code <= self.high_code <= 65535:
            raise ValueError("invalid packet lambda codes")
        h, w = self.y_shape
        if (h % 4 or w % 4 or self.hyper_shape != (h // 4, w // 4)
                or self.original_shape[0] > h * 64 or self.original_shape[1] > w * 64):
            raise ValueError("inconsistent image/latent/hyper dimensions")
        for payload in self.payloads:
            if not isinstance(payload, bytes):
                raise TypeError("packet payloads must be bytes")
        if not self.hyper_stream or not self.main_stream:
            raise ValueError("entropy streams cannot be empty")
        unpack_codes(self.allocation, 3, h * w)
        if self.backend == "D":
            unpack_codes(self.scales, 4, h * w)
        elif self.scales:
            raise ValueError("C packets do not contain a scale map")
        if self.mask:
            unpack_codes(self.mask, 1, h * w)
        elif self.low_code != self.high_code:
            raise ValueError("two different lambdas require a mask")

    @property
    def payloads(self):
        return (self.hyper_stream, self.main_stream, self.allocation, self.scales, self.mask)

    def to_bytes(self):
        header = _HEADER.pack(MAGIC, self.version, BACKENDS[self.backend], bool(self.mask),
                              *self.original_shape, *self.y_shape, *self.hyper_shape,
                              self.low_code, self.high_code)
        return header + _LENGTHS.pack(*(len(p) for p in self.payloads)) + b"".join(self.payloads)

    @classmethod
    def from_bytes(cls, data):
        if not isinstance(data, bytes) or len(data) < FIXED_HEADER_BYTES:
            raise ValueError("truncated BT packet")
        fields = _HEADER.unpack(data[:_HEADER.size])
        magic, version, backend, mixed, oh, ow, yh, yw, hh, hw, low, high = fields
        if (magic != MAGIC or version not in (VERSION, PRECISION_CONSISTENT_VERSION)
                or backend not in BACKENDS.values() or mixed not in (0, 1)):
            raise ValueError("unsupported BT packet header")
        lengths = _LENGTHS.unpack(data[_HEADER.size:FIXED_HEADER_BYTES])
        if len(data) != FIXED_HEADER_BYTES + sum(lengths):
            raise ValueError("truncated packet or trailing bytes")
        fd = io.BytesIO(data[FIXED_HEADER_BYTES:])
        payloads = [fd.read(size) for size in lengths]
        if bool(payloads[-1]) != bool(mixed):
            raise ValueError("mask/header mismatch")
        return cls("C" if backend == 0 else "D", (oh, ow), (yh, yw), (hh, hw),
                   low, high, *payloads, version=version)

    def rate_control(self, device="cpu"):
        import torch

        if not self.mask:
            return RateControl(self.low_code / 65535)
        h, w = self.y_shape
        mask = torch.tensor(unpack_codes(self.mask, 1, h*w), dtype=torch.bool,
                            device=device).reshape(1, 1, h, w)
        return RateControl(self.low_code / 65535, self.high_code / 65535, mask)

    def byte_breakdown(self):
        return dict(zip(("hyper", "main", "allocation", "scales", "mask"),
                        map(len, self.payloads)), header=FIXED_HEADER_BYTES)


def fixed_side_bits(height, width, backend, mixed=False):
    count = height * width
    return 8 * (FIXED_HEADER_BYTES + (3*count+7)//8
                + ((4*count+7)//8 if backend == "D" else 0)
                + ((count+7)//8 if mixed else 0))
