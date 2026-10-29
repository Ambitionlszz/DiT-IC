"""Self-describing container for variable-rate latent-codec streams."""

from dataclasses import dataclass
import struct
from typing import BinaryIO


PACKET_MAGIC = b"DVR1"
PACKET_FORMAT_VERSION = 1
QUALITY_LEVELS = 6
_HEADER = struct.Struct(">4sBBIIH")
_GROUP_COUNT = struct.Struct(">H")
_STREAM_LENGTH = struct.Struct(">I")


@dataclass(frozen=True)
class CodecPacket:
    strings: list[list[bytes]]
    shape: tuple[int, int]
    quality_index: int
    format_version: int = PACKET_FORMAT_VERSION

    def __post_init__(self):
        if self.format_version != PACKET_FORMAT_VERSION:
            raise ValueError(
                f"format_version must be {PACKET_FORMAT_VERSION}, "
                f"got {self.format_version}"
            )
        if isinstance(self.quality_index, bool) or not isinstance(
            self.quality_index, int
        ):
            raise TypeError("quality_index must be an integer")
        if not 0 <= self.quality_index < QUALITY_LEVELS:
            raise ValueError(
                f"quality_index must be in [0, {QUALITY_LEVELS - 1}], "
                f"got {self.quality_index}"
            )
        if len(self.shape) != 2 or any(int(value) <= 0 for value in self.shape):
            raise ValueError("shape must contain two positive integers")
        if not self.strings:
            raise ValueError("strings must contain at least one stream group")
        for group in self.strings:
            if not isinstance(group, list) or not group:
                raise ValueError("each stream group must be a non-empty list")
            if any(not isinstance(stream, bytes) for stream in group):
                raise TypeError("all encoded streams must be bytes")


def _read_exact(fd: BinaryIO, size: int) -> bytes:
    value = fd.read(size)
    if len(value) != size:
        raise EOFError("truncated variable-rate codec packet")
    return value


def write_codec_packet(fd: BinaryIO, packet: CodecPacket) -> int:
    """Write a packet and return the exact number of billed bytes."""
    written = fd.write(
        _HEADER.pack(
            PACKET_MAGIC,
            packet.format_version,
            packet.quality_index,
            int(packet.shape[0]),
            int(packet.shape[1]),
            len(packet.strings),
        )
    )
    for group in packet.strings:
        written += fd.write(_GROUP_COUNT.pack(len(group)))
        for stream in group:
            written += fd.write(_STREAM_LENGTH.pack(len(stream)))
            written += fd.write(stream)
    return written


def read_codec_packet(fd: BinaryIO) -> CodecPacket:
    """Read one self-describing variable-rate packet."""
    magic, version, quality_index, height, width, group_count = _HEADER.unpack(
        _read_exact(fd, _HEADER.size)
    )
    if magic != PACKET_MAGIC:
        raise ValueError("not a DiT-IC variable-rate codec packet")
    if version != PACKET_FORMAT_VERSION:
        raise ValueError(f"unsupported codec packet format version: {version}")

    strings = []
    for _ in range(group_count):
        (stream_count,) = _GROUP_COUNT.unpack(_read_exact(fd, _GROUP_COUNT.size))
        group = []
        for _ in range(stream_count):
            (stream_length,) = _STREAM_LENGTH.unpack(
                _read_exact(fd, _STREAM_LENGTH.size)
            )
            group.append(_read_exact(fd, stream_length))
        strings.append(group)

    return CodecPacket(
        strings=strings,
        shape=(height, width),
        quality_index=quality_index,
        format_version=version,
    )
