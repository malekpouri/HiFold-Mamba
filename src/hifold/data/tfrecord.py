"""Minimal TFRecord + tf.train.Example reader (no TensorFlow dependency).

TFRecord framing per record:
    uint64 length | uint32 masked_crc32c(length) | bytes data | uint32 masked_crc32c(data)
Files may be ZLIB- or GZIP-compressed as a whole stream (Basenji/Akita writes ZLIB).

tf.train.Example is decoded with a small protobuf wire-format parser that supports exactly
the fields Example uses: Features{map<string, Feature>}, Feature{bytes_list|float_list|int64_list}.
"""

import gzip
import struct
import zlib
from pathlib import Path

import numpy as np

# --- CRC-32C (Castagnoli), used by TFRecord framing ---------------------------------------

_CRC_TABLE = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = (_c >> 1) ^ 0x82F63B78 if _c & 1 else _c >> 1
    _CRC_TABLE.append(_c)


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    table = _CRC_TABLE
    for b in data:
        crc = table[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def masked_crc(data: bytes) -> int:
    crc = crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


# --- Stream opening ---------------------------------------------------------------------

class _ZlibReader:
    """File-like wrapper that inflates a ZLIB stream on the fly."""

    def __init__(self, fh):
        self._fh = fh
        self._d = zlib.decompressobj()
        self._buf = b""

    def read(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self._fh.read(1 << 20)
            if not chunk:
                self._buf += self._d.flush()
                break
            self._buf += self._d.decompress(chunk)
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def close(self):
        self._fh.close()


def _open(path: Path, compression: str):
    if compression == "auto":
        with open(path, "rb") as fh:
            head = fh.read(2)
        if head[:2] == b"\x1f\x8b":
            compression = "gzip"
        elif len(head) == 2 and head[0] == 0x78 and (head[0] * 256 + head[1]) % 31 == 0:
            compression = "zlib"
        else:
            compression = "none"
    if compression == "gzip":
        return gzip.open(path, "rb")
    if compression == "zlib":
        return _ZlibReader(open(path, "rb"))
    return open(path, "rb")


def iter_records(path, compression: str = "auto", check_crc: bool = True):
    """Yield the raw bytes of each record in a TFRecord file."""
    fh = _open(Path(path), compression)
    try:
        while True:
            header = fh.read(12)
            if not header:
                return
            if len(header) < 12:
                raise ValueError(f"{path}: truncated record header")
            (length,) = struct.unpack("<Q", header[:8])
            (len_crc,) = struct.unpack("<I", header[8:])
            if check_crc and masked_crc(header[:8]) != len_crc:
                raise ValueError(f"{path}: length CRC mismatch")
            data = fh.read(length)
            footer = fh.read(4)
            if len(data) < length or len(footer) < 4:
                raise ValueError(f"{path}: truncated record")
            if check_crc and masked_crc(data) != struct.unpack("<I", footer)[0]:
                raise ValueError(f"{path}: data CRC mismatch")
            yield data
    finally:
        fh.close()


# --- Protobuf wire format ---------------------------------------------------------------

def _varint(buf: bytes, pos: int):
    result = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def _fields(buf: bytes):
    """Yield (field_number, wire_type, value) for one message; value is bytes or int."""
    pos, end = 0, len(buf)
    while pos < end:
        key, pos = _varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            val, pos = _varint(buf, pos)
        elif wire == 2:
            n, pos = _varint(buf, pos)
            val, pos = buf[pos:pos + n], pos + n
        elif wire == 5:
            val, pos = buf[pos:pos + 4], pos + 4
        elif wire == 1:
            val, pos = buf[pos:pos + 8], pos + 8
        else:
            raise ValueError(f"unsupported wire type {wire}")
        yield field, wire, val


def _decode_feature(buf: bytes):
    # Feature: 1 = BytesList, 2 = FloatList, 3 = Int64List
    for field, _, val in _fields(buf):
        if field == 1:
            return [v for f, _, v in _fields(val) if f == 1]
        if field == 2:
            out = []
            for f, wire, v in _fields(val):
                if f == 1:
                    out.append(np.frombuffer(v, dtype="<f4"))  # packed (wire 2) or single (wire 5)
            return np.concatenate(out) if out else np.zeros(0, np.float32)
        if field == 3:
            out = []
            for f, wire, v in _fields(val):
                if f == 1 and wire == 0:
                    out.append(v)
                elif f == 1 and wire == 2:
                    p = 0
                    while p < len(v):
                        x, p = _varint(v, p)
                        out.append(x)
            # two's complement for negative int64
            return np.array([x - (1 << 64) if x >= 1 << 63 else x for x in out], dtype=np.int64)
    return None


def parse_example(record: bytes) -> dict:
    """Decode a serialized tf.train.Example into {name: list[bytes] | ndarray}."""
    out = {}
    for field, _, features in _fields(record):
        if field != 1:  # Example.features
            continue
        for f, _, entry in _fields(features):
            if f != 1:  # Features.feature (map entry)
                continue
            key = value = None
            for ef, _, ev in _fields(entry):
                if ef == 1:
                    key = ev.decode()
                elif ef == 2:
                    value = _decode_feature(ev)
            out[key] = value
    return out


def iter_examples(path, compression: str = "auto"):
    for rec in iter_records(path, compression):
        yield parse_example(rec)
