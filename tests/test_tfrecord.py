"""TFRecord reader tests on a hand-built file (no TensorFlow needed)."""

import struct
import zlib

import numpy as np
import pytest

from hifold.data import tfrecord as T


def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _ld(field, payload):  # length-delimited field
    return _varint((field << 3) | 2) + _varint(len(payload)) + payload


def _example(features):
    entries = b""
    for key, value in features.items():
        if isinstance(value, bytes):
            feat = _ld(1, _ld(1, value))  # BytesList
        else:
            feat = _ld(2, _ld(1, np.asarray(value, "<f4").tobytes()))  # packed FloatList
        entries += _ld(1, _ld(1, key.encode()) + _ld(2, feat))
    return _ld(1, entries)


def _frame(data):
    length = struct.pack("<Q", len(data))
    return (length + struct.pack("<I", T.masked_crc(length)) + data
            + struct.pack("<I", T.masked_crc(data)))


def test_crc32c_known_vector():
    # RFC 3720 test vector: 32 bytes of zeros -> 0x8A9136AA
    assert T.crc32c(b"\x00" * 32) == 0x8A9136AA
    assert T.crc32c(b"123456789") == 0xE3069283


@pytest.mark.parametrize("compress", [False, True])
def test_roundtrip(tmp_path, compress):
    seq = np.eye(4, dtype=np.uint8)[np.random.default_rng(0).integers(0, 4, 64)]
    tgt = np.random.default_rng(1).normal(size=(10, 5)).astype(np.float16)
    recs = [_example({"sequence": seq.tobytes(), "target": tgt.tobytes(), "f": [1.5, -2.0]})
            for _ in range(3)]
    raw = b"".join(_frame(r) for r in recs)
    path = tmp_path / "x.tfr"
    path.write_bytes(zlib.compress(raw) if compress else raw)
    out = list(T.iter_examples(path))
    assert len(out) == 3
    ex = out[0]
    assert np.array_equal(np.frombuffer(ex["sequence"][0], np.uint8).reshape(-1, 4), seq)
    assert np.array_equal(np.frombuffer(ex["target"][0], np.float16).reshape(10, 5), tgt)
    assert np.allclose(ex["f"], [1.5, -2.0])


def test_corruption_detected(tmp_path):
    raw = bytearray(_frame(_example({"a": b"hello"})))
    raw[-6] ^= 0xFF
    path = tmp_path / "bad.tfr"
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="CRC"):
        list(T.iter_records(path))


def test_all_zero_map_is_marked_missing(tmp_path):
    import json
    import numpy as np
    from hifold.data.dataset import ContactCache
    d = tmp_path / "train"; d.mkdir()
    np.save(d / "seq.u8.npy", np.zeros((2, 64), np.uint8))
    t = np.random.default_rng(0).normal(size=(2, 10, 2)).astype(np.float16)
    t[1, :, 1] = 0
    np.save(d / "target.f16.npy", t)
    (d / "windows.tsv").write_text("chr1\t0\t64\nchr1\t64\t128\n")
    (d / "meta.json").write_text(json.dumps({"lines": ["A", "B"]}))
    ds = ContactCache(tmp_path, "train")
    assert not np.isnan(ds[0][1].numpy()).any()
    y = ds[1][1].numpy()
    assert np.isnan(y[:, 1]).all() and not np.isnan(y[:, 0]).any()
