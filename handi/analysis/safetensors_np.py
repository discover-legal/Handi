"""Raw safetensors reader. Never goes through a model class.

Format: 8-byte little-endian u64 header length N, then N bytes of JSON, then the
byte buffer. Each JSON entry: {"dtype": "BF16", "shape": [..], "data_offsets": [a, b]}
with offsets relative to the start of the byte buffer.
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

_NP_DTYPES = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "BF16": np.dtype("<u2"),  # raw bits; converted below
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("i1"),
    "U8": np.dtype("u1"),
    "BOOL": np.dtype("?"),
}


def read_header(path: str | os.PathLike) -> tuple[dict, int]:
    """Return (header dict without the __metadata__ key, byte offset of the data buffer)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n).decode("utf-8"))
    header.pop("__metadata__", None)
    return header, 8 + n


def tensor_names(path: str | os.PathLike) -> list[str]:
    header, _ = read_header(path)
    return sorted(header)


def _bf16_to_f32(raw_u16: np.ndarray) -> np.ndarray:
    return (raw_u16.astype(np.uint32) << 16).view(np.float32)


def _decode(buf: bytes | memoryview, info: dict) -> np.ndarray:
    dtype = info["dtype"]
    if dtype not in _NP_DTYPES:
        raise ValueError(f"unsupported safetensors dtype {dtype}")
    arr = np.frombuffer(buf, dtype=_NP_DTYPES[dtype])
    if dtype == "BF16":
        arr = _bf16_to_f32(arr)
    else:
        arr = arr.astype(np.float32)
    return arr.reshape(info["shape"])


def load_tensor(path: str | os.PathLike, name: str) -> np.ndarray:
    """Load one tensor as float32 (bf16 widened exactly via uint16 << 16)."""
    header, data_offset = read_header(path)
    if name not in header:
        raise KeyError(f"{name} not in {path}")
    info = header[name]
    a, b = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_offset + a)
        buf = f.read(b - a)
    return _decode(buf, info)


def load_all(
    path: str | os.PathLike, filter: Callable[[str], bool] | None = None
) -> dict[str, np.ndarray]:
    """Load every tensor (optionally those whose name passes `filter`) as float32."""
    header, data_offset = read_header(path)
    out: dict[str, np.ndarray] = {}
    with open(path, "rb") as f:
        for name in sorted(header):
            if filter is not None and not filter(name):
                continue
            info = header[name]
            a, b = info["data_offsets"]
            f.seek(data_offset + a)
            out[name] = _decode(f.read(b - a), info)
    return out


def open_checkpoint(path: str | os.PathLike) -> list[Path]:
    """Return the list of safetensors shard paths for a file or a snapshot directory.

    A directory is resolved via model.safetensors.index.json if present, else
    model.safetensors, else every *.safetensors file sorted by name.
    """
    p = Path(path)
    if p.is_file():
        return [p]
    index = p / "model.safetensors.index.json"
    if index.exists():
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        return [p / s for s in sorted(set(weight_map.values()))]
    single = p / "model.safetensors"
    if single.exists():
        return [single]
    shards = sorted(p.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors files under {p}")
    return shards


def load_sharded(
    path: str | os.PathLike, filter: Callable[[str], bool] | None = None
) -> dict[str, np.ndarray]:
    """load_all over every shard of a checkpoint directory (or single file)."""
    out: dict[str, np.ndarray] = {}
    for shard in open_checkpoint(path):
        out.update(load_all(shard, filter))
    return out


def sharded_tensor_names(path: str | os.PathLike) -> list[str]:
    names: list[str] = []
    for shard in open_checkpoint(path):
        names.extend(tensor_names(shard))
    return sorted(names)


def convert_bin_to_safetensors(bin_path: str | os.PathLike, out_path: str | os.PathLike) -> Path:
    """One-time conversion of a pytorch_model.bin into a safetensors file.

    Tensors are made contiguous; tensors sharing storage are cloned so that
    safetensors (which refuses shared storage) accepts them.
    """
    import torch
    from safetensors.torch import save_file

    state = torch.load(str(bin_path), map_location="cpu", weights_only=True)
    seen: dict[tuple[int, int], str] = {}  # (data_ptr, nbytes) -> first name
    clean: dict[str, "torch.Tensor"] = {}
    for name, t in state.items():
        if not isinstance(t, torch.Tensor):
            continue
        t = t.detach().contiguous()
        key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
        if key in seen:
            t = t.clone()
        else:
            seen[key] = name
        clean[name] = t
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(clean, str(out_path), metadata={"format": "pt", "converted_from": Path(bin_path).name})
    return out_path
