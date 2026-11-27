# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import json

# Third Party
import numpy as np
import pytest
import torch

# First Party
from lmcache.inprocess.v1.storage_backend.gds_backend import (
    _DATA_FILE_SUFFIX,
    _METADATA_FILE_SUFFIX,
    _METADATA_MAX_SIZE,
    _METADATA_VERSION,
    GdsBackend,
    pack_metadata,
)
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat

NUM_LAYERS = 3
CHUNK_SIZE = 8
KV_HEADS = 8
HEAD_DIM = 4
BUCKET = 8
SCALE = 0.25


def _write_archive(backend, key, kv):
    path = Path(backend._key_to_path(key)[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    header = pack_metadata(
        kv, MemoryFormat.KV_2LTD, lmcache_version=str(_METADATA_VERSION)
    )
    path.write_bytes(header + kv.view(torch.uint8).numpy().tobytes())
    Path(str(path) + _METADATA_FILE_SUFFIX).write_bytes(header)
    return path


@pytest.mark.parametrize("tp_size", [2, 4])
def test_qkpack_packs_head_shards_in_token_rows(tmp_path, tp_size):
    # Packing only consumes on-disk archives. No CuFile or GPU pool is needed.
    backend = GdsBackend.__new__(GdsBackend)
    backend.gds_path = str(tmp_path)
    backend.gds_paths = [str(tmp_path)]
    backend.data_suffix = _DATA_FILE_SUFFIX
    backend._aissd_qkpack_tp_size = tp_size
    backend._aissd_qkpack_immutable_source_kv = True
    backend._aissd_qkpack_fsync = False
    backend._aissd_qkpack_abi_path = str(tmp_path / "abi.json")
    backend._aissd_qkpack_model_profile = "test_model"
    backend._aissd_qkpack_abi = {
        "num_layers": NUM_LAYERS,
        "num_q_heads": 40,
        "buckets": {
            str(BUCKET): {
                "packed_k_chunk_abi": {
                    "chunk_size": CHUNK_SIZE,
                    "num_kv_heads": KV_HEADS,
                    "head_dim": HEAD_DIM,
                    "row_stride_bytes": KV_HEADS * HEAD_DIM * 2,
                    "chunk_bytes": CHUNK_SIZE * KV_HEADS * HEAD_DIM * 2,
                    "scale": SCALE,
                    "zero_point": -3,
                },
            },
        },
    }
    backend._aissd_qkpack_abi_identity = backend._qkpack_abi_identity(
        backend._aissd_qkpack_abi
    )
    rank_width = KV_HEADS * HEAD_DIM // tp_size
    full_kv = (
        torch.arange(2 * NUM_LAYERS * CHUNK_SIZE * KV_HEADS * HEAD_DIM)
        .reshape(2, NUM_LAYERS, CHUNK_SIZE, KV_HEADS * HEAD_DIM)
        .div(16)
        .to(torch.bfloat16)
    )
    key = CacheEngineKey("tp-model", tp_size, 0, 54321, torch.bfloat16)
    paths = [
        _write_archive(backend, key.with_new_worker_id(rank), rank_kv.contiguous())
        for rank, rank_kv in enumerate(full_kv.split(rank_width, dim=-1))
    ]

    def prepare():
        return Path(
            backend.prepare_tp_qkpack_sidecar(
                key,
                str(paths[0]),
                layer=1,
                bucket=BUCKET,
                num_q_heads=40,
                kv_shape=(NUM_LAYERS, CHUNK_SIZE, KV_HEADS, HEAD_DIM),
            )
        )

    sidecar = prepare()
    data = sidecar.read_bytes()
    header = json.loads(data[:_METADATA_MAX_SIZE].split(b"\0", 1)[0])
    assert header["tp_size"] == tp_size
    assert header["num_kv_heads"] == KV_HEADS
    assert header["shape_per_chunk"] == [CHUNK_SIZE, KV_HEADS * HEAD_DIM]
    expected = torch.round(full_kv[0, 1].float() / SCALE - 3).to(torch.int16)
    np.testing.assert_array_equal(
        np.frombuffer(data[_METADATA_MAX_SIZE:], dtype="<i2"),
        expected.numpy().reshape(-1),
    )
    inode = sidecar.stat().st_ino
    assert prepare() == sidecar
    assert sidecar.stat().st_ino == inode

    missing = paths[-1]
    missing.unlink()
    with pytest.raises(
        RuntimeError, match=f"Missing SSD Sparse KV TP shard.*{tp_size - 1}"
    ):
        prepare()
    updated_shard = full_kv[..., -rank_width:].contiguous() + 5
    _write_archive(backend, key.with_new_worker_id(tp_size - 1), updated_shard)
    updated = prepare()
    assert updated != sidecar
    expected[..., -rank_width:] = torch.round(
        updated_shard[0, 1].float() / SCALE - 3
    ).to(torch.int16)
    np.testing.assert_array_equal(
        np.frombuffer(updated.read_bytes()[_METADATA_MAX_SIZE:], dtype="<i2"),
        expected.numpy().reshape(-1),
    )
