# SPDX-License-Identifier: Apache-2.0
"""Stride-aware page copies for AISSD KV buffers.

Physical IDs are checked on the host, then reused as a device index. Neither
operation materializes a contiguous copy of the paged cache.
"""

from operator import index
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Third Party
    import torch


def page_major(
    cache: "torch.Tensor", block_size: int, heads: int, head_dim: int
) -> "torch.Tensor":
    """Return a [page, K/V, token, head, dim] view of a BF16 cache."""
    import torch

    if (
        cache.ndim == 4
        and tuple(cache.shape[1:]) == (heads, block_size, 2 * head_dim)
        and cache.dtype == torch.bfloat16
    ):
        return cache.unflatten(-1, (2, head_dim)).permute(0, 3, 2, 1, 4)
    if (cache.ndim != 5 or tuple(cache.shape[2:]) != (block_size, heads, head_dim)
            or cache.dtype != torch.bfloat16
            or (cache.shape[1] != 2 and cache.shape[0] != 2)):
        raise ValueError(f"Unsupported AISSD BF16 KV page layout: {tuple(cache.shape)}")
    return cache if cache.shape[1] == 2 else cache.transpose(0, 1)


def make_page_index(block_ids, num_pages, device, *, expected):
    """Reject null, duplicate, out-of-range and truncated physical page lists."""
    import torch

    ids = tuple(index(i) for i in block_ids)
    if len(ids) != expected or len(set(ids)) != len(ids):
        raise ValueError("AISSD page table has incorrect length or duplicate pages")
    if any(i <= 0 or i >= num_pages for i in ids):
        raise ValueError("AISSD page table references a null or out-of-range page")
    return torch.tensor(ids, dtype=torch.long, device=device)


def scatter_kv_pages(paged, ids, kv):
    """Copy [K/V, tokens, heads, dim] into the indexed physical pages."""
    blocks = ids.numel()
    block_size, heads, dim = paged.shape[2:]
    if (blocks == 0 or tuple(kv.shape) != (2, blocks * block_size, heads, dim)
            or not kv.is_contiguous() or kv.dtype != paged.dtype
            or kv.device != paged.device or ids.device != paged.device):
        raise ValueError("AISSD page scatter source, destination or index mismatch")
    values = kv.view(2, blocks, block_size, heads, dim).transpose(0, 1)
    paged.index_copy_(0, ids, values)


def gather_kv_pages_into(paged, ids, destination):
    """Gather directly into a contiguous staging buffer, including tail padding."""
    import torch

    blocks = ids.numel()
    block_size, heads, dim = paged.shape[2:]
    if (blocks == 0 or tuple(destination.shape) != (2, blocks * block_size, heads, dim)
            or not all(destination[p].is_contiguous() for p in (0, 1))
            or destination.dtype != paged.dtype
            or destination.device != paged.device or ids.device != paged.device):
        raise ValueError("AISSD page gather source, destination or index mismatch")
    for plane in (0, 1):
        torch.index_select(paged[:, plane], 0, ids,
                           out=destination[plane].view(blocks, block_size, heads, dim))
