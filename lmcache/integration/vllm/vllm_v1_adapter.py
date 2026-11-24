# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry points for vLLM's in-process LMCache integration."""

# First Party
from lmcache.inprocess.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorV1Impl,
    get_sparse_kv_connector,
)

__all__ = ["LMCacheConnectorV1Impl", "get_sparse_kv_connector"]
