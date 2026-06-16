# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for per-group BlockPool construction and routing in the KV cache
coordinator (Task G3).

When ``KVCacheConfig.per_group_num_blocks`` is set (gemma4 per-group depths),
the coordinator must build one ``BlockPool`` per distinct budget and route each
per-group manager to its pool. When it is ``None`` (uniform), there must be
exactly one pool, byte-identical to the legacy single-pool path.
"""
from math import lcm

import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    get_kv_cache_coordinator,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

# Synthetic gemma4 shape: 5 sliding-window groups (shallow D_sw pool) + 1 global
# full-attention group (deep D_global pool). Mirrors the 6-entry
# ``per_group_num_blocks`` the gemma4 allocator emits.
D_SW = 32
D_GLOBAL = 512
BLOCK_SIZE = 16
NUM_SLIDING_GROUPS = 5


def _gemma4_kv_cache_config(per_group: bool) -> KVCacheConfig:
    """Build a synthetic gemma4-shaped hybrid config.

    Args:
        per_group: If True, set ``per_group_num_blocks`` to
            ``5 * [D_SW] + [D_GLOBAL]``. If False, leave it None (uniform).
    """
    sliding_groups = [
        KVCacheGroupSpec(
            [f"sliding_layer_{i}"],
            SlidingWindowSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                sliding_window=2 * BLOCK_SIZE,
            ),
        )
        for i in range(NUM_SLIDING_GROUPS)
    ]
    global_group = KVCacheGroupSpec(
        ["global_layer"],
        FullAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
        ),
    )
    # Sliding groups first, then the global group (group id == index).
    kv_cache_groups = sliding_groups + [global_group]
    per_group_num_blocks = (
        [D_SW] * NUM_SLIDING_GROUPS + [D_GLOBAL] if per_group else None
    )
    return KVCacheConfig(
        # num_blocks is the max depth (mirrors the gemma4 allocator).
        num_blocks=max(D_SW, D_GLOBAL),
        kv_cache_tensors=[],
        kv_cache_groups=kv_cache_groups,
        per_group_num_blocks=per_group_num_blocks,
    )


def _make_coordinator(kv_cache_config: KVCacheConfig):
    return get_kv_cache_coordinator(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
    )


def test_per_group_pools_built():
    """Per-group config yields exactly two distinct pools, each sized and
    routed to its budget, each with its own null block."""
    config = _gemma4_kv_cache_config(per_group=True)
    coordinator = _make_coordinator(config)
    assert isinstance(coordinator, HybridKVCacheCoordinator)

    # Exactly two distinct BlockPool objects.
    assert len(coordinator.block_pools) == 2
    assert len({id(p) for p in coordinator.block_pools}) == 2

    # Each sliding group routes to the shallow (D_sw) pool, the global group to
    # the deep (D_global) pool. Manager and pool routing must agree.
    sw_pools = {
        id(coordinator.pool_for_group(i)) for i in range(NUM_SLIDING_GROUPS)
    }
    global_pool = coordinator.pool_for_group(NUM_SLIDING_GROUPS)
    assert len(sw_pools) == 1, "all sliding groups must share one pool"
    assert id(global_pool) not in sw_pools, "global pool must differ from sliding"

    sw_pool = coordinator.pool_for_group(0)
    assert sw_pool.num_gpu_blocks == D_SW
    assert global_pool.num_gpu_blocks == D_GLOBAL

    # Each manager points at its routed pool.
    for i in range(NUM_SLIDING_GROUPS):
        assert coordinator.single_type_managers[i].block_pool is sw_pool
    assert (
        coordinator.single_type_managers[NUM_SLIDING_GROUPS].block_pool
        is global_pool
    )

    # Each pool has its own null block (id 0, distinct objects per pool).
    assert sw_pool.null_block is not global_pool.null_block
    assert sw_pool.null_block.block_id == 0
    assert global_pool.null_block.block_id == 0
    assert sw_pool.null_block.is_null
    assert global_pool.null_block.is_null

    # Primary pool is the first built pool (sliding, first-seen).
    assert coordinator.block_pool is coordinator.block_pools[0]
    assert coordinator.block_pool is sw_pool


def test_shared_kv_same_pool():
    """Groups that share KV (here: all sliding groups share one spec/budget)
    must map to the identical pool object; the global group to the other.

    This is the R3 invariant: layers/groups sharing KV land in the same pool.
    """
    config = _gemma4_kv_cache_config(per_group=True)
    coordinator = _make_coordinator(config)

    first_sw_pool = coordinator.pool_for_group(0)
    # All sliding groups map to the *same* object, not just an equal one.
    for i in range(1, NUM_SLIDING_GROUPS):
        assert coordinator.pool_for_group(i) is first_sw_pool

    global_pool = coordinator.pool_for_group(NUM_SLIDING_GROUPS)
    assert global_pool is not first_sw_pool


def test_legacy_single_pool_unchanged():
    """A uniform config (per_group_num_blocks=None) yields exactly one pool,
    and ``block_pool`` is that pool (byte-identical legacy path)."""
    config = _gemma4_kv_cache_config(per_group=False)
    assert config.per_group_num_blocks is None
    coordinator = _make_coordinator(config)

    assert len(coordinator.block_pools) == 1
    assert coordinator.block_pool is coordinator.block_pools[0]
    assert coordinator.block_pool.num_gpu_blocks == config.num_blocks

    # Every group routes to the single pool.
    for i in range(len(config.kv_cache_groups)):
        assert coordinator.pool_for_group(i) is coordinator.block_pool
        assert (
            coordinator.single_type_managers[i].block_pool
            is coordinator.block_pool
        )


# ---------------------------------------------------------------------------
# Task G4: per-pool admission gate.
# ---------------------------------------------------------------------------


def _make_manager(kv_cache_config: KVCacheConfig) -> KVCacheManager:
    """Build a real ``KVCacheManager`` from the (per-group) config so the
    actual ``allocate_slots`` admission gate is exercised end to end."""
    init_none_hash(sha256)
    return KVCacheManager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        scheduler_block_size=lcm(
            *(g.kv_cache_spec.block_size for g in kv_cache_config.kv_cache_groups)
        ),
        hash_block_size=BLOCK_SIZE,
        max_num_batched_tokens=8192,
        enable_caching=True,
    )


def _make_request(request_id: str, num_tokens: int) -> Request:
    sampling_params = SamplingParams(max_tokens=1)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(num_tokens)),
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _drain_to_free(pool, target_free: int) -> None:
    """Drain ``pool`` down to exactly ``target_free`` free blocks."""
    to_take = pool.get_num_free_blocks() - target_free
    assert to_take >= 0
    if to_take:
        pool.get_new_blocks(to_take)
    assert pool.get_num_free_blocks() == target_free


def test_get_num_blocks_to_allocate_per_pool():
    """Per-pool sums attribute each group's demand to its routed pool, and the
    legacy ``get_num_blocks_to_allocate`` equals their sum and stays ``int``."""
    config = _gemma4_kv_cache_config(per_group=True)
    coordinator = _make_coordinator(config)
    sw_pool = coordinator.pool_for_group(0)
    global_pool = coordinator.pool_for_group(NUM_SLIDING_GROUPS)

    num_tokens = 4 * BLOCK_SIZE  # 4 blocks per group, no cache hits
    empty = tuple([] for _ in range(len(config.kv_cache_groups)))
    per_pool = coordinator.get_num_blocks_to_allocate_per_pool(
        request_id="r",
        num_tokens=num_tokens,
        new_computed_blocks=empty,
        num_encoder_tokens=0,
        total_computed_tokens=0,
        num_tokens_main_model=num_tokens,
    )
    # Sliding pool accumulates all 5 sliding groups (5 * 4); global pool 1 * 4.
    assert per_pool[sw_pool] == NUM_SLIDING_GROUPS * 4
    assert per_pool[global_pool] == 4

    # Scheduler contract: the scalar method equals the sum and is an int.
    total = coordinator.get_num_blocks_to_allocate(
        request_id="r",
        num_tokens=num_tokens,
        new_computed_blocks=empty,
        num_encoder_tokens=0,
        total_computed_tokens=0,
        num_tokens_main_model=num_tokens,
    )
    assert total == sum(per_pool.values())
    assert total == NUM_SLIDING_GROUPS * 4 + 4
    assert type(total) is int

    # The global pool is identified as the full-attention pool.
    assert coordinator.global_pool is global_pool


def test_per_pool_admission_gate():
    """The real ``allocate_slots`` gate rejects a request whose demand overflows
    EITHER pool and admits one that fits BOTH.

    Demand per kv cache group for a fresh request of N tokens is
    ``cdiv(N, BLOCK_SIZE)``; the sliding pool accumulates all 5 sliding groups,
    the global pool just the 1 full-attention group. So a 2-block request needs
    10 sliding-pool blocks and 2 global-pool blocks.
    """
    block = BLOCK_SIZE
    num_tokens = 2 * block  # 2 blocks/group -> sliding demand 10, global demand 2
    # Sliding pool depth D_SW=32 (31 free max); global pool D_GLOBAL=512.

    # (a) Sliding demand (10) exceeds sliding free (9) while global has slack.
    manager = _make_manager(_gemma4_kv_cache_config(per_group=True))
    coord = manager.coordinator
    sw_pool = coord.pool_for_group(0)
    global_pool = coord.global_pool
    _drain_to_free(sw_pool, 9)
    _drain_to_free(global_pool, 100)
    assert manager.allocate_slots(_make_request("a", num_tokens), num_tokens) is None

    # (b) Global demand (2) exceeds global free (1) while sliding has slack.
    manager = _make_manager(_gemma4_kv_cache_config(per_group=True))
    coord = manager.coordinator
    sw_pool = coord.pool_for_group(0)
    global_pool = coord.global_pool
    _drain_to_free(sw_pool, 31)  # full sliding pool (>= demand 10)
    _drain_to_free(global_pool, 1)
    assert manager.allocate_slots(_make_request("b", num_tokens), num_tokens) is None

    # (c) Both pools have enough free -> admitted (non-None).
    manager = _make_manager(_gemma4_kv_cache_config(per_group=True))
    coord = manager.coordinator
    sw_pool = coord.pool_for_group(0)
    global_pool = coord.global_pool
    _drain_to_free(sw_pool, 10)
    _drain_to_free(global_pool, 2)
    assert manager.allocate_slots(_make_request("c", num_tokens), num_tokens) is not None


def test_reserved_blocks_charged_to_global_pool_only():
    """``reserved_blocks`` is subtracted from the global pool's free count only;
    the sliding pool is checked on its raw free count.

    For the single (legacy) pool, ``global_pool`` is that pool, so the full
    reservation applies there -- byte-identical to the pre-G4 behavior.
    """
    block = BLOCK_SIZE
    num_tokens = 2 * block  # sliding demand 10, global demand 2

    # reserved_blocks=2 eats into the global pool: global free 3 -> effective 1
    # < demand 2 => reject, even though sliding (free 10 >= 10) fits.
    manager = _make_manager(_gemma4_kv_cache_config(per_group=True))
    coord = manager.coordinator
    _drain_to_free(coord.pool_for_group(0), 10)
    _drain_to_free(coord.global_pool, 3)
    assert (
        manager.allocate_slots(
            _make_request("d", num_tokens), num_tokens, reserved_blocks=2
        )
        is None
    )

    # Same reservation is NOT charged to the sliding pool: sliding free 10
    # (== demand 10) with reserved=2 still admits, because reserved hits global
    # (free 5 - 2 = 3 >= demand 2) not sliding.
    manager = _make_manager(_gemma4_kv_cache_config(per_group=True))
    coord = manager.coordinator
    _drain_to_free(coord.pool_for_group(0), 10)
    _drain_to_free(coord.global_pool, 5)
    assert (
        manager.allocate_slots(
            _make_request("e", num_tokens), num_tokens, reserved_blocks=2
        )
        is not None
    )
