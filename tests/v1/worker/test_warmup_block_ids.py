# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for warmup KV-cache block-id allocation (Task G5).

The warmup forward pass (``vllm/v1/worker/gpu/warmup.py``) runs on dummy data to
JIT-compile kernels and picks its own block ids -- it bypasses the runtime
``BlockPool`` admission gate. The legacy code used a SINGLE monotonic counter
shared across ALL KV cache groups, so with gemma4's per-group pool depths
(5 shallow sliding pools of depth ``D_sw`` + 1 deep global pool of depth
``D_global``) the later groups received ids that climbed past a shallow group's
``D_sw`` -- indexing that group's depth-``D_sw`` KV tensor OUT OF BOUNDS.

These tests pin the fix (``_WarmupBlockAllocator``):

* per-group mode confines every group's ids to its own ``[1, depth)`` space, and
* legacy mode (``per_group_num_blocks is None``) is byte-identical to the old
  single-monotonic-counter scheme.
"""

import pytest

from vllm.utils.math_utils import cdiv
from vllm.v1.worker.gpu.warmup import _WarmupBlockAllocator

# gemma4-shaped depths: 5 sliding groups at a shallow D_sw, 1 global group deep.
D_SW = 386
D_GLOBAL = 25080
GEMMA4_DEPTHS = [D_SW, D_SW, D_SW, D_SW, D_SW, D_GLOBAL]


def _simulate_warmup_alloc(allocator, per_group_block_counts, num_reqs):
    """Drive the allocator exactly the way ``warmup_kernels`` does: one alloc per
    group per request (prefill), then the same again (decode delta). Returns the
    flat list of (group, id) pairs that were handed out."""
    handed_out: list[tuple[int, int]] = []
    for _ in range(num_reqs):
        for g, n in enumerate(per_group_block_counts):
            for bid in allocator.alloc(g, n):
                handed_out.append((g, bid))
    return handed_out


def test_warmup_block_ids_within_pool_depth():
    """ESSENTIAL guard: every group's warmup ids must be < that group's pool
    depth (sliding ids < D_sw). This is the case the OLD monotonic counter
    violated."""
    allocator = _WarmupBlockAllocator(GEMMA4_DEPTHS, len(GEMMA4_DEPTHS))

    # 1 block per request per group is realistic for the tiny warmup seqs, but
    # use enough requests that a single shared counter would blow past D_sw.
    num_reqs = 80
    per_group_block_counts = [1] * len(GEMMA4_DEPTHS)

    handed_out = _simulate_warmup_alloc(allocator, per_group_block_counts, num_reqs)

    # Core invariant: in-bounds for the assigned group's tensor, never null.
    for group, bid in handed_out:
        assert 0 < bid < GEMMA4_DEPTHS[group], (
            f"group {group} got id {bid} not in [1, {GEMMA4_DEPTHS[group]})"
        )

    # Sanity: we actually exercised the sliding groups enough that the OLD
    # shared counter (1, 2, 3, ... monotonic across all groups) WOULD have
    # exceeded D_sw -- otherwise the test would pass vacuously.
    total_allocs = num_reqs * sum(per_group_block_counts)
    assert total_allocs > D_SW, (
        "test does not exercise the overflow case the fix targets"
    )


def test_warmup_sliding_group_cycles_within_shallow_depth():
    """A shallow sliding group asked for far more ids than its depth must cycle
    inside [1, depth), never emitting an out-of-bounds id."""
    shallow = 8
    allocator = _WarmupBlockAllocator([shallow], num_groups=1)

    seen = set()
    for _ in range(100):  # way more than `shallow` allocations
        (bid,) = allocator.alloc(0, 1)
        assert 0 < bid < shallow
        seen.add(bid)

    # It should reuse the full usable id range [1, shallow).
    assert seen == set(range(1, shallow))


def test_warmup_multi_block_request_stays_in_bounds():
    """A single request needing several blocks in a shallow group stays in
    bounds even when the request size approaches the depth."""
    depth = 5
    allocator = _WarmupBlockAllocator([depth], num_groups=1)
    for _ in range(20):
        ids = allocator.alloc(0, 3)
        assert all(0 < b < depth for b in ids), ids


def test_warmup_degenerate_depth_falls_back_to_null():
    """A pathological depth<=1 pool has no usable non-null id; the allocator
    must emit the null block (0, always in-bounds) rather than an OOB id."""
    for depth in (0, 1):
        allocator = _WarmupBlockAllocator([depth], num_groups=1)
        assert allocator.alloc(0, 3) == [0, 0, 0]


def test_warmup_legacy_path_byte_identical():
    """Legacy (per_group_num_blocks=None) must reproduce the OLD single
    monotonic counter EXACTLY: ids advance globally in call order, ignoring the
    group argument, starting at 1."""
    allocator = _WarmupBlockAllocator(None, num_groups=3)

    # Reproduce the exact OLD reference behavior with a local counter.
    next_block_id = 1

    def _legacy_alloc(num_blocks):
        nonlocal next_block_id
        out = list(range(next_block_id, next_block_id + num_blocks))
        next_block_id += num_blocks
        return out

    # Drive both through a representative warmup schedule:
    #   prefill: per req, per group counts [2, 1, 3]
    #   decode:  per req, per group deltas [1, 0, 1]
    prefill_counts = [2, 1, 3]
    decode_deltas = [1, 0, 1]
    num_reqs = 4

    for _ in range(num_reqs):
        for g, n in enumerate(prefill_counts):
            assert allocator.alloc(g, n) == _legacy_alloc(n)
    for _ in range(num_reqs):
        for g, n in enumerate(decode_deltas):
            assert allocator.alloc(g, n) == _legacy_alloc(n)


def test_warmup_legacy_zero_delta_advances_nothing():
    """Legacy zero-block alloc returns [] and does not advance the counter
    (matches the old ``_alloc_blocks(0)``)."""
    allocator = _WarmupBlockAllocator(None, num_groups=2)
    assert allocator.alloc(0, 0) == []
    assert allocator.alloc(0, 2) == [1, 2]
    assert allocator.alloc(1, 0) == []
    assert allocator.alloc(1, 1) == [3]


@pytest.mark.parametrize("block_size", [16, 32])
def test_warmup_realistic_gemma4_schedule_in_bounds(block_size):
    """End-to-end-shaped check: build the per-group prefill/decode counts the way
    ``warmup_kernels`` does and assert every id assigned across a full prefill +
    decode warmup is in-bounds for its group's depth."""
    num_spec_steps = 0
    prompt_len = 2 + num_spec_steps
    decode_len = prompt_len + 1 + num_spec_steps

    # 6 groups, all using the same block size here for simplicity.
    depths = GEMMA4_DEPTHS
    prefill_counts = [cdiv(prompt_len, block_size) for _ in depths]
    decode_counts = [cdiv(decode_len, block_size) for _ in depths]
    decode_deltas = [d - p for d, p in zip(decode_counts, prefill_counts)]

    allocator = _WarmupBlockAllocator(depths, len(depths))
    num_reqs = 128

    prefill = _simulate_warmup_alloc(allocator, prefill_counts, num_reqs)
    decode = _simulate_warmup_alloc(allocator, decode_deltas, num_reqs)
    for group, bid in prefill + decode:
        assert 0 <= bid < depths[group], (group, bid, depths[group])
        # Non-degenerate pools never hand out the null block.
        if depths[group] > 1:
            assert bid > 0
