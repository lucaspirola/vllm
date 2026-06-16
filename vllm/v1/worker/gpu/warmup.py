# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from vllm import PoolingParams, SamplingParams
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.request import Request
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


class _WarmupBlockAllocator:
    """Hands out warmup block ids that stay in-bounds for each KV cache group's
    own pool depth.

    Background (the OOB this guards against): warmup runs a real forward pass on
    dummy data to JIT-compile kernels. It bypasses the runtime BlockPools, so it
    must pick the dummy block ids itself. The legacy code used a SINGLE monotonic
    counter shared across ALL groups, so with a per-group-depth config (gemma4:
    5 shallow sliding pools of depth ``D_sw`` plus 1 deep global pool of depth
    ``D_global``) the later groups received ids that climbed well past a shallow
    group's ``D_sw``. The forward pass then indexed that group's depth-``D_sw``
    KV tensor OUT OF BOUNDS (illegal address / silent corruption).

    Two modes:

    * Legacy (``per_group_depths is None``): a single monotonic counter shared
      across all groups, starting at 1 (id 0 is the reserved null block). This
      reproduces the old behavior BYTE-IDENTICALLY -- the ``group`` argument is
      ignored and ids advance globally in call order.
    * Per-group (gemma4): each group ``i`` draws from its OWN ``[1, depth_i)``
      space, cycling within that range. Because warmup data is dummy and only
      needs to be in-bounds for graph capture, reusing ids within a group across
      requests is fine -- the only invariant is ``id < depth_i``.
    """

    def __init__(self, per_group_depths: list[int] | None, num_groups: int) -> None:
        self._per_group = per_group_depths is not None
        if self._per_group:
            assert per_group_depths is not None  # narrow for type checkers
            self._depths = per_group_depths
            # Per-group monotonic cursor; start at 1 to reserve id 0 (null).
            self._cursors = [1] * num_groups
        else:
            # Single shared counter -- exact legacy behavior.
            self._next_block_id = 1

    def alloc(self, group: int, num_blocks: int) -> list[int]:
        """Return ``num_blocks`` ids for ``group``.

        Legacy mode: ids come from one global monotonic counter (``group``
        ignored), byte-identical to the previous implementation.

        Per-group mode: ids advance monotonically within ``[1, depth)`` and wrap
        back to 1 before reaching the group's depth, so every id is strictly
        in-bounds for that group's KV tensor and never the null block (0).
        """
        if not self._per_group:
            start = self._next_block_id
            self._next_block_id = start + num_blocks
            return list(range(start, self._next_block_id))

        depth = self._depths[group]
        if depth <= 1:
            # Degenerate pool with no non-null id; the null block (0) is always
            # in-bounds. A real KV pool is sized for at least one request, so
            # this only guards pathological configs from emitting an OOB id.
            return [0] * num_blocks
        ids = []
        cur = self._cursors[group]
        for _ in range(num_blocks):
            if cur >= depth:
                # Wrap: dummy warmup data, so id reuse within a group is fine;
                # we only require every id to be in-bounds for that group's
                # KV tensor.
                cur = 1
            ids.append(cur)
            cur += 1
        # Keep the cursor inside [1, depth) so the NEXT alloc for this group also
        # starts in-bounds (matters when the usable space is a single id).
        if cur >= depth:
            cur = 1
        self._cursors[group] = cur
        return ids


@torch.inference_mode()
def warmup_kernels(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
) -> None:
    """Run two execute_model + sample_tokens iterations to JIT compile
    triton kernels. We must call the provided worker's execute_model for
    pipeline parallel coordination.

    The first iteration simulates a prefill with requests of
    decode_query_len + 1 prompt tokens each. The second iteration simulates
    a decode step with all requests generating decode_query_len tokens.
    """
    num_spec_steps = model_runner.num_speculative_steps
    decode_query_len = model_runner.decode_query_len
    # Use decode_query_len + 1 tokens so the prefill batch's per-request query
    # length exceeds decode_query_len, preventing it from being misclassified as
    # a uniform decode batch.
    prompt_len = decode_query_len + 1
    prompt_token_ids = list(range(prompt_len))
    # After prefill, decode generates decode_query_len tokens.
    decode_len = prompt_len + decode_query_len

    kv_cache_groups = model_runner.kv_cache_config.kv_cache_groups
    num_kv_cache_groups = len(kv_cache_groups)

    # Compute per-request block counts for each KV cache group.
    group_block_sizes = [g.kv_cache_spec.block_size for g in kv_cache_groups]
    prefill_block_counts = [cdiv(prompt_len, bs) for bs in group_block_sizes]
    decode_block_counts = [cdiv(decode_len, bs) for bs in group_block_sizes]
    decode_block_deltas = [
        d - p for d, p in zip(decode_block_counts, prefill_block_counts)
    ]
    max_blocks_per_req = sum(decode_block_counts)

    # ``num_blocks`` is the GLOBAL budget (``max(per_group_num_blocks)`` ==
    # ``D_global`` when per-group depths are set), so this bound can over-count
    # capacity for a SHALLOW per-group pool. That is harmless now: the allocator
    # confines each group's ids to its OWN depth (cycling within ``[1, depth)``),
    # so a too-large ``num_reqs`` only means more id REUSE inside a shallow group,
    # never an out-of-bounds id. For the legacy single-pool path this is exactly
    # the previous formula, unchanged.
    num_reqs = min(
        model_runner.scheduler_config.max_num_seqs,
        model_runner.scheduler_config.max_num_batched_tokens
        // max(prompt_len, decode_query_len),
        # Reserve block 0 (null block) and ensure we have enough blocks.
        max(1, (model_runner.kv_cache_config.num_blocks - 1) // max_blocks_per_req),
    )

    req_ids = [f"_warmup_{i}_" for i in range(num_reqs)]

    # SamplingParams exercising all sampling features.
    if model_runner.is_pooling_model:
        sampling_params = None
        pooling_params = PoolingParams()
    else:
        sampling_params = SamplingParams.for_sampler_warmup()
        pooling_params = None

    # Assign block IDs per request per group. Id 0 is the null block; real ids
    # start at 1. Each group's ids are confined to its OWN pool depth so the
    # warmup forward pass never indexes a (shallow) per-group KV tensor out of
    # bounds. The legacy single-pool path keeps a single monotonic counter and
    # is byte-identical (see ``_WarmupBlockAllocator``).
    allocator = _WarmupBlockAllocator(
        model_runner.kv_cache_config.per_group_num_blocks,
        num_kv_cache_groups,
    )

    # Step 1: Prefill all requests with 1 + decode_query_len prompt tokens each.
    new_reqs = [
        NewRequestData.from_request(
            Request(req_ids[i], prompt_token_ids, sampling_params, pooling_params),
            block_ids=tuple(
                allocator.alloc(g, n) for g, n in enumerate(prefill_block_counts)
            ),
            prefill_token_ids=prompt_token_ids,
        )
        for i in range(num_reqs)
    ]

    prefill_output = SchedulerOutput.make_empty()
    prefill_output.scheduled_new_reqs = new_reqs
    prefill_output.num_scheduled_tokens = {rid: prompt_len for rid in req_ids}
    prefill_output.total_num_scheduled_tokens = prompt_len * num_reqs
    prefill_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

    # Disable KV connector for warmup run.
    model_runner.kv_connector.set_disabled(True)
    worker_execute_model(prefill_output)

    if not model_runner.is_pooling_model:
        # Warm up sampler and perform a decode step for non-pooling models.

        grammar_output = None
        if model_runner.is_last_pp_rank:
            # Build a GrammarOutput to exercise the structured output bitmask
            # kernel during the prefill step.
            vocab_size = model_runner.model_config.get_vocab_size()
            bitmask_width = (vocab_size + 31) // 32
            grammar_bitmask = np.full(
                (len(req_ids), bitmask_width), fill_value=-1, dtype=np.int32
            )
            grammar_output = GrammarOutput(
                structured_output_request_ids=req_ids, grammar_bitmask=grammar_bitmask
            )

        worker_sample_tokens(grammar_output)

        # Step 2: Decode all requests with decode_query_len tokens each.
        cached_req_data = CachedRequestData.make_empty()
        cached_req_data.req_ids = list(req_ids)
        cached_req_data.num_computed_tokens = [prompt_len] * num_reqs
        cached_req_data.num_output_tokens = [1] * num_reqs
        new_block = any(decode_block_deltas)
        cached_req_data.new_block_ids = [
            tuple(
                allocator.alloc(g, n) for g, n in enumerate(decode_block_deltas)
            )
            if new_block
            else None
            for _ in range(num_reqs)
        ]

        decode_output = SchedulerOutput.make_empty()
        decode_output.scheduled_cached_reqs = cached_req_data
        decode_output.num_scheduled_tokens = {
            req_id: decode_query_len for req_id in req_ids
        }
        if num_spec_steps > 0:
            decode_output.scheduled_spec_decode_tokens = {
                req_id: [0] * num_spec_steps for req_id in req_ids
            }
        decode_output.total_num_scheduled_tokens = sum(
            decode_output.num_scheduled_tokens.values()
        )
        decode_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

        worker_execute_model(decode_output)
        worker_sample_tokens(None)

    # Clean up - process finish_req_ids.
    cleanup_output = SchedulerOutput.make_empty()
    cleanup_output.finished_req_ids = set(req_ids)
    worker_execute_model(cleanup_output)
    model_runner.kv_connector.set_disabled(False)
    torch.accelerator.synchronize()
