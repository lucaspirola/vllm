# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import importlib
from collections.abc import Callable
from typing import Any

import pytest
import torch

import vllm.v1.core.kv_cache_utils as kv_cache_utils
from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
from vllm.config.kv_events import KVEventsConfig
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256, sha256_cbor
from vllm.utils.math_utils import cdiv
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    estimate_max_model_len,
    generate_block_hash_extra_keys,
    generate_scheduler_kv_cache_config,
    get_kv_cache_capacity,
    get_kv_cache_configs,
    get_max_concurrency_for_kv_cache_config,
    get_request_block_hasher,
    hash_block_tokens,
    init_none_hash,
    is_kv_cache_spec_uniform,
    make_block_hash_with_group_id,
    tensor_data,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheSpecKind,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SinkFullAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
    get_kv_cache_spec_kind,
    get_kv_cache_spec_sliding_window,
)
from vllm.v1.metrics.stats import CachingMetrics, PrefixCacheStats
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _auto_init_hash_fn(request):
    hash_fn: Callable
    if "hash_fn" in request.fixturenames:
        hash_fn = request.getfixturevalue("hash_fn")
    else:
        hash_fn = sha256
    init_none_hash(hash_fn)


def make_request(
    request_id: str,
    prompt_token_ids: list[int] | None,
    block_size: int = 3,
    hash_fn: Callable = hash,
    mm_positions: list[PlaceholderRange] | None = None,
    mm_hashes: list[str] | None = None,
    cache_salt: str | None = None,
    prompt_embeds: torch.Tensor | None = None,
):
    mm_features = []
    if mm_positions is not None:
        for j, position in enumerate(mm_positions):
            identifier = mm_hashes[j] if mm_hashes else f"hash_{j}"
            mm_feature = MultiModalFeatureSpec(
                data=MultiModalKwargsItem.dummy(),
                mm_position=position,
                identifier=identifier,
                modality="image",
            )
            mm_features.append(mm_feature)

    sampling_params = SamplingParams(max_tokens=17)
    sampling_params.update_from_generation_config({}, eos_token_id=100)

    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=mm_features if mm_features else None,
        sampling_params=sampling_params,
        pooling_params=None,
        lora_request=None,
        cache_salt=cache_salt,
        block_hasher=get_request_block_hasher(block_size, hash_fn),
        prompt_embeds=prompt_embeds,
    )


def new_kv_cache_spec(
    block_size=16,
    num_kv_heads=2,
    head_size=64,
    dtype=torch.float32,
    page_size_padded=None,
    sliding_window=None,
    attention_chunk_size=None,
):
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        page_size_padded=page_size_padded,
        sliding_window=sliding_window,
        attention_chunk_size=attention_chunk_size,
    )


def new_sliding_window_spec(
    block_size=16,
    num_kv_heads=2,
    head_size=64,
    dtype=torch.float32,
    page_size_padded=None,
    sliding_window=1,
):
    return SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        page_size_padded=page_size_padded,
        sliding_window=sliding_window,
    )


def new_chunked_local_attention_spec(
    block_size=16,
    num_kv_heads=2,
    head_size=64,
    dtype=torch.float32,
    page_size_padded=None,
    attention_chunk_size=4,
):
    return ChunkedLocalAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        page_size_padded=page_size_padded,
        attention_chunk_size=attention_chunk_size,
    )


def new_mamba_spec(
    block_size=16,
    shapes=((2, 512), (3, 32, 32)),
    dtypes=(torch.float32, torch.float32),
    num_speculative_blocks=2,
    mamba_cache_mode="none",
    page_size_padded=None,
):
    return MambaSpec(
        block_size=block_size,
        shapes=shapes,
        dtypes=dtypes,
        page_size_padded=page_size_padded,
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_blocks,
    )


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_none_hash(monkeypatch, hash_fn):
    import vllm.v1.core.kv_cache_utils

    # case 1: PYTHONHASHSEED is not set, use random
    with monkeypatch.context() as m:
        m.delenv("PYTHONHASHSEED", raising=False)
        reloaded_kv_cache_utils = importlib.reload(vllm.v1.core.kv_cache_utils)
        reloaded_kv_cache_utils.init_none_hash(hash_fn)
        assert reloaded_kv_cache_utils.NONE_HASH is not None
        assert isinstance(reloaded_kv_cache_utils.NONE_HASH, bytes)
        assert reloaded_kv_cache_utils.NONE_HASH != b""

    # case 2: PYTHONHASHSEED is set, use the seed and hash_fn
    with monkeypatch.context() as m:
        m.setenv("PYTHONHASHSEED", "python hash seed")
        reloaded_kv_cache_utils = importlib.reload(vllm.v1.core.kv_cache_utils)
        reloaded_kv_cache_utils.init_none_hash(hash_fn)
        assert reloaded_kv_cache_utils.NONE_HASH is not None
        assert isinstance(reloaded_kv_cache_utils.NONE_HASH, bytes)
        assert hash_fn("python hash seed") == reloaded_kv_cache_utils.NONE_HASH


def test_kv_cache_block():
    # Test KVCacheBlock initialization
    block = KVCacheBlock(block_id=0)
    assert block.block_id == 0
    assert block.ref_cnt == 0
    assert block.block_hash is None

    # Test reference count manipulation
    block.ref_cnt += 1
    assert block.ref_cnt == 1
    block.ref_cnt -= 1
    assert block.ref_cnt == 0

    # Test block hash setting and resetting
    block_hash = make_block_hash_with_group_id(BlockHash(b"abc"), 0)
    block.block_hash = block_hash
    assert block.block_hash == block_hash

    block.reset_hash()
    assert block.block_hash is None


def test_kv_cache_block_uses_slots():
    block = KVCacheBlock(block_id=0)

    # Slots eliminate per-instance __dict__, saving ~264 bytes per block.
    # At 100K+ blocks this avoids tens of MB of overhead and GC pressure.
    assert not hasattr(block, "__dict__")

    # Verify that slots actually prevent dynamic attribute assignment.
    with pytest.raises(AttributeError):
        block.unexpected_field = True


def test_free_kv_cache_block_queue_initialization():
    # Test with a single block
    block = KVCacheBlock(block_id=0)
    queue = FreeKVCacheBlockQueue([block])
    assert queue.num_free_blocks == 1
    assert queue.fake_free_list_head.next_free_block is block
    assert queue.fake_free_list_tail.prev_free_block is block


def test_free_kv_cache_block_queue_operations():
    # Create a list of KVCacheBlock objects
    blocks = [KVCacheBlock(block_id=i) for i in range(5)]

    # Create a FreeKVCacheBlockQueue with these blocks
    queue = FreeKVCacheBlockQueue(blocks)

    # Check initial state
    assert queue.num_free_blocks == 5
    assert queue.fake_free_list_head.next_free_block is blocks[0]
    assert queue.fake_free_list_tail.prev_free_block is blocks[4]

    # Pop the first block
    block1 = queue.popleft()
    assert block1 == blocks[0]
    assert queue.num_free_blocks == 4
    assert queue.fake_free_list_head.next_free_block is blocks[1]
    assert queue.fake_free_list_tail.prev_free_block is blocks[4]

    # Remove a block from the middle
    block_to_remove = blocks[2]
    queue.remove(block_to_remove)
    assert queue.num_free_blocks == 3
    assert blocks[1].next_free_block is blocks[3]
    assert blocks[3].prev_free_block is blocks[1]

    # Append a block back
    queue.append(block_to_remove)
    assert queue.num_free_blocks == 4
    assert queue.fake_free_list_tail.prev_free_block is block_to_remove
    assert block_to_remove.prev_free_block is blocks[4]
    assert block_to_remove.next_free_block is queue.fake_free_list_tail

    # Pop blocks until empty
    for _ in range(4):
        queue.popleft()
    assert queue.num_free_blocks == 0
    assert queue.fake_free_list_head.next_free_block is queue.fake_free_list_tail
    assert queue.fake_free_list_tail.prev_free_block is queue.fake_free_list_head

    # Attempt to pop from an empty queue
    with pytest.raises(ValueError) as e:
        queue.popleft()
    assert str(e.value) == "No free blocks available"


def test_free_kv_cache_block_queue_append_n():
    # Create an empty FreeKVCacheBlockQueue with these blocks
    queue = FreeKVCacheBlockQueue([])
    blocks = [KVCacheBlock(block_id=i) for i in range(6)]
    # Append 0 block
    # fake_head->fake_tail
    queue.append_n([])
    assert queue.num_free_blocks == 0
    assert queue.fake_free_list_head.next_free_block is queue.fake_free_list_tail
    assert queue.fake_free_list_tail.prev_free_block is queue.fake_free_list_head
    # Append 1 block
    # fake_head->b0->fake_tail
    queue.append_n(blocks[0:1])
    assert queue.num_free_blocks == 1
    assert queue.fake_free_list_head.next_free_block is blocks[0]
    assert blocks[0].prev_free_block is queue.fake_free_list_head
    assert blocks[0].next_free_block is queue.fake_free_list_tail
    assert queue.fake_free_list_tail.prev_free_block is blocks[0]
    # Append 2 blocks
    # fake_head->b0->b4->b5->fake_tail
    queue.append_n(blocks[4:6])
    assert queue.num_free_blocks == 3
    assert queue.fake_free_list_head.next_free_block is blocks[0]
    assert blocks[0].prev_free_block is queue.fake_free_list_head
    assert blocks[0].next_free_block is blocks[4]
    assert blocks[4].prev_free_block is blocks[0]
    assert blocks[4].next_free_block is blocks[5]
    assert blocks[5].prev_free_block is blocks[4]
    assert blocks[5].next_free_block is queue.fake_free_list_tail
    assert queue.fake_free_list_tail.prev_free_block is blocks[5]
    # Append 3 blocks
    # fake_head->b0->b4->b5->b1->b2->b3->fake_tail
    queue.append_n(blocks[1:4])
    assert queue.num_free_blocks == 6
    assert queue.fake_free_list_head.next_free_block is blocks[0]
    assert blocks[0].prev_free_block is queue.fake_free_list_head
    assert blocks[0].next_free_block is blocks[4]
    assert blocks[4].prev_free_block is blocks[0]
    assert blocks[4].next_free_block is blocks[5]
    assert blocks[5].prev_free_block is blocks[4]
    assert blocks[5].next_free_block is blocks[1]
    assert blocks[1].prev_free_block is blocks[5]
    assert blocks[1].next_free_block is blocks[2]
    assert blocks[2].prev_free_block is blocks[1]
    assert blocks[2].next_free_block is blocks[3]
    assert blocks[3].prev_free_block is blocks[2]
    assert blocks[3].next_free_block is queue.fake_free_list_tail
    assert queue.fake_free_list_tail.prev_free_block is blocks[3]

    # Create an empty FreeKVCacheBlockQueue
    invalid_queue = FreeKVCacheBlockQueue([])
    # set prev_free_block to None and this will cause assertion in append_n
    invalid_queue.fake_free_list_tail.prev_free_block = None
    with pytest.raises(AssertionError):
        # Append 1 block
        # fake_head->fake_tail
        invalid_queue.append_n(blocks[0:1])
    assert invalid_queue.num_free_blocks == 0
    assert (
        invalid_queue.fake_free_list_head.next_free_block
        == invalid_queue.fake_free_list_tail
    )


def test_free_kv_cache_block_queue_prepend_n():
    # Seed the queue with one block so prepend has an existing head to splice
    # in front of (fake_head->b0->fake_tail).
    blocks = [KVCacheBlock(block_id=i) for i in range(6)]
    queue = FreeKVCacheBlockQueue(blocks[0:1])

    # Prepend 0 blocks is a no-op.
    queue.prepend_n([])
    assert queue.num_free_blocks == 1
    assert queue.fake_free_list_head.next_free_block is blocks[0]

    # Prepend 2 blocks; they land in front of the existing head, in order.
    # fake_head->b4->b5->b0->fake_tail
    queue.prepend_n(blocks[4:6])
    assert queue.num_free_blocks == 3
    assert queue.fake_free_list_head.next_free_block is blocks[4]
    assert blocks[4].prev_free_block is queue.fake_free_list_head
    assert blocks[4].next_free_block is blocks[5]
    assert blocks[5].prev_free_block is blocks[4]
    assert blocks[5].next_free_block is blocks[0]
    assert blocks[0].prev_free_block is blocks[5]
    assert blocks[0].next_free_block is queue.fake_free_list_tail
    assert queue.fake_free_list_tail.prev_free_block is blocks[0]

    # A second prepend goes ahead of everything previously prepended.
    # fake_head->b1->b2->b4->b5->b0->fake_tail
    queue.prepend_n(blocks[1:3])
    assert queue.num_free_blocks == 5
    assert queue.fake_free_list_head.next_free_block is blocks[1]
    assert blocks[1].next_free_block is blocks[2]
    assert blocks[2].next_free_block is blocks[4]

    # The popleft order reflects the front-to-back queue order.
    assert [queue.popleft().block_id for _ in range(5)] == [1, 2, 4, 5, 0]
    assert queue.num_free_blocks == 0


def test_free_kv_cache_block_queue_popleft_n():
    blocks = [KVCacheBlock(block_id=i) for i in range(6)]
    # Create an empty FreeKVCacheBlockQueue with these blocks
    queue = FreeKVCacheBlockQueue(
        [blocks[1], blocks[3], blocks[5], blocks[4], blocks[0], blocks[2]]
    )
    assert queue.num_free_blocks == 6
    assert queue.fake_free_list_head.next_free_block is blocks[1]
    assert blocks[1].prev_free_block is queue.fake_free_list_head
    assert blocks[1].next_free_block is blocks[3]
    assert blocks[3].prev_free_block is blocks[1]
    assert blocks[3].next_free_block is blocks[5]
    assert blocks[5].prev_free_block is blocks[3]
    assert blocks[5].next_free_block is blocks[4]
    assert blocks[4].prev_free_block is blocks[5]
    assert blocks[4].next_free_block is blocks[0]
    assert blocks[0].prev_free_block is blocks[4]
    assert blocks[0].next_free_block is blocks[2]
    assert blocks[2].prev_free_block is blocks[0]
    assert blocks[2].next_free_block is queue.fake_free_list_tail
    assert queue.fake_free_list_tail.prev_free_block is blocks[2]

    # Pop 0 block
    # fake_head->b1->b3->b5->b4->b0->b2->fake_tail
    assert len(queue.popleft_n(0)) == 0
    assert queue.num_free_blocks == 6
    # Pop 1 block
    # fake_head->b3->b5->b4->b0->b2->fake_tail
    result_blocks = queue.popleft_n(1)
    assert queue.num_free_blocks == 5
    assert len(result_blocks) == 1
    assert result_blocks[0] is blocks[1]
    for block in result_blocks:
        assert block.prev_free_block is None
        assert block.next_free_block is None
    # Pop 2 blocks
    # fake_head->b4->b0->b2->fake_tail
    result_blocks = queue.popleft_n(2)
    assert len(result_blocks) == 2
    assert queue.num_free_blocks == 3
    assert result_blocks[0] is blocks[3]
    assert result_blocks[1] is blocks[5]
    for block in result_blocks:
        assert block.prev_free_block is None
        assert block.next_free_block is None
    # Pop 3 blocks
    # fake_head->fake_tail
    result_blocks = queue.popleft_n(3)
    assert len(result_blocks) == 3
    assert queue.num_free_blocks == 0
    assert result_blocks[0] is blocks[4]
    assert result_blocks[1] is blocks[0]
    assert result_blocks[2] is blocks[2]
    for block in result_blocks:
        assert block.prev_free_block is None
        assert block.next_free_block is None


def test_free_kv_cache_block_queue_get_all_free_blocks():
    # Create a list of KVCacheBlock objects
    blocks = [KVCacheBlock(block_id=i) for i in range(5)]

    # Create a FreeKVCacheBlockQueue with these blocks
    queue = FreeKVCacheBlockQueue(blocks)

    # Check all blocks are correctly retrieved
    assert queue.get_all_free_blocks() == blocks

    # Pop a block and check again
    queue.popleft()
    assert queue.get_all_free_blocks() == blocks[1:]

    # Remove a block and check again
    block_to_remove = blocks[2]
    queue.remove(block_to_remove)
    assert queue.get_all_free_blocks() == blocks[1:2] + blocks[3:]

    # Append a block back and check again
    queue.append(block_to_remove)
    assert queue.get_all_free_blocks() == blocks[1:2] + blocks[3:] + [block_to_remove]


def test_generate_block_hash_extra_keys():
    request = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(20)],
        mm_positions=[
            PlaceholderRange(offset=0, length=5),
            PlaceholderRange(offset=10, length=5),
        ],
        mm_hashes=["hash1", "hash2"],
    )

    # Test with no extra keys
    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request, 0, 5, 0)
    assert extra_keys == (("hash1", 0),)
    assert next_mm_idx == 1

    # Test with partial overlap
    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request, 3, 8, 0)
    assert extra_keys == (("hash1", -3),)
    assert next_mm_idx == 1

    # Test with no overlap
    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request, 6, 10, 0)
    assert extra_keys is None
    assert next_mm_idx == 1

    # Test with multiple extra keys
    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request, 0, 15, 0)
    assert extra_keys == (("hash1", 0), ("hash2", 10))
    assert next_mm_idx == 2


def test_generate_block_hash_extra_keys_no_mm_inputs():
    request = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(6)],
        mm_positions=None,
        mm_hashes=None,
    )

    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request, 0, 5, 0)
    assert extra_keys is None
    assert next_mm_idx == 0


def test_generate_block_hash_extra_keys_cache_salt():
    request = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(6)],
        mm_positions=None,
        mm_hashes=None,
        cache_salt="salt",
    )

    # salt is added for the first token
    extra_keys, _ = generate_block_hash_extra_keys(request, 0, 1, 0)
    assert extra_keys == ("salt",)
    extra_keys, _ = generate_block_hash_extra_keys(request, 0, 10, 0)
    assert extra_keys == ("salt",)

    # no salt added for other tokens
    extra_keys, _ = generate_block_hash_extra_keys(request, 1, 2, 0)
    assert extra_keys is None
    extra_keys, _ = generate_block_hash_extra_keys(request, 6, 10, 0)
    assert extra_keys is None

    # works together with other extra keys
    request_mm = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(20)],
        mm_positions=[
            PlaceholderRange(offset=0, length=5),
        ],
        mm_hashes=["hash1"],
        cache_salt="salt",
    )

    # Test with no extra keys
    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request_mm, 0, 5, 0)
    assert extra_keys == (("hash1", 0), "salt")
    assert next_mm_idx == 1


def test_generate_block_hash_extra_keys_prompt_embeds():
    prompt_embeds = torch.randn(10, 3)
    request = make_request(
        request_id="0",
        prompt_token_ids=None,
        mm_positions=None,
        mm_hashes=None,
        prompt_embeds=prompt_embeds,
    )

    # Test with prompt embeds for the first block
    extra_keys, _ = generate_block_hash_extra_keys(request, 0, 5, 0)
    expected_embeds = prompt_embeds[0:5]
    expected_hash = hashlib.sha256(kv_cache_utils.tensor_data(expected_embeds)).digest()
    assert extra_keys == (expected_hash,)

    # Test with prompt embeds for the second block
    extra_keys, _ = generate_block_hash_extra_keys(request, 5, 10, 0)
    expected_embeds = prompt_embeds[5:10]
    expected_hash = hashlib.sha256(kv_cache_utils.tensor_data(expected_embeds)).digest()
    assert extra_keys == (expected_hash,)


def test_generate_block_hash_extra_keys_prompt_embeds_cached(monkeypatch):
    prompt_embeds = torch.randn(10, 3)
    request = make_request(
        request_id="0",
        prompt_token_ids=None,
        mm_positions=None,
        mm_hashes=None,
        prompt_embeds=prompt_embeds,
        block_size=20,
    )

    num_tensor_data_calls = 0
    original_tensor_data = kv_cache_utils.tensor_data

    def counting_tensor_data(tensor: torch.Tensor):
        nonlocal num_tensor_data_calls
        num_tensor_data_calls += 1
        return original_tensor_data(tensor)

    monkeypatch.setattr(kv_cache_utils, "tensor_data", counting_tensor_data)

    extra_keys_1, _ = generate_block_hash_extra_keys(request, 0, 5, 0)
    extra_keys_2, _ = generate_block_hash_extra_keys(request, 0, 5, 0)
    assert extra_keys_1 == extra_keys_2
    assert num_tensor_data_calls == 1


def test_generate_block_hash_extra_keys_different_prompt_embeds():
    prompt_embeds1 = torch.randn(10, 3)
    prompt_embeds2 = torch.randn(10, 3)
    request1 = make_request(
        request_id="0",
        prompt_token_ids=None,
        mm_positions=None,
        mm_hashes=None,
        prompt_embeds=prompt_embeds1,
    )
    request2 = make_request(
        request_id="1",
        prompt_token_ids=None,
        mm_positions=None,
        mm_hashes=None,
        prompt_embeds=prompt_embeds2,
    )

    extra_keys1, _ = generate_block_hash_extra_keys(request1, 0, 5, 0)
    extra_keys2, _ = generate_block_hash_extra_keys(request2, 0, 5, 0)
    assert extra_keys1 != extra_keys2


def test_generate_block_hash_extra_keys_lora():
    request = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(6)],
    )

    request.lora_request = LoRARequest(
        lora_name="test_lora_adapter", lora_int_id=1, lora_path="/path/to/lora"
    )

    extra_keys, _ = generate_block_hash_extra_keys(request, 0, 3, 0)
    assert extra_keys == ("test_lora_adapter",)

    request.lora_request = None
    extra_keys, _ = generate_block_hash_extra_keys(request, 0, 3, 0)
    assert extra_keys is None


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_hash_block_tokens(hash_fn):
    parent_block_hash = BlockHash(b"123")
    curr_block_token_ids = (1, 2, 3)
    extra_keys = ("key1", "key2")

    block_hash = hash_block_tokens(
        hash_fn, parent_block_hash, curr_block_token_ids, extra_keys
    )
    expected = hash_fn((parent_block_hash, curr_block_token_ids, extra_keys))
    assert block_hash == expected


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_request_block_hasher(hash_fn):
    request = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(6)],
        block_size=3,
        hash_fn=hash_fn,
        mm_positions=[
            PlaceholderRange(offset=0, length=3),
            PlaceholderRange(offset=3, length=3),
        ],
        mm_hashes=["hash1", "hash2"],
    )

    block_hashes = request.block_hashes
    assert len(block_hashes) == 2
    assert block_hashes[0] == hash_fn(
        (kv_cache_utils.NONE_HASH, (0, 1, 2), (("hash1", 0),))
    )
    assert block_hashes[1] == hash_fn((block_hashes[0], (3, 4, 5), (("hash2", 0),)))


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_hash_tokens_different_mm_input(hash_fn):
    request1 = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(6)],
        block_size=3,
        hash_fn=hash_fn,
        mm_positions=[
            PlaceholderRange(offset=0, length=3),
            PlaceholderRange(offset=3, length=3),
        ],
        mm_hashes=["hash1", "hash2"],
    )
    request2 = make_request(
        request_id="1",
        prompt_token_ids=[_ for _ in range(6)],
        mm_positions=[
            PlaceholderRange(offset=0, length=3),
            PlaceholderRange(offset=3, length=3),
        ],
        mm_hashes=["hash3", "hash2"],
    )
    block_hashes1 = request1.block_hashes
    block_hashes2 = request2.block_hashes
    assert block_hashes1[0] != block_hashes2[0]
    assert block_hashes1[1] != block_hashes2[1]


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_hash_request_tokens_no_mm_inputs(hash_fn):
    request = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(6)],
        block_size=3,
        hash_fn=hash_fn,
        mm_positions=None,
        mm_hashes=None,
    )

    block_hashes = request.block_hashes

    assert len(block_hashes) == 2
    assert block_hashes[0] == hash_fn((kv_cache_utils.NONE_HASH, (0, 1, 2), None))
    assert block_hashes[1] == hash_fn((block_hashes[0], (3, 4, 5), None))


def _stats(requests: int, queries: int, hits: int) -> PrefixCacheStats:
    return PrefixCacheStats(requests=requests, queries=queries, hits=hits)


def test_metrics():
    """
    Test the prefix caching metrics.
    """
    metrics = CachingMetrics(max_recent_requests=5)
    assert metrics.hit_rate == 0.0

    metrics.observe(_stats(1, 20, 9))
    # 9 / 20 = 0.45
    assert metrics.hit_rate == 0.45

    metrics.observe(_stats(4, 80, 16))

    # 25 / 100 = 0.25
    assert metrics.hit_rate == 0.25

    metrics.observe(_stats(1, 10, 2))

    # Remove (20, 9) and add (10, 2): 18 / 90 = 0.2
    assert metrics.aggregated_requests == 5
    assert metrics.aggregated_query_total == 90
    assert metrics.aggregated_query_hit == 18
    assert metrics.hit_rate == 0.2

    metrics.reset()
    assert metrics.hit_rate == 0.0
    assert metrics.aggregated_requests == 0
    assert metrics.aggregated_query_total == 0
    assert metrics.aggregated_query_hit == 0
    assert not metrics.query_queue


def test_metrics_empty_stats():
    """
    Test the prefix caching metrics with empty stats.
    """
    metrics = CachingMetrics(max_recent_requests=5)
    metrics.observe(_stats(0, 0, 0))
    metrics.observe(_stats(1, 20, 9))
    metrics.observe(_stats(0, 0, 0))
    metrics.observe(_stats(4, 80, 16))
    metrics.observe(_stats(0, 0, 0))
    metrics.observe(_stats(1, 10, 2))
    # Remove (20, 9) and add (10, 2): 18 / 90 = 0.2
    assert metrics.aggregated_requests == 5
    assert metrics.aggregated_query_total == 90
    assert metrics.aggregated_query_hit == 18
    assert metrics.hit_rate == 0.2

    # Only the latest added stats preserved 10 / 20 = 0.5
    metrics.observe(_stats(11, 20, 10))
    assert metrics.aggregated_requests == 11
    assert metrics.aggregated_query_total == 20
    assert metrics.aggregated_query_hit == 10
    assert metrics.hit_rate == 0.5

    # Only the latest added stats preserved 30 / 40 = 0.75
    metrics.observe(_stats(22, 40, 30))
    assert metrics.aggregated_requests == 22
    assert metrics.aggregated_query_total == 40
    assert metrics.aggregated_query_hit == 30
    assert metrics.hit_rate == 0.75


def test_get_kv_cache_configs_multiple_workers():
    model_config = ModelConfig(max_model_len=16)
    vllm_config = VllmConfig(model_config=model_config)

    ref_kv_cache_spec = new_kv_cache_spec()
    same_kv_cache_specs = [
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_kv_cache_spec(),
        },
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_kv_cache_spec(),
        },
    ]

    # Basic case. All things are the same.
    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        same_kv_cache_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
        ],
    )
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer1"]
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer2"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1", "layer2"], ref_kv_cache_spec),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer1"]
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer2"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1", "layer2"], ref_kv_cache_spec),
            ],
        ),
    ]

    # Different available memory. This is the case for TP.
    # Use the smallest memory available.
    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        same_kv_cache_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 20,
        ],
    )
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer1"]
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer2"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1", "layer2"], ref_kv_cache_spec),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer1"]
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer2"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1", "layer2"], ref_kv_cache_spec),
            ],
        ),
    ]

    # Different KV cache specs. This is the case for PP.
    different_layer_specs = [
        {
            "layer1": new_kv_cache_spec(),
        },
        {
            "layer2": new_kv_cache_spec(),
            "layer3": new_kv_cache_spec(),
        },
    ]

    # Different workers have different layers.
    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        different_layer_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
        ],
    )
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer1"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1"], new_kv_cache_spec()),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer2"]
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer3"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer2", "layer3"], new_kv_cache_spec()),
            ],
        ),
    ]

    # Some layers are the same, some are different. This is the case for TP+PP
    tp_pp_kv_cache_specs = [
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_kv_cache_spec(),
        },
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_kv_cache_spec(),
        },
        {
            "layer3": new_kv_cache_spec(),
        },
        {
            "layer3": new_kv_cache_spec(),
        },
    ]

    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        tp_pp_kv_cache_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
        ],
    )
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer1"]
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer2"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1", "layer2"], ref_kv_cache_spec),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer1"]
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer2"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1", "layer2"], ref_kv_cache_spec),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer3"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer3"], ref_kv_cache_spec),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer3"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer3"], ref_kv_cache_spec),
            ],
        ),
    ]

    # Different workers have different types of layers. This is the case for
    # hybrid models + PP.
    different_type_layer_specs = [
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_kv_cache_spec(),
        },
        {
            "layer3": new_sliding_window_spec(),
            "layer4": new_sliding_window_spec(),
        },
    ]
    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        different_type_layer_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
        ],
    )
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer1"]
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer2"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1", "layer2"], ref_kv_cache_spec),
                KVCacheGroupSpec([], new_sliding_window_spec()),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer3"]
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer4"]
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec([], ref_kv_cache_spec),
                KVCacheGroupSpec(["layer3", "layer4"], new_sliding_window_spec()),
            ],
        ),
    ]

    # When divided into multiple KVCacheGroups, need to ensure the number of
    # layers per group is similar.
    different_type_layer_specs = [
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_sliding_window_spec(),
            "layer3": new_sliding_window_spec(),
        },
        {
            "layer4": new_kv_cache_spec(),
            "layer5": new_sliding_window_spec(),
            "layer6": new_sliding_window_spec(),
        },
    ]
    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        different_type_layer_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 10,
            ref_kv_cache_spec.page_size_bytes * 10,
        ],
    )
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10,
                    shared_by=["layer1", "layer2", "layer3"],
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1"], ref_kv_cache_spec),
                KVCacheGroupSpec(["layer2"], new_sliding_window_spec()),
                KVCacheGroupSpec(["layer3"], new_sliding_window_spec()),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10,
                    shared_by=["layer4", "layer5", "layer6"],
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer4"], ref_kv_cache_spec),
                KVCacheGroupSpec(["layer5"], new_sliding_window_spec()),
                KVCacheGroupSpec(["layer6"], new_sliding_window_spec()),
            ],
        ),
    ]

    # Have conflicting layers. Need to raise an error.
    conflicting_layer_specs = [
        {
            "layer1": new_kv_cache_spec(),
        },
        {
            "layer1": new_sliding_window_spec(),
        },
    ]
    with pytest.raises(AssertionError):
        get_kv_cache_configs(
            vllm_config,
            conflicting_layer_specs,
            [
                ref_kv_cache_spec.page_size_bytes * 2 * 10,
                ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ],
        )


@pytest.mark.parametrize(
    "asymmetric_memory",
    [False, True],
    ids=["symmetric", "asymmetric"],
)
def test_get_kv_cache_configs_pp_sharding(asymmetric_memory):
    model_config = ModelConfig(max_model_len=512)
    vllm_config = VllmConfig(model_config=model_config)

    ref_kv_cache_spec = new_kv_cache_spec()
    pp_kv_cache_specs = [
        {"layer1": ref_kv_cache_spec},
        {"layer2": ref_kv_cache_spec},
    ]

    expected_num_blocks = model_config.max_model_len // ref_kv_cache_spec.block_size + 1
    avail_memory = ref_kv_cache_spec.page_size_bytes * expected_num_blocks

    # With per-worker validation, each worker only needs memory for its own
    # layers. Worker 2 having more memory shouldn't affect worker 1's config.
    available_memory = (
        [avail_memory, avail_memory * 2] if asymmetric_memory else [avail_memory] * 2
    )

    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        pp_kv_cache_specs,
        available_memory,
    )

    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=expected_num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * expected_num_blocks,
                    shared_by=["layer1"],
                ),
            ],
            kv_cache_groups=[KVCacheGroupSpec(["layer1"], ref_kv_cache_spec)],
        ),
        KVCacheConfig(
            num_blocks=expected_num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * expected_num_blocks,
                    shared_by=["layer2"],
                ),
            ],
            kv_cache_groups=[KVCacheGroupSpec(["layer2"], ref_kv_cache_spec)],
        ),
    ]


def test_project_kv_cache_groups_to_worker():
    spec_a = new_kv_cache_spec()
    spec_b = new_kv_cache_spec(num_kv_heads=4)

    global_groups = [
        KVCacheGroupSpec(["layer1", "layer2", "layer3"], spec_a),
    ]
    worker_spec = {"layer1": spec_a, "layer2": spec_a}
    projected = kv_cache_utils._project_kv_cache_groups_to_worker(
        global_groups, worker_spec
    )
    assert len(projected) == 1
    assert projected[0].layer_names == ["layer1", "layer2"]
    assert projected[0].kv_cache_spec is spec_a

    projected = kv_cache_utils._project_kv_cache_groups_to_worker(
        global_groups, {"layer4": spec_a}
    )
    assert len(projected) == 1
    assert projected[0].layer_names == []
    assert projected[0].kv_cache_spec is spec_a

    uniform_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={"layer1": spec_a, "layer2": spec_b, "layer3": spec_a},
    )
    global_groups_uniform = [
        KVCacheGroupSpec(["layer1", "layer2", "layer3"], uniform_spec),
    ]
    projected = kv_cache_utils._project_kv_cache_groups_to_worker(
        global_groups_uniform, {"layer1": spec_a, "layer3": spec_a}
    )
    assert len(projected) == 1
    assert projected[0].layer_names == ["layer1", "layer3"]
    proj_spec = projected[0].kv_cache_spec
    assert isinstance(proj_spec, UniformTypeKVCacheSpecs)
    assert set(proj_spec.kv_cache_specs.keys()) == {"layer1", "layer3"}


def test_merge_kv_cache_spec():
    same_layer_specs = [
        new_kv_cache_spec(num_kv_heads=32),
        new_kv_cache_spec(num_kv_heads=32),
    ]
    merged_layer_spec = same_layer_specs[0].merge(same_layer_specs)
    assert merged_layer_spec.block_size == 16
    assert merged_layer_spec.num_kv_heads == 32
    assert merged_layer_spec.head_size == 64
    assert merged_layer_spec.dtype == torch.float32
    assert merged_layer_spec.sliding_window is None

    different_layer_specs = [
        new_kv_cache_spec(num_kv_heads=32),
        new_kv_cache_spec(num_kv_heads=16),
    ]
    with pytest.raises(AssertionError):
        different_layer_specs[0].merge(different_layer_specs)

    full_spec = new_kv_cache_spec(num_kv_heads=32)
    different_type_layer_specs = [
        full_spec,
        SlidingWindowSpec(
            block_size=full_spec.block_size,
            num_kv_heads=full_spec.num_kv_heads,
            head_size=full_spec.head_size,
            dtype=full_spec.dtype,
            sliding_window=1,
        ),
    ]
    with pytest.raises(AssertionError):
        different_type_layer_specs[0].merge(different_type_layer_specs)
    with pytest.raises(AssertionError):
        different_type_layer_specs[1].merge(different_type_layer_specs)

    different_sliding_window_layer_specs = [
        new_kv_cache_spec(num_kv_heads=32),
        new_kv_cache_spec(num_kv_heads=32, sliding_window=1),
        new_kv_cache_spec(num_kv_heads=32, sliding_window=2),
    ]
    with pytest.raises(ValueError):
        different_sliding_window_layer_specs[0].merge(
            different_sliding_window_layer_specs
        )

    same_sliding_window_layer_specs = [
        new_kv_cache_spec(num_kv_heads=32, sliding_window=1),
        new_kv_cache_spec(num_kv_heads=32, sliding_window=1),
    ]
    merged_layer_spec = same_sliding_window_layer_specs[0].merge(
        same_sliding_window_layer_specs
    )
    assert merged_layer_spec.sliding_window == 1

    same_sliding_window_layer_spec_with_none = [
        new_kv_cache_spec(num_kv_heads=32, sliding_window=1),
        new_kv_cache_spec(num_kv_heads=32, sliding_window=None),
    ]
    merged_layer_spec = same_sliding_window_layer_spec_with_none[0].merge(
        same_sliding_window_layer_spec_with_none
    )
    assert merged_layer_spec.sliding_window == 1


def test_is_kv_cache_spec_uniform():
    kv_cache_spec = {
        "layer_1": new_kv_cache_spec(num_kv_heads=32),
        "layer_2": new_kv_cache_spec(num_kv_heads=32),
    }
    assert is_kv_cache_spec_uniform(kv_cache_spec)

    kv_cache_spec = {
        "layer_1": new_kv_cache_spec(num_kv_heads=32),
        "layer_2": new_kv_cache_spec(num_kv_heads=32, sliding_window=1),
    }
    assert is_kv_cache_spec_uniform(kv_cache_spec)

    kv_cache_spec = {
        "layer_1": new_kv_cache_spec(num_kv_heads=32),
        "layer_2": new_sliding_window_spec(num_kv_heads=32, sliding_window=1),
    }
    assert not is_kv_cache_spec_uniform(kv_cache_spec)

    kv_cache_spec = {
        "layer_1": new_sliding_window_spec(num_kv_heads=32, sliding_window=1),
        "layer_2": new_sliding_window_spec(num_kv_heads=32, sliding_window=1),
    }
    assert is_kv_cache_spec_uniform(kv_cache_spec)

    kv_cache_spec = {
        "layer_1": new_sliding_window_spec(num_kv_heads=32, sliding_window=1),
        "layer_2": new_sliding_window_spec(num_kv_heads=32, sliding_window=2),
    }
    assert not is_kv_cache_spec_uniform(kv_cache_spec)


@pytest.mark.parametrize(
    ("model_id", "max_model_len", "want_estimated_max_len"),
    [
        ("Qwen/Qwen1.5-7B", 16385, 16384),
        ("Qwen/Qwen1.5-7B", 16383, 16383),
    ],
)
def test_estimate_max_model_len(model_id, max_model_len, want_estimated_max_len):
    # Create a VllmConfig
    model_config = ModelConfig(
        model_id,
        runner="generate",
        dtype="float16",
        max_model_len=max_model_len,
    )
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=32768,
        max_model_len=model_config.max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
    )

    # Create KV cache specs
    kv_cache_spec = {}
    for i in range(32):
        layer_name = f"layer_{i}"
        kv_cache_spec[layer_name] = FullAttentionSpec(
            block_size=16,
            num_kv_heads=32,
            head_size=128,
            dtype=torch.float16,
        )
    # Estimate the maximum model length, 16384 model_len need 8GB
    estimated_max_len = estimate_max_model_len(
        vllm_config, kv_cache_spec, 8 * GiB_bytes
    )
    assert estimated_max_len == want_estimated_max_len


def test_get_max_concurrency_for_kv_cache_config():
    # Create a VllmConfig
    model_id = "Qwen/Qwen1.5-7B"
    max_model_len = 16384
    model_config = ModelConfig(
        model_id,
        runner="generate",
        dtype="float16",
        max_model_len=max_model_len,
    )
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
        max_model_len=model_config.max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
    )

    full_attention_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=32,
        head_size=128,
        dtype=torch.float16,
    )

    sliding_window_spec = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=32,
        head_size=128,
        dtype=torch.float16,
        sliding_window=1024,
    )

    kv_cache_config_full_attention = KVCacheConfig(
        num_blocks=int(1024 * 1.5),
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer_{i}" for i in range(32)], full_attention_spec),
        ],
    )
    max_concurrency_full_attention = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config_full_attention
    )
    assert max_concurrency_full_attention == 1.5

    kv_cache_config_sliding_window = KVCacheConfig(
        num_blocks=129 * 3,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer_{i}" for i in range(32)], sliding_window_spec),
        ],
    )
    max_concurrency_sliding_window = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config_sliding_window
    )
    assert max_concurrency_sliding_window == 3

    kv_cache_config_hybrid_model = KVCacheConfig(
        num_blocks=(1024 + 129) * 3,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer_{i}" for i in range(32)], full_attention_spec),
            KVCacheGroupSpec(
                [f"layer_{i}" for i in range(32, 64)], sliding_window_spec
            ),
        ],
    )
    max_concurrency_hybrid_model = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config_hybrid_model
    )
    assert max_concurrency_hybrid_model == 3
    num_tokens, max_concurrency = get_kv_cache_capacity(
        vllm_config, kv_cache_config_hybrid_model
    )
    assert num_tokens == max_concurrency_hybrid_model * max_model_len
    assert max_concurrency == max_concurrency_hybrid_model


def test_allocate_with_lookahead():
    """Verify that lookahead tokens correctly affect block allocation"""
    block_size = 4
    config = KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[
            KVCacheTensor(size=100, shared_by=["layer1"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer1"], new_kv_cache_spec(block_size=block_size)),
        ],
    )

    request = make_request(
        request_id="0",
        prompt_token_ids=[],
        block_size=block_size,
        mm_positions=None,
        mm_hashes=None,
    )

    # Test case 1: Requires additional lookahead tokens
    kv_cache_manager = KVCacheManager(
        kv_cache_config=config,
        max_model_len=100,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
    )
    blocks = kv_cache_manager.allocate_slots(
        request,
        num_new_tokens=3,
        num_lookahead_tokens=2,  # Total required: 3+2=5 tokens
    )
    assert len(blocks.get_block_ids()[0]) == 2  # ceil(5/4)=2 blocks

    # Test case 2: With precomputed blocks
    kv_cache_manager = KVCacheManager(
        kv_cache_config=config,
        max_model_len=100,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
    )
    # required_blocks = ceil((3 + 2) /4) = 2
    blocks = kv_cache_manager.allocate_slots(
        request,
        num_new_tokens=3,
        num_lookahead_tokens=2,
    )
    assert len(blocks.get_block_ids()[0]) == 2

    # Test case 3: With precomputed blocks
    # required_blocks = ceil((3 + 4) / 4) = 2
    kv_cache_manager = KVCacheManager(
        kv_cache_config=config,
        max_model_len=100,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
    )
    blocks = kv_cache_manager.allocate_slots(
        request,
        num_new_tokens=3,
        num_lookahead_tokens=4,
    )
    assert len(blocks.get_block_ids()[0]) == 2


def test_get_kv_cache_config_one_worker():
    # pass max_model_len to pass check_enough_kv_cache_memory
    model_config = ModelConfig(max_model_len=16)
    vllm_config = VllmConfig(model_config=model_config)

    mem_per_block_per_layer = 16 * 2 * 64 * 4 * 2
    # all layers are full attention -> single group
    kv_cache_specs_full = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
    }
    kv_cache_config_full = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_full], [mem_per_block_per_layer * 2 * 32]
    )[0]
    print(kv_cache_config_full)
    assert kv_cache_config_full == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(size=mem_per_block_per_layer * 32, shared_by=["layer_1"]),
            KVCacheTensor(size=mem_per_block_per_layer * 32, shared_by=["layer_2"]),
        ],
        kv_cache_groups=[KVCacheGroupSpec(["layer_1", "layer_2"], new_kv_cache_spec())],
    )

    # all layers are sliding window -> single group
    kv_cache_specs_sliding = {
        "layer_1": new_sliding_window_spec(),
        "layer_2": new_sliding_window_spec(),
    }
    kv_cache_config_sliding = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_sliding], [mem_per_block_per_layer * 2 * 32]
    )[0]
    assert kv_cache_config_sliding == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(size=mem_per_block_per_layer * 32, shared_by=["layer_1"]),
            KVCacheTensor(size=mem_per_block_per_layer * 32, shared_by=["layer_2"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1", "layer_2"], new_sliding_window_spec())
        ],
    )

    # full + sliding, but disable_hybrid_kv_cache_manager
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = True
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_sliding_window_spec(),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 2 * 32]
    )[0]
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(size=mem_per_block_per_layer * 32, shared_by=["layer_1"]),
            KVCacheTensor(size=mem_per_block_per_layer * 32, shared_by=["layer_2"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer_1", "layer_2"], new_kv_cache_spec(sliding_window=1)
            ),
        ],
    )
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False

    # full + sliding, with hybrid_kv_cache_manager
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_sliding_window_spec(),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 2 * 32]
    )[0]
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 64, shared_by=["layer_1", "layer_2"]
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1"], new_kv_cache_spec()),
            KVCacheGroupSpec(["layer_2"], new_sliding_window_spec()),
        ],
    )

    # 2 full + 4 sliding, 2 layers per group
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
        "layer_3": new_sliding_window_spec(),
        "layer_4": new_sliding_window_spec(),
        "layer_5": new_sliding_window_spec(),
        "layer_6": new_sliding_window_spec(),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 2 * 32]
    )[0]
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_1", "layer_3", "layer_4"],
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_2", "layer_5", "layer_6"],
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1", "layer_2"], new_kv_cache_spec()),
            KVCacheGroupSpec(["layer_3", "layer_5"], new_sliding_window_spec()),
            KVCacheGroupSpec(["layer_4", "layer_6"], new_sliding_window_spec()),
        ],
    )

    # 3 full + 7 sliding, pad to 3 full + 9 sliding
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
        "layer_3": new_kv_cache_spec(),
        "layer_4": new_sliding_window_spec(),
        "layer_5": new_sliding_window_spec(),
        "layer_6": new_sliding_window_spec(),
        "layer_7": new_sliding_window_spec(),
        "layer_8": new_sliding_window_spec(),
        "layer_9": new_sliding_window_spec(),
        "layer_10": new_sliding_window_spec(),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 3 * 32]
    )[0]
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_1", "layer_4", "layer_5", "layer_6"],
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_2", "layer_7", "layer_8", "layer_9"],
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32, shared_by=["layer_3", "layer_10"]
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1", "layer_2", "layer_3"], new_kv_cache_spec()),
            KVCacheGroupSpec(
                ["layer_4", "layer_7", "layer_10"], new_sliding_window_spec()
            ),
            KVCacheGroupSpec(["layer_5", "layer_8"], new_sliding_window_spec()),
            KVCacheGroupSpec(["layer_6", "layer_9"], new_sliding_window_spec()),
        ],
    )

    # 6 full + 5 sliding, pad to 6 full + 6 sliding. This is a typical case for gpt-oss
    # eagle where there is only one more full attention layer than sliding window layers
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
        "layer_3": new_kv_cache_spec(),
        "layer_4": new_kv_cache_spec(),
        "layer_5": new_kv_cache_spec(),
        "layer_6": new_kv_cache_spec(),
        "layer_7": new_sliding_window_spec(),
        "layer_8": new_sliding_window_spec(),
        "layer_9": new_sliding_window_spec(),
        "layer_10": new_sliding_window_spec(),
        "layer_11": new_sliding_window_spec(),
    }

    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 6 * 32]
    )[0]
    print(kv_cache_config_hybrid)
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_1", "layer_7"],
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_2", "layer_8"],
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_3", "layer_9"],
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_4", "layer_10"],
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_5", "layer_11"],
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                shared_by=["layer_6"],
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer_1", "layer_2", "layer_3", "layer_4", "layer_5", "layer_6"],
                new_kv_cache_spec(),
            ),
            KVCacheGroupSpec(
                ["layer_7", "layer_8", "layer_9", "layer_10", "layer_11"],
                new_sliding_window_spec(),
            ),
        ],
    )

    # different hidden size but same type, use UniformTypeKVCacheSpecs
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(head_size=128),
        "layer_2": new_kv_cache_spec(head_size=64),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 3 * 32]
    )[0]
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(size=mem_per_block_per_layer * 32 * 2, shared_by=["layer_1"]),
            KVCacheTensor(size=mem_per_block_per_layer * 32, shared_by=["layer_2"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer_1", "layer_2"],
                UniformTypeKVCacheSpecs(
                    block_size=16, kv_cache_specs=kv_cache_specs_hybrid
                ),
            )
        ],
    )

    # Different hidden size and different type (full + sliding window with
    # different natural page sizes, the Gemma-4 shape, #39133). This hits the
    # gemma-4 per-group allocator (G2): keep each group's natural page size and
    # emit one tensor per (slot, page_size), but size each bucket to its *class*
    # depth -- a shallow window-bounded pool (D_sw) for the sliding group and a
    # deep pool (D_global) for the global group -- reported via
    # `per_group_num_blocks`.
    #
    # full page 16384, sliding page 8192; max_model_len=16 -> maxlen_blocks=1;
    # admit_blocks (window=1, max_num_batched_tokens=2048) = 2.
    # g_per_block = 16384, s_per_block = 8192, denom = 1*16384 + 2*8192 = 32768;
    # C = min(max_num_seqs=128, 524288 // 32768) = 16; D_sw = 2*16 = 32;
    # D_global = (524288 - 32*8192) // 16384 = 16. (Here the tiny max_model_len
    # makes the global pool the *shallow* one and the window-bounded sliding
    # pool the deep one -- the inverse of the gemma-4 200K case, but the
    # per-class sizing machinery is identical.)
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(head_size=64),
        "layer_2": new_sliding_window_spec(head_size=32),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 32]
    )[0]
    # D_sw = 2*16 + 1 = 33 (+1 for the null block so free=D_sw-1=32 >= one
    # request's sliding demand sw_req=2); D_global = (524288 - 33*8192) //
    # 16384 = 15 (free=14 >= gl_req=1). num_blocks = max(33, 15) = 33.
    # (C=16 here, so free has ample slack; the tight C=1 regime where the
    # null-block off-by-one actually deadlocks is covered by the dedicated
    # test_gemma4_sliding_pool_no_admission_deadlock / _per_group_pool_depths.)
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=33,
        kv_cache_tensors=[
            # full (global) layer: page 16384 x D_global=15
            KVCacheTensor(size=mem_per_block_per_layer * 15, shared_by=["layer_1"]),
            # sliding layer: page 8192 x D_sw=33
            KVCacheTensor(
                size=(mem_per_block_per_layer // 2) * 33, shared_by=["layer_2"]
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1"], new_kv_cache_spec(head_size=64)),
            KVCacheGroupSpec(["layer_2"], new_sliding_window_spec(head_size=32)),
        ],
        per_group_num_blocks=[15, 33],
    )

    # Different hidden size that cannot be page-aligned by adjusting block_size
    # (page 24576 is not a multiple of 16384). The bucketed (slot, page_size)
    # layout handles any set of page sizes; with the gemma-4 per-group allocator
    # each group is sized to its class depth.
    # full page 16384, sliding page 24576; max_model_len=16 -> maxlen_blocks=1;
    # admit_blocks = 2; g_per_block = 16384, s_per_block = 24576;
    # denom = 1*16384 + 2*24576 = 65536; available = 16384*2*32 = 1048576;
    # C = min(128, 1048576 // 65536) = 16; D_sw = 2*16 + 1 = 33 (+1 null block,
    # free=32 >= sw_req=2); D_global = (1048576 - 33*24576) // 16384 = 14
    # (free=13 >= gl_req=1); num_blocks = max(33, 14) = 33.
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(head_size=64),
        "layer_2": new_sliding_window_spec(head_size=96),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 2 * 32]
    )[0]
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=33,
        kv_cache_tensors=[
            # full (global) layer: page 16384 x D_global=14
            KVCacheTensor(size=mem_per_block_per_layer * 14, shared_by=["layer_1"]),
            # sliding layer: page 24576 x D_sw=33
            KVCacheTensor(
                size=(mem_per_block_per_layer * 3 // 2) * 33, shared_by=["layer_2"]
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1"], new_kv_cache_spec(head_size=64)),
            KVCacheGroupSpec(["layer_2"], new_sliding_window_spec(head_size=96)),
        ],
        per_group_num_blocks=[14, 33],
    )

    # Test num_gpu_blocks_override
    vllm_config.cache_config.num_gpu_blocks_override = 16
    kv_cache_config_override_blocks = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_full], [mem_per_block_per_layer * 2 * 32]
    )[0]
    assert kv_cache_config_override_blocks == KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[
            KVCacheTensor(size=mem_per_block_per_layer * 16, shared_by=["layer_1"]),
            KVCacheTensor(size=mem_per_block_per_layer * 16, shared_by=["layer_2"]),
        ],
        kv_cache_groups=[KVCacheGroupSpec(["layer_1", "layer_2"], new_kv_cache_spec())],
    )


def _physical_bytes_per_layer(kv_cache_config: KVCacheConfig) -> dict[str, float]:
    """Attribute each KVCacheTensor's physical bytes across the layers sharing it.

    A ``KVCacheTensor`` is a single physical buffer shared by the layers listed
    in ``shared_by`` (they have disjoint block-id namespaces). We split its
    ``size`` evenly across those layers to get a per-layer physical-byte budget.
    """
    per_layer: dict[str, float] = {}
    for tensor in kv_cache_config.kv_cache_tensors:
        share = len(tensor.shared_by)
        assert share > 0
        for layer_name in tensor.shared_by:
            per_layer[layer_name] = per_layer.get(layer_name, 0.0) + tensor.size / share
    return per_layer


def test_gemma4_hybrid_swa_physically_capped():
    """Gemma-4 hybrid (SWA + global) with different page sizes (#39133).

    Gemma-4 mixes 40 sliding-window layers (head_dim=256, 8 KV heads) with 8
    global/full-attention layers (head_dim=512, 1 KV head). The two spec types
    have *different* page sizes, so this model falls through the general
    multi-group allocator path. The sliding layers' KV is window-bounded
    (``SlidingWindowSpec.max_memory_usage_bytes`` is capped at the 1024-token
    window), while the global layers scale with ``max_model_len``. The allocator
    must therefore give each sliding layer a *physically smaller* block budget
    than each global layer instead of one shared num_blocks for every layer.
    """
    # max_model_len well above the sliding window so the window cap dominates:
    # a global layer's budget scales with 32768 tokens while a sliding layer's
    # is bounded by the 1024-token window (+ chunked-prefill batch).
    model_config = ModelConfig(max_model_len=32768)
    vllm_config = VllmConfig(model_config=model_config)

    # 48 layers; the 8 global layers sit at the Gemma-4 global indices.
    global_indices = {5, 11, 17, 23, 29, 35, 41, 47}
    kv_cache_spec: dict[str, KVCacheSpec] = {}
    for i in range(48):
        name = f"layer_{i}"
        if i in global_indices:
            kv_cache_spec[name] = new_kv_cache_spec(num_kv_heads=1, head_size=512)
        else:
            kv_cache_spec[name] = new_sliding_window_spec(
                num_kv_heads=8, head_size=256, sliding_window=1024
            )

    a_global = "layer_5"
    a_sliding = "layer_0"
    global_spec = kv_cache_spec[a_global]
    sliding_spec = kv_cache_spec[a_sliding]
    # Precondition for the bug: the per-spec window cap is strictly below the
    # full-attention budget, so a correct allocator must allocate less for the
    # sliding layer.
    assert sliding_spec.max_memory_usage_bytes(
        vllm_config
    ) < global_spec.max_memory_usage_bytes(vllm_config)

    available_memory = 4 * GiB_bytes
    kv_cache_config = get_kv_cache_configs(
        vllm_config, [kv_cache_spec], [available_memory]
    )[0]

    per_layer_bytes = _physical_bytes_per_layer(kv_cache_config)
    sliding_bytes = per_layer_bytes[a_sliding]
    global_bytes = per_layer_bytes[a_global]

    # The core #39133 contract: a window-bounded sliding layer must receive a
    # smaller per-layer KV-cache allocation than a global layer. Before the fix
    # the general allocator padded every layer up to the (larger) sliding page
    # and shared one num_blocks across all groups, so a sliding layer and a
    # global layer received identical bytes. The fix keeps each group's natural
    # page size and emits one tensor per (slot, page_size) -- the DeepseekV4
    # bucketed layout -- so the global layers are no longer over-padded.
    #
    # NOTE: vLLM uses a single shared block pool of `num_blocks` blocks indexed
    # by a global block-id space, so every layer's physical tensor must be sized
    # for `num_blocks` to stay in bounds; the sliding-window memory saving is
    # realized through the runtime admission cap (more concurrency), not a
    # physically under-sized tensor. The per-layer figure above is the amortized
    # share of each (slot, page_size) tensor across the layers sharing it.
    assert sliding_bytes < global_bytes, (
        f"sliding layer got {sliding_bytes} bytes, global layer got "
        f"{global_bytes} bytes; sliding KV must be smaller than global KV"
    )


def test_gemma4_per_group_pool_depths():
    """Gemma-4 per-group KV pool depths (G2, #39133).

    The per-group allocator must give the window-bounded sliding groups a
    *shallow* pool (D_sw) and the global groups a *deep* pool (D_global),
    reported via ``KVCacheConfig.per_group_num_blocks``. Each (slot, page_size)
    tensor is physically sized to its class depth, and the total must never
    exceed the available pool.
    """
    # max_model_len well above the sliding window (1024) so the sliding pool is
    # window-bounded (shallow) while the global pool must hold a full-length
    # request (deep): D_sw << D_global.
    max_model_len = 32768
    model_config = ModelConfig(max_model_len=max_model_len)
    vllm_config = VllmConfig(model_config=model_config)
    block_size = vllm_config.cache_config.block_size

    # 48 layers: 8 global at the Gemma-4 global indices, 40 sliding-window.
    global_indices = {5, 11, 17, 23, 29, 35, 41, 47}
    kv_cache_spec: dict[str, KVCacheSpec] = {}
    for i in range(48):
        name = f"layer_{i}"
        if i in global_indices:
            kv_cache_spec[name] = new_kv_cache_spec(num_kv_heads=1, head_size=512)
        else:
            kv_cache_spec[name] = new_sliding_window_spec(
                num_kv_heads=8, head_size=256, sliding_window=1024
            )

    sliding_page = kv_cache_spec["layer_0"].page_size_bytes
    global_page = kv_cache_spec["layer_5"].page_size_bytes
    assert sliding_page != global_page

    available_memory = 4 * GiB_bytes
    cfg = get_kv_cache_configs(vllm_config, [kv_cache_spec], [available_memory])[0]

    # 6 kv_cache_groups: 5 SlidingWindowSpec + 1 FullAttentionSpec.
    assert len(cfg.kv_cache_groups) == 6
    n_sliding_groups = sum(
        isinstance(g.kv_cache_spec, SlidingWindowSpec) for g in cfg.kv_cache_groups
    )
    assert n_sliding_groups == 5

    # per_group_num_blocks populated with 6 entries: 5 x D_sw + 1 x D_global.
    assert cfg.per_group_num_blocks is not None
    assert len(cfg.per_group_num_blocks) == 6
    sliding_depths = {
        n
        for g, n in zip(cfg.kv_cache_groups, cfg.per_group_num_blocks)
        if isinstance(g.kv_cache_spec, SlidingWindowSpec)
    }
    global_depths = {
        n
        for g, n in zip(cfg.kv_cache_groups, cfg.per_group_num_blocks)
        if not isinstance(g.kv_cache_spec, SlidingWindowSpec)
    }
    assert len(sliding_depths) == 1 and len(global_depths) == 1
    d_sw = sliding_depths.pop()
    d_global = global_depths.pop()

    # NO-DEADLOCK INVARIANT (#39133 per-group follow-up): the sliding pool must
    # hold at least ONE request's full sliding demand = n_sliding_groups x
    # admit_blocks. Each sliding group has its own block table and reserves up to
    # admit_blocks per request, so the shared sliding pool needs the SUM across
    # groups. If D_sw < this, the full-sequence admission gate rejects every
    # request and the scheduler spins forever at 0% GPU.
    admit_blocks = kv_cache_spec["layer_0"].max_admission_blocks_per_request(
        vllm_config.scheduler_config.max_num_batched_tokens, max_model_len
    )
    # free = D_sw - 1 (block id 0 is the reserved null block); the FIRST
    # request's full sliding demand must fit that free count, else the gate
    # rejects request #1 and the scheduler spins (observed free=964 < 965).
    assert d_sw - 1 >= n_sliding_groups * admit_blocks, (
        f"free(D_sw-1)={d_sw - 1} < per-request sliding demand "
        f"{n_sliding_groups} x {admit_blocks} = {n_sliding_groups * admit_blocks}"
        " -> admission deadlock"
    )
    # ...and the per-request, per-group sliding reservation is window-bounded:
    # each sliding group reserves only `admit_blocks` (~window + one batch),
    # strictly less than a full-length request's `maxlen_blocks` -- that
    # per-request saving is the whole point of Gap B. (D_sw itself scales with
    # concurrency C, so D_sw vs maxlen_blocks is NOT a fixed relationship; the
    # genuine invariant is the per-request reservation below.)
    assert admit_blocks < cdiv(max_model_len, block_size), (
        f"sliding per-group reservation admit_blocks={admit_blocks} not "
        f"window-bounded below full-length {cdiv(max_model_len, block_size)}"
    )
    # Global pool must hold one full-length request.
    assert d_global >= cdiv(max_model_len, block_size)

    # Each tensor is sized page x (D_sw for sliding-page, D_global for global).
    sliding_sizes = {
        t.size for t in cfg.kv_cache_tensors if t.size % sliding_page == 0
        and t.size // sliding_page == d_sw
    }
    global_sizes = {
        t.size for t in cfg.kv_cache_tensors if t.size % global_page == 0
        and t.size // global_page == d_global
    }
    for t in cfg.kv_cache_tensors:
        if t.shared_by[0] == "layer_0":
            assert t.size == sliding_page * d_sw
        if t.shared_by[0] == "layer_5":
            assert t.size == global_page * d_global
    assert sliding_sizes == {sliding_page * d_sw}
    assert global_sizes == {global_page * d_global}

    # Hard budget: physical tensors must never exceed the available pool.
    total = sum(t.size for t in cfg.kv_cache_tensors)
    assert total <= available_memory, (
        f"total KV tensor bytes {total} exceed available {available_memory}"
    )

    # Legacy num_blocks is the deep (global) depth.
    assert cfg.num_blocks == max(d_sw, d_global) == d_global


def test_gemma4_sliding_int4_per_group_pool_depths():
    """Gemma-4 per-group pools with int4 TurboQuant on BOTH sliding and global
    layers (TQSlidingWindowSpec + TQFullAttentionSpec; VLLM_TQ_SLIDING_WINDOW=1).

    Verifies the int4 sliding spec (a) groups with itself and separately from
    the int4 globals (6 groups, 2 distinct page sizes), (b) uses the packed TQ
    slot page -- far smaller than bf16 -- and (c) keeps the Gap-B per-group
    depths + the no-admission-deadlock invariant.
    """
    from vllm.v1.kv_cache_interface import (
        TQFullAttentionSpec,
        TQSlidingWindowSpec,
    )

    max_model_len = 32768
    model_config = ModelConfig(max_model_len=max_model_len)
    vllm_config = VllmConfig(model_config=model_config)
    block_size = vllm_config.cache_config.block_size

    # turboquant_4bit_nc slot sizes: head_dim=512 -> 518, head_dim=256 -> 262.
    GLOBAL_SLOT = 518
    SLIDING_SLOT = 262

    global_indices = {5, 11, 17, 23, 29, 35, 41, 47}
    kv_cache_spec: dict[str, KVCacheSpec] = {}
    for i in range(48):
        name = f"layer_{i}"
        if i in global_indices:
            kv_cache_spec[name] = TQFullAttentionSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=512,
                dtype=torch.bfloat16,
                tq_slot_size=GLOBAL_SLOT,
            )
        else:
            kv_cache_spec[name] = TQSlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=8,
                head_size=256,
                dtype=torch.bfloat16,
                sliding_window=1024,
                tq_slot_size=SLIDING_SLOT,
            )

    sliding_page = kv_cache_spec["layer_0"].page_size_bytes
    global_page = kv_cache_spec["layer_5"].page_size_bytes
    # int4 TQ page uses the packed slot, NOT head_size * dtype.
    assert sliding_page == block_size * 8 * SLIDING_SLOT
    assert global_page == block_size * 1 * GLOBAL_SLOT
    assert sliding_page != global_page
    # The point of sliding-int4: the page is ~4x smaller than the bf16 sliding
    # page (block_size * 8 * (256 + 256) * 2 bytes).
    bf16_sliding_page = block_size * 8 * (256 + 256) * 2
    assert sliding_page < bf16_sliding_page / 3

    available_memory = 4 * GiB_bytes
    cfg = get_kv_cache_configs(vllm_config, [kv_cache_spec], [available_memory])[0]

    # 6 groups: 5 sliding (TQSlidingWindowSpec) + 1 global (TQFullAttentionSpec).
    assert len(cfg.kv_cache_groups) == 6
    n_sliding_groups = sum(
        isinstance(g.kv_cache_spec, SlidingWindowSpec) for g in cfg.kv_cache_groups
    )
    assert n_sliding_groups == 5
    # Exactly two distinct page sizes across the groups (sliding-int4 vs
    # global-int4), and they are the TQ-slot pages.
    page_sizes = {g.kv_cache_spec.page_size_bytes for g in cfg.kv_cache_groups}
    assert page_sizes == {sliding_page, global_page}

    assert cfg.per_group_num_blocks is not None
    assert len(cfg.per_group_num_blocks) == 6
    sliding_depths = {
        n
        for g, n in zip(cfg.kv_cache_groups, cfg.per_group_num_blocks)
        if isinstance(g.kv_cache_spec, SlidingWindowSpec)
    }
    global_depths = {
        n
        for g, n in zip(cfg.kv_cache_groups, cfg.per_group_num_blocks)
        if not isinstance(g.kv_cache_spec, SlidingWindowSpec)
    }
    assert len(sliding_depths) == 1 and len(global_depths) == 1
    d_sw = sliding_depths.pop()
    d_global = global_depths.pop()

    # Same no-deadlock invariant as the bf16 sliding case: the shared sliding
    # pool must hold one request's full sliding demand across all 5 groups.
    admit_blocks = kv_cache_spec["layer_0"].max_admission_blocks_per_request(
        vllm_config.scheduler_config.max_num_batched_tokens, max_model_len
    )
    assert d_sw - 1 >= n_sliding_groups * admit_blocks
    assert d_global >= cdiv(max_model_len, block_size)

    # Hard budget: physical tensors never exceed the available pool.
    total = sum(t.size for t in cfg.kv_cache_tensors)
    assert total <= available_memory


def test_gemma4_sliding_pool_no_admission_deadlock(monkeypatch):
    """Regression for the #39133 per-group follow-up deadlock.

    At large max_model_len the achievable concurrency C is small (1-2). The
    original sizing set ``D_sw = admit_blocks * C`` -- a SINGLE group's demand --
    but the sliding pool is shared by all ``n_sliding_groups`` groups (gemma4: 5),
    so one request's true demand is ``n_sliding_groups * admit_blocks``. With
    C < n_sliding_groups, ``D_sw`` fell BELOW one request's demand, the
    full-sequence admission gate rejected every request, and the v1 scheduler
    spun forever at 0% GPU (observed: demand=1330 > D_sw free=1153 at
    max_model_len=98304). D_sw must be >= the per-request sliding demand.
    """
    # The deadlock was observed in production at max_model_len=98304 (demand
    # 1330 > D_sw 1153). We test at 65536, which physically FITS at 4.66 GiB so
    # the config loads (rather than vLLM's _check_enough_kv_cache_memory
    # rejecting it upfront) and we can assert the positive no-deadlock invariant.
    # 65536 > the dummy model's max_position_embeddings (40960), so allow it.
    monkeypatch.setenv("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    max_model_len = 65536
    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=max_model_len))
    global_indices = {5, 11, 17, 23, 29, 35, 41, 47}
    kv_cache_spec: dict[str, KVCacheSpec] = {}
    for i in range(48):
        if i in global_indices:
            kv_cache_spec[f"layer_{i}"] = new_kv_cache_spec(num_kv_heads=1, head_size=512)
        else:
            kv_cache_spec[f"layer_{i}"] = new_sliding_window_spec(
                num_kv_heads=8, head_size=256, sliding_window=1024
            )
    # ~4.66 GiB: the real RTX-5080 KV pool after ~10 GB weights. With the old
    # formula this produced D_sw=1154 < demand 2885 (deadlock); the fix gives
    # C=1, D_sw = n_sliding_groups * admit_blocks.
    available_memory = int(4.66 * GiB_bytes)
    cfg = get_kv_cache_configs(vllm_config, [kv_cache_spec], [available_memory])[0]

    n_sliding_groups = sum(
        isinstance(g.kv_cache_spec, SlidingWindowSpec) for g in cfg.kv_cache_groups
    )
    admit_blocks = kv_cache_spec["layer_0"].max_admission_blocks_per_request(
        vllm_config.scheduler_config.max_num_batched_tokens, max_model_len
    )
    d_sw = next(
        n
        for g, n in zip(cfg.kv_cache_groups, cfg.per_group_num_blocks)
        if isinstance(g.kv_cache_spec, SlidingWindowSpec)
    )
    # The invariant the deadlock violated: one request must be admissible
    # against the pool's FREE count (D_sw - 1, excluding the null block).
    assert d_sw - 1 >= n_sliding_groups * admit_blocks, (
        f"free(D_sw-1)={d_sw - 1} < per-request demand "
        f"{n_sliding_groups}*{admit_blocks}={n_sliding_groups * admit_blocks}"
        " -> admission deadlock"
    )
    # And the config must still physically fit the pool.
    assert sum(t.size for t in cfg.kv_cache_tensors) <= available_memory


def test_get_kv_cache_configs_attention_free():
    kv_cache_specs: dict[str, KVCacheSpec] = {}
    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=16))
    kv_cache_configs = get_kv_cache_configs(vllm_config, [kv_cache_specs], [0])
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=[],
        )
    ]


def test_generate_uniform_type_kv_cache_specs():
    # All layers are full attention, can be merged
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(head_size=128),
    }
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(kv_cache_specs)
    assert uniform_spec == UniformTypeKVCacheSpecs(
        block_size=16, kv_cache_specs=kv_cache_specs
    )

    # Full attention + sliding window, cannot be merged
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_sliding_window_spec(sliding_window=1),
    }
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(kv_cache_specs)
    assert uniform_spec is None

    # different order of full attention + sliding window, cannot be merged
    kv_cache_specs = {
        "layer_1": new_sliding_window_spec(sliding_window=1),
        "layer_2": new_kv_cache_spec(),
    }
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(kv_cache_specs)
    assert uniform_spec is None

    # Same-size sliding window, can be merged
    kv_cache_specs = {
        "layer_1": new_sliding_window_spec(sliding_window=1),
        "layer_2": new_sliding_window_spec(sliding_window=1, head_size=128),
    }
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(kv_cache_specs)
    assert uniform_spec == UniformTypeKVCacheSpecs(
        block_size=16, kv_cache_specs=kv_cache_specs
    )

    # different block sizes, cannot be merged
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(block_size=16),
        "layer_2": new_kv_cache_spec(block_size=32),
    }
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(kv_cache_specs)
    assert uniform_spec is None


def test_generate_scheduler_kv_cache_config():
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(head_size=128),
    }
    kv_cache_configs = [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer_1", "layer_2"],
                    UniformTypeKVCacheSpecs(
                        block_size=16, kv_cache_specs=kv_cache_specs
                    ),
                ),
            ],
        )
    ]
    scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
    assert scheduler_kv_cache_config == KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer_1", "layer_2"], new_kv_cache_spec())],
    )


def new_mla_spec(cache_dtype_str=None):
    # head_size = kv_lora_rank(512) + qk_rope_head_dim(64) = 576
    return MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.float32,
        cache_dtype_str=cache_dtype_str,
    )


def test_get_kv_cache_spec_kind_prefers_specific_attention_subclasses():
    assert get_kv_cache_spec_kind(new_mla_spec()) == KVCacheSpecKind.MLA_ATTENTION

    sliding_window_mla_spec = SlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.float32,
        sliding_window=128,
    )
    assert (
        get_kv_cache_spec_kind(sliding_window_mla_spec)
        == KVCacheSpecKind.SLIDING_WINDOW_MLA
    )

    sink_full_attention_spec = SinkFullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.float32,
        sink_len=4,
    )
    assert (
        get_kv_cache_spec_kind(sink_full_attention_spec)
        == KVCacheSpecKind.SINK_FULL_ATTENTION
    )


def test_get_kv_cache_spec_kind_unwraps_uniform_type_specs():
    uniform_mla_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={
            "layer_1": new_mla_spec(),
            "layer_2": new_mla_spec(cache_dtype_str="fp8"),
        },
    )
    assert get_kv_cache_spec_kind(uniform_mla_spec) == KVCacheSpecKind.MLA_ATTENTION

    uniform_swa_mla_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={
            "layer_1": SlidingWindowMLASpec(
                block_size=16,
                num_kv_heads=1,
                head_size=576,
                dtype=torch.float32,
                sliding_window=128,
            ),
            "layer_2": SlidingWindowMLASpec(
                block_size=16,
                num_kv_heads=1,
                head_size=1024,
                dtype=torch.float32,
                sliding_window=128,
            ),
        },
    )
    assert (
        get_kv_cache_spec_kind(uniform_swa_mla_spec)
        == KVCacheSpecKind.SLIDING_WINDOW_MLA
    )


def test_get_kv_cache_spec_kind_unknown_for_mixed_uniform_type_specs():
    uniform_mixed_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={
            "layer_1": new_mla_spec(),
            "layer_2": SlidingWindowMLASpec(
                block_size=16,
                num_kv_heads=1,
                head_size=576,
                dtype=torch.float32,
                sliding_window=128,
            ),
        },
    )
    assert get_kv_cache_spec_kind(uniform_mixed_spec) == KVCacheSpecKind.UNKNOWN


def test_get_kv_cache_spec_sliding_window_reads_windowed_specs():
    full_attention_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.float32,
    )
    sliding_window_spec = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.float32,
        sliding_window=128,
    )

    assert get_kv_cache_spec_sliding_window(full_attention_spec) is None
    assert get_kv_cache_spec_sliding_window(sliding_window_spec) == 128


def test_get_kv_cache_spec_sliding_window_unwraps_uniform_type_specs():
    uniform_window_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={
            "layer_1": SlidingWindowSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=64,
                dtype=torch.float32,
                sliding_window=128,
            ),
            "layer_2": SlidingWindowSpec(
                block_size=16,
                num_kv_heads=2,
                head_size=64,
                dtype=torch.float32,
                sliding_window=128,
            ),
        },
    )
    mixed_window_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={
            "layer_1": SlidingWindowSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=64,
                dtype=torch.float32,
                sliding_window=128,
            ),
            "layer_2": SlidingWindowSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=64,
                dtype=torch.float32,
                sliding_window=256,
            ),
        },
    )

    assert get_kv_cache_spec_sliding_window(uniform_window_spec) == 128
    assert get_kv_cache_spec_sliding_window(mixed_window_spec) is None


def test_merge_mla_spec():
    kv_cache_specs = [
        new_mla_spec(),
        new_mla_spec(),
    ]
    mla_spec = kv_cache_specs[0].merge(kv_cache_specs)
    assert mla_spec == new_mla_spec()

    kv_cache_specs = [
        new_mla_spec(cache_dtype_str="fp8_ds_mla"),
        new_mla_spec(cache_dtype_str="fp8_ds_mla"),
    ]
    mla_spec = kv_cache_specs[0].merge(kv_cache_specs)
    assert mla_spec == new_mla_spec(cache_dtype_str="fp8_ds_mla")

    kv_cache_specs = [
        new_mla_spec(cache_dtype_str="fp8_ds_mla"),
        new_mla_spec(cache_dtype_str=None),
    ]
    with pytest.raises(AssertionError):
        kv_cache_specs[0].merge(kv_cache_specs)

    kv_cache_specs = [
        new_kv_cache_spec(),
        new_mla_spec(),
    ]
    with pytest.raises(AssertionError):
        kv_cache_specs[0].merge(kv_cache_specs)

    kv_cache_specs = [
        new_mla_spec(cache_dtype_str="fp8_ds_mla"),
        new_kv_cache_spec(),
    ]
    with pytest.raises(AssertionError):
        kv_cache_specs[0].merge(kv_cache_specs)


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_request_block_hasher_with_prompt_embeds(hash_fn: Callable[[Any], bytes]):
    block_size = 3
    num_tokens = 2 * block_size
    prompt_token_ids = [_ for _ in range(num_tokens)]
    hidden_size = 5
    prompt_embeds = torch.randn((num_tokens, hidden_size))

    request = make_request(
        request_id="0",
        prompt_token_ids=prompt_token_ids,
        block_size=block_size,
        hash_fn=hash_fn,
        prompt_embeds=prompt_embeds,
    )

    block_hashes = request.block_hashes
    assert len(block_hashes) == 2

    block1_embeds_hash = hashlib.sha256(
        tensor_data(prompt_embeds[:block_size])
    ).digest()
    expected_hash1 = hash_fn(
        (
            kv_cache_utils.NONE_HASH,
            tuple(prompt_token_ids[:block_size]),
            (block1_embeds_hash,),
        )
    )
    assert block_hashes[0] == expected_hash1

    block2_embeds_hash = hashlib.sha256(
        tensor_data(prompt_embeds[block_size:num_tokens])
    ).digest()
    expected_hash2 = hash_fn(
        (
            block_hashes[0],
            tuple(prompt_token_ids[block_size:num_tokens]),
            (block2_embeds_hash,),
        )
    )
    assert block_hashes[1] == expected_hash2


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_request_with_prompt_embeds_and_mm_inputs(hash_fn: Callable[[Any], bytes]):
    block_size = 3
    num_tokens = 2 * block_size
    prompt_token_ids = [_ for _ in range(num_tokens)]
    hidden_size = 5
    prompt_embeds = torch.randn((num_tokens, hidden_size))

    request = make_request(
        request_id="0",
        prompt_token_ids=prompt_token_ids,
        block_size=block_size,
        hash_fn=hash_fn,
        mm_positions=[
            PlaceholderRange(offset=0, length=3),
            PlaceholderRange(offset=3, length=3),
        ],
        mm_hashes=["hash1", "hash2"],
        prompt_embeds=prompt_embeds,
    )

    block_hashes = request.block_hashes
    assert len(block_hashes) == 2

    block1_embeds_hash = hashlib.sha256(
        tensor_data(prompt_embeds[:block_size])
    ).digest()
    expected_hash1 = hash_fn(
        (
            kv_cache_utils.NONE_HASH,
            tuple(prompt_token_ids[:block_size]),
            (("hash1", 0), block1_embeds_hash),
        )
    )
    assert block_hashes[0] == expected_hash1

    block2_embeds_hash = hashlib.sha256(
        tensor_data(prompt_embeds[block_size:num_tokens])
    ).digest()
    expected_hash2 = hash_fn(
        (
            block_hashes[0],
            tuple(prompt_token_ids[block_size:num_tokens]),
            (("hash2", 0), block2_embeds_hash),
        )
    )
    assert block_hashes[1] == expected_hash2


def test_auto_fit_max_model_len():
    """Test that max_model_len=-1 auto-fits to available GPU memory."""
    # Create config with original_max_model_len=-1 to trigger auto-fit
    model_config = ModelConfig(max_model_len=1024)
    # Simulate the user passing -1 by setting original_max_model_len
    model_config.original_max_model_len = -1
    vllm_config = VllmConfig(model_config=model_config)

    mem_per_block_per_layer = 16 * 2 * 64 * 4 * 2  # 16KB per block per layer
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
    }

    # With enough memory, max_model_len stays at the derived max
    large_available_memory = mem_per_block_per_layer * 2 * 1024  # plenty of memory
    _kv_cache_configs = get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [large_available_memory]
    )
    assert vllm_config.model_config.max_model_len == 1024

    # Reset for next test
    model_config = ModelConfig(max_model_len=1024)
    model_config.original_max_model_len = -1
    vllm_config = VllmConfig(model_config=model_config)

    # With limited memory, max_model_len should be reduced
    # Need memory for at least max_model_len tokens
    # 32 blocks worth of memory for 2 layers = can fit 32*16=512 tokens
    limited_memory = mem_per_block_per_layer * 2 * 32
    _kv_cache_configs = get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [limited_memory]
    )
    # Should be reduced to fit in memory
    assert vllm_config.model_config.max_model_len < 1024
    assert vllm_config.model_config.max_model_len > 0


def test_auto_fit_max_model_len_with_hybrid():
    """Test that auto-fit works with hybrid KV cache specs."""
    # Create config with original_max_model_len=-1 to trigger auto-fit
    model_config = ModelConfig(max_model_len=8192)
    # Simulate the user passing -1 by setting original_max_model_len
    model_config.original_max_model_len = -1
    vllm_config = VllmConfig(model_config=model_config)

    mem_per_block_per_layer = 16 * 2 * 64 * 4 * 2  # 16KB per block per layer
    gamma = 2
    kv_cache_specs = {
        "layer_1": new_mamba_spec(num_speculative_blocks=gamma),
        "layer_2": new_kv_cache_spec(),
    }

    available_memory = mem_per_block_per_layer * (1024 // 16 + 1 + gamma)
    _kv_cache_configs = get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [available_memory]
    )
    assert vllm_config.model_config.max_model_len == 1024


def test_auto_fit_max_model_len_not_triggered():
    """Test that auto-fit is not triggered when original_max_model_len is not -1."""
    model_config = ModelConfig(max_model_len=16)
    # original_max_model_len should be None by default, not -1
    vllm_config = VllmConfig(model_config=model_config)

    mem_per_block_per_layer = 16 * 2 * 64 * 4 * 2
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
    }

    # This should work normally without auto-fit
    _kv_cache_configs = get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [mem_per_block_per_layer * 2 * 32]
    )
    assert vllm_config.model_config.max_model_len == 16


def test_auto_fit_max_model_len_respects_num_gpu_blocks_override():
    """Auto-fit must size max_model_len against the override-clamped pool, not
    the raw `available_memory`. Without this, auto-fit could pick a
    max_model_len that no longer fits once `num_gpu_blocks_override` is applied.
    """
    model_config = ModelConfig(max_model_len=16384)
    model_config.original_max_model_len = -1  # request auto-fit
    vllm_config = VllmConfig(model_config=model_config)
    # Cap the cache to 32 blocks regardless of available memory.
    vllm_config.cache_config.num_gpu_blocks_override = 32

    mem_per_block_per_layer = 16 * 2 * 64 * 4 * 2
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),  # block_size=16
        "layer_2": new_kv_cache_spec(),
    }
    # Plenty of raw memory (1024 blocks per layer would fit max_model_len=16384).
    large_available_memory = mem_per_block_per_layer * 2 * 1024

    get_kv_cache_configs(vllm_config, [kv_cache_specs], [large_available_memory])

    # 32 blocks * block_size 16 = 512 token slots, so max_model_len must
    # auto-fit at or below that.
    assert 0 < vllm_config.model_config.max_model_len <= 32 * 16


def test_check_enough_kv_cache_memory_respects_num_gpu_blocks_override():
    """Admission check must use the override-clamped pool size, not raw
    `available_memory`. Without this, startup could accept a max_model_len
    that does not actually fit in `num_gpu_blocks_override` blocks.
    """
    model_config = ModelConfig(max_model_len=16384)
    vllm_config = VllmConfig(model_config=model_config)
    # 32 blocks is far too small for max_model_len=16384 (would need 1024).
    vllm_config.cache_config.num_gpu_blocks_override = 32

    mem_per_block_per_layer = 16 * 2 * 64 * 4 * 2
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
    }
    # Plenty of raw memory: a bytes-only check against this would pass.
    large_available_memory = mem_per_block_per_layer * 2 * 1024

    with pytest.raises(ValueError, match="max seq len"):
        get_kv_cache_configs(vllm_config, [kv_cache_specs], [large_available_memory])


def test_unify_hybrid_kv_cache_specs():
    # 1. has_full_attention and has_sliding_window
    before_spec_1 = new_kv_cache_spec()
    before_spec_2 = new_sliding_window_spec(
        page_size_padded=32 * 1024, sliding_window=1024
    )
    kv_cache_spec = {
        "layer_1": before_spec_1,
        "layer_2": before_spec_2,
    }
    kv_cache_utils.unify_hybrid_kv_cache_specs(kv_cache_spec)
    expected_spec_1 = new_kv_cache_spec()
    expected_spec_2 = new_kv_cache_spec(page_size_padded=32 * 1024, sliding_window=1024)
    assert kv_cache_spec["layer_1"] == expected_spec_1
    assert kv_cache_spec["layer_2"] == expected_spec_2

    # 2. has_full_attention and has_chunked_local_attention
    before_spec_1 = new_kv_cache_spec()
    before_spec_2 = new_chunked_local_attention_spec(
        page_size_padded=32 * 1024, attention_chunk_size=512
    )
    kv_cache_spec = {
        "layer_1": before_spec_1,
        "layer_2": before_spec_2,
    }
    kv_cache_utils.unify_hybrid_kv_cache_specs(kv_cache_spec)
    expected_spec_1 = new_kv_cache_spec()
    expected_spec_2 = new_kv_cache_spec(
        page_size_padded=32 * 1024, attention_chunk_size=512
    )

    assert kv_cache_spec["layer_1"] == expected_spec_1
    assert kv_cache_spec["layer_2"] == expected_spec_2

    # 3. has_full_attention, has_sliding_window and has_chunked_local_attention
    before_spec_1 = new_kv_cache_spec()
    before_spec_2 = new_sliding_window_spec(
        page_size_padded=32 * 1024, sliding_window=1024
    )
    before_spec_3 = new_chunked_local_attention_spec(
        page_size_padded=32 * 1024, attention_chunk_size=512
    )
    kv_cache_spec = {
        "layer_1": before_spec_1,
        "layer_2": before_spec_2,
        "layer_3": before_spec_3,
    }
    kv_cache_utils.unify_hybrid_kv_cache_specs(kv_cache_spec)
    expected_spec_1 = new_kv_cache_spec()
    expected_spec_2 = new_kv_cache_spec(page_size_padded=32 * 1024, sliding_window=1024)
    expected_spec_3 = new_kv_cache_spec(
        page_size_padded=32 * 1024, attention_chunk_size=512
    )
    assert kv_cache_spec["layer_1"] == expected_spec_1
    assert kv_cache_spec["layer_2"] == expected_spec_2
    assert kv_cache_spec["layer_3"] == expected_spec_3

    # 4. No FullAttentionSpec, should not convert
    kv_cache_spec = {
        "layer_1": new_sliding_window_spec(sliding_window=1024),
        "layer_2": new_chunked_local_attention_spec(attention_chunk_size=512),
    }

    with pytest.raises(ValueError):
        kv_cache_utils.unify_hybrid_kv_cache_specs(kv_cache_spec)


def test_hma_not_disabled_when_kv_events_enabled():
    """
    Test enabling KV events must not force disable_hybrid_kv_cache_manager to True.

    This test guards against that regression by verifying that a VllmConfig
    with kv_events_config set still resolves disable_hybrid_kv_cache_manager
    to False (i.e. HMA remains enabled) when no other condition requires it
    to be disabled.
    """
    model_config = ModelConfig(max_model_len=16)
    kv_events_config = KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="null",
    )

    # Leave disable_hybrid_kv_cache_manager as None (the default) so that
    # VllmConfig.__post_init__ resolves it automatically.
    vllm_config = VllmConfig(
        model_config=model_config,
        kv_events_config=kv_events_config,
    )

    assert vllm_config.scheduler_config.disable_hybrid_kv_cache_manager is False, (
        "kv_events_config must not force-disable the hybrid KV cache manager."
    )


def test_kvcacheconfig_per_group_num_blocks_defaults():
    ref_kv_cache_spec = new_kv_cache_spec()

    # (a) Built WITHOUT the new field -> per_group_num_blocks is None.
    config = KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[
            KVCacheTensor(
                size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer1"]
            ),
            KVCacheTensor(
                size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer2"]
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer1"], ref_kv_cache_spec),
            KVCacheGroupSpec(["layer2"], ref_kv_cache_spec),
        ],
    )
    assert config.per_group_num_blocks is None

    # (b) Accessor falls back to num_blocks when the field is unset.
    assert config.num_blocks_for_group(0) == config.num_blocks
    assert config.num_blocks_for_group(1) == config.num_blocks

    # (b) Accessor returns per_group_num_blocks[i] when set.
    config_with_per_group = KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[
            KVCacheTensor(
                size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer1"]
            ),
            KVCacheTensor(
                size=ref_kv_cache_spec.page_size_bytes * 10, shared_by=["layer2"]
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer1"], ref_kv_cache_spec),
            KVCacheGroupSpec(["layer2"], ref_kv_cache_spec),
        ],
        per_group_num_blocks=[3, 7],
    )
    assert config_with_per_group.per_group_num_blocks == [3, 7]
    assert config_with_per_group.num_blocks_for_group(0) == 3
    assert config_with_per_group.num_blocks_for_group(1) == 7
