# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the Qwen3.5 MoE GGUF weight-name mapping added to
gguf_loader._get_gguf_weights_map().

These tests call GGUFModelLoader._build_qwen35moe_name_map() directly so that
any change to production mapping logic immediately breaks the test.  The
staticmethod has no vllm C-extension dependency, so tests run without a GPU.
"""

import pytest

from vllm.model_executor.model_loader.gguf_loader import GGUFModelLoader


@pytest.mark.parametrize(
    "model_type,num_layers",
    [
        ("qwen3_5_moe_text", 4),
        ("qwen3_5_moe", 4),
    ],
)
def test_expert_weight_mapping(model_type, num_layers):
    (
        name_map,
        _,
    ) = GGUFModelLoader._build_qwen35moe_name_map(num_layers)

    for idx in range(num_layers):
        assert name_map[f"blk.{idx}.ffn_down_exps.weight"] == (
            f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
        )
        assert name_map[f"blk.{idx}.ffn_gate_exps.weight"] == (
            f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
        )
        assert name_map[f"blk.{idx}.ffn_up_exps.weight"] == (
            f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
        )


def test_gdn_tensor_mapping():
    name_map, _ = GGUFModelLoader._build_qwen35moe_name_map(4)

    for idx in range(4):
        assert name_map[f"blk.{idx}.ssm_dt.bias"] == (
            f"model.layers.{idx}.linear_attn.dt_bias"
        ), "ssm_dt.bias must map to dt_bias (no .weight/.bias suffix)"
        assert name_map[f"blk.{idx}.ssm_a"] == (
            f"model.layers.{idx}.linear_attn.A_log"
        ), "ssm_a must map to A_log (bare name, no trailing dot)"


def test_sideload_regex_matches_all_expert_indices():
    _, sideloads = GGUFModelLoader._build_qwen35moe_name_map(4)

    assert len(sideloads) == 4

    pattern = sideloads[0]
    for proj in ("gate", "up", "down"):
        assert pattern.fullmatch(f"model.layers.0.mlp.experts.0.{proj}_proj.weight"), (
            f"regex missed expert 0 {proj}"
        )
        assert pattern.fullmatch(
            f"model.layers.0.mlp.experts.255.{proj}_proj.weight"
        ), f"regex missed expert 255 {proj}"

    assert not pattern.fullmatch("model.layers.1.mlp.experts.0.gate_proj.weight")


def test_lm_prefix_is_text_only():
    """All HF names must start with 'model.layers.', not 'model.language_model.'"""
    name_map, _ = GGUFModelLoader._build_qwen35moe_name_map(2)
    for hf_name in name_map.values():
        assert hf_name.startswith("model.layers."), (
            f"Unexpected prefix (multimodal path leaked?): {hf_name}"
        )


def test_per_layer_key_count():
    """Each layer contributes exactly 5 manual overrides (3 expert + 2 GDN)."""
    num_layers = 3
    name_map, sideloads = GGUFModelLoader._build_qwen35moe_name_map(num_layers)
    assert len(name_map) == num_layers * 5
    assert len(sideloads) == num_layers
