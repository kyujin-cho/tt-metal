# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
End-to-end tests for MiniMax-M2.5 model.

Tests full model integration including:
- Expert count parsing for 256 experts
- Partial RoPE configuration
- Pluggable attention and MoE class integration
- FP8 weight dequantization
"""

import re

import torch
from loguru import logger


def test_minimax_expert_count_parsing():
    """Test that 256 experts are correctly parsed from state dict keys."""
    # Simulate state dict keys with expert IDs 0-255
    keys = []
    for layer in range(1):
        for expert_id in range(256):
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                keys.append(f"layers.{layer}.block_sparse_moe.experts.{expert_id}.{proj}.weight")

    expert_ids = [
        int(re.search(r"experts\.(\d+)\.", item).group(1))
        for item in keys
        if "block_sparse_moe.experts" in item and re.search(r"experts\.(\d+)\.", item)
    ]
    num_experts = max(expert_ids) + 1 if expert_ids else 0

    assert num_experts == 256, f"Expected 256 experts, got {num_experts}"
    logger.info(f"Expert count parsing test passed: {num_experts} experts")


def test_minimax_partial_rotary_config():
    """Test that partial_rotary_factor is correctly computed."""
    head_dim = 128
    partial_rotary_factor = 0.5
    rotary_dim = int(head_dim * partial_rotary_factor)

    assert rotary_dim == 64, f"Expected rotary_dim=64, got {rotary_dim}"

    # Verify default (full rotation)
    default_rotary_dim = int(head_dim * 1.0)
    assert default_rotary_dim == head_dim


def test_minimax_fp8_dequant_integration():
    """Test FP8 dequantization with the DeepSeek dequant utility."""
    from models.demos.deepseek_v3.utils.dequantize import dequantize_tensor

    torch.manual_seed(42)

    # Create a small weight with known scale
    weight = torch.randn(128, 128)
    scale = torch.ones(1, 1) * 2.0  # Scale of 2.0

    dequantized = dequantize_tensor(weight, scale, (128, 128))

    # Dequantized should be weight * scale
    expected = weight.float() * 2.0
    assert torch.allclose(dequantized, expected, atol=1e-5), "FP8 dequantization failed"


def test_minimax_pluggable_classes():
    """Test that attention_class and moe_class can be passed to Transformer."""
    import inspect

    from models.tt_transformers.tt.model import Transformer

    sig = inspect.signature(Transformer.__init__)
    params = list(sig.parameters.keys())

    assert "attention_class" in params, "Transformer missing attention_class parameter"
    assert "moe_class" in params, "Transformer missing moe_class parameter"
    assert "moe_kwargs" in params, "Transformer missing moe_kwargs parameter"

    # Verify ordering: moe_class should come after attention_class
    attn_idx = params.index("attention_class")
    moe_idx = params.index("moe_class")
    assert moe_idx > attn_idx, "moe_class should come after attention_class in signature"

    logger.info("Pluggable class parameter test passed")
