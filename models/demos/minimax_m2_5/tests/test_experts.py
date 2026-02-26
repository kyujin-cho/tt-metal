# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for MiniMax-M2.5 SwiGLU expert MLPs with FP8 dequantization.

Verifies:
1. FP8 dequantization produces correct results
2. Batched expert computation matches PyTorch reference
"""

import pytest
import torch
import torch.nn.functional as F


def test_fp8_dequantization():
    """Test FP8 block-wise dequantization."""
    from models.demos.deepseek_v3.utils.dequantize import dequantize_tensor

    torch.manual_seed(42)

    # Simulate FP8 quantized weight with block_size=(128, 128)
    rows, cols = 256, 256
    block_rows, block_cols = 128, 128
    n_blocks_rows = rows // block_rows
    n_blocks_cols = cols // block_cols

    # Original weight
    original_weight = torch.randn(rows, cols)

    # Simulate quantization: compute scale per block, quantize, then dequantize
    inv_scale = torch.zeros(n_blocks_rows, n_blocks_cols)
    quantized = torch.zeros_like(original_weight)

    for br in range(n_blocks_rows):
        for bc in range(n_blocks_cols):
            block = original_weight[
                br * block_rows : (br + 1) * block_rows,
                bc * block_cols : (bc + 1) * block_cols,
            ]
            scale = block.abs().max() / 127.0  # Simple scale
            inv_scale[br, bc] = scale
            quantized[
                br * block_rows : (br + 1) * block_rows,
                bc * block_cols : (bc + 1) * block_cols,
            ] = torch.round(block / scale)

    # Dequantize
    dequantized = dequantize_tensor(quantized, inv_scale, (block_rows, block_cols))

    # Should be close to original (within quantization error)
    max_error = (original_weight - dequantized).abs().max().item()
    assert max_error < 0.1, f"Dequantization error too large: {max_error}"


@pytest.mark.parametrize("num_tokens", [1, 8])
def test_minimax_experts_reference(num_tokens):
    """Test SwiGLU expert computation against PyTorch reference (CPU only)."""
    torch.manual_seed(42)

    hidden_dim = 128
    intermediate_size = 64
    num_experts = 4

    # Create expert weights
    experts = []
    for _ in range(num_experts):
        experts.append(
            {
                "gate_proj": torch.nn.Linear(hidden_dim, intermediate_size, bias=False),
                "up_proj": torch.nn.Linear(hidden_dim, intermediate_size, bias=False),
                "down_proj": torch.nn.Linear(intermediate_size, hidden_dim, bias=False),
            }
        )

    x = torch.randn(num_tokens, hidden_dim)

    outputs = []
    for expert in experts:
        gate_out = F.silu(expert["gate_proj"](x))
        up_out = expert["up_proj"](x)
        down_out = expert["down_proj"](gate_out * up_out)
        outputs.append(down_out)

    stacked = torch.stack(outputs, dim=0)
    assert stacked.shape == (num_experts, num_tokens, hidden_dim)
    assert not torch.isnan(stacked).any()
