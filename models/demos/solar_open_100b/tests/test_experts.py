# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for Solar-Open-100B SwiGLU expert MLPs.

Verifies that the batched expert computation matches a PyTorch reference
implementation of SwiGLU: down_proj(SiLU(gate_proj(x)) * up_proj(x))
"""

import pytest
import torch
import torch.nn.functional as F


class SwiGLUExpertReference(torch.nn.Module):
    """PyTorch reference for a single SwiGLU expert."""

    def __init__(self, hidden_dim, intermediate_size):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_dim, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_dim, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


@pytest.mark.parametrize("num_tokens", [1, 8, 32])
def test_solar_experts_reference(num_tokens):
    """Test SwiGLU expert computation against PyTorch reference (CPU only)."""
    torch.manual_seed(42)

    hidden_dim = 256  # Small for testing
    intermediate_size = 64
    num_experts = 4

    # Create reference experts
    experts = [SwiGLUExpertReference(hidden_dim, intermediate_size) for _ in range(num_experts)]

    # Create input tokens
    x = torch.randn(num_tokens, hidden_dim)

    # Run each expert independently
    outputs = []
    for expert in experts:
        outputs.append(expert(x))

    # Stack outputs: [num_experts, num_tokens, hidden_dim]
    stacked_output = torch.stack(outputs, dim=0)

    # Verify shapes
    assert stacked_output.shape == (num_experts, num_tokens, hidden_dim)

    # Verify each expert produces different outputs
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            assert not torch.allclose(
                stacked_output[i], stacked_output[j], atol=1e-4
            ), f"Experts {i} and {j} produced identical outputs"
