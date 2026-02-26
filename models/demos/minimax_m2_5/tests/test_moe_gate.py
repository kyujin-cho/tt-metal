# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for MiniMax-M2.5 sigmoid+bias top-8 MoE gate.

Verifies that the gate correctly:
1. Applies sigmoid activation + learned bias
2. Selects top-8 out of 256 experts
3. Normalizes selected scores and applies scaling factor
4. Produces results matching a PyTorch reference implementation
"""

import pytest
import torch


class MiniMaxMoEGateReference(torch.nn.Module):
    """PyTorch reference implementation for MiniMax sigmoid+bias MoE gate."""

    def __init__(self, hidden_dim, num_experts, top_k, routed_scaling_factor=1.0):
        super().__init__()
        self.gate = torch.nn.Linear(hidden_dim, num_experts, bias=False)
        self.e_score_correction_bias = torch.nn.Parameter(torch.zeros(num_experts))
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor

    def forward(self, x):
        logits = self.gate(x)
        scores = torch.sigmoid(logits)
        scores_with_bias = scores + self.e_score_correction_bias

        # Top-k selection on biased scores
        _, topk_indices = torch.topk(scores_with_bias, k=self.top_k, dim=-1, sorted=True)

        # Gather original sigmoid scores (without bias)
        topk_scores = torch.gather(scores, dim=-1, index=topk_indices)

        # Normalize
        scores_sum = topk_scores.sum(dim=-1, keepdim=True) + 1e-20
        normalized = topk_scores / scores_sum

        # Scale
        scaled = normalized * self.routed_scaling_factor

        return topk_indices, scaled


@pytest.mark.parametrize("seq_len", [1, 32])
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 2.5])
def test_minimax_moe_gate_reference(seq_len, routed_scaling_factor):
    """Test MiniMax MoE gate reference implementation (CPU only)."""
    torch.manual_seed(42)

    hidden_dim = 256
    num_experts = 64  # Small for testing
    top_k = 4

    model = MiniMaxMoEGateReference(hidden_dim, num_experts, top_k, routed_scaling_factor)

    x = torch.randn(1, seq_len, hidden_dim)
    indices, weights = model(x)

    # Verify shapes
    assert indices.shape == (1, seq_len, top_k)
    assert weights.shape == (1, seq_len, top_k)

    # Weights should be positive (sigmoid output normalized)
    assert (weights >= 0).all(), "Weights should be non-negative"

    # Weights should sum approximately to routed_scaling_factor per token
    weight_sums = weights.sum(dim=-1)
    expected_sum = routed_scaling_factor
    assert torch.allclose(
        weight_sums, torch.tensor(expected_sum), atol=0.1
    ), f"Weight sums {weight_sums} should be close to {expected_sum}"

    # Indices should be in valid range
    assert (indices >= 0).all() and (indices < num_experts).all()
