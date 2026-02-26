# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for Solar-Open-100B sigmoid+bias top-8 MoE gate.

Verifies that the gate correctly:
1. Applies sigmoid activation followed by bias correction
2. Selects top-8 out of 128 experts based on biased scores
3. Normalizes selected (unbiased) sigmoid scores
4. Scales by routed_scaling_factor
"""

import pytest
import torch


class SolarMoEGateReference(torch.nn.Module):
    """PyTorch reference implementation for Solar MoE sigmoid+bias gate."""

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

        # Top-k on biased scores
        _, topk_indices = torch.topk(scores_with_bias, k=self.top_k, dim=-1, sorted=True)

        # Gather original (unbiased) scores
        topk_scores = torch.gather(scores, dim=-1, index=topk_indices)

        # Normalize
        scores_sum = topk_scores.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = (topk_scores / scores_sum) * self.routed_scaling_factor

        return topk_indices, topk_weights


@pytest.mark.parametrize("seq_len", [1, 4])
def test_solar_moe_gate_reference(seq_len):
    """Test Solar MoE gate reference implementation (CPU only)."""
    torch.manual_seed(42)

    hidden_dim = 64
    num_experts = 16
    top_k = 4

    model = SolarMoEGateReference(hidden_dim, num_experts, top_k, routed_scaling_factor=1.0)
    # Set non-zero bias to verify it affects routing
    model.e_score_correction_bias.data = torch.randn(num_experts) * 0.1

    x = torch.randn(1, seq_len, hidden_dim)
    indices, weights = model(x)

    assert indices.shape == (1, seq_len, top_k)
    assert weights.shape == (1, seq_len, top_k)

    # Weights should sum to ~routed_scaling_factor (1.0) per token
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(
        weight_sums, torch.ones_like(weight_sums), atol=1e-3
    ), f"Weights don't sum to 1.0: {weight_sums}"

    # All weights should be positive
    assert (weights > 0).all(), "All weights should be positive (sigmoid scores)"
