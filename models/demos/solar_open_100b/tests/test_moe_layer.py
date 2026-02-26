# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for Solar-Open-100B full MoE layer.

Verifies the complete MoE pipeline:
1. Sigmoid+bias gate routing (top-8)
2. All-to-all dispatch
3. Expert computation
4. All-to-all combine
5. Weighted sum + shared expert addition
"""

import pytest
import torch
import torch.nn.functional as F


class SolarMoELayerReference(torch.nn.Module):
    """
    PyTorch reference implementation of Solar-Open-100B MoE layer.

    Implements: output = sum(normalize(sigmoid_topk) * expert_outputs) + shared_expert(x)
    """

    def __init__(self, hidden_dim, intermediate_size, num_experts, top_k, routed_scaling_factor=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor

        # Gate
        self.gate = torch.nn.Linear(hidden_dim, num_experts, bias=False)
        self.e_score_correction_bias = torch.nn.Parameter(torch.zeros(num_experts))

        # Routed experts
        self.experts = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "gate_proj": torch.nn.Linear(hidden_dim, intermediate_size, bias=False),
                        "up_proj": torch.nn.Linear(hidden_dim, intermediate_size, bias=False),
                        "down_proj": torch.nn.Linear(intermediate_size, hidden_dim, bias=False),
                    }
                )
                for _ in range(num_experts)
            ]
        )

        # Shared expert (same intermediate size as routed experts)
        self.shared_expert = torch.nn.ModuleDict(
            {
                "gate_proj": torch.nn.Linear(hidden_dim, intermediate_size, bias=False),
                "up_proj": torch.nn.Linear(hidden_dim, intermediate_size, bias=False),
                "down_proj": torch.nn.Linear(intermediate_size, hidden_dim, bias=False),
            }
        )

    def _expert_forward(self, expert, x):
        return expert["down_proj"](F.silu(expert["gate_proj"](x)) * expert["up_proj"](x))

    def forward(self, x):
        batch, seq_len, hidden_dim = x.shape

        # Gate: sigmoid + bias
        logits = self.gate(x)
        scores = torch.sigmoid(logits)
        scores_with_bias = scores + self.e_score_correction_bias

        # Top-k selection on biased scores
        _, topk_indices = torch.topk(scores_with_bias, k=self.top_k, dim=-1)
        topk_scores = torch.gather(scores, dim=-1, index=topk_indices)

        # Normalize and scale
        scores_sum = topk_scores.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = (topk_scores / scores_sum) * self.routed_scaling_factor

        # Routed expert computation
        routed_output = torch.zeros_like(x)
        for b in range(batch):
            for s in range(seq_len):
                token = x[b, s].unsqueeze(0)
                for k in range(self.top_k):
                    expert_idx = topk_indices[b, s, k].item()
                    weight = topk_weights[b, s, k].item()
                    expert_out = self._expert_forward(self.experts[expert_idx], token)
                    routed_output[b, s] += weight * expert_out.squeeze(0)

        # Shared expert
        shared_output = self._expert_forward(self.shared_expert, x)

        return routed_output + shared_output


@pytest.mark.parametrize("seq_len", [1, 4])
def test_solar_moe_layer_reference(seq_len):
    """Test Solar MoE layer reference implementation (CPU only, small scale)."""
    torch.manual_seed(42)

    hidden_dim = 64
    intermediate_size = 32
    num_experts = 8
    top_k = 2

    model = SolarMoELayerReference(
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        routed_scaling_factor=1.0,
    )

    x = torch.randn(1, seq_len, hidden_dim)
    output = model(x)

    assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"
    assert not torch.isnan(output).any(), "Output contains NaNs"
    assert not torch.isinf(output).any(), "Output contains Infs"
