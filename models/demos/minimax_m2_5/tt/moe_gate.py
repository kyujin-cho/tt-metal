# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Sigmoid+Bias Top-8 Router for MiniMax-M2.5 MoE.

Routes tokens to the top-8 out of 256 routed experts using sigmoid activation
with a learned bias, followed by top-k selection and normalization.

Reference: models/demos/deepseek_v3/tt/moe_gate.py (sigmoid routing, without hierarchical grouping)
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class MiniMaxMoEGate(LightweightModule):
    """
    Sigmoid+bias top-k router for MiniMax-M2.5.

    Computes:
    1. logits = x @ gate_weight
    2. scores = sigmoid(logits) + bias
    3. top-8 selection
    4. Normalize selected scores (divide by sum)
    5. Scale by routed_scaling_factor

    Attributes:
        num_experts: Total number of routed experts (256).
        top_k: Number of experts per token (8).
        routed_scaling_factor: Scaling factor for normalized weights.
    """

    def __init__(self, mesh_device, args, state_dict, layer_num, dtype):
        super().__init__()

        self.mesh_device = mesh_device
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.hidden_dim = args.dim
        self.routed_scaling_factor = getattr(args, "routed_scaling_factor", 1.0)

        gate_name = f"layers.{layer_num}.block_sparse_moe.gate.weight"
        bias_name = f"layers.{layer_num}.block_sparse_moe.e_score_correction_bias"

        if args.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: args.weight_cache_path(dtype) / (name + "_minimax_gate")

        # Gate projection weight: [hidden_dim, num_experts]
        gate_weight_torch = state_dict[gate_name].transpose(0, 1).unsqueeze(0).unsqueeze(0)
        self.gate_weight = ttnn.as_tensor(
            gate_weight_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(gate_name),
        )

        # Score correction bias: [1, 1, 1, num_experts]
        if bias_name in state_dict:
            bias_torch = state_dict[bias_name].unsqueeze(0).unsqueeze(0).unsqueeze(0)
        else:
            bias_torch = torch.zeros(1, 1, 1, self.num_experts)

        self.score_bias = ttnn.from_torch(
            bias_torch,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # Scaling factor tensor
        scale_torch = torch.tensor([self.routed_scaling_factor]).repeat(1, self.top_k).unsqueeze(0).unsqueeze(0)
        self.expert_scale = ttnn.from_torch(
            scale_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def forward(self, x):
        """
        Route tokens to top-k experts using sigmoid+bias scoring.

        Args:
            x: Hidden states [1, 1, seq_len, hidden_dim]

        Returns:
            expert_indices: [1, 1, seq_len, top_k] indices of selected experts
            expert_weights: [1, 1, seq_len, top_k] normalized and scaled weights
        """
        # Linear projection
        logits = ttnn.linear(
            x,
            self.gate_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Sigmoid activation
        scores = ttnn.sigmoid(logits)
        ttnn.deallocate(logits)

        # Add bias (expand to match dynamic seq_len)
        bias_expanded = ttnn.repeat(self.score_bias, ttnn.Shape((1, 1, scores.shape[2], 1)))
        bias_expanded = ttnn.to_layout(bias_expanded, ttnn.TILE_LAYOUT)
        scores_with_bias = ttnn.add(scores, bias_expanded, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Ensure bfloat16 for topk
        if scores_with_bias.dtype != ttnn.bfloat16:
            scores_with_bias = ttnn.typecast(scores_with_bias, dtype=ttnn.bfloat16)

        # Top-k selection (on scores+bias for expert selection)
        topk_scores_biased, topk_indices = ttnn.topk(scores_with_bias, k=self.top_k, dim=-1, sorted=True)
        ttnn.deallocate(scores_with_bias)
        ttnn.deallocate(topk_scores_biased)

        # Gather original sigmoid scores (without bias) for the selected experts
        topk_scores = ttnn.gather(scores, dim=3, index=topk_indices)
        ttnn.deallocate(scores)

        # Normalize: divide by sum
        scores_sum = ttnn.sum(topk_scores, dim=3, keepdim=True) + 1e-20
        normalized_scores = ttnn.div(topk_scores, scores_sum)
        ttnn.deallocate(scores_sum)
        ttnn.deallocate(topk_scores)

        # Scale by routed_scaling_factor
        scale_expanded = ttnn.repeat(self.expert_scale, ttnn.Shape((1, 1, normalized_scores.shape[2], 1)))
        scale_expanded = ttnn.to_layout(scale_expanded, ttnn.TILE_LAYOUT)
        expert_weights = ttnn.mul(normalized_scores, scale_expanded, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(normalized_scores)

        return topk_indices, expert_weights
