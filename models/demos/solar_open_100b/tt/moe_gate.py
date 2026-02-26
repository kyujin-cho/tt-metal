# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Sigmoid+Bias Top-8 Router for Solar-Open-100B MoE.

Routes tokens to the top-8 out of 128 routed experts using:
1. Linear projection → sigmoid → add e_score_correction_bias
2. Top-8 selection on biased scores
3. Gather original (unbiased) sigmoid scores for selected experts
4. Normalize selected scores and scale by routed_scaling_factor

Reference: models/demos/deepseek_v3/tt/moe_gate.py
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class SolarMoEGate(LightweightModule):
    """
    Sigmoid+bias top-k router for Solar-Open-100B.

    Computes: logits = x @ gate_weight → sigmoid → add bias → topk selection
    Then gathers original sigmoid scores (without bias) for normalization.

    Attributes:
        num_experts: Total number of routed experts (128).
        top_k: Number of experts per token (8).
        gate_weight: Linear projection weight [hidden_dim, num_experts].
        e_score_correction_bias: Bias added to sigmoid scores for selection [1, 1, 1, num_experts].
        routed_scaling_factor: Scaling factor for final weights.
    """

    def __init__(self, mesh_device, args, state_dict, layer_num, dtype):
        super().__init__()

        self.mesh_device = mesh_device
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.hidden_dim = args.dim
        self.routed_scaling_factor = getattr(args, "routed_scaling_factor", 1.0)

        # Gate weight: meta-format key after HF-to-meta conversion
        gate_name = f"layers.{layer_num}.feed_forward.gate.weight"
        gate_weight_torch = state_dict[gate_name].transpose(0, 1).unsqueeze(0).unsqueeze(0)

        # Score correction bias
        bias_name = f"layers.{layer_num}.feed_forward.gate.e_score_correction_bias"
        bias_torch = state_dict[bias_name].float().unsqueeze(0).unsqueeze(0).unsqueeze(0)

        if args.dummy_weights:
            cache_name_gate = None
            cache_name_bias = None
        else:
            cache_name_gate = args.weight_cache_path(dtype) / (gate_name + "_solar_gate")
            cache_name_bias = args.weight_cache_path(dtype) / (bias_name + "_solar_bias")

        self.gate_weight = ttnn.as_tensor(
            gate_weight_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name_gate,
        )

        self.e_score_correction_bias = ttnn.as_tensor(
            bias_torch,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name_bias,
        )

        # Scaling factor tensor
        if self.routed_scaling_factor != 1.0:
            scale_torch = torch.tensor([self.routed_scaling_factor]).repeat(1, self.top_k).unsqueeze(0).unsqueeze(0)
            self.scale_tensor = ttnn.as_tensor(
                scale_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.scale_tensor = None

        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def forward(self, x):
        """
        Route tokens to top-k experts using sigmoid+bias routing.

        Args:
            x: Hidden states [1, 1, seq_len, hidden_dim]

        Returns:
            expert_indices: [1, 1, seq_len, top_k] indices of selected experts
            expert_weights: [1, 1, seq_len, top_k] normalized+scaled weights
        """
        # Linear projection: [1, 1, seq_len, hidden_dim] @ [1, 1, hidden_dim, num_experts]
        logits = ttnn.linear(
            x,
            self.gate_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Sigmoid activation
        scores = ttnn.sigmoid(logits)
        ttnn.deallocate(logits)

        # Add score correction bias (expand bias to match dynamic seq_len)
        bias = ttnn.repeat(self.e_score_correction_bias, ttnn.Shape((1, 1, scores.shape[2], 1)))
        bias = ttnn.to_layout(bias, ttnn.TILE_LAYOUT)
        scores_with_bias = ttnn.add(scores, bias, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)

        # Ensure bfloat16 for topk
        if scores_with_bias.dtype != ttnn.bfloat16:
            scores_with_bias = ttnn.typecast(scores_with_bias, dtype=ttnn.bfloat16)

        # Select top-k experts based on biased scores
        _, expert_indices = ttnn.topk(scores_with_bias, k=self.top_k, dim=-1, sorted=True)
        ttnn.deallocate(scores_with_bias)

        # Gather original (unbiased) sigmoid scores for selected experts
        expert_weights = ttnn.gather(scores, dim=3, index=expert_indices)
        ttnn.deallocate(scores)

        # Normalize: divide by sum of selected scores
        scores_sum = ttnn.sum(expert_weights, dim=3, keepdim=True) + 1e-20
        expert_weights = ttnn.div(expert_weights, scores_sum)
        ttnn.deallocate(scores_sum)

        # Scale by routed_scaling_factor
        if self.scale_tensor is not None:
            scale = ttnn.repeat(self.scale_tensor, ttnn.Shape((1, 1, expert_weights.shape[2], 1)))
            scale = ttnn.to_layout(scale, ttnn.TILE_LAYOUT)
            expert_weights = ttnn.mul(expert_weights, scale, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)

        return expert_indices, expert_weights
