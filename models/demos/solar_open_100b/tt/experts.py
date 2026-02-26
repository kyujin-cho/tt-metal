# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
SwiGLU Expert MLPs for Solar-Open-100B MoE.

128 routed experts, each with w1 (gate_proj), w3 (up_proj), w2 (down_proj).
Expert dimensions: [hidden_size=4096, moe_intermediate_size=1280].
Weights are stacked per-device for efficient batched computation.

Meta-format keys (after HF-to-meta conversion):
  layers.{i}.feed_forward.experts.{j}.w1.weight  (gate_proj)
  layers.{i}.feed_forward.experts.{j}.w3.weight  (up_proj)
  layers.{i}.feed_forward.experts.{j}.w2.weight  (down_proj)

Reference: models/demos/deepseek_v3/tt/experts.py
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class SolarExperts(LightweightModule):
    """
    Batched SwiGLU experts for Solar-Open-100B.

    Each expert has:
      - w1 (gate_proj): [hidden_size, moe_intermediate_size]
      - w3 (up_proj):   [hidden_size, moe_intermediate_size]
      - w2 (down_proj):  [moe_intermediate_size, hidden_size]

    Weights are stacked as [num_experts, hidden_size, moe_intermediate_size]
    and sharded across devices.
    """

    def __init__(self, mesh_device, args, state_dict, layer_num, dtype):
        super().__init__()

        self.mesh_device = mesh_device
        self.args = args
        self.num_experts = args.num_experts
        self.num_devices = args.num_devices
        self.num_experts_per_device = self.num_experts // self.num_devices
        self.hidden_dim = args.dim
        self.moe_intermediate_size = args.moe_intermediate_size

        base_name = lambda expert_num: f"layers.{layer_num}.feed_forward.experts.{expert_num}"

        if args.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: args.weight_cache_path(dtype) / (f"layers.{layer_num}.solar_moe_experts.{name}")

        # Stack expert weights: [num_experts, dim, intermediate] -> shard across devices
        def load_expert_weight(proj_name):
            """Stack all expert weights for a given projection."""
            weights = []
            for expert_id in range(self.num_experts):
                w = state_dict[f"{base_name(expert_id)}.{proj_name}.weight"]
                weights.append(w.transpose(0, 1).unsqueeze(0))  # [1, dim, intermediate]
            return torch.cat(weights, dim=0).unsqueeze(0)  # [1, num_experts, dim, intermediate]

        def as_expert_tensor(proj_name, name):
            weight = load_expert_weight(proj_name)
            return ttnn.as_tensor(
                weight,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_name(name),
            )

        self.w1 = as_expert_tensor("w1", "w1")  # gate_proj
        self.w2 = as_expert_tensor("w2", "w2")  # down_proj
        self.w3 = as_expert_tensor("w3", "w3")  # up_proj

    def forward(self, x, expert_indices):
        """
        Compute expert outputs for dispatched tokens.

        Args:
            x: Dispatched activations [1, num_experts_per_device, num_tokens, hidden_dim]
            expert_indices: Not used here (dispatch already handled by all-to-all)

        Returns:
            Expert outputs [1, num_experts_per_device, num_tokens, hidden_dim]
        """
        # Gate projection with SiLU activation
        w1_out = ttnn.linear(x, self.w1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Up projection
        w3_out = ttnn.linear(x, self.w3, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # SiLU(gate) * up
        activated = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)

        # Down projection
        output = ttnn.linear(activated, self.w2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(activated)

        return output
