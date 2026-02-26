# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
SwiGLU Expert MLPs for MiniMax-M2.5 MoE with FP8 weight support.

256 routed experts, each with gate_proj, up_proj, down_proj.
Weights may be stored in float8_e4m3fn format and dequantized during conversion.
Reference: models/demos/deepseek_v3/tt/experts.py
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class MiniMaxExperts(LightweightModule):
    """
    Batched SwiGLU experts for MiniMax-M2.5.

    256 experts with FP8 weight support:
      - gate_proj (w1): [hidden_dim, intermediate_size]
      - up_proj   (w3): [hidden_dim, intermediate_size]
      - down_proj (w2): [intermediate_size, hidden_dim]

    Weights are dequantized from FP8 during weight conversion and stored as BFP8/BF16.
    Sharded across devices: experts_per_device = 256 / num_devices.
    """

    def __init__(self, mesh_device, args, state_dict, layer_num, dtype):
        super().__init__()

        self.mesh_device = mesh_device
        self.args = args
        self.num_experts = args.num_experts
        self.num_devices = args.num_devices
        self.num_experts_per_device = self.num_experts // self.num_devices
        self.hidden_dim = args.dim
        self.intermediate_size = getattr(args, "moe_intermediate_size", args.hidden_dim)

        base_name = lambda expert_num: f"layers.{layer_num}.block_sparse_moe.experts.{expert_num}"

        if args.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: args.weight_cache_path(dtype) / (f"layers.{layer_num}.minimax_moe_experts.{name}")

        def load_expert_weight(proj_name):
            """Stack all expert weights for a given projection, handling FP8 dequant if needed."""
            weights = []
            for expert_id in range(self.num_experts):
                key = f"{base_name(expert_id)}.{proj_name}.weight"
                w = state_dict[key]

                # If weight has an associated scale, dequantize from FP8
                scale_key = f"{base_name(expert_id)}.{proj_name}.weight_scale_inv"
                if scale_key in state_dict:
                    from models.demos.deepseek_v3.utils.dequantize import dequantize_tensor

                    scale = state_dict[scale_key]
                    block_size = getattr(args, "weight_block_size", (128, 128))
                    w = dequantize_tensor(w, scale, block_size)

                weights.append(w.transpose(0, 1).unsqueeze(0))
            return torch.cat(weights, dim=0).unsqueeze(0)

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
            expert_indices: Not used (dispatch already handled by all-to-all)

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
