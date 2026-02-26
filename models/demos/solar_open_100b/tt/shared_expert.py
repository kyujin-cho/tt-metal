# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Shared Expert MLP for Solar-Open-100B.

A single SwiGLU MLP applied to every token unconditionally.
Its output is added to the weighted sum of routed expert outputs.

Shared expert dimensions: [hidden_size=4096, moe_intermediate_size=1280].

Meta-format keys (after HF-to-meta conversion):
  layers.{i}.feed_forward.shared_experts.w1.weight  (gate_proj)
  layers.{i}.feed_forward.shared_experts.w3.weight  (up_proj)
  layers.{i}.feed_forward.shared_experts.w2.weight  (down_proj)

Reference: models/tt_transformers/tt/mlp.py
"""

import ttnn
from models.common.lightweightmodule import LightweightModule


class SolarSharedExpert(LightweightModule):
    """
    Shared expert for Solar-Open-100B.

    A standard SwiGLU MLP: down_proj(SiLU(gate_proj(x)) * up_proj(x))
    Applied to every token regardless of routing decisions.
    """

    def __init__(self, mesh_device, args, state_dict, layer_num, dtype):
        super().__init__()

        self.mesh_device = mesh_device
        self.args = args

        prefix = f"layers.{layer_num}.feed_forward.shared_experts"

        if args.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: args.weight_cache_path(dtype) / (f"layers.{layer_num}.solar_shared_expert.{name}")

        def load_weight(proj_name):
            return state_dict[f"{prefix}.{proj_name}.weight"].transpose(0, 1).unsqueeze(0).unsqueeze(0)

        def as_tensor(proj_name, name):
            weight = load_weight(proj_name)
            return ttnn.as_tensor(
                weight,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_name(name),
            )

        self.w1 = as_tensor("w1", "w1")  # gate_proj
        self.w2 = as_tensor("w2", "w2")  # down_proj
        self.w3 = as_tensor("w3", "w3")  # up_proj

    def forward(self, x):
        """
        Apply shared expert MLP.

        Args:
            x: Input hidden states [1, 1, seq_len, hidden_dim]

        Returns:
            Output [1, 1, seq_len, hidden_dim]
        """
        # Gate projection with SiLU activation
        w1_out = ttnn.linear(x, self.w1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Up projection
        w3_out = ttnn.linear(x, self.w3, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # SiLU(gate) * up
        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)

        # Down projection
        output = ttnn.linear(w2_in, self.w2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(w2_in)

        return output
