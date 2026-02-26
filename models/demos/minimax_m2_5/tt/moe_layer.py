# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
MoE Layer for MiniMax-M2.5 with all-to-all dispatch/combine.

Same pattern as Solar MoE but simpler (no shared expert):
1. Gate → sigmoid+bias top-8
2. All-to-all dispatch → expert computation → combine
3. Weight, sum, return

Reference: models/demos/deepseek_v3/tt/moe.py, models/demos/solar_open_100b/tt/moe_layer.py
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.minimax_m2_5.tt.experts import MiniMaxExperts
from models.demos.minimax_m2_5.tt.moe_gate import MiniMaxMoEGate


class MiniMaxMoELayer(LightweightModule):
    """
    Full MoE layer for MiniMax-M2.5.

    Uses all-to-all dispatch/combine for distributing tokens across devices.
    No shared expert (unlike Solar-Open-100B).

    Interface matches decoder.py expectations: forward(hidden_states, mode).
    """

    def __init__(self, mesh_device, state_dict, args, layer_num, dtype, tt_ccl, **kwargs):
        super().__init__()

        self.mesh_device = mesh_device
        self.args = args
        self.dtype = dtype
        self.tt_ccl = tt_ccl
        self.num_devices = args.num_devices
        self.num_experts = args.num_experts
        self.num_experts_per_device = self.num_experts // self.num_devices
        self.top_k = args.num_experts_per_tok
        self.hidden_dim = args.dim

        # Determine cluster_axis from mesh shape for all-to-all communication
        mesh_shape = mesh_device.shape
        if mesh_shape[0] == 1 and mesh_shape[1] > 1:
            self.cluster_axis = 1
            self.num_dispatch_devices = mesh_shape[1]
        elif mesh_shape[1] == 1 and mesh_shape[0] > 1:
            self.cluster_axis = 0
            self.num_dispatch_devices = mesh_shape[0]
        else:
            # 2D mesh: dispatch along row axis by default
            self.cluster_axis = 0
            self.num_dispatch_devices = mesh_shape[0]

        from loguru import logger

        logger.info(
            f"MiniMaxMoELayer: mesh_shape={tuple(mesh_shape)}, cluster_axis={self.cluster_axis}, "
            f"num_dispatch_devices={self.num_dispatch_devices}, "
            f"num_experts={self.num_experts}, experts_per_device={self.num_experts_per_device}"
        )

        # Gate (sigmoid+bias top-8 router)
        self.gate = MiniMaxMoEGate(
            mesh_device=mesh_device,
            args=args,
            state_dict=state_dict,
            layer_num=layer_num,
            dtype=dtype,
        )

        # Routed experts (256 SwiGLU MLPs with FP8 dequant, distributed across devices)
        self.experts = MiniMaxExperts(
            mesh_device=mesh_device,
            args=args,
            state_dict=state_dict,
            layer_num=layer_num,
            dtype=dtype,
        )

        # Expert mapping tensor: maps expert IDs to devices for all-to-all
        expert_mapping = (
            torch.eye(self.num_devices, dtype=torch.int32)
            .repeat_interleave(self.num_experts_per_device, dim=0)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.expert_mapping_tensors = ttnn.from_torch(
            expert_mapping,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def forward(self, hidden_states, mode):
        """
        Full MoE forward pass (no shared expert).

        Args:
            hidden_states: Input tensor [1, 1, seq_len, hidden_dim]
            mode: "decode" or "prefill"

        Returns:
            Combined output [1, 1, seq_len, hidden_dim]
        """
        # 1. Gate → sigmoid+bias top-8 expert selection
        expert_indices, expert_weights = self.gate.forward(hidden_states)

        # 2. All-to-all dispatch: send tokens to expert-owning devices
        batch_size_per_device = hidden_states.shape[-2]
        seq_len = 1  # all-to-all expects DP format

        x_rm = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        x_rm = ttnn.reshape(x_rm, shape=(batch_size_per_device, 1, seq_len, self.hidden_dim))

        indices_rm = ttnn.to_layout(expert_indices, ttnn.ROW_MAJOR_LAYOUT)
        indices_rm = ttnn.reshape(indices_rm, shape=(batch_size_per_device, 1, seq_len, self.top_k))

        dispatch_output, dispatch_metadata = ttnn.all_to_all_dispatch(
            x_rm,
            indices_rm,
            self.expert_mapping_tensors,
            cluster_axis=self.cluster_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x_rm)
        ttnn.deallocate(indices_rm)

        # 3. Expert computation
        batch_size = batch_size_per_device * self.num_dispatch_devices

        dispatch_reshaped = ttnn.reshape(dispatch_output, shape=(1, 1, batch_size * seq_len, self.hidden_dim))
        dispatch_reshaped = ttnn.repeat(dispatch_reshaped, ttnn.Shape((1, self.num_experts_per_device, 1, 1)))
        dispatch_reshaped = ttnn.to_layout(dispatch_reshaped, ttnn.TILE_LAYOUT)
        ttnn.deallocate(dispatch_output)

        experts_output = self.experts.forward(dispatch_reshaped, None)
        ttnn.deallocate(dispatch_reshaped)

        # 4. All-to-all combine: gather expert outputs back
        experts_output = ttnn.to_layout(experts_output, ttnn.ROW_MAJOR_LAYOUT)
        experts_output = ttnn.reshape(
            experts_output,
            shape=(self.num_experts_per_device, batch_size, seq_len, self.hidden_dim),
        )

        dispatch_metadata = ttnn.reshape(dispatch_metadata, shape=(1, batch_size, seq_len, self.top_k))

        combined_output = ttnn.all_to_all_combine(
            experts_output,
            dispatch_metadata,
            self.expert_mapping_tensors,
            cluster_axis=self.cluster_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(experts_output)
        ttnn.deallocate(dispatch_metadata)

        # 5. Weight and sum routed outputs
        combined_reshaped = ttnn.reshape(
            combined_output,
            shape=(self.top_k, 1, batch_size_per_device * seq_len, self.hidden_dim),
        )
        combined_reshaped = ttnn.to_layout(combined_reshaped, ttnn.TILE_LAYOUT)

        # Repeat weights to match hidden dim for element-wise multiply
        weights_rm = ttnn.to_layout(expert_weights, ttnn.ROW_MAJOR_LAYOUT)
        weights_rm = ttnn.repeat(weights_rm, ttnn.Shape((self.hidden_dim, 1, 1, 1)))
        weights_rm = ttnn.permute(weights_rm, (3, 1, 2, 0))
        weights_tiled = ttnn.to_layout(weights_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(weights_rm)

        weighted_output = ttnn.mul(combined_reshaped, weights_tiled, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(combined_reshaped)
        ttnn.deallocate(weights_tiled)

        # Sum across top-k dimension
        output = ttnn.sum(weighted_output, dim=0, keepdim=True)
        ttnn.deallocate(weighted_output)

        # Reshape back to [1, 1, seq_len, hidden_dim]
        output = ttnn.reshape(output, shape=(1, 1, batch_size_per_device, self.hidden_dim))

        return output
