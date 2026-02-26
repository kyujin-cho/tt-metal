# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
MoE Layer for Solar-Open-100B with all-to-all dispatch/combine.

Orchestrates:
1. Gate → top-8 expert indices/weights (sigmoid+bias routing)
2. All-to-all dispatch → send tokens to expert-owning devices
3. Expert computation (batched SwiGLU across 16 experts/device)
4. All-to-all combine → gather outputs back to originating devices
5. Weight and sum routed outputs
6. Add shared expert output
7. Return combined output

Reference: models/demos/deepseek_v3/tt/moe.py (proven all-to-all pattern)
"""

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.solar_open_100b.tt.experts import SolarExperts
from models.demos.solar_open_100b.tt.moe_gate import SolarMoEGate
from models.demos.solar_open_100b.tt.shared_expert import SolarSharedExpert


class SolarMoELayer(LightweightModule):
    """
    Full MoE layer for Solar-Open-100B.

    Uses all-to-all dispatch/combine for distributing tokens across devices
    to their assigned experts, then combines results with the shared expert output.

    Interface matches decoder.py expectations: forward(hidden_states, mode).
    """

    def __init__(self, mesh_device, state_dict, args, layer_num, dtype, tt_ccl=None, **kwargs):
        super().__init__()

        self.mesh_device = mesh_device
        self.args = args
        self.dtype = dtype
        self.num_devices = args.num_devices
        self.num_experts = args.num_experts
        self.num_experts_per_device = self.num_experts // self.num_devices
        self.top_k = args.num_experts_per_tok
        self.hidden_dim = args.dim

        # Determine cluster_axis from mesh shape for all-to-all communication
        mesh_shape = mesh_device.shape
        if mesh_shape[0] == 1 and mesh_shape[1] > 1:
            # (1, N) mesh: dispatch along column axis
            self.cluster_axis = 1
            self.num_dispatch_devices = mesh_shape[1]
        elif mesh_shape[1] == 1 and mesh_shape[0] > 1:
            # (N, 1) mesh: dispatch along row axis
            self.cluster_axis = 0
            self.num_dispatch_devices = mesh_shape[0]
        else:
            # 2D mesh: dispatch along row axis by default
            self.cluster_axis = 0
            self.num_dispatch_devices = mesh_shape[0]

        logger.info(
            f"SolarMoELayer: mesh_shape={tuple(mesh_shape)}, cluster_axis={self.cluster_axis}, "
            f"num_dispatch_devices={self.num_dispatch_devices}, "
            f"num_experts={self.num_experts}, experts_per_device={self.num_experts_per_device}"
        )

        # Gate (sigmoid+bias top-8 router)
        self.gate = SolarMoEGate(
            mesh_device=mesh_device,
            args=args,
            state_dict=state_dict,
            layer_num=layer_num,
            dtype=dtype,
        )

        # Routed experts (128 SwiGLU MLPs distributed across devices)
        self.experts = SolarExperts(
            mesh_device=mesh_device,
            args=args,
            state_dict=state_dict,
            layer_num=layer_num,
            dtype=dtype,
        )

        # Shared expert (1 MLP applied to all tokens)
        self.shared_expert = SolarSharedExpert(
            mesh_device=mesh_device,
            args=args,
            state_dict=state_dict,
            layer_num=layer_num,
            dtype=dtype,
        )

        # Expert mapping tensor: maps expert IDs to devices for all-to-all
        # Shape [1, 1, num_experts, num_devices] - one-hot encoding
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
        Full MoE forward pass.

        Args:
            hidden_states: Input tensor [1, 1, batch_per_device, hidden_dim]
            mode: Mode.DECODE or Mode.PREFILL

        Returns:
            Combined output [1, 1, batch_per_device, hidden_dim]
        """
        # 1. Compute shared expert output (applied to all tokens unconditionally)
        shared_output = self.shared_expert.forward(hidden_states)

        # 2. Gate → top-8 expert selection
        expert_indices, expert_weights = self.gate.forward(hidden_states)

        # 3. Prepare for all-to-all dispatch
        # All-to-all expects: input [B, 1, S, H], indices [B, 1, S, K] in ROW_MAJOR
        batch_per_device = hidden_states.shape[-2]
        seq_len = 1  # all-to-all uses DP format: batch dim holds tokens

        x_rm = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        x_rm = ttnn.reshape(x_rm, shape=(batch_per_device, 1, seq_len, self.hidden_dim))

        indices_rm = ttnn.to_layout(expert_indices, ttnn.ROW_MAJOR_LAYOUT)
        indices_rm = ttnn.reshape(indices_rm, shape=(batch_per_device, 1, seq_len, self.top_k))
        # Expert indices must be uint16 for all-to-all dispatch
        if indices_rm.dtype != ttnn.uint16:
            indices_rm = ttnn.typecast(indices_rm, ttnn.uint16)

        # 4. All-to-all dispatch: route tokens to expert-hosting devices
        dispatch_output, dispatch_metadata = ttnn.all_to_all_dispatch(
            x_rm,
            indices_rm,
            self.expert_mapping_tensors,
            cluster_axis=self.cluster_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x_rm)
        ttnn.deallocate(indices_rm)

        # 5. Expert computation
        # Dispatch output: [1, B*D_dispatch, S, H] per device (with output_concat_dim=1 default)
        total_batch = batch_per_device * self.num_dispatch_devices

        # Reshape to [1, 1, total_tokens, H] then repeat for all local experts
        dispatch_reshaped = ttnn.reshape(dispatch_output, shape=(1, 1, total_batch * seq_len, self.hidden_dim))
        dispatch_reshaped = ttnn.repeat(dispatch_reshaped, ttnn.Shape((1, self.num_experts_per_device, 1, 1)))
        dispatch_reshaped = ttnn.to_layout(dispatch_reshaped, ttnn.TILE_LAYOUT)
        ttnn.deallocate(dispatch_output)

        # Run batched expert MLPs: [1, experts_per_device, total_tokens, H]
        experts_output = self.experts.forward(dispatch_reshaped, None)
        ttnn.deallocate(dispatch_reshaped)

        # 6. All-to-all combine: gather expert outputs back to originating devices
        # Reshape expert output: [experts_per_device, total_batch, S, H]
        experts_output = ttnn.to_layout(experts_output, ttnn.ROW_MAJOR_LAYOUT)
        experts_output = ttnn.reshape(
            experts_output,
            shape=(self.num_experts_per_device, total_batch, seq_len, self.hidden_dim),
        )

        # Reshape metadata: [1, total_batch, S, K]
        dispatch_metadata = ttnn.reshape(dispatch_metadata, shape=(1, total_batch, seq_len, self.top_k))

        combined_output = ttnn.all_to_all_combine(
            experts_output,
            dispatch_metadata,
            self.expert_mapping_tensors,
            cluster_axis=self.cluster_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(experts_output)
        ttnn.deallocate(dispatch_metadata)

        # 7. Weight and sum routed outputs
        # Combine output: [K, batch_per_device, S, H] per device
        combined_reshaped = ttnn.reshape(
            combined_output,
            shape=(self.top_k, 1, batch_per_device * seq_len, self.hidden_dim),
        )
        combined_reshaped = ttnn.to_layout(combined_reshaped, ttnn.TILE_LAYOUT)
        ttnn.deallocate(combined_output)

        # Expand weights: [1, 1, tokens, K] -> [H, 1, tokens, K] -> [K, 1, tokens, H]
        weights_rm = ttnn.to_layout(expert_weights, ttnn.ROW_MAJOR_LAYOUT)
        weights_rm = ttnn.repeat(weights_rm, ttnn.Shape((self.hidden_dim, 1, 1, 1)))
        weights_rm = ttnn.permute(weights_rm, (3, 1, 2, 0))
        weights_tiled = ttnn.to_layout(weights_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(weights_rm)
        ttnn.deallocate(expert_weights)

        # Element-wise multiply: [K, 1, tokens, H] * [K, 1, tokens, H]
        weighted_output = ttnn.mul(combined_reshaped, weights_tiled, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(combined_reshaped)
        ttnn.deallocate(weights_tiled)

        # Sum across top-k dimension: [K, 1, tokens, H] -> [1, 1, tokens, H]
        routed_output = ttnn.sum(weighted_output, dim=0, keepdim=True)
        ttnn.deallocate(weighted_output)

        # Reshape to match input: [1, 1, batch_per_device, hidden_dim]
        routed_output = ttnn.reshape(routed_output, shape=(1, 1, batch_per_device, self.hidden_dim))

        # 8. Add shared expert output (no all-reduce needed for 1D mesh without TP)
        output = ttnn.add(routed_output, shared_output, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(routed_output)
        ttnn.deallocate(shared_output)

        return output
