# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Hardware test for Solar-Open-100B MoE layer on Galaxy.

Tests the Solar MoE gate on hardware against a PyTorch reference:
1. Loads layer 0 weights from /data/models/Solar-Open-100B via ModelArgs
2. Creates TT MoE gate (sigmoid+bias top-8 router)
3. Runs forward pass on mesh device
4. Compares against PyTorch reference

Requires: HF_MODEL=/data/models/Solar-Open-100B MESH_DEVICE=TG (or T3K)
Run: source python_env/bin/activate && HF_MODEL=/data/models/Solar-Open-100B pytest models/demos/solar_open_100b/tests/test_solar_moe_hw.py -v
"""

import os

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.solar_open_100b.tt.moe_gate import SolarMoEGate
from models.tt_transformers.tt.model_config import ModelArgs

CKPT_DIR = os.getenv("HF_MODEL", "/data/models/Solar-Open-100B")


class SolarGateReference(torch.nn.Module):
    """PyTorch reference for sigmoid+bias gate routing."""

    def __init__(self, gate_weight, e_score_correction_bias, top_k, routed_scaling_factor=1.0):
        super().__init__()
        self.gate_weight = gate_weight.float()
        self.e_score_correction_bias = e_score_correction_bias.float()
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor

    def forward(self, x):
        x = x.float()
        logits = F.linear(x, self.gate_weight)
        scores = torch.sigmoid(logits)
        scores_with_bias = scores + self.e_score_correction_bias

        _, topk_indices = torch.topk(scores_with_bias, k=self.top_k, dim=-1, sorted=True)
        topk_scores = torch.gather(scores, dim=-1, index=topk_indices)

        scores_sum = topk_scores.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = (topk_scores / scores_sum) * self.routed_scaling_factor

        return topk_indices, topk_weights


@pytest.mark.skipif(
    not os.path.exists(CKPT_DIR),
    reason=f"Solar checkpoint not found at {CKPT_DIR}",
)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_solar_moe_gate_hw(mesh_device, device_params):
    """Test Solar MoE gate on hardware against reference implementation."""
    torch.manual_seed(42)
    pcc_threshold = 0.98

    mesh_device.disable_and_clear_program_cache()

    # Load model args and state dict
    model_args = ModelArgs(mesh_device, max_batch_size=1, max_seq_len=4096)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    layer_num = 0
    hidden_dim = model_args.dim
    num_experts = model_args.num_experts
    top_k = model_args.num_experts_per_tok
    routed_scaling_factor = model_args.routed_scaling_factor

    logger.info(
        f"Config: hidden_dim={hidden_dim}, num_experts={num_experts}, "
        f"top_k={top_k}, routed_scaling_factor={routed_scaling_factor}"
    )

    # Create reference model
    gate_weight = state_dict[f"layers.{layer_num}.feed_forward.gate.weight"]
    bias = state_dict[f"layers.{layer_num}.feed_forward.gate.e_score_correction_bias"]
    ref_gate = SolarGateReference(gate_weight, bias, top_k, routed_scaling_factor)

    # Create test input
    pt_input = (torch.rand(1, 1, 1, hidden_dim) * 2) - 1  # [1, 1, 1, hidden_dim] for decode

    # Reference output
    with torch.no_grad():
        ref_indices, ref_weights = ref_gate(pt_input)
    logger.info(f"Reference: indices={ref_indices.squeeze()}, weights sum={ref_weights.sum().item():.4f}")

    # TT gate
    dtype = ttnn.bfloat8_b
    tt_gate = SolarMoEGate(
        mesh_device=mesh_device,
        args=model_args,
        state_dict=state_dict,
        layer_num=layer_num,
        dtype=dtype,
    )

    # TT input
    tt_input = ttnn.from_torch(
        pt_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Run TT gate
    logger.info("Running TT MoE gate...")
    tt_indices, tt_weights = tt_gate.forward(tt_input)

    # Convert back to torch
    tt_weights_torch = ttnn.to_torch(tt_weights, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[
        0:1
    ]  # Take first device's output (replicated)

    tt_indices_torch = ttnn.to_torch(tt_indices, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1]

    logger.info(f"TT output: indices={tt_indices_torch.squeeze()}, weights sum={tt_weights_torch.sum().item():.4f}")

    # Compare weights (indices may differ for equal-valued scores)
    passing, pcc_message = comp_pcc(ref_weights.float(), tt_weights_torch.float(), pcc_threshold)
    logger.info(f"PCC: {pcc_message}")
    logger.info(comp_allclose(ref_weights.float(), tt_weights_torch.float()))

    if passing:
        logger.info(f"Solar MoE gate test PASSED! (PCC >= {pcc_threshold})")
    else:
        logger.warning(f"Solar MoE gate test FAILED!")

    assert passing, f"PCC check failed: {pcc_message}"
