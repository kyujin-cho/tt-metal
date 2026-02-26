# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Hardware test for Solar-Open-100B MoE layer components on Galaxy.

Tests:
1. test_solar_full_moe_pipeline_hw: Full MoE pipeline (gate + 128 experts + shared expert)
   - Runs all components on TT hardware
   - Combines expert outputs in Python (no all-to-all fabric required)
   - Validates full pipeline against PyTorch reference
2. test_solar_experts_hw: Batched expert SwiGLU computation
3. test_solar_shared_expert_hw: Shared expert SwiGLU computation
4. test_solar_moe_layer_hw: Full MoE with all-to-all dispatch/combine (requires fabric)

Requires: HF_MODEL=/data/models/Solar-Open-100B
Run: source python_env/bin/activate && HF_MODEL=/data/models/Solar-Open-100B pytest models/demos/solar_open_100b/tests/test_solar_moe_layer_hw.py -v
"""

import os

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tt.model_config import ModelArgs

CKPT_DIR = os.getenv("HF_MODEL", "/data/models/Solar-Open-100B")


class SolarMoEReference(torch.nn.Module):
    """Full PyTorch reference for Solar MoE layer (gate + 128 experts + shared expert)."""

    def __init__(self, state_dict, layer_num, num_experts, top_k, routed_scaling_factor):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor

        # Gate weights
        self.gate_weight = state_dict[f"layers.{layer_num}.feed_forward.gate.weight"].float()
        self.bias = state_dict[f"layers.{layer_num}.feed_forward.gate.e_score_correction_bias"].float()

        # Expert weights (128 routed experts)
        self.experts_w1 = []
        self.experts_w2 = []
        self.experts_w3 = []
        for i in range(self.num_experts):
            prefix = f"layers.{layer_num}.feed_forward.experts.{i}"
            self.experts_w1.append(state_dict[f"{prefix}.w1.weight"].float())
            self.experts_w2.append(state_dict[f"{prefix}.w2.weight"].float())
            self.experts_w3.append(state_dict[f"{prefix}.w3.weight"].float())

        # Shared expert
        shared_prefix = f"layers.{layer_num}.feed_forward.shared_experts"
        self.shared_w1 = state_dict[f"{shared_prefix}.w1.weight"].float()
        self.shared_w2 = state_dict[f"{shared_prefix}.w2.weight"].float()
        self.shared_w3 = state_dict[f"{shared_prefix}.w3.weight"].float()

    def _expert_mlp(self, x, w1, w2, w3):
        """SwiGLU: down(silu(gate(x)) * up(x))"""
        gate = F.silu(F.linear(x, w1))
        up = F.linear(x, w3)
        return F.linear(gate * up, w2)

    def forward(self, x):
        """
        Args:
            x: [1, 1, 1, hidden_dim]
        Returns:
            output: [1, 1, 1, hidden_dim]
        """
        x_float = x.float()

        # Gate: sigmoid + bias + topk
        logits = F.linear(x_float, self.gate_weight)
        scores = torch.sigmoid(logits)
        biased = scores + self.bias
        _, indices = torch.topk(biased, k=self.top_k, dim=-1, sorted=True)
        selected_scores = torch.gather(scores, dim=-1, index=indices)
        normalized = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-20)
        weights = normalized * self.routed_scaling_factor

        # Run selected experts
        routed_output = torch.zeros_like(x_float)
        for k in range(self.top_k):
            expert_id = indices[0, 0, 0, k].item()
            expert_out = self._expert_mlp(
                x_float, self.experts_w1[expert_id], self.experts_w2[expert_id], self.experts_w3[expert_id]
            )
            routed_output = routed_output + weights[0, 0, 0, k] * expert_out

        # Shared expert
        shared_out = self._expert_mlp(x_float, self.shared_w1, self.shared_w2, self.shared_w3)

        return routed_output + shared_out


@pytest.mark.skipif(
    not os.path.exists(CKPT_DIR),
    reason=f"Solar checkpoint not found at {CKPT_DIR}",
)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_solar_full_moe_pipeline_hw(mesh_device, device_params):
    """
    Test full Solar MoE pipeline on hardware: gate + 128 experts + shared expert.

    Runs each component on TT hardware independently, then combines expert outputs
    in Python to validate the complete MoE output matches the PyTorch reference.
    This tests that all TT components produce numerically correct outputs that,
    when combined, produce the correct MoE result.

    Does NOT require fabric (no all-to-all dispatch/combine).
    """
    torch.manual_seed(42)
    pcc_threshold = 0.95

    mesh_device.disable_and_clear_program_cache()

    # Load model args and state dict
    logger.info("Loading model args and state dict...")
    model_args = ModelArgs(mesh_device, max_batch_size=1, max_seq_len=4096)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    layer_num = 0
    hidden_dim = model_args.dim
    num_experts = model_args.num_experts
    top_k = model_args.num_experts_per_tok
    num_devices = model_args.num_devices
    experts_per_device = num_experts // num_devices
    routed_scaling_factor = model_args.routed_scaling_factor

    logger.info(
        f"Config: hidden_dim={hidden_dim}, num_experts={num_experts}, top_k={top_k}, "
        f"experts_per_device={experts_per_device}, routed_scaling_factor={routed_scaling_factor}"
    )

    # PyTorch reference
    logger.info("Creating PyTorch reference model...")
    ref_model = SolarMoEReference(state_dict, layer_num, num_experts, top_k, routed_scaling_factor)

    pt_input = (torch.rand(1, 1, 1, hidden_dim) * 2) - 1

    logger.info("Computing reference output...")
    with torch.no_grad():
        ref_output = ref_model(pt_input)
    logger.info(f"Reference: mean={ref_output.mean().item():.6f}, std={ref_output.std().item():.6f}")

    # ---- TT Components ----
    dtype = ttnn.bfloat8_b

    # 1. Gate
    from models.demos.solar_open_100b.tt.moe_gate import SolarMoEGate

    tt_gate = SolarMoEGate(
        mesh_device=mesh_device, args=model_args, state_dict=state_dict, layer_num=layer_num, dtype=dtype
    )

    # 2. Experts (16 per device, 128 total across 8 devices)
    from models.demos.solar_open_100b.tt.experts import SolarExperts

    tt_experts = SolarExperts(
        mesh_device=mesh_device, args=model_args, state_dict=state_dict, layer_num=layer_num, dtype=dtype
    )

    # 3. Shared expert
    from models.demos.solar_open_100b.tt.shared_expert import SolarSharedExpert

    tt_shared = SolarSharedExpert(
        mesh_device=mesh_device, args=model_args, state_dict=state_dict, layer_num=layer_num, dtype=dtype
    )

    # TT input (replicated to all devices)
    tt_input = ttnn.from_torch(
        pt_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # ---- Run TT Gate ----
    logger.info("Running TT gate...")
    tt_indices, tt_weights = tt_gate.forward(tt_input)

    indices_torch = ttnn.to_torch(tt_indices, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[
        0:1
    ]  # [1, 1, 1, 8] - expert IDs from device 0

    weights_torch = ttnn.to_torch(tt_weights, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[
        0:1
    ]  # [1, 1, 1, 8] - weights from device 0

    logger.info(f"Gate: indices={indices_torch.squeeze().int().tolist()}, weights_sum={weights_torch.sum().item():.4f}")

    # ---- Run TT Shared Expert ----
    logger.info("Running TT shared expert...")
    tt_shared_out = tt_shared.forward(tt_input)
    shared_torch = ttnn.to_torch(tt_shared_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[
        0:1
    ]  # [1, 1, 1, hidden_dim]

    # ---- Run TT Experts (all 128 via batched computation per device) ----
    logger.info("Running TT experts on all devices...")
    # Create expert input: replicate input for each local expert
    # [1, 1, 1, hidden_dim] -> [1, experts_per_device, 1, hidden_dim]
    tt_expert_input = ttnn.repeat(tt_input, ttnn.Shape((1, experts_per_device, 1, 1)))

    tt_expert_out = tt_experts.forward(tt_expert_input, None)

    # Get ALL 128 expert outputs by collecting from all 8 devices
    # Each device has [1, 16, 1, 4096], concatenating on dim=0 gives [8, 16, 1, 4096]
    all_expert_out_torch = ttnn.to_torch(
        tt_expert_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )  # [8, 16, 1, 4096]

    # Reshape to [128, 1, 4096] - device 0 has experts 0-15, device 1 has 16-31, etc.
    all_expert_out = all_expert_out_torch.reshape(num_experts, 1, hidden_dim)
    logger.info(f"All expert outputs: shape={all_expert_out.shape}, mean={all_expert_out.mean().item():.6f}")

    # ---- Combine in Python: select top-k expert outputs, weight, add shared ----
    logger.info("Combining outputs...")
    tt_combined = torch.zeros(1, 1, 1, hidden_dim)
    for k in range(top_k):
        expert_id = indices_torch[0, 0, 0, k].int().item()
        weight = weights_torch[0, 0, 0, k].float().item()
        expert_output = all_expert_out[expert_id : expert_id + 1].unsqueeze(0)  # [1, 1, 1, hidden_dim]
        tt_combined = tt_combined + weight * expert_output
        logger.debug(f"  Expert {expert_id}: weight={weight:.4f}, output_mean={expert_output.mean().item():.6f}")

    # Add shared expert
    tt_combined = tt_combined + shared_torch.float()

    logger.info(f"TT combined: mean={tt_combined.mean().item():.6f}, std={tt_combined.std().item():.6f}")

    # ---- Compare ----
    passing, pcc_message = comp_pcc(ref_output.float(), tt_combined.float(), pcc_threshold)
    logger.info(f"PCC: {pcc_message}")
    logger.info(comp_allclose(ref_output.float(), tt_combined.float()))

    if passing:
        logger.info(f"Solar full MoE pipeline test PASSED! (PCC >= {pcc_threshold})")
    else:
        logger.warning(f"Solar full MoE pipeline test FAILED!")

    assert passing, f"PCC check failed: {pcc_message}"


@pytest.mark.skipif(
    not os.path.exists(CKPT_DIR),
    reason=f"Solar checkpoint not found at {CKPT_DIR}",
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_solar_moe_layer_hw(mesh_device, device_params):
    """
    Test Solar full MoE layer with all-to-all dispatch/combine on hardware.

    Requires working fabric (all-to-all communication between devices).
    Uses (8,4) full TG mesh with FABRIC_1D.
    """
    torch.manual_seed(42)
    pcc_threshold = 0.90

    mesh_device.disable_and_clear_program_cache()

    logger.info("Loading model args and state dict...")
    model_args = ModelArgs(mesh_device, max_batch_size=1, max_seq_len=4096)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    layer_num = 0
    hidden_dim = model_args.dim
    num_experts = model_args.num_experts
    top_k = model_args.num_experts_per_tok
    routed_scaling_factor = model_args.routed_scaling_factor

    logger.info(
        f"Config: hidden_dim={hidden_dim}, num_experts={num_experts}, top_k={top_k}, "
        f"mesh_shape={tuple(mesh_device.shape)}"
    )

    ref_model = SolarMoEReference(state_dict, layer_num, num_experts, top_k, routed_scaling_factor)

    pt_input = (torch.rand(1, 1, 1, hidden_dim) * 2) - 1
    with torch.no_grad():
        ref_output = ref_model(pt_input)
    logger.info(f"Reference: mean={ref_output.mean().item():.6f}")

    from models.demos.solar_open_100b.tt.moe_layer import SolarMoELayer

    dtype = ttnn.bfloat8_b
    tt_moe = SolarMoELayer(
        mesh_device=mesh_device,
        state_dict=state_dict,
        args=model_args,
        layer_num=layer_num,
        dtype=dtype,
    )

    tt_input = ttnn.from_torch(
        pt_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    from models.tt_transformers.tt.common import Mode

    logger.info("Running TT MoE layer...")
    tt_output = tt_moe.forward(tt_input, Mode.DECODE)

    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1]

    logger.info(f"TT output: mean={tt_output_torch.mean().item():.6f}")

    passing, pcc_message = comp_pcc(ref_output.float(), tt_output_torch.float(), pcc_threshold)
    logger.info(f"PCC: {pcc_message}")
    logger.info(comp_allclose(ref_output.float(), tt_output_torch.float()))

    assert passing, f"PCC check failed: {pcc_message}"


@pytest.mark.skipif(
    not os.path.exists(CKPT_DIR),
    reason=f"Solar checkpoint not found at {CKPT_DIR}",
)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_solar_experts_hw(mesh_device, device_params):
    """Test Solar experts computation without all-to-all (per-device validation)."""
    torch.manual_seed(42)
    pcc_threshold = 0.95

    mesh_device.disable_and_clear_program_cache()

    logger.info("Loading model args and state dict...")
    model_args = ModelArgs(mesh_device, max_batch_size=1, max_seq_len=4096)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    layer_num = 0
    hidden_dim = model_args.dim
    num_experts = model_args.num_experts
    num_devices = model_args.num_devices
    experts_per_device = num_experts // num_devices

    logger.info(
        f"Testing experts: {num_experts} total, {experts_per_device} per device, "
        f"hidden_dim={hidden_dim}, moe_intermediate_size={model_args.moe_intermediate_size}"
    )

    from models.demos.solar_open_100b.tt.experts import SolarExperts

    dtype = ttnn.bfloat8_b
    tt_experts = SolarExperts(
        mesh_device=mesh_device, args=model_args, state_dict=state_dict, layer_num=layer_num, dtype=dtype
    )

    pt_input = (torch.rand(1, experts_per_device, 1, hidden_dim) * 2) - 1

    # PyTorch reference for device 0's experts (experts 0..experts_per_device-1)
    ref_output = torch.zeros(1, experts_per_device, 1, hidden_dim)
    for j in range(experts_per_device):
        w1 = state_dict[f"layers.{layer_num}.feed_forward.experts.{j}.w1.weight"].float()
        w2 = state_dict[f"layers.{layer_num}.feed_forward.experts.{j}.w2.weight"].float()
        w3 = state_dict[f"layers.{layer_num}.feed_forward.experts.{j}.w3.weight"].float()
        x = pt_input[:, j : j + 1, :, :].float()
        gate = F.silu(F.linear(x, w1))
        up = F.linear(x, w3)
        ref_output[:, j : j + 1, :, :] = F.linear(gate * up, w2)

    logger.info(f"Reference output: mean={ref_output.mean().item():.6f}")

    tt_input = ttnn.from_torch(
        pt_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    logger.info("Running TT experts...")
    tt_output = tt_experts.forward(tt_input, None)

    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1]

    logger.info(f"TT output: mean={tt_output_torch.mean().item():.6f}")

    passing, pcc_message = comp_pcc(ref_output.float(), tt_output_torch.float(), pcc_threshold)
    logger.info(f"PCC: {pcc_message}")
    logger.info(comp_allclose(ref_output.float(), tt_output_torch.float()))

    if passing:
        logger.info(f"Solar experts test PASSED!")
    else:
        logger.warning(f"Solar experts test FAILED!")

    assert passing, f"PCC check failed: {pcc_message}"


@pytest.mark.skipif(
    not os.path.exists(CKPT_DIR),
    reason=f"Solar checkpoint not found at {CKPT_DIR}",
)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_solar_shared_expert_hw(mesh_device, device_params):
    """Test Solar shared expert computation."""
    torch.manual_seed(42)
    pcc_threshold = 0.95

    mesh_device.disable_and_clear_program_cache()

    logger.info("Loading model args and state dict...")
    model_args = ModelArgs(mesh_device, max_batch_size=1, max_seq_len=4096)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    layer_num = 0
    hidden_dim = model_args.dim

    prefix = f"layers.{layer_num}.feed_forward.shared_experts"
    w1 = state_dict[f"{prefix}.w1.weight"].float()
    w2 = state_dict[f"{prefix}.w2.weight"].float()
    w3 = state_dict[f"{prefix}.w3.weight"].float()

    pt_input = (torch.rand(1, 1, 1, hidden_dim) * 2) - 1
    x = pt_input.float()
    gate = F.silu(F.linear(x, w1))
    up = F.linear(x, w3)
    ref_output = F.linear(gate * up, w2)

    logger.info(f"Reference output: mean={ref_output.mean().item():.6f}")

    from models.demos.solar_open_100b.tt.shared_expert import SolarSharedExpert

    dtype = ttnn.bfloat8_b
    tt_shared = SolarSharedExpert(
        mesh_device=mesh_device, args=model_args, state_dict=state_dict, layer_num=layer_num, dtype=dtype
    )

    tt_input = ttnn.from_torch(
        pt_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    logger.info("Running TT shared expert...")
    tt_output = tt_shared.forward(tt_input)

    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1]

    logger.info(f"TT output: mean={tt_output_torch.mean().item():.6f}")

    passing, pcc_message = comp_pcc(ref_output.float(), tt_output_torch.float(), pcc_threshold)
    logger.info(f"PCC: {pcc_message}")
    logger.info(comp_allclose(ref_output.float(), tt_output_torch.float()))

    if passing:
        logger.info(f"Solar shared expert test PASSED!")
    else:
        logger.warning(f"Solar shared expert test FAILED!")

    assert passing, f"PCC check failed: {pcc_message}"
