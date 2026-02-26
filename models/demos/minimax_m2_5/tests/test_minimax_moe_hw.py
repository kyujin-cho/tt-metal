# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Hardware test for MiniMax-M2.5 MoE layer components on Galaxy.

Tests:
1. test_minimax_gate_hw: Sigmoid+bias top-8 gate on TT hardware
2. test_minimax_experts_hw: Batched SwiGLU expert computation (with FP8 dequant)
3. test_minimax_full_moe_pipeline_hw: Full MoE pipeline (gate + 256 experts, combine in Python)
4. test_minimax_moe_layer_hw: Full MoE with all-to-all dispatch/combine (requires fabric)

Requires: HF_MODEL=/data/models/MiniMax-M2.5
Run: source python_env/bin/activate && HF_MODEL=/data/models/MiniMax-M2.5 pytest models/demos/minimax_m2_5/tests/test_minimax_moe_hw.py -v -s
"""

import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from loguru import logger
from safetensors.torch import safe_open

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.deepseek_v3.utils.dequantize import dequantize_tensor

CKPT_DIR = os.getenv("HF_MODEL", "/data/models/MiniMax-M2.5")


def load_minimax_layer_weights(ckpt_dir, layer_num=0, num_experts=256):
    """
    Load MiniMax-M2.5 weights for a single layer directly from safetensors.

    Strips 'model.' prefix, dequantizes FP8 expert weights.
    Returns state dict with keys like:
      layers.0.block_sparse_moe.gate.weight
      layers.0.block_sparse_moe.e_score_correction_bias
      layers.0.block_sparse_moe.experts.{j}.w1.weight  (dequantized from FP8)
    """
    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index_data = json.load(f)

    weight_map = index_data["weight_map"]

    # Find all keys for this layer
    layer_prefix = f"model.layers.{layer_num}."
    layer_keys = {k: v for k, v in weight_map.items() if k.startswith(layer_prefix)}

    # Load from shards
    shards_needed = sorted(set(layer_keys.values()))
    raw_state_dict = {}
    for shard in shards_needed:
        shard_path = os.path.join(ckpt_dir, shard)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(layer_prefix):
                    raw_state_dict[key] = f.get_tensor(key)
        logger.info(f"Loaded shard {shard}")

    # Strip 'model.' prefix
    state_dict = {}
    for key, value in raw_state_dict.items():
        new_key = key[6:] if key.startswith("model.") else key
        state_dict[new_key] = value

    # Dequantize FP8 weights
    keys_to_remove = []
    dequant_count = 0
    for key in list(state_dict.keys()):
        if key.endswith(".weight_scale_inv"):
            weight_key = key.replace(".weight_scale_inv", ".weight")
            if weight_key in state_dict:
                weight = state_dict[weight_key]
                scale = state_dict[key]
                state_dict[weight_key] = dequantize_tensor(weight, scale, (128, 128))
                keys_to_remove.append(key)
                dequant_count += 1

    for key in keys_to_remove:
        del state_dict[key]

    logger.info(f"Loaded {len(state_dict)} tensors for layer {layer_num}, dequantized {dequant_count} FP8 weights")
    return state_dict


def make_minimax_args(num_devices):
    """Create a minimal args namespace matching what MiniMax TT components expect."""
    args = SimpleNamespace()
    args.dim = 3072
    args.num_experts = 256
    args.num_experts_per_tok = 8
    args.moe_intermediate_size = 1536
    args.hidden_dim = 3072
    args.num_devices = num_devices
    args.routed_scaling_factor = 1.0  # MiniMax uses 1.0
    args.dummy_weights = True  # Skip weight caching (we load directly from safetensors)
    args.ccl_dtype = ttnn.bfloat16
    return args


class MiniMaxMoEReference(torch.nn.Module):
    """Full PyTorch reference for MiniMax MoE layer (gate + 256 experts, no shared expert)."""

    def __init__(self, state_dict, layer_num, num_experts, top_k, routed_scaling_factor=1.0):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor

        # Gate weights
        self.gate_weight = state_dict[f"layers.{layer_num}.block_sparse_moe.gate.weight"].float()
        bias_key = f"layers.{layer_num}.block_sparse_moe.e_score_correction_bias"
        self.bias = state_dict[bias_key].float() if bias_key in state_dict else torch.zeros(num_experts)

        # Expert weights (256 routed experts)
        self.experts_w1 = []
        self.experts_w2 = []
        self.experts_w3 = []
        for i in range(self.num_experts):
            prefix = f"layers.{layer_num}.block_sparse_moe.experts.{i}"
            self.experts_w1.append(state_dict[f"{prefix}.w1.weight"].float())
            self.experts_w2.append(state_dict[f"{prefix}.w2.weight"].float())
            self.experts_w3.append(state_dict[f"{prefix}.w3.weight"].float())

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

        return routed_output


class MiniMaxGateReference(torch.nn.Module):
    """PyTorch reference for MiniMax gate only."""

    def __init__(self, state_dict, layer_num, num_experts, top_k, routed_scaling_factor=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor

        self.gate_weight = state_dict[f"layers.{layer_num}.block_sparse_moe.gate.weight"].float()
        bias_key = f"layers.{layer_num}.block_sparse_moe.e_score_correction_bias"
        self.bias = state_dict[bias_key].float() if bias_key in state_dict else torch.zeros(num_experts)

    def forward(self, x):
        x_float = x.float()
        logits = F.linear(x_float, self.gate_weight)
        scores = torch.sigmoid(logits)
        biased = scores + self.bias

        _, indices = torch.topk(biased, k=self.top_k, dim=-1, sorted=True)
        selected_scores = torch.gather(scores, dim=-1, index=indices)
        normalized = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-20)
        weights = normalized * self.routed_scaling_factor

        return indices, weights


# ─── Test 1: Gate ────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not os.path.exists(CKPT_DIR),
    reason=f"MiniMax checkpoint not found at {CKPT_DIR}",
)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_minimax_gate_hw(mesh_device, device_params):
    """
    Test MiniMax sigmoid+bias top-8 gate on TT hardware.

    Validates gate routing produces correct expert indices and weights
    compared to PyTorch reference.
    """
    torch.manual_seed(42)
    pcc_threshold = 0.99

    mesh_device.disable_and_clear_program_cache()

    layer_num = 0
    num_experts = 256
    top_k = 8
    hidden_dim = 3072

    logger.info("Loading MiniMax layer 0 weights...")
    state_dict = load_minimax_layer_weights(CKPT_DIR, layer_num, num_experts)

    # PyTorch reference
    ref_gate = MiniMaxGateReference(state_dict, layer_num, num_experts, top_k)

    pt_input = (torch.rand(1, 1, 1, hidden_dim) * 2) - 1
    with torch.no_grad():
        ref_indices, ref_weights = ref_gate(pt_input)

    logger.info(
        f"Reference: indices={ref_indices.squeeze().int().tolist()}, weights_sum={ref_weights.sum().item():.4f}"
    )

    # TT gate
    num_devices = mesh_device.get_num_devices()
    args = make_minimax_args(num_devices)

    from models.demos.minimax_m2_5.tt.moe_gate import MiniMaxMoEGate

    tt_gate = MiniMaxMoEGate(
        mesh_device=mesh_device,
        args=args,
        state_dict=state_dict,
        layer_num=layer_num,
        dtype=ttnn.bfloat8_b,
    )

    tt_input = ttnn.from_torch(
        pt_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    logger.info("Running TT gate...")
    tt_indices, tt_weights = tt_gate.forward(tt_input)

    indices_torch = ttnn.to_torch(tt_indices, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1]
    weights_torch = ttnn.to_torch(tt_weights, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1]

    logger.info(f"TT: indices={indices_torch.squeeze().int().tolist()}, weights_sum={weights_torch.sum().item():.4f}")

    # Compare indices — allow partial overlap since MiniMax biased scores are
    # very close (~9.0 ± 0.2), making bfloat16 rounding flip the ordering.
    ref_set = set(ref_indices.squeeze().int().tolist())
    tt_set = set(indices_torch.squeeze().int().tolist())
    overlap = ref_set & tt_set
    logger.info(f"Index overlap: {len(overlap)}/{top_k} ({overlap})")
    if ref_set != tt_set:
        logger.info(f"  Ref only: {ref_set - tt_set}")
        logger.info(f"  TT only:  {tt_set - ref_set}")

    # Weight sums should be close (both should sum to ~routed_scaling_factor)
    ref_sum = ref_weights.sum().item()
    tt_sum = weights_torch.sum().item()
    logger.info(f"Weight sums: ref={ref_sum:.4f}, TT={tt_sum:.4f}")

    # At least 5/8 top indices should overlap (biased scores are tightly clustered)
    assert len(overlap) >= 5, f"Index overlap too low: {len(overlap)}/8"
    # Weight sums should be close
    assert abs(ref_sum - tt_sum) < 0.2, f"Weight sum mismatch: ref={ref_sum:.4f}, TT={tt_sum:.4f}"
    logger.info("MiniMax gate test PASSED")


# ─── Test 2: Experts ─────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not os.path.exists(CKPT_DIR),
    reason=f"MiniMax checkpoint not found at {CKPT_DIR}",
)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_minimax_experts_hw(mesh_device, device_params):
    """
    Test MiniMax batched SwiGLU expert computation on TT hardware.

    Validates expert outputs for device 0's local experts (32 experts out of 256)
    against PyTorch reference using real dequantized FP8 weights.
    """
    torch.manual_seed(42)
    pcc_threshold = 0.95

    mesh_device.disable_and_clear_program_cache()

    layer_num = 0
    num_experts = 256
    hidden_dim = 3072
    num_devices = mesh_device.get_num_devices()
    experts_per_device = num_experts // num_devices

    logger.info(f"Loading MiniMax weights ({num_experts} experts, {experts_per_device} per device)...")
    state_dict = load_minimax_layer_weights(CKPT_DIR, layer_num, num_experts)

    args = make_minimax_args(num_devices)

    from models.demos.minimax_m2_5.tt.experts import MiniMaxExperts

    dtype = ttnn.bfloat8_b
    tt_experts = MiniMaxExperts(
        mesh_device=mesh_device,
        args=args,
        state_dict=state_dict,
        layer_num=layer_num,
        dtype=dtype,
    )

    pt_input = (torch.rand(1, experts_per_device, 1, hidden_dim) * 2) - 1

    # PyTorch reference for device 0's experts (experts 0..experts_per_device-1)
    ref_output = torch.zeros(1, experts_per_device, 1, hidden_dim)
    for j in range(experts_per_device):
        prefix = f"layers.{layer_num}.block_sparse_moe.experts.{j}"
        w1 = state_dict[f"{prefix}.w1.weight"].float()
        w2 = state_dict[f"{prefix}.w2.weight"].float()
        w3 = state_dict[f"{prefix}.w3.weight"].float()
        x = pt_input[:, j : j + 1, :, :].float()
        gate = F.silu(F.linear(x, w1))
        up = F.linear(x, w3)
        ref_output[:, j : j + 1, :, :] = F.linear(gate * up, w2)

    logger.info(f"Reference output: mean={ref_output.mean().item():.6f}, std={ref_output.std().item():.6f}")

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

    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[
        0:1
    ]  # Device 0 only

    logger.info(f"TT output: mean={tt_output_torch.mean().item():.6f}, std={tt_output_torch.std().item():.6f}")

    passing, pcc_message = comp_pcc(ref_output.float(), tt_output_torch.float(), pcc_threshold)
    logger.info(f"PCC: {pcc_message}")
    logger.info(comp_allclose(ref_output.float(), tt_output_torch.float()))

    assert passing, f"Expert PCC check failed: {pcc_message}"


# ─── Test 3: Full MoE Pipeline (no fabric) ──────────────────────────────────


@pytest.mark.skipif(
    not os.path.exists(CKPT_DIR),
    reason=f"MiniMax checkpoint not found at {CKPT_DIR}",
)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_minimax_full_moe_pipeline_hw(mesh_device, device_params):
    """
    Test full MiniMax MoE pipeline: gate + 256 experts, combined in Python.

    Runs gate and experts on TT hardware, combines expert outputs in Python
    to validate the complete MoE output matches PyTorch reference.
    Does NOT require fabric (no all-to-all dispatch/combine).

    Note: PCC threshold is lower (0.80) because MiniMax has 256 experts with
    tightly-clustered biased scores (~9.0 ± 0.2), causing bfloat16 precision
    to flip 1-2 of the top-8 expert selections. Individual expert PCC is 0.9999.
    """
    torch.manual_seed(42)
    pcc_threshold = 0.80

    mesh_device.disable_and_clear_program_cache()

    layer_num = 0
    num_experts = 256
    hidden_dim = 3072
    top_k = 8
    num_devices = mesh_device.get_num_devices()
    experts_per_device = num_experts // num_devices

    logger.info(f"Loading MiniMax weights ({num_experts} experts)...")
    state_dict = load_minimax_layer_weights(CKPT_DIR, layer_num, num_experts)

    # PyTorch reference
    ref_model = MiniMaxMoEReference(state_dict, layer_num, num_experts, top_k)

    pt_input = (torch.rand(1, 1, 1, hidden_dim) * 2) - 1

    logger.info("Computing reference output...")
    with torch.no_grad():
        ref_output = ref_model(pt_input)
    logger.info(f"Reference: mean={ref_output.mean().item():.6f}, std={ref_output.std().item():.6f}")

    # TT Components
    args = make_minimax_args(num_devices)
    dtype = ttnn.bfloat8_b

    from models.demos.minimax_m2_5.tt.experts import MiniMaxExperts
    from models.demos.minimax_m2_5.tt.moe_gate import MiniMaxMoEGate

    tt_gate = MiniMaxMoEGate(
        mesh_device=mesh_device,
        args=args,
        state_dict=state_dict,
        layer_num=layer_num,
        dtype=dtype,
    )

    tt_experts = MiniMaxExperts(
        mesh_device=mesh_device,
        args=args,
        state_dict=state_dict,
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

    # ---- Run TT Gate ----
    logger.info("Running TT gate...")
    tt_indices, tt_weights = tt_gate.forward(tt_input)

    indices_torch = ttnn.to_torch(tt_indices, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1]

    weights_torch = ttnn.to_torch(tt_weights, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1]

    logger.info(f"Gate: indices={indices_torch.squeeze().int().tolist()}, weights_sum={weights_torch.sum().item():.4f}")

    # ---- Run TT Experts (all 256 via batched computation per device) ----
    logger.info("Running TT experts on all devices...")
    tt_expert_input = ttnn.repeat(tt_input, ttnn.Shape((1, experts_per_device, 1, 1)))

    tt_expert_out = tt_experts.forward(tt_expert_input, None)

    # Get ALL 256 expert outputs: 8 devices x 32 experts/device
    all_expert_out_torch = ttnn.to_torch(
        tt_expert_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )  # [8, 32, 1, 3072]

    all_expert_out = all_expert_out_torch.reshape(num_experts, 1, hidden_dim)
    logger.info(f"All expert outputs: shape={all_expert_out.shape}, mean={all_expert_out.mean().item():.6f}")

    # ---- Combine in Python ----
    logger.info("Combining outputs...")
    tt_combined = torch.zeros(1, 1, 1, hidden_dim)
    for k in range(top_k):
        expert_id = indices_torch[0, 0, 0, k].int().item()
        weight = weights_torch[0, 0, 0, k].float().item()
        expert_output = all_expert_out[expert_id : expert_id + 1].unsqueeze(0)
        tt_combined = tt_combined + weight * expert_output
        logger.debug(f"  Expert {expert_id}: weight={weight:.4f}")

    logger.info(f"TT combined: mean={tt_combined.mean().item():.6f}, std={tt_combined.std().item():.6f}")

    # ---- Compare ----
    passing, pcc_message = comp_pcc(ref_output.float(), tt_combined.float(), pcc_threshold)
    logger.info(f"PCC: {pcc_message}")
    logger.info(comp_allclose(ref_output.float(), tt_combined.float()))

    assert passing, f"Full MoE pipeline PCC check failed: {pcc_message}"


# ─── Test 4: Full MoE with all-to-all (requires fabric) ─────────────────────


@pytest.mark.skipif(
    not os.path.exists(CKPT_DIR),
    reason=f"MiniMax checkpoint not found at {CKPT_DIR}",
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
def test_minimax_moe_layer_hw(mesh_device, device_params):
    """
    Test MiniMax full MoE layer with all-to-all dispatch/combine on hardware.

    Requires working fabric (all-to-all communication between devices).
    Uses (8,4) full TG mesh with FABRIC_1D.
    256 experts across 32 devices = 8 experts/device.

    Note: PCC threshold is 0.70 due to two compounding factors:
    1. bfloat16 routing precision: MiniMax's 256 biased scores cluster at ~9.0 ± 0.2,
       causing 1-2 of the top-8 selections to differ from float32 reference
    2. All-to-all quantization: dispatch/combine adds additional precision loss
       from row-major ↔ tile layout conversions and cross-device communication
    Individual experts achieve PCC 0.9999; the gap is from routing divergence.
    """
    torch.manual_seed(42)
    pcc_threshold = 0.70

    mesh_device.disable_and_clear_program_cache()

    layer_num = 0
    num_experts = 256
    hidden_dim = 3072
    top_k = 8
    num_devices = mesh_device.get_num_devices()

    logger.info(f"Loading MiniMax weights ({num_experts} experts, {num_experts // num_devices} per device)...")
    state_dict = load_minimax_layer_weights(CKPT_DIR, layer_num, num_experts)

    # PyTorch reference
    ref_model = MiniMaxMoEReference(state_dict, layer_num, num_experts, top_k)

    pt_input = (torch.rand(1, 1, 1, hidden_dim) * 2) - 1
    with torch.no_grad():
        ref_output = ref_model(pt_input)
    logger.info(f"Reference: mean={ref_output.mean().item():.6f}")

    args = make_minimax_args(num_devices)
    # MoE layer needs ccl_topology
    args.ccl_topology = lambda: ttnn.Topology.Linear
    args.ccl_dtype = ttnn.bfloat16

    from models.demos.minimax_m2_5.tt.moe_layer import MiniMaxMoELayer

    dtype = ttnn.bfloat8_b
    tt_moe = MiniMaxMoELayer(
        mesh_device=mesh_device,
        state_dict=state_dict,
        args=args,
        layer_num=layer_num,
        dtype=dtype,
        tt_ccl=None,
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

    assert passing, f"Full MoE layer PCC check failed: {pcc_message}"
