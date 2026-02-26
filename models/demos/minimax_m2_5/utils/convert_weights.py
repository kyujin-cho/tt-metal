# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Weight conversion utilities for MiniMax-M2.5.

Handles FP8 (float8_e4m3fn) weight dequantization and conversion to the format
expected by the TT implementation.

HF weight key patterns:
- Attention: model.layers.{i}.self_attn.{q_proj|k_proj|v_proj|o_proj}.weight
- MoE gate: model.layers.{i}.block_sparse_moe.gate.weight
- Gate bias: model.layers.{i}.block_sparse_moe.e_score_correction_bias
- Routed experts: model.layers.{i}.block_sparse_moe.experts.{j}.{w1|w2|w3}.weight
- Expert scales: model.layers.{i}.block_sparse_moe.experts.{j}.{w1|w2|w3}.weight_scale_inv
- Norms: model.layers.{i}.{input_layernorm|post_attention_layernorm}.weight
- Embedding: model.embed_tokens.weight
- LM head: lm_head.weight

FP8 dequantization reuses: models/demos/deepseek_v3/utils/dequantize.py
"""

from loguru import logger

from models.demos.deepseek_v3.utils.dequantize import dequantize_tensor


def dequantize_expert_weights(state_dict, num_experts=256, block_size=(128, 128)):
    """
    Dequantize FP8 expert weights in-place.

    Converts float8_e4m3fn quantized weights using block-wise inverse scales.

    Args:
        state_dict: State dict containing FP8 weights and their scales
        num_experts: Number of routed experts
        block_size: Block size for dequantization (default [128, 128])

    Returns:
        State dict with dequantized weights (scales removed)
    """
    dequantized_count = 0
    keys_to_remove = []

    for key in list(state_dict.keys()):
        if key.endswith(".weight_scale_inv"):
            weight_key = key.replace(".weight_scale_inv", ".weight")
            if weight_key in state_dict:
                weight = state_dict[weight_key]
                scale = state_dict[key]

                state_dict[weight_key] = dequantize_tensor(weight, scale, (1, *block_size))
                keys_to_remove.append(key)
                dequantized_count += 1

    for key in keys_to_remove:
        del state_dict[key]

    logger.info(f"Dequantized {dequantized_count} FP8 weight tensors for MiniMax-M2.5")
    return state_dict


def convert_minimax_weights(hf_state_dict, num_experts=256, block_size=(128, 128)):
    """
    Convert HuggingFace MiniMax-M2.5 weights to tt-metal format.

    1. Strip "model." prefix
    2. Dequantize FP8 expert weights

    Args:
        hf_state_dict: HuggingFace state dict
        num_experts: Number of routed experts
        block_size: FP8 dequantization block size

    Returns:
        Converted state dict
    """
    state_dict = {}

    for key, value in hf_state_dict.items():
        new_key = key
        if new_key.startswith("model."):
            new_key = new_key[6:]
        state_dict[new_key] = value

    # Dequantize FP8 weights
    state_dict = dequantize_expert_weights(state_dict, num_experts, block_size)

    logger.info(f"Converted {len(state_dict)} weight tensors for MiniMax-M2.5")
    return state_dict


def validate_minimax_weights(state_dict, n_layers, num_experts=256):
    """
    Validate that all expected MiniMax-M2.5 weights are present.

    Args:
        state_dict: Converted state dict
        n_layers: Number of transformer layers
        num_experts: Number of routed experts
    """
    missing = []

    for layer in range(n_layers):
        prefix = f"layers.{layer}"

        # Check attention weights
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            key = f"{prefix}.self_attn.{proj}.weight"
            if key not in state_dict:
                missing.append(key)

        # Check MoE gate
        gate_key = f"{prefix}.block_sparse_moe.gate.weight"
        if gate_key not in state_dict:
            missing.append(gate_key)

        # Check routed experts (weights should be dequantized, no scale keys)
        for expert_id in range(num_experts):
            for proj in ["w1", "w2", "w3"]:
                key = f"{prefix}.block_sparse_moe.experts.{expert_id}.{proj}.weight"
                if key not in state_dict:
                    missing.append(key)

        # Check norms
        for norm in ["input_layernorm", "post_attention_layernorm"]:
            key = f"{prefix}.{norm}.weight"
            if key not in state_dict:
                missing.append(key)

    if missing:
        logger.warning(f"Missing {len(missing)} weight tensors. First 10: {missing[:10]}")
    else:
        logger.info(f"All expected MiniMax-M2.5 weights present for {n_layers} layers")

    return missing
