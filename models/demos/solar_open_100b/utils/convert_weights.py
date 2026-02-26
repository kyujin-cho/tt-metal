# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Weight conversion utilities for Solar-Open-100B.

Converts HuggingFace checkpoint weights to the format expected by the TT implementation.

HF weight key patterns (before conversion):
- Attention: model.layers.{i}.self_attn.{q_proj|k_proj|v_proj|o_proj}.weight
- MoE gate: model.layers.{i}.mlp.gate.weight
- Gate bias: model.layers.{i}.mlp.gate.e_score_correction_bias
- Routed experts: model.layers.{i}.mlp.experts.{j}.{gate_proj|up_proj|down_proj}.weight
- Shared expert: model.layers.{i}.mlp.shared_experts.{gate_proj|up_proj|down_proj}.weight
- Norms: model.layers.{i}.{input_layernorm|post_attention_layernorm}.weight
- Embedding: model.embed_tokens.weight
- LM head: lm_head.weight

Meta-format key patterns (after convert_hf_to_meta):
- Attention: layers.{i}.attention.{wq|wk|wv|wo}.weight
- MoE gate: layers.{i}.feed_forward.gate.weight
- Gate bias: layers.{i}.feed_forward.gate.e_score_correction_bias
- Routed experts: layers.{i}.feed_forward.experts.{j}.{w1|w2|w3}.weight
- Shared expert: layers.{i}.feed_forward.shared_experts.{w1|w2|w3}.weight
- Norms: layers.{i}.{attention_norm|ffn_norm}.weight
"""

from loguru import logger


def validate_solar_weights(state_dict, n_layers, num_experts=128):
    """
    Validate that all expected Solar-Open-100B weights are present (meta-format keys).

    Args:
        state_dict: Converted state dict (after HF-to-meta conversion)
        n_layers: Number of transformer layers
        num_experts: Number of routed experts
    """
    missing = []

    for layer in range(n_layers):
        prefix = f"layers.{layer}"

        # Check attention weights
        for proj in ["wq", "wk", "wv", "wo"]:
            key = f"{prefix}.attention.{proj}.weight"
            if key not in state_dict:
                missing.append(key)

        # Check MoE gate
        gate_key = f"{prefix}.feed_forward.gate.weight"
        if gate_key not in state_dict:
            missing.append(gate_key)

        # Check gate bias
        bias_key = f"{prefix}.feed_forward.gate.e_score_correction_bias"
        if bias_key not in state_dict:
            missing.append(bias_key)

        # Check routed experts
        for expert_id in range(num_experts):
            for proj in ["w1", "w2", "w3"]:
                key = f"{prefix}.feed_forward.experts.{expert_id}.{proj}.weight"
                if key not in state_dict:
                    missing.append(key)

        # Check shared expert
        for proj in ["w1", "w2", "w3"]:
            key = f"{prefix}.feed_forward.shared_experts.{proj}.weight"
            if key not in state_dict:
                missing.append(key)

        # Check norms
        for norm in ["attention_norm", "ffn_norm"]:
            key = f"{prefix}.{norm}.weight"
            if key not in state_dict:
                missing.append(key)

    if missing:
        logger.warning(f"Missing {len(missing)} weight tensors. First 10: {missing[:10]}")
    else:
        logger.info(f"All expected Solar-Open-100B weights present for {n_layers} layers")

    return missing
