# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Solar-Open-100B specific configuration helpers.

Solar-Open-100B uses:
- 128 routed experts + 1 shared expert
- Sigmoid+bias top-8 routing (with e_score_correction_bias)
- YaRN RoPE scaling (already supported by shared rope.py)
- GQA attention (already supported by shared attention.py)
- SwiGLU activation
- moe_intermediate_size: 1280 (for both routed and shared experts)
"""


def get_solar_model_config(args):
    """
    Apply Solar-specific configuration overrides.

    Solar-Open-100B key parameters:
    - num_experts: 128 (n_routed_experts)
    - num_experts_per_tok: 8
    - n_shared_experts: 1
    - moe_intermediate_size: 1280 (expert intermediate dim)
    - hidden_size: 4096
    - num_attention_heads: 64
    - num_key_value_heads: 8
    - head_dim: 128
    - rope_scaling: {type: "yarn", factor: 2.0}
    - routed_scaling_factor: 1.0
    - norm_topk_prob: True
    """
    # Validate expected MoE config
    assert hasattr(args, "num_experts"), "Solar model requires num_experts in config"
    assert args.num_experts == 128, f"Expected 128 routed experts, got {args.num_experts}"

    # num_experts_per_tok should be set from HF config
    if not hasattr(args, "num_experts_per_tok"):
        args.num_experts_per_tok = 8

    return args
