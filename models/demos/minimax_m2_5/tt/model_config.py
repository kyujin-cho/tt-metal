# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
MiniMax-M2.5 specific configuration helpers.

MiniMax-M2.5 uses:
- 256 routed experts (no shared expert)
- Sigmoid+bias top-8 routing with routed_scaling_factor
- Partial RoPE (partial_rotary_factor=0.5, rotary_dim=64 with head_dim=128)
- FP8 quantized weights (float8_e4m3fn with [128,128] block dequant)
- GQA attention with QK norm
- SwiGLU activation
- MTP (Multi-Token Prediction, 3 modules) — deferred to future work
"""


def get_minimax_model_config(args):
    """
    Apply MiniMax-specific configuration overrides.

    MiniMax-M2.5 key parameters:
    - num_experts: 256
    - num_experts_per_tok: 8
    - intermediate_size (expert): 1536
    - hidden_size: 3072
    - num_attention_heads: 24
    - num_key_value_heads: 8
    - head_dim: 128
    - partial_rotary_factor: 0.5 → rotary_dim=64
    - routed_scaling_factor: configurable
    - weight_block_size: [128, 128] (for FP8 dequantization)
    """
    # Validate expected MoE config
    assert hasattr(args, "num_experts"), "MiniMax model requires num_experts in config"
    assert args.num_experts == 256, f"Expected 256 routed experts, got {args.num_experts}"

    # Ensure partial RoPE is properly set
    assert args.partial_rotary_factor == 0.5, f"Expected partial_rotary_factor=0.5, got {args.partial_rotary_factor}"
    assert args.rotary_dim == 64, f"Expected rotary_dim=64, got {args.rotary_dim}"

    # num_experts_per_tok should be set from HF config
    if not hasattr(args, "num_experts_per_tok"):
        args.num_experts_per_tok = 8

    return args
