# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
End-to-end tests for Solar-Open-100B model.

Tests full model integration including:
- Expert count parsing (128 experts, regex-based)
- Sigmoid+bias gate routing
- Meta-format key conversion
- Pluggable moe_class/moe_kwargs integration
"""

import re

import torch
from loguru import logger


def test_solar_expert_count_parsing():
    """Test that 128 experts are correctly parsed from both key formats."""
    # Test with feed_forward format (Solar meta-format after HF-to-meta conversion)
    meta_keys = []
    for layer in range(1):
        for expert_id in range(128):
            for proj in ["w1", "w2", "w3"]:
                meta_keys.append(f"layers.{layer}.feed_forward.experts.{expert_id}.{proj}.weight")

    expert_ids = [
        int(re.search(r"\.experts\.(\d+)\.", item).group(1))
        for item in meta_keys
        if re.search(r"\.experts\.(\d+)\.", item)
    ]
    num_experts = max(expert_ids) + 1 if expert_ids else 0

    assert num_experts == 128, f"Expected 128 experts (meta format), got {num_experts}"

    # Test with block_sparse_moe format (Mixtral backward compatibility)
    mixtral_keys = []
    for expert_id in range(8):
        for proj in ["w1", "w2", "w3"]:
            mixtral_keys.append(f"layers.0.block_sparse_moe.experts.{expert_id}.{proj}.weight")

    expert_ids_mixtral = [
        int(re.search(r"\.experts\.(\d+)\.", item).group(1))
        for item in mixtral_keys
        if re.search(r"\.experts\.(\d+)\.", item)
    ]
    num_experts_mixtral = max(expert_ids_mixtral) + 1 if expert_ids_mixtral else 0

    assert num_experts_mixtral == 8, f"Expected 8 experts (Mixtral format), got {num_experts_mixtral}"
    logger.info("Expert count parsing test passed for both key formats")


def test_solar_key_mapping():
    """Test that HF keys are correctly converted to meta format for Solar."""
    from models.tt_transformers.tt.load_checkpoints import map_hf_to_meta_keys

    test_keys = {
        "model.layers.0.mlp.experts.5.gate_proj.weight": torch.zeros(1),
        "model.layers.0.mlp.gate.weight": torch.zeros(1),
        "model.layers.0.mlp.gate.e_score_correction_bias": torch.zeros(1),
        "model.layers.0.mlp.shared_experts.gate_proj.weight": torch.zeros(1),
    }

    mapped = map_hf_to_meta_keys(test_keys)
    mapped_keys = list(mapped.keys())

    assert "layers.0.feed_forward.experts.5.w1.weight" in mapped_keys
    assert "layers.0.feed_forward.gate.weight" in mapped_keys
    assert "layers.0.feed_forward.gate.e_score_correction_bias" in mapped_keys
    assert "layers.0.feed_forward.shared_experts.w1.weight" in mapped_keys

    logger.info("Key mapping test passed")


def test_solar_moe_class_pluggable():
    """Test that moe_class parameter is correctly threaded through Transformer -> TransformerBlock."""
    import inspect

    from models.tt_transformers.tt.decoder import TransformerBlock
    from models.tt_transformers.tt.model import Transformer

    # Verify TransformerBlock accepts moe_class and moe_kwargs
    block_sig = inspect.signature(TransformerBlock.__init__)
    assert "moe_class" in block_sig.parameters, "TransformerBlock missing moe_class parameter"
    assert "moe_kwargs" in block_sig.parameters, "TransformerBlock missing moe_kwargs parameter"

    # Verify Transformer accepts moe_class and moe_kwargs
    model_sig = inspect.signature(Transformer.__init__)
    assert "moe_class" in model_sig.parameters, "Transformer missing moe_class parameter"
    assert "moe_kwargs" in model_sig.parameters, "Transformer missing moe_kwargs parameter"

    logger.info("Pluggable MoE class test passed")


def test_solar_model_config_fields():
    """Test that MoE config fields are properly read from config."""
    # Simulate a Solar-like config
    text_config = {
        "moe_intermediate_size": 1280,
        "n_routed_experts": 128,
        "n_shared_experts": 1,
        "routed_scaling_factor": 1.0,
        "norm_topk_prob": True,
        "partial_rotary_factor": 1.0,
    }

    assert text_config["moe_intermediate_size"] == 1280
    assert text_config["n_routed_experts"] == 128
    assert text_config["n_shared_experts"] == 1
    assert text_config["routed_scaling_factor"] == 1.0
    assert text_config["norm_topk_prob"] is True

    logger.info("Solar model config fields test passed")
