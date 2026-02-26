# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
vLLM Generator for Solar-Open-100B.

Provides the standard vLLM integration interface: initialize, prefill, decode, allocate_kv_cache.
Uses the shared Transformer with pluggable moe_class=SolarMoELayer.

Since model_type "solar_open" is not recognized by transformers, this generator
loads weights directly from safetensors files and applies the standard key conversion.
"""

import json
import os

from loguru import logger
from safetensors.torch import safe_open

import ttnn
from models.demos.solar_open_100b.tt.moe_layer import SolarMoELayer
from models.tt_transformers.tt.generator import Generator, create_submeshes
from models.tt_transformers.tt.generator_vllm import allocate_vllm_kv_cache
from models.tt_transformers.tt.load_checkpoints import convert_hf_to_meta, standardize_hf_keys
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs


def load_solar_state_dict(ckpt_dir, head_dim, n_heads, n_kv_heads):
    """
    Load Solar-Open-100B weights directly from safetensors, bypassing AutoModel.

    The standard load_state_dict path uses AutoModelForCausalLM.from_pretrained(),
    which fails because model_type "solar_open" is not recognized by transformers.
    This function loads weights directly from safetensors and applies the standard
    key conversion pipeline.

    Args:
        ckpt_dir: Path to the checkpoint directory
        head_dim: Head dimension for QKV conversion
        n_heads: Number of attention heads
        n_kv_heads: Number of KV heads

    Returns:
        state_dict: Converted state dict in meta format
    """
    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")

    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index_data = json.load(f)

        weight_map = index_data["weight_map"]
        shards = sorted(set(weight_map.values()))

        state_dict = {}
        for shard in shards:
            shard_path = os.path.join(ckpt_dir, shard)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            logger.info(f"Loaded shard {shard} ({len(state_dict)} total keys)")
    else:
        single_path = os.path.join(ckpt_dir, "model.safetensors")
        with safe_open(single_path, framework="pt", device="cpu") as f:
            state_dict = {key: f.get_tensor(key) for key in f.keys()}

    logger.info(f"Loaded {len(state_dict)} weight tensors from {ckpt_dir}")

    # Apply standard HF-to-meta key conversion
    state_dict = standardize_hf_keys(state_dict)
    state_dict = convert_hf_to_meta(state_dict, head_dim, n_heads, n_kv_heads)

    return state_dict


class SolarOpenForCausalLM(Generator):
    """
    vLLM-compatible generator for Solar-Open-100B.

    Uses the shared Transformer infrastructure with SolarMoELayer
    plugged in via the moe_class parameter.
    """

    model_capabilities = {
        "supports_prefix_caching": True,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations: str = "performance",
    ):
        submesh_devices = create_submeshes(mesh_device, tt_data_parallel)

        model_args = []
        for submesh in submesh_devices:
            model_args_i = ModelArgs(
                submesh,
                instruct=False,
                max_batch_size=max_batch_size // tt_data_parallel,
                optimizations=lambda model_args: DecodersPrecision.from_string(optimizations)(
                    model_args.n_layers, model_args.model_name
                )
                if optimizations is not None
                else DecodersPrecision.performance(model_args.n_layers, model_args.model_name),
                max_seq_len=max_seq_len,
            )
            if n_layers is not None:
                model_args_i.n_layers = n_layers
            model_args.append(model_args_i)

        # Load state dict directly from safetensors (bypasses AutoModel)
        args0 = model_args[0]
        state_dict = load_solar_state_dict(
            ckpt_dir=args0.CKPT_DIR,
            head_dim=args0.head_dim,
            n_heads=args0.n_heads,
            n_kv_heads=args0.n_kv_heads,
        )

        # Set MoE flags that would normally be set by load_state_dict
        for args_i in model_args:
            args_i.is_mixture_of_experts = any(".experts." in k for k in state_dict.keys())
            if args_i.is_mixture_of_experts:
                args_i.moe = True
                import re

                expert_ids = [
                    int(re.search(r"\.experts\.(\d+)\.", item).group(1))
                    for item in state_dict.keys()
                    if re.search(r"\.experts\.(\d+)\.", item)
                ]
                args_i.num_experts = max(expert_ids) + 1 if expert_ids else 0

        tt_model = []
        for i, submesh in enumerate(submesh_devices):
            tt_model_i = Transformer(
                args=model_args[i],
                mesh_device=submesh,
                dtype=ttnn.bfloat8_b,
                state_dict=state_dict,
                weight_cache_path=model_args[i].weight_cache_path(ttnn.bfloat8_b),
                use_paged_kv_cache=True,
                moe_class=SolarMoELayer,
            )
            tt_model.append(tt_model_i)

        return cls(tt_model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args[0].model_cache_path

    def prefill_forward(self, *args, **kwargs):
        return super().prefill_forward_text(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward(*args, **kwargs)

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(*args, **kwargs, dp_model=self.model, tt_cache_path=self.cache_path)
