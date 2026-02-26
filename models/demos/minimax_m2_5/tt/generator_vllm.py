# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
vLLM Generator for MiniMax-M2.5.

Provides the standard vLLM integration interface: initialize, prefill, decode, allocate_kv_cache.
Uses the shared Transformer with:
- moe_class=MiniMaxMoELayer (sigmoid+bias routing, 256 experts)
- attention_class=PartialRoPEAttention (partial RoPE, rotary_dim=64)
"""

import ttnn
from models.demos.minimax_m2_5.tt.attention import PartialRoPEAttention
from models.demos.minimax_m2_5.tt.moe_layer import MiniMaxMoELayer
from models.tt_transformers.tt.generator import Generator, create_submeshes
from models.tt_transformers.tt.generator_vllm import allocate_vllm_kv_cache
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs


class MiniMaxForCausalLM(Generator):
    """
    vLLM-compatible generator for MiniMax-M2.5.

    Uses the shared Transformer infrastructure with:
    - PartialRoPEAttention for partial rotary embeddings
    - MiniMaxMoELayer for sigmoid+bias routing with 256 experts
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

        state_dict = model_args[0].load_state_dict()

        tt_model = []
        for i, submesh in enumerate(submesh_devices):
            tt_model_i = Transformer(
                args=model_args[i],
                mesh_device=submesh,
                dtype=ttnn.bfloat8_b,
                state_dict=state_dict,
                weight_cache_path=model_args[i].weight_cache_path(ttnn.bfloat8_b),
                use_paged_kv_cache=True,
                attention_class=PartialRoPEAttention,
                moe_class=MiniMaxMoELayer,
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
