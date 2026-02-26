# Solar-Open-100B: Mixture of Experts Language Model

Inference implementation for [Solar-Open-100B](https://huggingface.co/upstage/Solar-Open-100B) on Tenstorrent Wormhole accelerators.

**Architecture**: 48-layer MoE with 128 routed + 1 shared SwiGLU expert per layer, sigmoid+bias top-8 routing, GQA (64Q/8KV), YaRN RoPE.

**Target Hardware**:
- **T3K**: 1x8 Wormhole mesh (16 experts/device)
- **TG Galaxy**: 8x4 Wormhole mesh (4 experts/device, all-to-all fabric)

## Quick Start

```bash
source python_env/bin/activate

# Set model path
export HF_MODEL="/data/models/Solar-Open-100B"

# Run component tests on T3K (1x8)
pytest models/demos/solar_open_100b/tests/test_solar_moe_layer_hw.py -v -k "test_solar_full_moe_pipeline_hw"

# Run full MoE with all-to-all on TG Galaxy (8x4, requires fabric)
pytest models/demos/solar_open_100b/tests/test_solar_moe_layer_hw.py -v -k "test_solar_moe_layer_hw"
```

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Layers | 48 (all MoE) |
| Hidden dim | 4096 |
| Attention heads | 64 Q / 8 KV (GQA) |
| Head dim | 128 |
| Routed experts | 128 |
| Shared experts | 1 |
| Top-k | 8 |
| Expert FFN dim | 1280 |
| Shared FFN dim | 10240 |
| Vocab size | 196608 |
| Routing | Sigmoid + bias correction, normalized, scaled |
| RoPE | YaRN (theta=1M, factor=2.0) |

## Architecture

### MoE Routing (Sigmoid + Bias Top-8)

Unlike standard softmax routing, Solar uses sigmoid activation with a learned bias correction:

1. **Linear projection**: `x @ gate_weight` → [batch, 128] logits
2. **Sigmoid activation**: element-wise sigmoid on logits
3. **Bias correction**: add `e_score_correction_bias` (learned per-expert bias)
4. **Top-8 selection**: select 8 experts with highest biased scores
5. **Normalize**: gather original (unbiased) sigmoid scores for selected experts, divide by sum
6. **Scale**: multiply by `routed_scaling_factor`

Implementation: `tt/moe_gate.py` — `SolarMoEGate`

### Expert Distribution (All-to-All)

On multi-device configurations, the 128 experts are sharded across devices:
- **T3K (1x8)**: 16 experts per device, `cluster_axis=1`
- **TG Galaxy (8x4)**: 4 experts per device, `cluster_axis=0`

The MoE layer uses `ttnn.all_to_all_dispatch` and `ttnn.all_to_all_combine` to route tokens to the devices hosting their assigned experts:

1. Gate selects top-8 experts per token
2. `all_to_all_dispatch` sends tokens to expert-owning devices
3. Batched SwiGLU expert computation on local experts
4. `all_to_all_combine` gathers results back to originating devices
5. Weighted sum of routed outputs + shared expert output

Implementation: `tt/moe_layer.py` — `SolarMoELayer`

### Shared Expert

A single SwiGLU MLP (`tt/shared_expert.py`) applied to every token unconditionally. Its output is added to the weighted routed expert output.

### vLLM Integration

`tt/generator_vllm.py` provides the standard vLLM interface. Since `model_type "solar_open"` is not recognized by the `transformers` library, weights are loaded directly from safetensors files using a custom loader that applies the standard HF-to-meta key conversion pipeline.

```python
# The generator plugs SolarMoELayer into the shared Transformer:
tt_model = Transformer(
    ...,
    moe_class=SolarMoELayer,
)
```

## Infrastructure Changes

This model required three changes to the shared transformer infrastructure:

1. **Pluggable MoE class** (`decoder.py`, `model.py`): Added `moe_class` and `moe_kwargs` parameters, mirroring the existing `attention_class` pattern. When `moe_class=None`, the existing Mixtral MoE is used (backward compatible).

2. **Expert count parsing fix** (`model_config.py`): Replaced single-character extraction (`int(item[-11])`) with regex `re.search(r"experts\.(\d+)\.", item)` to handle expert IDs >= 10.

3. **Partial rotary factor** (`model_config.py`): Added `partial_rotary_factor` and `rotary_dim` to ModelArgs (defaults to 1.0, used by MiniMax-M2.5).

## Testing

### Component Tests (CPU reference comparison)

```bash
# Gate routing accuracy
pytest models/demos/solar_open_100b/tests/test_moe_gate.py -v

# Expert SwiGLU computation
pytest models/demos/solar_open_100b/tests/test_experts.py -v

# Full MoE layer
pytest models/demos/solar_open_100b/tests/test_moe_layer.py -v
```

### Hardware Tests

```bash
# Individual components on T3K (1x8)
pytest models/demos/solar_open_100b/tests/test_solar_moe_layer_hw.py -v -k "test_solar_experts_hw"
pytest models/demos/solar_open_100b/tests/test_solar_moe_layer_hw.py -v -k "test_solar_shared_expert_hw"
pytest models/demos/solar_open_100b/tests/test_solar_moe_layer_hw.py -v -k "test_solar_full_moe_pipeline_hw"

# Gate routing on T3K
pytest models/demos/solar_open_100b/tests/test_solar_moe_hw.py -v

# Full MoE with all-to-all on TG Galaxy (8x4)
pytest models/demos/solar_open_100b/tests/test_solar_moe_layer_hw.py -v -k "test_solar_moe_layer_hw"
```

### Hardware Test Results

| Test | Hardware | Mesh | PCC | Status |
|------|----------|------|-----|--------|
| Gate (sigmoid+bias top-8) | T3K | 1x8 | 0.9992 | PASS |
| Experts (batched SwiGLU) | T3K | 1x8 | 0.9999 | PASS |
| Shared Expert (SwiGLU) | T3K | 1x8 | 0.9999 | PASS |
| Full MoE Pipeline (no fabric) | T3K | 1x8 | 0.9999 | PASS |
| Full MoE Layer (all-to-all) | TG Galaxy | 8x4 + FABRIC_1D | 0.9704 | PASS |

All tests validated against PyTorch reference using Pearson Correlation Coefficient (PCC).

## File Structure

```
models/demos/solar_open_100b/
├── README.md
├── tt/
│   ├── moe_gate.py          # Sigmoid+bias top-8 router
│   ├── moe_layer.py          # All-to-all MoE orchestration + shared expert
│   ├── experts.py             # 128 batched SwiGLU expert MLPs
│   ├── shared_expert.py       # Single shared SwiGLU MLP
│   ├── model_config.py        # Solar-specific config helpers
│   └── generator_vllm.py      # vLLM integration (custom weight loader)
├── utils/
│   └── convert_weights.py     # Weight conversion/sharding utilities
└── tests/
    ├── test_moe_gate.py       # Gate unit test (CPU reference)
    ├── test_experts.py        # Expert unit test (CPU reference)
    ├── test_moe_layer.py      # MoE layer unit test (CPU reference)
    ├── test_solar_moe_hw.py   # Gate hardware test (T3K)
    └── test_solar_moe_layer_hw.py  # Full MoE hardware tests (T3K + TG Galaxy)
```

## Known Issues

### TG Galaxy ETH Training Timeout

On TG Galaxy systems with firmware < 19.6.0, `ttnn.open_mesh_device()` can hang indefinitely due to broken ETH core training on specific cores (observed on eth cores 1,6 and 8,6). Each stuck core times out after 15 minutes (900s), but the system may still hang at "Opening user mode device driver" afterward.

**Resolution**: Upgrade firmware to >= 19.6.0 and update the kernel module (TT-KMD).

**Diagnostic tools**:
- `tests/scripts/galaxy_health_check.py` — standalone Galaxy health check (eltwise + matmul on all devices)
- `tests/scripts/galaxy_eth_repro.py` — minimal reproducer for the ETH training hang
- `pyluwen.detect_chips_fallible()` can verify hardware health without triggering ETH training
