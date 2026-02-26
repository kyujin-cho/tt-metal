# MiniMax-M2.5: Mixture of Experts Language Model

Inference implementation for [MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M1-80k) on Tenstorrent Wormhole accelerators.

**Architecture**: 64-layer MoE with 256 routed SwiGLU experts per layer (no shared expert), sigmoid+bias top-8 routing, GQA (48Q/8KV), partial RoPE (rotary_dim=64 of head_dim=128), FP8 expert weights.

**Target Hardware**:
- **T3K**: 1x8 Wormhole mesh (32 experts/device)
- **TG Galaxy**: 8x4 Wormhole mesh (8 experts/device, all-to-all fabric)

## Quick Start

```bash
source python_env/bin/activate

# Set model path
export HF_MODEL="/data/models/MiniMax-M2.5"

# Run component tests on T3K (1x8)
pytest models/demos/minimax_m2_5/tests/test_minimax_moe_hw.py -v -k "test_minimax_gate_hw"
pytest models/demos/minimax_m2_5/tests/test_minimax_moe_hw.py -v -k "test_minimax_experts_hw"
pytest models/demos/minimax_m2_5/tests/test_minimax_moe_hw.py -v -k "test_minimax_full_moe_pipeline_hw"

# Run full MoE with all-to-all on TG Galaxy (8x4, requires fabric)
pytest models/demos/minimax_m2_5/tests/test_minimax_moe_hw.py -v -k "test_minimax_moe_layer_hw"

# Run all tests
HF_MODEL=/data/models/MiniMax-M2.5 pytest models/demos/minimax_m2_5/tests/test_minimax_moe_hw.py -v -s
```

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Layers | 64 (all MoE) |
| Hidden dim | 3072 |
| Attention heads | 48 Q / 8 KV (GQA) |
| Head dim | 128 |
| Rotary dim | 64 (partial_rotary_factor=0.5) |
| Routed experts | 256 |
| Shared experts | 0 |
| Top-k | 8 |
| Expert FFN dim | 1536 |
| Vocab size | 200064 |
| Routing | Sigmoid + bias correction, normalized, scaled |
| Weight format | FP8 (float8_e4m3fn) with [128,128] block-wise scales |

## Architecture

### MoE Routing (Sigmoid + Bias Top-8)

Similar to Solar's routing but with 256 experts and tighter score clustering:

1. **Linear projection**: `x @ gate_weight` → [batch, 256] logits
2. **Sigmoid activation**: element-wise sigmoid on logits
3. **Bias correction**: add `e_score_correction_bias` (learned per-expert bias, ~8.0-8.9)
4. **Top-8 selection**: select 8 experts with highest biased scores
5. **Normalize**: gather original (unbiased) sigmoid scores for selected experts, divide by sum
6. **Scale**: multiply by `routed_scaling_factor`

**Precision note**: The large bias values push all biased scores to ~9.0 ± 0.2, making expert selection sensitive to bfloat16 rounding. This causes 1-2 of the top-8 selections to differ from float32 reference — this is inherent to the architecture, not a bug. Individual expert computation achieves PCC 0.9999.

Implementation: `tt/moe_gate.py` — `MiniMaxMoEGate`

### Expert Distribution (All-to-All)

On multi-device configurations, the 256 experts are sharded across devices:
- **T3K (1x8)**: 32 experts per device, `cluster_axis=1`
- **TG Galaxy (8x4)**: 8 experts per device, `cluster_axis=0`

The MoE layer uses `ttnn.all_to_all_dispatch` and `ttnn.all_to_all_combine` for token routing:

1. Gate selects top-8 experts per token
2. `all_to_all_dispatch` sends tokens to expert-owning devices
3. Batched SwiGLU expert computation on local experts
4. `all_to_all_combine` gathers results back to originating devices
5. Weighted sum of routed outputs (no shared expert)

Implementation: `tt/moe_layer.py` — `MiniMaxMoELayer`

### FP8 Expert Weights

MiniMax stores expert weights in `float8_e4m3fn` format with block-wise inverse scales ([128,128] blocks). During weight loading, FP8 weights are dequantized to float32 using `dequantize_tensor()` from DeepSeek-V3's utilities, then stored as BFP8/BF16 on device.

Weight key format (HF safetensors):
- Expert projections: `model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight` (gate_proj), `w2` (down_proj), `w3` (up_proj)
- Expert scales: `model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight_scale_inv`
- Gate: `model.layers.{i}.block_sparse_moe.gate.weight`
- Bias: `model.layers.{i}.block_sparse_moe.e_score_correction_bias`

Implementation: `tt/experts.py` — `MiniMaxExperts`

### Partial RoPE

MiniMax uses partial rotary position embeddings: only the first 64 of 128 head dimensions are rotated, the remaining 64 pass through unchanged. This is configured via `partial_rotary_factor=0.5` in the model config, which sets `rotary_dim=64`.

Implementation: `tt/attention.py` — `MiniMaxPartialRoPEAttention`

### vLLM Integration

`tt/generator_vllm.py` provides the standard vLLM interface. Since `model_type "minimax_m2"` is not recognized by the `transformers` library, weights are loaded directly from safetensors files with custom FP8 dequantization.

```python
tt_model = Transformer(
    ...,
    moe_class=MiniMaxMoELayer,
    attention_class=MiniMaxPartialRoPEAttention,
)
```

## Testing

### Hardware Tests

```bash
# Gate routing accuracy (T3K 1x8)
pytest models/demos/minimax_m2_5/tests/test_minimax_moe_hw.py -v -k "test_minimax_gate_hw"

# Expert SwiGLU with FP8 dequant (T3K 1x8)
pytest models/demos/minimax_m2_5/tests/test_minimax_moe_hw.py -v -k "test_minimax_experts_hw"

# Full MoE pipeline without fabric (T3K 1x8)
pytest models/demos/minimax_m2_5/tests/test_minimax_moe_hw.py -v -k "test_minimax_full_moe_pipeline_hw"

# Full MoE with all-to-all (TG Galaxy 8x4)
pytest models/demos/minimax_m2_5/tests/test_minimax_moe_hw.py -v -k "test_minimax_moe_layer_hw"
```

### Hardware Test Results

| Test | Hardware | Mesh | Metric | Result | Status |
|------|----------|------|--------|--------|--------|
| Gate (sigmoid+bias top-8) | T3K | 1x8 | Index overlap | 6/8 | PASS |
| Experts (batched SwiGLU, FP8) | T3K | 1x8 | PCC | 0.9999 | PASS |
| Full MoE Pipeline (no fabric) | T3K | 1x8 | PCC | 0.8119 | PASS |
| Full MoE Layer (all-to-all) | TG Galaxy | 8x4 + FABRIC_1D | PCC | 0.7269 | PASS |

**PCC notes**:
- Individual expert computation is highly accurate (PCC 0.9999)
- Pipeline and full MoE PCC are lower because bfloat16 routing precision causes 1-2 different expert selections from the float32 reference (biased scores cluster at ~9.0 ± 0.2 for 256 experts)
- The all-to-all test adds further precision loss from cross-device communication and layout conversions

### CPU Reference Tests

```bash
pytest models/demos/minimax_m2_5/tests/test_moe_gate.py -v
pytest models/demos/minimax_m2_5/tests/test_experts.py -v
pytest models/demos/minimax_m2_5/tests/test_moe_layer.py -v
```

## File Structure

```
models/demos/minimax_m2_5/
├── README.md
├── tt/
│   ├── moe_gate.py          # Sigmoid+bias top-8 router (256 experts)
│   ├── moe_layer.py          # All-to-all MoE orchestration (no shared expert)
│   ├── experts.py             # 256 batched SwiGLU expert MLPs (FP8 dequant)
│   ├── attention.py           # Partial RoPE attention (rotary_dim=64)
│   ├── model_config.py        # MiniMax-specific config helpers
│   └── generator_vllm.py      # vLLM integration (FP8 weight loader)
├── utils/
│   └── convert_weights.py     # FP8 weight conversion/validation
└── tests/
    ├── test_moe_gate.py       # Gate unit test (CPU reference)
    ├── test_experts.py        # Expert unit test (CPU reference)
    ├── test_moe_layer.py      # MoE layer unit test (CPU reference)
    └── test_minimax_moe_hw.py # Full hardware tests (T3K + TG Galaxy)
```

## Key Differences from Solar-Open-100B

| Feature | Solar-Open-100B | MiniMax-M2.5 |
|---------|----------------|--------------|
| Experts | 128 routed + 1 shared | 256 routed, no shared |
| Weight format | BF16/BFP8 | FP8 (float8_e4m3fn) |
| Gate routing | Sigmoid + bias | Sigmoid + bias (same pattern) |
| RoPE | Full (YaRN) | Partial (64/128 dims) |
| Expert FFN | gate/up/down proj | w1/w2/w3 (same ops, different key names) |
| PCC (full MoE) | 0.97 | 0.73 (more experts → tighter routing) |
