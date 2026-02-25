# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TT-Metal is Tenstorrent's open-source framework for AI accelerator programming. It has two layers:
- **TT-Metalium** (`tt_metal/`): Low-level C++/RISC-V runtime — device APIs, allocators, kernels
- **TT-NN** (`ttnn/`): High-level Python/C++ neural network ops (matmul, attention, RoPE, etc.)
- **Models** (`models/`): Production model implementations (Llama, Qwen, DeepSeek-V3, Mixtral)
- **Tools** (`tools/`): Debugging and scaleout utilities
- **TT-Train** (`tt-train/`): Training library built on TT-NN

Hardware targets: Wormhole, Blackhole, Galaxy (multi-chip) clusters.

## Build Commands

```bash
# Full build (C++ and Python)
./build_metal.sh --build-all

# Build with tests
./build_metal.sh --build-tests

# Build types
./build_metal.sh --release          # Release
./build_metal.sh --development      # RelWithDebInfo (default)
./build_metal.sh --debug            # Debug with symbols

# Build with ccache (faster rebuilds)
./build_metal.sh --build-all --enable-ccache

# Python virtual environment setup
./create_venv.sh
source python_env/bin/activate
```

## Testing

```bash
# Post-commit regressions (must pass before any commit)
./tests/scripts/run_python_api_unit_tests.sh
./tests/scripts/run_cpp_unit_tests.sh

# Run specific Python test
pytest tests/path/to/test_file.py -vvv

# Run specific test function
pytest tests/path/to/test_file.py::test_function_name -vvv

# C++ gtest with filter
./build/test/tt_metal/unit_tests_api --gtest_filter="FixtureName.TestName"

# Slow dispatch mode for C++ tests
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests/unit_tests_api --gtest_filter="TestName"

# Model performance tests
pytest models/ -m models_performance_bare_metal
pytest models/ -m models_performance_virtual_machine
```

Pytest config: 300s timeout, `-vvs -rA` defaults. Key markers: `post_commit`, `slow`, `models_performance_bare_metal`, `model_perf_t3000`, `model_perf_tg`.

## Linting & Formatting

```bash
# C++ static analysis
cmake --preset clang-tidy
cmake --build --preset clang-tidy

# Pre-commit hooks (Black, isort, clang-format, custom validators)
pre-commit install
pre-commit run --all-files
```

- **Python**: Black (120 char), isort (black profile), ruff (120 char)
- **C++**: clang-format (Google-based, 120 char), clang-tidy (bugprone, performance, modernize, readability, cppcoreguidelines)
- **CMake**: gersemi formatting

## Architecture

### Model Execution: Three-Stage Pipeline
1. **`convert_weights()`**: HuggingFace weights → TTNN tensor files
2. **`model_config()`**: HF config → operator configs (memory layout, compute kernel, precision)
3. **`create_state()` + `forward()`**: Load tensors to device, execute forward pass

### Shared Transformer Infrastructure (`models/tt_transformers/tt/`)
- `attention.py` — GQA attention with auto-detected QK norm, paged KV cache, RoPE
- `mlp.py` — Standard gate/up → SiLU → down MLP with device sharding
- `decoder.py` — TransformerBlock composing attention + MLP/MoE
- `model.py` — Full model (embedding + decoder stack + LM head), pluggable `rope_setup_class`
- `model_config.py` — ModelArgs, per-layer precision control (TensorGroup, PrecisionSetting)
- `generator_vllm.py` — vLLM integration base (Llama, Qwen, Mistral, Gemma3 subclasses)
- `mixtral_moe.py` — Simple MoE reference (8 experts, top-2, softmax routing)
- `load_checkpoints.py` — HF weight conversion, QKV splitting, key remapping

### MoE Models (`models/demos/`)
- `deepseek_v3/tt/` — Complex MoE: 256 experts, sigmoid+group routing, all-to-all dispatch/combine, FP8 dequantization
- `gpt_oss/tt/` — Modular MoE: generic ExpertConfig, TopKRouter, throughput/standard expert modes

### Device Mesh
- T3K: 8 devices (medium models)
- Galaxy (TG): 32 devices (large models like DeepSeek-V3)
- `num_devices == 32` triggers TG-specific code paths
- Expert distribution: `n_experts / num_devices = experts_per_device`

### vLLM Integration
Every model must provide a generator class implementing: `initialize_vllm_model()`, `allocate_kv_cache()`, `prefill_forward()`, `decode_forward()`.

## Code Standards

- C++20 codebase — prefer compile-time safety, clarity, simplicity over cleverness
- Avoid macros (use templates/constexpr), SFINAE/enable_if, recursive template instantiations
- Every source file needs an SPDX license header (Apache-2.0)
- Python logging: Loguru. C++ logging: Tenstorrent logger (`TT_LOGGER_LEVEL=Debug`)
- Keep compile times in mind — avoid unnecessary includes, prefer forward declarations
- Respect clang-tidy profile; when suggesting changes, cite the relevant rule

## Debugging

```bash
# Enable watcher for device hang debugging
TT_METAL_WATCHER=10 ./your_program

# Debug build for GDB
CONFIG=Debug ./build_metal.sh
gdb --args ./build/test/tt_metal/test_binary

# Kernel debug prints
TT_METAL_DPRINT_CORES=(0,0)-(4,4) ./your_program

# Board reset
tt-smi -r 0          # single-card
tt-smi -r 0,1,2,3    # T3000
```

## CI/CD

- All post-commit regressions must pass before merging
- Use `[skip ci]` prefix in commit message for documentation-only changes
- Platform-specific workflows exist for T3000, Galaxy, Blackhole, single-card
- Squash and merge is the standard merge strategy
- Linear history enforced (no merge commits on main)
