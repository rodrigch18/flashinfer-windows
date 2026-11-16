<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/flashinfer-ai/web-data/blob/main/logo/FlashInfer-black-background.png?raw=true">
    <img alt="FlashInfer" src="https://github.com/flashinfer-ai/web-data/blob/main/logo/FlashInfer-white-background.png?raw=true" width=55%>
  </picture>
</p>
<h1 align="center">
High-Performance GPU Kernels for Inference
</h1>

<p align="center">
| <a href="https://docs.flashinfer.ai"><b>Documentation</b></a> | <a href="https://github.com/flashinfer-ai/flashinfer/releases/latest"><b>Latest Release</b></a> | <a href="https://flashinfer.ai"><b>Blog</b></a> | <a href="https://join.slack.com/t/flashinfer/shared_invite/zt-379wct3hc-D5jR~1ZKQcU00WHsXhgvtA"><b>Slack</b></a> |  <a href="https://github.com/orgs/flashinfer-ai/discussions"><b>Discussion Forum</b></a> |
</p>

[![Build Status](https://ci.tlcpack.ai/job/flashinfer-ci/job/main/badge/icon)](https://ci.tlcpack.ai/job/flashinfer-ci/job/main/)
[![Documentation](https://github.com/flashinfer-ai/flashinfer/actions/workflows/build-doc.yml/badge.svg)](https://github.com/flashinfer-ai/flashinfer/actions/workflows/build-doc.yml)

## FlashInfer for Windows

Native Windows x64 build of FlashInfer (MSVC + CUDA 13), maintained as the companion of **[vLLM for Windows](https://github.com/rodrigch18/vllm-windows/tree/vllm-for-windows)**, which pins the FlashInfer version built here.

### How this fork is organized

- This repository is a fork of the official [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer). `main` tracks the latest upstream `main` and is never modified.
- The `flashinfer-for-windows` branch is an official upstream release tag plus exactly **one** squashed Windows commit, so the whole port can be reviewed in a single diff.
- The Windows changes started from [SystemPanic/flashinfer-windows](https://github.com/SystemPanic/flashinfer-windows) and were audited: only changes needed to build and run on Windows are kept, and Linux behavior is identical to upstream.

### How this fork differs

| | SystemPanic/flashinfer-windows | This fork |
|---|---|---|
| FlashInfer | 0.7.0.post1 (latest release) | **0.7.2.post1**, the version vLLM 0.33 pins |
| Base | FlashInfer source with Windows changes in several commits | **Official upstream release tag + one Windows commit** |

Windows fixes in this fork, in addition to the ported MSVC fixes:

- JIT works without administrator rights or Developer Mode: directory junctions or file copies replace symlinks.
- JIT object file names are shortened so nvcc output paths stay under the Windows 260-character limit.
- Kernels new in 0.7.2 that MSVC rejected are ported (preprocessor directives inside macro arguments, GCC-only lambda attributes in the DeepSeek V4.1 sparse MLA kernels).
- The NCCL-only `dcp_lse_reduce` module is skipped on Windows (NCCL has no official Windows build).
- Release JIT builds define `NDEBUG` again, and compiler flags keep a stable order, so ninja does not rebuild modules for no reason.
- `FLASHINFER_NINJA_KEEP_GOING=1` lets a build report every failing kernel in one run instead of stopping at the first.

### Windows instructions:

#### Installing a release:

Each release of this fork is the source code at its tag (the `flashinfer-for-windows` commit for that FlashInfer version). A release may also attach prebuilt wheels built for one specific environment, listed in its release notes (Python, torch, CUDA and GPU architecture). If they match your environment, install them with `pip install`; otherwise build the wheels from source as described below.

#### Building from source:

##### Pre-requisites

A Visual Studio 2019 or newer is required to launch the compiler x64 environment. The installation path is referred in the instructions as VISUAL_STUDIO_INSTALL_PATH. For example, for Visual Studio 2022 default installation, replace VISUAL_STUDIO_INSTALL_PATH with C:\Program Files\Microsoft Visual Studio\2022\Community

CUDA path will be found automatically if you have the bin folder in your PATH, or have the CUDA installation path settled on well-known environment vars like CUDA_ROOT, CUDA_HOME or CUDA_PATH.

If none of these are present, make sure to set the environment variable before starting the build:
set CUDA_HOME=CUDA_INSTALLATION_PATH

##### IMPORTANT FOR ALL CUDA 13 BUILDS:
CUDA 13 `cuda.h` declares `CUtensorMap` with 128 byte alignment (still the case in CUDA 13.5). MSVC does not support passing over-aligned types like alignas(128) by value as function parameters (`error C2719`), so the header must be patched to use 64 byte alignment under MSVC. Use `fix_cuda_13_align.py` from [vLLM for Windows](https://github.com/rodrigch18/vllm-windows/blob/vllm-for-windows/fix_cuda_13_align.py): in an elevated command line (cmd.exe as Administrator), set `CUDA_HOME` to the CUDA toolkit you build with and run `python fix_cuda_13_align.py` once. Repeat after installing or reinstalling any CUDA 13 toolkit.

##### Instructions

1. Open a Command Line (cmd.exe)
2. Execute ```VISUAL_STUDIO_INSTALL_PATH\VC\Auxiliary\Build\vcvarsall.bat x64```
3. **Clone this repository from the flashinfer-for-windows branch (NOT MAIN): ```cd C:\ & git clone --single-branch --branch flashinfer-for-windows --recurse-submodules https://github.com/rodrigch18/flashinfer-windows.git```**
4. Change the working directory to the cloned repository path, for example: ```cd C:\flashinfer-windows```
5. Set the following environment variables:
```
set DISTUTILS_USE_SDK=1
#(replace 10 with your desired cpu threads to use in parallel to speed up compilation)
set MAX_JOBS=10

#(Optional) To build only against your specific GPU CUDA arch (to speed up compilation),
#replace YOUR_GPU_ARCH with your GPU arch number. For example, for RTX 50XX: set TORCH_CUDA_ARCH_LIST=12.0
set TORCH_CUDA_ARCH_LIST=YOUR_GPU_ARCH
set FLASHINFER_CUDA_ARCH_LIST=YOUR_GPU_ARCH

#If sccache is installed (e.g. via rustup), disable it for the build:
set RUSTC_WRAPPER=
```

6. Build & install:
```
#For JIT wheel:
python -m build --no-isolation --wheel
#Replace FLASHINFERVERSION with the corresponding flashinfer version, for example: 0.7.2.post1
pip install dist\flashinfer_python-FLASHINFERVERSION-py3-none-any.whl

#For cache (AOT) wheel:
#replace YOUR_GPU_ARCH with your GPU arch, using the family-specific "f" suffix on Blackwell. For example, for RTX 50XX: 12.0f
set FLASHINFER_JIT_CACHE_PROVIDER_ARCH=YOUR_GPU_ARCH
set FLASHINFER_JIT_CACHE_PROVIDER_ARCHS=YOUR_GPU_ARCH
#(Optional) keep compiling after a failure to see every failing kernel at once:
set FLASHINFER_NINJA_KEEP_GOING=1
python -m build --no-isolation --wheel flashinfer-jit-cache-provider
python -m build --no-isolation --wheel flashinfer-jit-cache
pip install flashinfer-jit-cache-provider\dist\flashinfer_jit_cache-FLASHINFERVERSION-cp39-abi3-win_amd64.whl flashinfer-jit-cache\dist\flashinfer_jit_cache-FLASHINFERVERSION-cp39-abi3-win_amd64.whl

#For Cubin wheel (platform independent, published by FlashInfer):
pip install flashinfer-cubin==FLASHINFERVERSION --extra-index-url https://flashinfer.ai/whl/
```
flashinfer-python, flashinfer-jit-cache and flashinfer-cubin must have exactly the same version, otherwise the mismatched package is ignored.

7. Build folder cleaning: Due to 260 chars path constraints on Windows, a custom build folder is generated at `C:\_fib` by default, and JIT-compiled kernels are cached under `C:\_fij`. To clean the build folder after wheel generation, remove it manually.

---

**FlashInfer** is a library and kernel generator for inference that delivers state-of-the-art performance across diverse GPU architectures. It provides unified APIs for attention, GEMM, and MoE operations with multiple backend implementations including FlashAttention-2/3, cuDNN, CUTLASS, and TensorRT-LLM.

## Why FlashInfer?

- **State-of-the-art Performance**: Optimized kernels for prefill, decode, and mixed batching scenarios
- **Multiple Backends**: Automatically selects the best backend for your hardware and workload
- **Modern Architecture Support**: Support for SM75 (Turing) and later (through Blackwell)
- **Low-Precision Compute**: FP8 and FP4 quantization for attention, GEMM, and MoE operations
- **Production-Ready**: CUDAGraph and torch.compile compatible for low-latency serving

## Core Features

### Attention Kernels
- **Paged and Ragged KV-Cache**: Efficient memory management for dynamic batch serving
- **Decode, Prefill, and Append**: Optimized kernels for all attention phases
- **MLA Attention**: Native support for DeepSeek's Multi-Latent Attention
- **Cascade Attention**: Memory-efficient hierarchical KV-Cache for shared prefixes
- **Sparse Attention**: Block-sparse and variable block-sparse patterns
- **POD-Attention**: Fused prefill+decode for mixed batching

### GEMM & Linear Operations
- **BF16 GEMM**: BF16 matrix multiplication for SM10.0+ GPUs.
- **FP8 GEMM**: Per-tensor and groupwise scaling
- **FP4 GEMM**: NVFP4 and MXFP4 matrix multiplication for Blackwell GPUs
- **Grouped GEMM**: Efficient batched matrix operations for LoRA and multi-expert routing

### Mixture of Experts (MoE)
- **Fused MoE Kernels**
- **Multiple Routing Methods**: DeepSeek-V3, Llama-4, and standard top-k routing
- **Quantized MoE**: FP8 and FP4 expert weights with block-wise scaling

### Sampling & Decoding
- **Sorting-Free Sampling**: Efficient Top-K, Top-P, and Min-P without sorting
- **Speculative Decoding**: Chain speculative sampling support

### Communication
- **AllReduce**: Custom implementations
- **Multi-Node NVLink**: MNNVL support for multi-node inference
- **NVSHMEM Integration**: For distributed memory operations

### Other Operators
- **RoPE**: LLaMA-style rotary position embeddings (including LLaMA 3.1)
- **Normalization**: RMSNorm, LayerNorm, Gemma-style fused operations
- **Activations**: SiLU, GELU with fused gating

## GPU Support

| Architecture | Compute Capability | Example GPUs |
|--------------|-------------------|------|
| Turing | SM 7.5 | T4, RTX 20 series |
| Ampere | SM 8.0, 8.6 | A100, A10, RTX 30 series |
| Ada Lovelace | SM 8.9 | L4, L40, RTX 40 series |
| Hopper | SM 9.0 | H100, H200 |
| Blackwell | SM 10.0, 10.3 | B200, B300 |
| Blackwell | SM 11.0 | Jetson Thor |
| Blackwell | SM 12.0, 12.1 | RTX 50 series, DGX Spark |

> **Note:** Not all features are supported across all compute capabilities.

## News

Latest: [![GitHub Release](https://img.shields.io/github/v/release/flashinfer-ai/flashinfer)](https://github.com/flashinfer-ai/flashinfer/releases/latest)

Notable updates:
- [2025-10-08] Blackwell support added in [v0.4.0](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.4.0)
- [2025-03-10] [Blog Post](https://flashinfer.ai/2025/03/10/sampling.html) Sorting-Free GPU Kernels for LLM Sampling, which explains the design of sampling kernels in FlashInfer.

## Getting Started

### Installation

**Quickstart:**

```bash
pip install flashinfer-python
```

**Package Options:**

- **flashinfer-python**: Core package that compiles/downloads kernels on first use
- **flashinfer-cubin**: Pre-compiled kernel binaries for all supported GPU architectures
- **flashinfer-jit-cache**: CUDA-specific shim that installs architecture-specific pre-built kernel providers

**For faster initialization and offline usage**, install the optional packages to have most kernels pre-compiled:

```bash
pip install flashinfer-python
flashinfer install-cubin-wheel
flashinfer install-jit-cache-wheel
```

**For Blackwell (SM100+) CuTe DSL kernels**, install with the CUDA 13 extra to enable Blackwell-optimized kernels:

```bash
pip install flashinfer-python[cu13]
```

### Verify Installation

```bash
flashinfer show-config
```

### Basic Usage

```python
import torch
import flashinfer

# Single decode attention
q = torch.randn(32, 128, device="cuda", dtype=torch.float16)  # [num_qo_heads, head_dim]
k = torch.randn(2048, 32, 128, device="cuda", dtype=torch.float16)  # [kv_len, num_kv_heads, head_dim]
v = torch.randn(2048, 32, 128, device="cuda", dtype=torch.float16)

output = flashinfer.single_decode_with_kv_cache(q, k, v)
```

See [documentation](https://docs.flashinfer.ai/) for comprehensive API reference and tutorials.

### Install from Source

```bash
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer
python -m pip install -v .
```

**For development**, install in editable mode:

```bash
python -m pip install --no-build-isolation -e . -v
```

> **Note:** When using `--no-build-isolation`, pip does not automatically install build dependencies. FlashInfer requires `setuptools>=77`. If you encounter an error like `AttributeError: module 'setuptools.build_meta' has no attribute 'prepare_metadata_for_build_editable'`, upgrade pip and setuptools first:
> ```bash
> python -m pip install --upgrade pip setuptools
> ```

Build optional packages:

```bash
# flashinfer-cubin
python -m build --no-isolation --wheel flashinfer-cubin
python -m pip install flashinfer-cubin/dist/*.whl
```

Build one JIT-cache provider for the target GPU, then build a shim that depends
on that provider. The example below builds an SM90a provider; both wheels must
use the same version settings.

```bash
export FLASHINFER_JIT_CACHE_PROVIDER_ARCH=9.0a
python -m build --no-isolation --wheel flashinfer-jit-cache-provider

export FLASHINFER_JIT_CACHE_PROVIDER_ARCHS="9.0a"
python -m build --no-isolation --wheel flashinfer-jit-cache

python -m pip install \
  flashinfer-jit-cache-provider/dist/*.whl \
  flashinfer-jit-cache/dist/*.whl
```

For more details, see the [Install from Source documentation](https://docs.flashinfer.ai/installation.html#install-from-source).

### Nightly Builds

```bash
pip install -U --pre flashinfer-python --index-url https://flashinfer.ai/whl/nightly/ --no-deps
pip install flashinfer-python  # Install dependencies from PyPI
flashinfer install-cubin-wheel --nightly
flashinfer install-jit-cache-wheel --nightly
```

### CLI Tools

FlashInfer provides several CLI commands for configuration, module management, and development:

```bash
# Verify installation and view configuration
flashinfer show-config

# List and inspect modules
flashinfer list-modules
flashinfer module-status

# Manage artifacts and cache
flashinfer download-cubin
flashinfer install-cubin-wheel
flashinfer install-jit-cache-wheel
flashinfer download-kernels
flashinfer clear-cache

# For developers: generate compile_commands.json for IDE integration
flashinfer export-compile-commands [output_path]
```

For complete documentation, see the [CLI reference](https://docs.flashinfer.ai/cli.html).

## API Logging

FlashInfer provides comprehensive API logging for debugging. Enable it using environment variables:

```bash
# Enable logging (levels: 0=off (default), 1=basic, 3=detailed, 5=statistics)
export FLASHINFER_LOGLEVEL=3

# Set log destination (stdout (default), stderr, or file path)
export FLASHINFER_LOGDEST=stdout
```

For detailed information about logging levels, configuration, and advanced features, see [Logging](https://docs.flashinfer.ai/logging.html) in our documentation.

## Custom Attention Variants

Users can customize their own attention variants with additional parameters. For more details, refer to our [JIT examples](https://github.com/flashinfer-ai/flashinfer/blob/main/tests/utils/test_jit_example.py).

## CUDA Support

**Supported CUDA Versions:** 12.9, 13.0, and 13.4 (PyTorch nightly)

> **Note:** FlashInfer strives to follow PyTorch's supported CUDA versions plus the latest CUDA release.
> CUDA 13.4 wheels are built with the preview toolkit and PyTorch nightly; runtime CI currently covers CUDA 12.9 and 13.0.

## Adoption

FlashInfer powers inference in:

- [SGLang](https://github.com/sgl-project/sglang)
- [vLLM](https://github.com/vllm-project/vllm)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [TGI (Text Generation Inference)](https://github.com/huggingface/text-generation-inference)
- [MLC-LLM](https://github.com/mlc-ai/mlc-llm)
- [LightLLM](https://github.com/ModelTC/lightllm)
- [lorax](https://github.com/predibase/lorax)
- [ScaleLLM](https://github.com/vectorch-ai/ScaleLLM)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, pull request guidelines
(PR template, performance numbers for optimizations), and how CI is triggered. Coding
agents pick up the same guidance from [CLAUDE.md](CLAUDE.md) / [AGENTS.md](AGENTS.md);
the review guidance is in [docs/code_review_guidance_human.md](docs/code_review_guidance_human.md).

## Acknowledgement

FlashInfer is inspired by [FlashAttention](https://github.com/dao-AILab/flash-attention/), [vLLM](https://github.com/vllm-project/vllm), [stream-K](https://arxiv.org/abs/2301.03598), [CUTLASS](https://github.com/nvidia/cutlass), and [AITemplate](https://github.com/facebookincubator/AITemplate).

## Citation

If you find FlashInfer helpful in your project or research, please consider citing our [paper](https://arxiv.org/abs/2501.01005):

```bibtex
@article{ye2025flashinfer,
    title = {FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving},
    author = {
      Ye, Zihao and
      Chen, Lequn and
      Lai, Ruihang and
      Lin, Wuwei and
      Zhang, Yineng and
      Wang, Stephanie and
      Chen, Tianqi and
      Kasikci, Baris and
      Grover, Vinod and
      Krishnamurthy, Arvind and
      Ceze, Luis
    },
    journal = {arXiv preprint arXiv:2501.01005},
    year = {2025},
    url = {https://arxiv.org/abs/2501.01005}
}
```
