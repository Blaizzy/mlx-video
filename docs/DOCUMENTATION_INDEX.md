# Wan2.2 Model Documentation Index

## Overview
Complete exploration of the Wan2.2 video diffusion model implementation in MLX for Apple Silicon. Three comprehensive documentation files have been created covering architecture, implementation, and references.

---

## 📄 Documentation Files

### 1. **WAN22_EXPLORATION_SUMMARY.md** (36 KB, 929 lines)
**Comprehensive technical deep-dive covering all aspects of Wan2.2 architecture and implementation.**

Sections:
- How Wan2.2 works (architecture overview, dual-model design)
- Denoising loop structure (timestep handling, batch processing, pre-computation)
- Transformer forward pass (block structure, modulation, self/cross-attention, FFN)
- Caching & acceleration mechanisms:
  - TeaCache (polynomial-based block skipping, 2-3x speedup)
  - Spectrum (Chebyshev feature prediction, 3.5-5x speedup)
  - Memory optimizations
- Model loading & configuration (config presets, weight conversion, quantization)
- Generate scripts (pipeline stages, generation commands)
- File structure (complete list with line counts)
- Architecture comparison table

**Best for**: Understanding the complete technical architecture, learning implementation details, debugging issues.

---

### 2. **WAN22_QUICK_REFERENCE.md** (11 KB, ~300 lines)
**Quick lookup guide for common tasks and key concepts.**

Sections:
- Core architecture at a glance (visual ASCII diagram)
- Key files to understand (quick navigation table)
- Denoising loop skeleton (pseudocode)
- Transformer block forward pass (code structure)
- 3-way factorized RoPE (explanation)
- TeaCache acceleration (3 lines of how it works)
- Spectrum acceleration (3 lines of how it works)
- Pre-computation checklist
- Configuration quick lookup (all model variants)
- Generation command cheat sheet (copy-paste examples)
- Weight conversion
- MoE architecture (clarification: NO MoE in Wan2.2)

**Best for**: Quick lookup, command examples, refreshing memory on how specific components work.

---

### 3. **WAN22_ARCHITECTURE_DIAGRAMS.md** (38 KB, ~845 lines)
**Detailed visual diagrams and data flow illustrations.**

Sections:
1. **High-Level Data Flow** (ASCII diagram)
   - Full generation pipeline (encoding → denoising → decoding)
   - Stage 1: Text/noise encoding
   - Stage 2: Denoising loop with model selection
   - Stage 3: VAE decoding

2. **Transformer Block Internals** (detailed ASCII)
   - Modulation vector extraction (6 vectors)
   - Self-attention with modulation
   - Cross-attention with text
   - FFN with modulation

3. **3-Way Factorized RoPE** (implementation details)
   - Frequency component splitting
   - Per-position rotation

4. **Cross-Attention K/V Caching** (optimization explanation)
   - Comparison: uncached vs cached
   - Storage requirements
   - Per-step computation savings

5. **TeaCache vs Spectrum vs No Caching** (comparison table)
   - Side-by-side speedup/quality metrics
   - Implementation complexity

6. **Latent Space Patchification** (step-by-step)
   - Input/output shapes
   - Reshape operations
   - Unpatchify (reverse operation)

7. **Frame Alignment & Extra Frames** (boundary handling)
   - Temporal frame computation
   - Spatial alignment requirements
   - Example walkthrough

8. **Memory Hierarchy** (resource planning)
   - Model weights breakdown
   - Temporary buffers
   - Pre-computed caches
   - Peak memory estimates
   - Memory timeline through phases

**Best for**: Visual learners, architecture design, resource planning, implementation references.

---

## 🗂️ File Locations

All files are in the project root:
```
/Users/daniel/Projects/mlx-video/
├── WAN22_EXPLORATION_SUMMARY.md      ← Deep technical details
├── WAN22_QUICK_REFERENCE.md          ← Quick lookup & examples
├── WAN22_ARCHITECTURE_DIAGRAMS.md    ← Visual diagrams & data flow
└── DOCUMENTATION_INDEX.md            ← This file
```

---

## 🎯 Key Findings Summary

### Architecture
- **Dual-Model**: Wan2.2 uses separate high-noise and low-noise transformers (boundary=875/1000 timesteps)
- **40 Transformer Blocks**: Each with self-attn, cross-attn, FFN, all modulated by 6 time-conditioned vectors
- **3-Way Factorized RoPE**: Temporal, height, and width frequency components kept separate
- **Learned Modulation**: DiT-style per-block time conditioning (not per-layer-pair)

### Denoising Loop
- **40 Diffusion Steps**: Flow matching with Euler/DPM++/UniPC schedulers
- **Classifier-Free Guidance**: CFG batch (B=2) or disabled (B=1, 2x faster)
- **Pre-computation**: Text embeddings, cross-attn K/V caches, RoPE computed once per generation
- **Model Switching**: Select transformer based on noise level for adaptive coarse-to-fine generation

### Acceleration
- **TeaCache**: Polynomial-based block skipping (~2-3x speedup)
  - Monitors time embedding distance
  - Pre-profiled coefficients per model
  - Threshold-based skip decision
  
- **Spectrum**: Chebyshev polynomial feature prediction (~3.5-5x speedup)
  - Warmup: 5 steps always compute
  - Fit: Chebyshev polynomials to cached features
  - Predict: Use fitted model + Taylor extrapolation
  - Adaptive windowing: Grows confidence over time

### No MoE
- **Important**: Wan2.2 uses standard **gated FFN**, NOT Mixture of Experts
- FFN: [5120 → 13824 → 5120] with GELU activation
- No expert routing or sparse computation

### Caching
- **Cross-attention K/V**: Pre-computed once (40 blocks × 2 models)
  - Eliminates 80 linear projections across all 40 denoising steps
  - Storage: ~50 MB, saves ~30% of transformer compute
  
- **RoPE Frequencies**: Pre-computed for constant grid sizes
  - Eliminates per-step broadcast/concat
  
- **Text Embeddings**: Pre-computed once
  - T5 encoder runs once, freed immediately
  - Reused 40 times across denoising

### Configuration Variants
```
Wan2.2 T2V 14B   (default, dual-model)    : dim=5120, heads=40, 40 layers
Wan2.2 I2V 14B   (image-to-video)         : in_dim=36, dual-model
Wan2.2 TI2V 5B   (text+image, single)     : dim=3072, heads=24, 30 layers
Wan2.1 T2V 14B   (backward compat)        : dim=5120, single-model
Wan2.1 T2V 1.3B  (smaller)                : dim=1536, 30 layers, single-model
```

---

## 🔍 Code Navigation

### Main Entry Points
- **Generation**: `mlx_video/generate_wan.py` (38 KB)
  - Main `generate_video()` function
  - Denoising loop (lines 589-693)
  
- **Model**: `mlx_video/models/wan/model.py` (518 lines)
  - `WanModel` class (main forward pass)
  - `TeaCacheState` (lines 15-59)
  - `_patchify()` and `unpatchify()` methods

### Architecture Components
- **Transformer Blocks**: `mlx_video/models/wan/transformer.py` (96 lines)
  - `WanAttentionBlock` (self-attn + cross-attn + FFN)
  - `WanFFN` (gated feed-forward)
  
- **Attention**: `mlx_video/models/wan/attention.py` (207 lines)
  - `WanSelfAttention` (QK norm + 3-way RoPE)
  - `WanCrossAttention` (pre-cached K/V support)
  
- **RoPE**: `mlx_video/models/wan/rope.py` (178 lines)
  - 3-way factorized positional encoding
  - Pre-computation utilities

### Acceleration
- **TeaCache**: Lines 450-507 in `model.py`
  - Block skipping logic
  - Residual reuse
  
- **Spectrum**: `mlx_video/models/wan/spectrum.py` (288 lines)
  - `ChebyshevForecaster` (polynomial fitting)
  - `SpectrumForecaster` (blended prediction)
  - `SpectrumState` (scheduling)

### Support
- **Configuration**: `mlx_video/models/wan/config.py` (157 lines)
  - Model presets for T2V, I2V, TI2V variants
  
- **Loading**: `mlx_video/models/wan/loading.py` (183 lines)
  - Model loading with quantization support
  - T5 encoder initialization
  - Text encoding utilities
  
- **Conversion**: `mlx_video/convert_wan.py` (27 KB)
  - PyTorch → MLX weight conversion
  - Quantization configuration
  - LoRA merging
  
- **Schedulers**: `mlx_video/models/wan/scheduler.py` (428 lines)
  - Flow matching: Euler, DPM++, UniPC

---

## 📊 At a Glance

| Aspect | Details |
|--------|---------|
| **Model Type** | Diffusion Transformer (DiT-style) |
| **Conditioning** | Text (T5), Time (sinusoidal), Image (optional) |
| **Architecture** | 40 blocks × 40 heads, dim=5120 |
| **Patch Size** | (1, 2, 2) temporal/height/width |
| **Dual Model** | Yes (high-noise, low-noise) |
| **Attention Types** | Self-attn (3-way RoPE), Cross-attn (text) |
| **FFN Type** | Standard gated (no MoE) |
| **Time Modulation** | 6 learned vectors per block |
| **Diffusion Steps** | 40 (customizable) |
| **Schedulers** | Euler, DPM++, UniPC |
| **CFG Support** | Yes (batch or disabled for 2x speedup) |
| **TeaCache** | 2-3x speedup with threshold |
| **Spectrum** | 3.5-5x speedup with adaptive windowing |
| **Quantization** | 4-bit or 8-bit on transformer blocks |
| **LoRA Support** | Yes (runtime or weight-merge) |
| **VAE** | Wan2.2: 48-dim latent (4×16×16), Wan2.1: 16-dim (4×8×8) |
| **Peak Memory** | ~74 GB (quantized: ~35 GB, with tiling: ~16 GB) |

---

## 🎓 Learning Path

**For beginners:**
1. Start with **Quick Reference** (architecture overview)
2. Read **Architecture Diagrams** (visual understanding)
3. Reference specific details in **Exploration Summary** as needed

**For implementers:**
1. **Exploration Summary** (complete context)
2. **Architecture Diagrams** (data flow verification)
3. Code navigation in **Quick Reference**

**For optimizers:**
1. Memory Hierarchy (Architecture Diagrams section 8)
2. TeaCache/Spectrum (Quick Reference sections 6-7)
3. Pre-computation (Quick Reference section 8)
4. Acceleration mechanisms (Exploration Summary section 4)

---

## 🔗 Cross-References

### Within Documentation
- Line numbers: Use "Ctrl+G" (VS Code) to jump to specific lines
- Search: Use "Ctrl+F" to find terms across files
- Sections: Use markdown outline navigation

### To Source Code
- Files referenced in documentation have absolute paths
- All paths start with `/Users/daniel/Projects/mlx-video/`
- Use symbolic links or environment variables for portability

### External References
- **Paper**: Spectrum (CVPR 2026, Han et al.)
- **Reference**: Wan2.2 official implementation (HuggingFace Diffusers)
- **Framework**: MLX (Apple Machine Learning Framework)

---

## ✅ Completeness Checklist

- [x] How Wan2.2 model works (architecture overview)
- [x] Denoising loop structure (complete walkthrough)
- [x] Transformer forward pass (all components)
- [x] 3-way factorized RoPE (detailed explanation)
- [x] TeaCache acceleration (implementation & theory)
- [x] Spectrum acceleration (Chebyshev + Taylor)
- [x] Model loading & configuration (all variants)
- [x] Generate scripts (pipeline stages)
- [x] MoE clarification (NO MoE present)
- [x] Pre-computation strategies (caching)
- [x] Memory hierarchy (resource planning)
- [x] File structure (complete navigation)
- [x] Visual diagrams (data flow)
- [x] Quick reference (command examples)
- [x] Cross-attention K/V caching
- [x] Frame alignment & VAE handling
- [x] Quantization & LoRA support

---

## 📝 Notes

- All documentation was created from source code analysis (no external copying)
- Line numbers and file paths are current as of March 6, 2025
- Code examples are pseudocode/simplified for clarity (refer to source for exact implementation)
- Performance metrics (speedup, memory) are approximate and system-dependent

---

Created: March 6, 2025
Project: mlx-video
Format: Markdown
Total Documentation: ~3 files, ~75 KB, ~2000 lines
