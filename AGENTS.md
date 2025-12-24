# CLAUDE.md

This file provides guidance to code agents when working with code in this repository.

## Code Style

### General Swift
- Avoid excessive comments. Code should be self-documenting through clear naming.
- Clarity at the point of use is the most important goal. Clarity over brevity.
- Use `let` over `var` whenever possible.
- Avoid force unwrapping (`!`). Use `guard let`, `if let`, or optional chaining.
- Prefer Swift native types (`Array`, `Dictionary`, `String`) over `NS*` types.
- Declare variables close to where they are first used.

### MLX-Specific
- **Lazy evaluation**: MLX operations are lazy. Arrays materialize only when needed.
- **Call `eval()` sparingly**: Evaluate at natural boundaries (e.g., end of denoising step), not after every operation. Each eval has fixed overhead.
- **Avoid implicit evals**: Printing arrays, `.item()`, or using array values in conditionals triggers evaluation and hurts performance.
- **Memory efficiency**: Leverage lazy evaluation for memory savings—loading weights doesn't consume memory until evaluation.
- **Unified memory**: MLX uses Apple Silicon's unified memory. No need to move data between CPU/GPU.

## Build Commands

**Always use xcodebuild** - `swift build` fails because Metal shaders require Xcode's build system.

```bash
# Build CLI (required before running)
xcodebuild -scheme QwenImageCLI -configuration Release -derivedDataPath .build/xcode -destination 'platform=macOS' build

# Run CLI
.build/xcode/Build/Products/Release/QwenImageCLI --help

# Run tests
xcodebuild -scheme QwenImage -configuration Debug -derivedDataPath .build/xcode -destination 'platform=macOS' test
```

## CLI Usage Examples

```bash
# Text-to-image generation
.build/xcode/Build/Products/Release/QwenImageCLI \
  --model mzbac/Qwen-Image-Edit-2509-8bit \
  --prompt "A photo of an astronaut" \
  --steps 20 --seed 42 -o output.png

# Image editing with reference images
.build/xcode/Build/Products/Release/QwenImageCLI \
  --model mzbac/Qwen-Image-Edit-2509-8bit \
  --reference-image input.png \
  --prompt "Edit description" \
  --steps 20 -o edited.png

# Layered image decomposition
.build/xcode/Build/Products/Release/QwenImageCLI \
  --model Qwen/Qwen-Image-Layered \
  --layered-image input.png \
  --layered-layers 2 \
  --layered-resolution 640 \
  --steps 20 --seed 42 -o ./layers/
```

## Architecture Overview

### Pipeline Flow
```
Input (Image/Text) → Tokenizer → Text Encoder → Transformer → VAE Decode → Output Image
```

### Core Components

| Directory | Purpose |
|-----------|---------|
| `Sources/QwenImage/Model/VAE/` | 3D Causal VAE (encode/decode images to/from latent space) |
| `Sources/QwenImage/Model/TextEncoder/` | Qwen2.5-VL text encoder with vision tower |
| `Sources/QwenImage/Model/Transformer/` | Dual-stream transformer (60 blocks, joint text-image attention) |
| `Sources/QwenImage/Pipeline/` | Orchestration: generation parameters, schedulers, pipelines |
| `Sources/QwenImage/Weights/` | SafeTensors loading, HuggingFace snapshot management, quantization |
| `Sources/QwenImageCLI/` | CLI entry point |

### Key Files

| File | Purpose |
|------|---------|
| `Pipeline/QwenImagePipeline.swift` | Main text-to-image and image editing pipeline |
| `Pipeline/QwenImageLayeredGeneration.swift` | Layer decomposition pipeline (QwenLayeredPipeline) |
| `Model/Transformer/QwenLayeredTransformerV2.swift` | Layered transformer with modulation and joint attention |
| `Model/Transformer/QwenLayeredJointAttention.swift` | Joint text-image attention mechanism |
| `Model/Transformer/LatentPacking.swift` | Pack/unpack latents for layer-aware processing |
| `Model/VAE/VAE.swift` | VAE wrapper with normalize/denormalize |
| `Pipeline/Scheduler.swift` | FlowMatchEulerDiscreteScheduler |

### Layered Generation Architecture

The layered pipeline decomposes images into foreground/background layers:

```
Input Image [B, 4, H, W]                    Prompt Text
    ↓ VAE Encode + Normalize                    ↓ Tokenizer (maxLength=256)
Image Latent [B, 1, 16, H/8, W/8]          Token IDs + Attention Mask
    ↓ + Random Noise for layers                 ↓ Text Encoder (Qwen2.5-VL)
Combined [B, L+1, 16, H/8, W/8]            Prompt Embeddings [B, seq, 3584]
    ↓ Pack Latents (2x2 patches)                ↓ Project to transformer dim
Packed [B, (L+1)*(H/16)*(W/16), 64]        Text Embeddings [B, seq, 3072]
    ↓                                           ↓
    └──────────────┬────────────────────────────┘
                   ↓
         Transformer (60 dual-stream blocks)
         - Joint text-image attention with RoPE
         - Text stream conditions image denoising
                   ↓
         Denoised Latents
                   ↓ Unpack + Denormalize
         Layer Latents [B, L+1, 16, H/8, W/8]
                   ↓ VAE Decode (per layer)
         Layer Images [B, L+1, 3, H, W]
```

Text encoding happens once before the denoising loop. For True CFG, both positive and negative prompts are encoded, and the transformer runs twice per step.

### Dual-Stream Transformer Pattern

The transformer processes text and image streams jointly:

```swift
// QwenLayeredTransformerBlockV2 pattern
1. Modulate image/text with adaptive layer norm
2. Joint attention (concatenate text+image → attention → split)
3. Feed-forward networks for each stream
4. Gated residual connections
```

## Dependencies

- **mlx-swift** (0.29.1+): Core ML array operations with Metal acceleration
- **swift-transformers** (0.1.21+): Tokenizer support
- **swift-log**: Logging framework

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `QWEN_IMAGE_LOG_LEVEL` | Log level: debug, info, warning, error (default: info) |
| `QWEN_IMAGE_LOG_FILE` | Optional log file path |
| `HF_HOME` | HuggingFace cache directory |

## Memory Considerations

- GPU cache limit is set to 2GB by default
- Text encoder + transformer requires ~34GB for 8bit layered generation
- Use quantized models (8-bit) for lower memory: `mzbac/Qwen-Image-Edit-2509-8bit`
