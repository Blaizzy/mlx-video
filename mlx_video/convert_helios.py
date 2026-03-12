"""Weight conversion for Helios models (HuggingFace diffusers → MLX)."""

import gc
import json
import logging
import math
from pathlib import Path
from typing import Dict

import mlx.core as mx
import mlx.utils

logger = logging.getLogger(__name__)


def sanitize_helios_transformer_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Convert HuggingFace Helios diffusers weight keys to MLX structure.

    HF diffusers keys → MLX keys:
        patch_embedding (Conv3d)                → patch_embedding (Linear, reshaped)
        patch_short/mid/long (Conv3d)           → patch_short/mid/long (Linear, reshaped)
        condition_embedder.time_embedder.linear_1/2 → time_embedding_0/1
        condition_embedder.time_proj            → time_projection
        condition_embedder.text_embedder.linear_1/2 → text_embedding_0/1
        blocks.{i}.attn1.to_q/k/v              → blocks.{i}.self_attn.q/k/v
        blocks.{i}.attn1.to_out.0              → blocks.{i}.self_attn.o
        blocks.{i}.attn1.norm_q/k              → blocks.{i}.self_attn.norm_q/k
        blocks.{i}.attn2.to_q/k/v              → blocks.{i}.cross_attn.q/k/v
        blocks.{i}.attn2.to_out.0              → blocks.{i}.cross_attn.o
        blocks.{i}.attn2.norm_q/k              → blocks.{i}.cross_attn.norm_q/k
        blocks.{i}.ffn.net.0.proj              → blocks.{i}.ffn.fc1
        blocks.{i}.ffn.net.2                   → blocks.{i}.ffn.fc2
        blocks.{i}.norm1/2/3                   → blocks.{i}.norm1/2/3
        blocks.{i}.scale_shift_table           → blocks.{i}.scale_shift_table
        norm_out.norm                           → output_norm
        norm_out.scale_shift_table             → output_norm_table
        proj_out                               → proj_out
    """
    sanitized = {}
    consumed = set()

    for key, value in weights.items():
        new_key = key

        # Conv3d patch embeddings: [O, I, D, H, W] → Linear [O, I*D*H*W]
        if key in ("patch_embedding.weight", "patch_short.weight",
                    "patch_mid.weight", "patch_long.weight"):
            if value.ndim == 5:
                value = value.reshape(value.shape[0], -1)
            sanitized[key] = value
            consumed.add(key)
            continue
        if key in ("patch_embedding.bias", "patch_short.bias",
                    "patch_mid.bias", "patch_long.bias"):
            sanitized[key] = value
            consumed.add(key)
            continue

        # condition_embedder.time_embedder → time_embedding
        if key.startswith("condition_embedder.time_embedder.linear_1."):
            suffix = key.split("condition_embedder.time_embedder.linear_1.")[-1]
            new_key = f"time_embedding_0.{suffix}"
            sanitized[new_key] = value
            consumed.add(key)
            continue
        if key.startswith("condition_embedder.time_embedder.linear_2."):
            suffix = key.split("condition_embedder.time_embedder.linear_2.")[-1]
            new_key = f"time_embedding_1.{suffix}"
            sanitized[new_key] = value
            consumed.add(key)
            continue

        # condition_embedder.time_proj → time_projection
        if key.startswith("condition_embedder.time_proj."):
            suffix = key.split("condition_embedder.time_proj.")[-1]
            new_key = f"time_projection.{suffix}"
            sanitized[new_key] = value
            consumed.add(key)
            continue

        # condition_embedder.text_embedder → text_embedding
        if key.startswith("condition_embedder.text_embedder.linear_1."):
            suffix = key.split("condition_embedder.text_embedder.linear_1.")[-1]
            new_key = f"text_embedding_0.{suffix}"
            sanitized[new_key] = value
            consumed.add(key)
            continue
        if key.startswith("condition_embedder.text_embedder.linear_2."):
            suffix = key.split("condition_embedder.text_embedder.linear_2.")[-1]
            new_key = f"text_embedding_1.{suffix}"
            sanitized[new_key] = value
            consumed.add(key)
            continue

        # norm_out.norm → output_norm
        if key.startswith("norm_out.norm."):
            suffix = key.split("norm_out.norm.")[-1]
            new_key = f"output_norm.{suffix}"
            sanitized[new_key] = value
            consumed.add(key)
            continue

        # norm_out.scale_shift_table → output_norm_table
        if key == "norm_out.scale_shift_table":
            sanitized["output_norm_table"] = value
            consumed.add(key)
            continue

        # Attention: attn1 → self_attn, attn2 → cross_attn
        new_key = new_key.replace(".attn1.", ".self_attn.")
        new_key = new_key.replace(".attn2.", ".cross_attn.")

        # to_q/k/v → q/k/v, to_out.0 → o
        new_key = new_key.replace(".to_q.", ".q.")
        new_key = new_key.replace(".to_k.", ".k.")
        new_key = new_key.replace(".to_v.", ".v.")
        new_key = new_key.replace(".to_out.0.", ".o.")

        # FFN: net.0.proj → fc1, net.2 → fc2
        new_key = new_key.replace(".ffn.net.0.proj.", ".ffn.fc1.")
        new_key = new_key.replace(".ffn.net.2.", ".ffn.fc2.")

        # Skip timesteps_proj (no trainable weights) and dropout layers
        if "timesteps_proj" in key or "to_out.1." in key:
            consumed.add(key)
            continue

        # Skip rope buffers (computed in model)
        if key.startswith("rope."):
            consumed.add(key)
            continue

        # Skip unused keys
        if "norm_added_q" in key or "norm_added_k" in key:
            consumed.add(key)
            continue
        if "add_k_proj" in key or "add_v_proj" in key:
            consumed.add(key)
            continue

        sanitized[new_key] = value
        consumed.add(key)

    unconsumed = set(weights.keys()) - consumed
    if unconsumed:
        logger.warning("Unconsumed transformer weight keys: %s", sorted(unconsumed))

    return sanitized


def _quantize_predicate(path: str, module) -> bool:
    """Return True for layers that should be quantized.

    Targets heavyweight Linear layers in attention and FFN blocks.
    Skips embeddings, norms, head, and modulation (small, precision-sensitive).
    """
    if not hasattr(module, "to_quantized"):
        return False
    quantize_patterns = (
        ".self_attn.q", ".self_attn.k", ".self_attn.v", ".self_attn.o",
        ".cross_attn.q", ".cross_attn.k", ".cross_attn.v", ".cross_attn.o",
        ".ffn.fc1", ".ffn.fc2",
    )
    return any(path.endswith(p) for p in quantize_patterns)


def sanitize_helios_t5_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Convert HuggingFace UMT5 encoder weight keys to MLX T5Encoder structure.

    HF UMT5 keys → MLX keys:
        shared.weight / encoder.embed_tokens.weight → token_embedding.weight
        encoder.final_layer_norm.weight              → norm.weight
        encoder.block.{i}.layer.0.SelfAttention.q.weight → blocks.{i}.attn.q.weight
        encoder.block.{i}.layer.0.SelfAttention.k.weight → blocks.{i}.attn.k.weight
        encoder.block.{i}.layer.0.SelfAttention.v.weight → blocks.{i}.attn.v.weight
        encoder.block.{i}.layer.0.SelfAttention.o.weight → blocks.{i}.attn.o.weight
        encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight
            → blocks.{i}.pos_embedding.embedding.weight
        encoder.block.{i}.layer.0.layer_norm.weight → blocks.{i}.norm1.weight
        encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight → blocks.{i}.ffn.gate_proj.weight
        encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight → blocks.{i}.ffn.fc1.weight
        encoder.block.{i}.layer.1.DenseReluDense.wo.weight   → blocks.{i}.ffn.fc2.weight
        encoder.block.{i}.layer.1.layer_norm.weight → blocks.{i}.norm2.weight
    """
    sanitized = {}

    for key, value in weights.items():
        # Token embedding
        if key in ("shared.weight", "encoder.embed_tokens.weight"):
            sanitized["token_embedding.weight"] = value
            continue

        # Final layer norm
        if key == "encoder.final_layer_norm.weight":
            sanitized["norm.weight"] = value
            continue

        # Skip decoder keys if present
        if key.startswith("decoder.") or key.startswith("lm_head."):
            continue

        # Block-level mappings: encoder.block.{i}.layer.{j}.*
        if key.startswith("encoder.block."):
            new_key = key
            # encoder.block.{i} → blocks.{i}
            new_key = new_key.replace("encoder.block.", "blocks.")

            # Self-attention (layer.0.SelfAttention)
            new_key = new_key.replace(".layer.0.SelfAttention.", ".attn.")
            new_key = new_key.replace(
                ".attn.relative_attention_bias.weight",
                ".pos_embedding.embedding.weight",
            )

            # Layer norms
            new_key = new_key.replace(".layer.0.layer_norm.", ".norm1.")
            new_key = new_key.replace(".layer.1.layer_norm.", ".norm2.")

            # FFN (layer.1.DenseReluDense)
            new_key = new_key.replace(".layer.1.DenseReluDense.wi_0.", ".ffn.gate_proj.")
            new_key = new_key.replace(".layer.1.DenseReluDense.wi_1.", ".ffn.fc1.")
            new_key = new_key.replace(".layer.1.DenseReluDense.wo.", ".ffn.fc2.")

            sanitized[new_key] = value
            continue

        logger.warning("Unhandled T5 key: %s", key)

    return sanitized


def sanitize_helios_vae_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Convert HF diffusers AutoencoderKLWan keys to MLX WanVAE keys.

    Handles key renaming and Conv3d/Conv2d weight transpositions.

    Key mapping:
        post_quant_conv → conv2
        quant_conv → conv1
        decoder.conv_in → decoder.conv1
        decoder.conv_out → decoder.head.2
        decoder.norm_out → decoder.head.0
        decoder.mid_block.resnets.{i} → decoder.middle.{i*2}
        decoder.mid_block.attentions.0 → decoder.middle.1
        decoder.up_blocks.{b}.resnets.{r} → decoder.upsamples.{flat}
        decoder.up_blocks.{b}.upsamplers.0 → decoder.upsamples.{flat}
        Within resnets: norm1→residual.0, conv1→residual.2,
                        norm2→residual.3, conv2→residual.6,
                        conv_shortcut→shortcut
        encoder.* keys are skipped (decoder-only loading)
    """
    # The WanVAE decoder.upsamples is a flat Sequential.
    # For dim_mult=[1,2,4,4], num_res_blocks=2 (→3 resnets/block):
    #   up_block 0: upsamples 0-2 (resnets) + 3 (upsampler with time_conv)
    #   up_block 1: upsamples 4-6 (resnets) + 7 (upsampler with time_conv)
    #   up_block 2: upsamples 8-10 (resnets) + 11 (upsampler, no time_conv)
    #   up_block 3: upsamples 12-14 (resnets only, no upsampler)
    block_offsets = {0: 0, 1: 4, 2: 8, 3: 12}
    upsampler_offsets = {0: 3, 1: 7, 2: 11}

    # Resnet sub-key mapping
    resnet_map = {
        "norm1": "residual.0",
        "conv1": "residual.2",
        "norm2": "residual.3",
        "conv2": "residual.6",
        "conv_shortcut": "shortcut",
    }

    sanitized = {}
    latents_mean = None
    latents_std = None

    for key, value in weights.items():
        # Transpose Conv3d [O, I, D, H, W] → MLX [O, D, H, W, I]
        if "weight" in key and value.ndim == 5:
            value = mx.transpose(value, (0, 2, 3, 4, 1))
        # Transpose Conv2d [O, I, H, W] → MLX [O, H, W, I]
        if "weight" in key and value.ndim == 4:
            value = mx.transpose(value, (0, 2, 3, 1))

        # Skip encoder keys
        if key.startswith("encoder."):
            continue

        # Top-level convolutions
        if key.startswith("post_quant_conv."):
            new_key = key.replace("post_quant_conv.", "conv2.")
            sanitized[new_key] = value
            continue
        if key.startswith("quant_conv."):
            new_key = key.replace("quant_conv.", "conv1.")
            sanitized[new_key] = value
            continue

        # Decoder conv_in → conv1
        if key.startswith("decoder.conv_in."):
            new_key = key.replace("decoder.conv_in.", "decoder.conv1.")
            sanitized[new_key] = value
            continue

        # Decoder conv_out → head.2
        if key.startswith("decoder.conv_out."):
            new_key = key.replace("decoder.conv_out.", "decoder.head.2.")
            sanitized[new_key] = value
            continue

        # Decoder norm_out → head.0
        if key.startswith("decoder.norm_out."):
            new_key = key.replace("decoder.norm_out.", "decoder.head.0.")
            sanitized[new_key] = value
            continue

        # Mid block resnets
        if key.startswith("decoder.mid_block.resnets."):
            rest = key[len("decoder.mid_block.resnets."):]
            # rest = "{i}.{subkey}"
            parts = rest.split(".", 1)
            resnet_idx = int(parts[0])
            subkey = parts[1]  # e.g. "norm1.gamma" or "conv1.weight"
            mid_idx = resnet_idx * 2  # resnets 0→middle.0, 1→middle.2

            sub_prefix = subkey.split(".")[0]
            if sub_prefix in resnet_map:
                mapped = resnet_map[sub_prefix]
                sub_rest = subkey[len(sub_prefix):]
                new_key = f"decoder.middle.{mid_idx}.{mapped}{sub_rest}"
            else:
                new_key = f"decoder.middle.{mid_idx}.{subkey}"
            sanitized[new_key] = value
            continue

        # Mid block attention
        if key.startswith("decoder.mid_block.attentions.0."):
            rest = key[len("decoder.mid_block.attentions.0."):]
            new_key = f"decoder.middle.1.{rest}"
            sanitized[new_key] = value
            continue

        # Up blocks
        if key.startswith("decoder.up_blocks."):
            rest = key[len("decoder.up_blocks."):]
            # Parse block index
            parts = rest.split(".", 1)
            block_idx = int(parts[0])
            sub = parts[1]

            if sub.startswith("resnets."):
                resnet_rest = sub[len("resnets."):]
                rparts = resnet_rest.split(".", 1)
                resnet_idx = int(rparts[0])
                subkey = rparts[1]
                flat_idx = block_offsets[block_idx] + resnet_idx

                sub_prefix = subkey.split(".")[0]
                if sub_prefix in resnet_map:
                    mapped = resnet_map[sub_prefix]
                    sub_rest = subkey[len(sub_prefix):]
                    new_key = f"decoder.upsamples.{flat_idx}.{mapped}{sub_rest}"
                else:
                    new_key = f"decoder.upsamples.{flat_idx}.{subkey}"
                sanitized[new_key] = value
                continue

            if sub.startswith("upsamplers.0."):
                upsampler_rest = sub[len("upsamplers.0."):]
                flat_idx = upsampler_offsets[block_idx]
                new_key = f"decoder.upsamples.{flat_idx}.{upsampler_rest}"
                sanitized[new_key] = value
                continue

        logger.debug("Skipped VAE key: %s", key)

    # Add latent statistics as buffers
    # WanVAE expects 'mean', 'std', 'inv_std' as registered buffers
    # Read from config if available, otherwise use zeros
    return sanitized


def convert_helios_checkpoint(
    checkpoint_dir: str,
    output_dir: str,
    dtype: str = "bfloat16",
    quantize: bool = False,
    bits: int = 4,
    group_size: int = 64,
):
    """Convert a HuggingFace Helios checkpoint to MLX format.

    Expected structure (HuggingFace diffusers format):
        checkpoint_dir/
            transformer/
                diffusion_pytorch_model*.safetensors
                config.json
            text_encoder/
                model*.safetensors
            vae/
                diffusion_pytorch_model*.safetensors
            tokenizer/
                ...

    Args:
        checkpoint_dir: Path to HF Helios checkpoint
        output_dir: Path to output MLX model directory
        dtype: Target dtype for transformer/text-encoder weights
        quantize: Whether to quantize the transformer
        bits: Quantization bits (4 or 8)
        group_size: Quantization group size
    """
    from mlx_video.convert_wan import (
        load_safetensors_weights,
    )

    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "float16": mx.float16,
        "float32": mx.float32,
        "bfloat16": mx.bfloat16,
    }
    target_dtype = dtype_map.get(dtype, mx.bfloat16)

    # 1. Convert transformer
    transformer_dir = checkpoint_dir / "transformer"
    if transformer_dir.exists():
        print("Converting Helios transformer...")
        weights = load_safetensors_weights(str(transformer_dir))
        weights = sanitize_helios_transformer_weights(weights)
        weights = {k: v.astype(target_dtype) for k, v in weights.items()}
        out_path = output_dir / "model.safetensors"
        mx.save_safetensors(str(out_path), weights)
        print(f"  Saved {len(weights)} weight tensors to {out_path}")
        del weights
        gc.collect()
    else:
        # Try loading safetensors directly from checkpoint_dir
        print("Converting Helios transformer (flat directory)...")
        weights = load_safetensors_weights(str(checkpoint_dir))
        if weights:
            weights = sanitize_helios_transformer_weights(weights)
            weights = {k: v.astype(target_dtype) for k, v in weights.items()}
            out_path = output_dir / "model.safetensors"
            mx.save_safetensors(str(out_path), weights)
            print(f"  Saved {len(weights)} weight tensors to {out_path}")
            del weights
            gc.collect()

    # 2. Save config
    from mlx_video.models.helios.config import HeliosModelConfig

    config = HeliosModelConfig.helios_distilled()
    config_path = output_dir / "config.json"

    # Try to read source config for any overrides
    src_config_path = transformer_dir / "config.json" if transformer_dir.exists() else None
    if src_config_path and src_config_path.exists():
        with open(src_config_path) as f:
            src_cfg = json.load(f)
        # Could update config from src_cfg if needed
        print(f"  Source config found: {src_config_path}")

    config_dict = {
        "model_type": "helios_distilled",
        "dim": config.dim,
        "ffn_dim": config.ffn_dim,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "patch_size": list(config.patch_size),
        "in_dim": config.in_dim,
        "out_dim": config.out_dim,
        "text_dim": config.text_dim,
        "freq_dim": config.freq_dim,
        "text_len": config.text_len,
        "rope_dim": list(config.rope_dim),
        "rope_theta": config.rope_theta,
        "history_sizes": config.history_sizes,
        "num_latent_frames_per_chunk": config.num_latent_frames_per_chunk,
        "vae_z_dim": config.vae_z_dim,
        "vae_stride": list(config.vae_stride),
        "shift": config.shift,
        "sample_fps": config.sample_fps,
    }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"  Saved config to {config_path}")

    # 3. Convert text encoder (T5)
    t5_dir = checkpoint_dir / "text_encoder"
    if t5_dir.exists():
        print("Converting T5 text encoder...")
        weights = load_safetensors_weights(str(t5_dir))
        weights = sanitize_helios_t5_weights(weights)
        weights = {k: v.astype(target_dtype) for k, v in weights.items()}
        out_path = output_dir / "t5_encoder.safetensors"
        mx.save_safetensors(str(out_path), weights)
        print(f"  Saved {len(weights)} weight tensors to {out_path}")
        del weights
        gc.collect()

    # 4. Convert VAE
    vae_dir = checkpoint_dir / "vae"
    if vae_dir.exists():
        print("Converting VAE...")
        weights = load_safetensors_weights(str(vae_dir))
        weights = sanitize_helios_vae_weights(weights)
        # VAE in float32 for quality
        weights = {k: v.astype(mx.float32) for k, v in weights.items()}
        out_path = output_dir / "vae.safetensors"
        mx.save_safetensors(str(out_path), weights)
        print(f"  Saved {len(weights)} weight tensors to {out_path} (float32)")
        del weights
        gc.collect()

    # 5. Quantize if requested
    if quantize:
        print(f"\nQuantizing transformer ({bits}-bit, group_size={group_size})...")
        _quantize_saved_model(output_dir, config, bits, group_size)

    print(f"\nConversion complete! Output: {output_dir}")


def _quantize_saved_model(
    output_dir: Path,
    config,
    bits: int,
    group_size: int,
):
    """Load saved bf16 model, quantize, and re-save."""
    import mlx.nn as nn
    from mlx_video.models.helios.transformer import HeliosModel

    model_path = output_dir / "model.safetensors"
    if not model_path.exists():
        print("  No model.safetensors found, skipping quantization")
        return

    model = HeliosModel(config)
    weights = mx.load(str(model_path))
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    del weights
    gc.collect()
    mx.clear_cache()

    nn.quantize(
        model,
        group_size=group_size,
        bits=bits,
        class_predicate=lambda path, m: _quantize_predicate(path, m),
    )

    weights_dict = dict(mlx.utils.tree_flatten(model.parameters()))

    # Validate for corruption
    bad_keys = []
    for k, v in weights_dict.items():
        if k.endswith(".bias") and not k.endswith(".biases"):
            mx.eval(v)
            if mx.any(mx.isnan(v)).item() or mx.any(mx.isinf(v)).item():
                bad_keys.append(k)
    if bad_keys:
        raise RuntimeError(
            f"Quantization produced corrupted weights: "
            f"{len(bad_keys)} bias tensors contain NaN/Inf"
        )

    mx.save_safetensors(str(model_path), weights_dict)
    n_quantized = sum(1 for k in weights_dict if ".scales" in k)
    print(f"    {n_quantized} layers quantized, {len(weights_dict)} tensors saved")

    del model, weights_dict
    gc.collect()
    mx.clear_cache()

    # Update config with quantization metadata
    config_path = output_dir / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)
    cfg["quantization"] = {"group_size": group_size, "bits": bits}
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Updated config.json with quantization metadata")


if __name__ == "__main__":
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description="Convert Helios weights to MLX format")
    subparsers = parser.add_subparsers(dest="command")

    # Original conversion command (default when no subcommand)
    convert_parser = subparsers.add_parser("convert", help="Convert HF checkpoint to MLX format")
    convert_parser.add_argument("checkpoint_dir", type=str, help="Path to HF Helios checkpoint")
    convert_parser.add_argument("output_dir", type=str, help="Output MLX model directory")
    convert_parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "float32", "bfloat16"])
    convert_parser.add_argument("--quantize", action="store_true", help="Quantize transformer weights")
    convert_parser.add_argument("--bits", type=int, default=4, help="Quantization bits")
    convert_parser.add_argument("--group-size", type=int, default=64, help="Quantization group size")

    # Quantize-only command for existing MLX models
    quant_parser = subparsers.add_parser("quantize", help="Quantize an existing MLX model")
    quant_parser.add_argument("model_dir", type=str, help="Path to existing MLX model directory")
    quant_parser.add_argument("--output-dir", type=str, default=None,
                              help="Output directory (default: <model_dir>-<bits>bit)")
    quant_parser.add_argument("--bits", type=int, default=4, help="Quantization bits (default: 4)")
    quant_parser.add_argument("--group-size", type=int, default=64, help="Quantization group size")

    args = parser.parse_args()

    # Backward compatibility: if no subcommand, treat positional args as convert
    if args.command is None:
        # Legacy mode: positional args
        parser.print_help()
        print("\nUse 'convert' or 'quantize' subcommand. Examples:")
        print("  python -m mlx_video.convert_helios convert <hf_dir> <output_dir> [--quantize]")
        print("  python -m mlx_video.convert_helios quantize <mlx_model_dir> [--bits 4]")
        raise SystemExit(1)

    if args.command == "convert":
        convert_helios_checkpoint(
            args.checkpoint_dir,
            args.output_dir,
            dtype=args.dtype,
            quantize=args.quantize,
            bits=args.bits,
            group_size=args.group_size,
        )
    elif args.command == "quantize":
        from mlx_video.models.helios.config import HeliosModelConfig

        model_dir = Path(args.model_dir)
        output_dir = Path(args.output_dir) if args.output_dir else model_dir.parent / f"{model_dir.name}-{args.bits}bit"

        # Load config
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No config.json found in {model_dir}")
        with open(config_path) as f:
            config_dict = json.load(f)
        if "quantization" in config_dict:
            raise ValueError(f"Model is already quantized: {config_dict['quantization']}")

        config = HeliosModelConfig(**{
            k: v for k, v in config_dict.items()
            if k in HeliosModelConfig.__dataclass_fields__
        })

        # Copy to output dir if different
        if output_dir != model_dir:
            print(f"Copying {model_dir} → {output_dir}")
            shutil.copytree(model_dir, output_dir, dirs_exist_ok=True)

        print(f"Quantizing to {args.bits}-bit (group_size={args.group_size})...")
        _quantize_saved_model(output_dir, config, bits=args.bits, group_size=args.group_size)
        print(f"\nDone! Quantized model saved to: {output_dir}")
