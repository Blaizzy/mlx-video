"""Weight conversion for Wan2.2 models (PyTorch -> MLX)."""

import gc
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.utils

logger = logging.getLogger(__name__)


def load_torch_weights(path: str) -> Dict[str, mx.array]:
    """Load PyTorch .pth weights and convert to MLX arrays.

    Args:
        path: Path to .pth file

    Returns:
        Dictionary of MLX arrays
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required to load .pth weights: pip install torch")

    logging.info(f"Loading weights from {path}")
    state_dict = torch.load(path, map_location="cpu", weights_only=True)

    weights = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            np_val = value.detach().float().numpy()
            weights[key] = mx.array(np_val)

    return weights


def load_safetensors_weights(path: str) -> Dict[str, mx.array]:
    """Load safetensors weights as MLX arrays.

    Args:
        path: Path to directory with safetensors files or single file

    Returns:
        Dictionary of MLX arrays
    """
    path = Path(path)
    weights = {}
    if path.is_file():
        weights = mx.load(str(path))
    elif path.is_dir():
        for sf in sorted(path.glob("*.safetensors")):
            weights.update(mx.load(str(sf)))
    return weights


def sanitize_wan_transformer_weights(
    weights: Dict[str, mx.array]
) -> Dict[str, mx.array]:
    """Convert Wan2.2 transformer weight keys to MLX model structure.

    Wan2.2 keys follow the pattern:
        patch_embedding.weight/bias
        text_embedding.{0,2}.weight/bias
        time_embedding.{0,2}.weight/bias
        time_projection.1.weight/bias
        blocks.{i}.norm1.weight
        blocks.{i}.self_attn.{q,k,v,o}.weight/bias
        blocks.{i}.self_attn.norm_q.weight
        blocks.{i}.self_attn.norm_k.weight
        blocks.{i}.norm3.weight/bias (if cross_attn_norm)
        blocks.{i}.cross_attn.{q,k,v,o}.weight/bias
        blocks.{i}.cross_attn.norm_q.weight
        blocks.{i}.cross_attn.norm_k.weight
        blocks.{i}.norm2.weight
        blocks.{i}.ffn.{0,2}.weight/bias
        blocks.{i}.modulation
        head.norm.weight
        head.head.weight/bias
        head.modulation
        freqs (buffer)

    MLX model uses:
        patch_embedding_proj.weight/bias (after patchify reshape)
        text_embedding_0.weight/bias, text_embedding_1.weight/bias
        time_embedding_0.weight/bias, time_embedding_1.weight/bias
        time_projection.weight/bias
        blocks.{i}.norm1.weight
        blocks.{i}.self_attn.{q,k,v,o}.weight/bias
        etc.
    """
    sanitized = {}
    consumed = set()

    for key, value in weights.items():
        new_key = key

        # Patch embedding: Conv3d(16, 5120, (1,2,2)) weight is [O, I, D, H, W]
        # MLX Linear expects [O, I*D*H*W] after we flatten in patchify
        if key == "patch_embedding.weight":
            # Original: [dim, in_dim, 1, 2, 2] -> reshape to [dim, in_dim*1*2*2]
            value = value.reshape(value.shape[0], -1)
            new_key = "patch_embedding_proj.weight"
            sanitized[new_key] = value
            consumed.add(key)
            continue
        if key == "patch_embedding.bias":
            new_key = "patch_embedding_proj.bias"
            sanitized[new_key] = value
            consumed.add(key)
            continue

        # Text embedding Sequential: 0=Linear, 1=GELU(no params), 2=Linear
        if key.startswith("text_embedding.0."):
            new_key = key.replace("text_embedding.0.", "text_embedding_0.")
            sanitized[new_key] = value
            consumed.add(key)
            continue
        if key.startswith("text_embedding.2."):
            new_key = key.replace("text_embedding.2.", "text_embedding_1.")
            sanitized[new_key] = value
            consumed.add(key)
            continue

        # Time embedding Sequential: 0=Linear, 1=SiLU(no params), 2=Linear
        if key.startswith("time_embedding.0."):
            new_key = key.replace("time_embedding.0.", "time_embedding_0.")
            sanitized[new_key] = value
            consumed.add(key)
            continue
        if key.startswith("time_embedding.2."):
            new_key = key.replace("time_embedding.2.", "time_embedding_1.")
            sanitized[new_key] = value
            consumed.add(key)
            continue

        # Time projection Sequential: 0=SiLU(no params), 1=Linear
        if key.startswith("time_projection.1."):
            new_key = key.replace("time_projection.1.", "time_projection.")
            sanitized[new_key] = value
            consumed.add(key)
            continue

        # FFN: Sequential(Linear, GELU, Linear) -> ffn.{0,2} -> ffn.fc1, ffn.fc2
        new_key = new_key.replace(".ffn.0.", ".ffn.fc1.")
        new_key = new_key.replace(".ffn.2.", ".ffn.fc2.")

        # Skip the freqs buffer (we compute it in the model)
        if key == "freqs":
            consumed.add(key)
            continue

        sanitized[new_key] = value
        consumed.add(key)

    unconsumed = set(weights.keys()) - consumed
    if unconsumed:
        logger.warning("Unconsumed transformer weight keys: %s", sorted(unconsumed))

    return sanitized


def sanitize_wan_s2v_weights(
    weights: Dict[str, mx.array],
) -> Dict[str, mx.array]:
    """Convert a Wan 2.2 S2V-14B PyTorch state dict to MLX WanS2VModel keys.

    Runs the base :func:`sanitize_wan_transformer_weights` for the shared
    T2V-compatible tensors (blocks, patch/text/time/head), then handles the
    165 S2V-specific keys:

        audio_injector.injector.<N>.{q,k,v,o,norm_q,norm_k}.*   -> unchanged
        audio_injector.injector_adain_layers.<N>.linear.*       -> unchanged
        casual_audio_encoder.*                                  -> unchanged
        frame_packer.{proj,proj_2x,proj_4x}.{weight,bias}       -> unchanged
                                                                   (Conv3d
                                                                   layout kept
                                                                   raw for the
                                                                   Phase-1 stub)
        cond_encoder.{weight,bias}                              -> unchanged
        trainable_cond_mask.weight                              -> unchanged

    The upstream typo ``casual_audio_encoder`` (instead of ``causal``) is
    preserved on both sides. If a future release fixes the typo, the sanitizer
    will alias ``causal_*`` -> ``casual_*`` automatically.
    """
    # Split into base-transformer keys vs S2V-only keys, so we can invoke the
    # existing T2V sanitizer without triggering its "unconsumed keys" warning
    # on legitimately-S2V keys.
    s2v_prefixes = (
        "audio_injector.",
        "casual_audio_encoder.",
        "causal_audio_encoder.",  # future-proofing against the upstream typo fix
        "frame_packer.",
        "cond_encoder.",
        "trainable_cond_mask.",
    )
    base_weights = {}
    s2v_weights = {}
    for k, v in weights.items():
        if any(k.startswith(p) for p in s2v_prefixes):
            s2v_weights[k] = v
        else:
            base_weights[k] = v

    sanitized = sanitize_wan_transformer_weights(base_weights)

    for key, value in s2v_weights.items():
        new_key = key
        # Alias the future-fixed spelling to the shipped-with-typo module name.
        if new_key.startswith("causal_audio_encoder."):
            new_key = "casual_audio_encoder." + new_key[len("causal_audio_encoder."):]
        # Every other S2V key already matches the MLX module attribute path.
        sanitized[new_key] = value

    return sanitized


def sanitize_wan_t5_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Convert Wan2.2 T5 encoder weight keys to MLX T5Encoder structure.

    Wan2.2 T5 keys:
        token_embedding.weight
        pos_embedding.embedding.weight (if shared_pos)
        blocks.{i}.norm1.weight
        blocks.{i}.attn.{q,k,v,o}.weight
        blocks.{i}.norm2.weight
        blocks.{i}.ffn.gate.0.weight  (gate linear)
        blocks.{i}.ffn.fc1.weight
        blocks.{i}.ffn.fc2.weight
        blocks.{i}.pos_embedding.embedding.weight (if not shared_pos)
        norm.weight

    MLX T5Encoder structure:
        token_embedding.weight
        blocks.{i}.norm1.weight
        blocks.{i}.attn.{q,k,v,o}.weight
        blocks.{i}.norm2.weight
        blocks.{i}.ffn.gate_proj.weight  (mapped from gate.0)
        blocks.{i}.ffn.fc1.weight
        blocks.{i}.ffn.fc2.weight
        blocks.{i}.pos_embedding.embedding.weight
        norm.weight
    """
    sanitized = {}
    consumed = set()

    for key, value in weights.items():
        new_key = key

        # Map gate.0 -> gate_proj (the GELU is a separate module, not a parameter)
        new_key = new_key.replace(".ffn.gate.0.", ".ffn.gate_proj.")

        sanitized[new_key] = value
        consumed.add(key)

    unconsumed = set(weights.keys()) - consumed
    if unconsumed:
        logger.warning("Unconsumed T5 weight keys: %s", sorted(unconsumed))

    return sanitized


def sanitize_wan_vae_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Convert Wan2.2 VAE weight keys to MLX WanVAE structure.

    Handles Conv3d and Conv2d weight transpositions for MLX format.
    """
    sanitized = {}
    consumed = set()

    for key, value in weights.items():
        new_key = key

        # Handle Conv3d: PyTorch [O, I, D, H, W] -> MLX CausalConv3d weight [O, D, H, W, I]
        if "weight" in key and value.ndim == 5:
            value = mx.transpose(value, (0, 2, 3, 4, 1))

        # Handle Conv2d: PyTorch [O, I, H, W] -> MLX [O, H, W, I]
        if "weight" in key and value.ndim == 4:
            value = mx.transpose(value, (0, 2, 3, 1))

        # Map decoder keys to MLX decoder structure
        # Wan2.2 uses encoder/decoder with downsamples/upsamples
        # Need to adapt naming for our simplified structure

        sanitized[new_key] = value
        consumed.add(key)

    unconsumed = set(weights.keys()) - consumed
    if unconsumed:
        logger.warning("Unconsumed VAE weight keys: %s", sorted(unconsumed))

    return sanitized


def _load_lora_configs(
    lora_configs: List[Tuple[str, float]],
) -> Dict[str, list]:
    """Load LoRA weights from config tuples, returning module_to_loras dict.

    Shared between weight-merging and runtime-wrapping paths.
    """
    from mlx_video.models.wan_2.generate import Colors
    from mlx_video.lora import LoRAConfig, load_multiple_loras

    print(f"\n{Colors.CYAN}Loading {len(lora_configs)} LoRA(s)...{Colors.RESET}")

    configs = []
    for lora_path, strength in lora_configs:
        try:
            config = LoRAConfig(path=lora_path, strength=strength)
            configs.append(config)
            print(f"  - {Path(lora_path).name} (strength: {strength})")
        except Exception as e:
            print(f"{Colors.RED}Error loading LoRA {lora_path}: {e}{Colors.RESET}")
            raise

    module_to_loras = load_multiple_loras(configs)

    if not module_to_loras:
        print(
            f"{Colors.YELLOW}Warning: No LoRA weights matched model layers{Colors.RESET}"
        )

    return module_to_loras


def load_and_apply_loras(
    model_weights: Dict[str, mx.array],
    lora_configs: Optional[List[Tuple[str, float]]] = None,
    verbose: bool = False,
    quantization_bits: int = 0,
) -> Dict[str, mx.array]:
    """Load and apply LoRA weights to model weights by merging into weight dict.

    For non-quantized (bf16) models. For quantized models, use apply_loras_to_model().
    """
    from mlx_video.models.wan_2.generate import Colors
    from mlx_video.lora import apply_loras_to_weights

    if not lora_configs:
        return model_weights

    module_to_loras = _load_lora_configs(lora_configs)
    if not module_to_loras:
        return model_weights

    print(
        f"{Colors.GREEN}Applying LoRAs to {len(module_to_loras)} modules...{Colors.RESET}"
    )
    if verbose:
        print(f"  Model has {len(model_weights)} weight keys")

    modified_weights = apply_loras_to_weights(
        model_weights,
        module_to_loras,
        verbose=verbose,
        quantization_bits=quantization_bits,
    )

    print(f"{Colors.GREEN}✓ LoRAs applied successfully{Colors.RESET}")

    return modified_weights


def convert_wan_checkpoint(
    checkpoint_dir: str,
    output_dir: str,
    dtype: str = "bfloat16",
    model_version: str = "auto",
    quantize: bool = False,
    bits: int = 4,
    group_size: int = 64,
):
    """Convert a Wan2.1 or Wan2.2 checkpoint directory to MLX format.

    Wan2.2 expected structure:
        checkpoint_dir/
            models_t5_umt5-xxl-enc-bf16.pth
            Wan2.1_VAE.pth
            low_noise_model/   (safetensors)
            high_noise_model/  (safetensors)

    Wan2.1 expected structure:
        checkpoint_dir/
            models_t5_umt5-xxl-enc-bf16.pth
            Wan2.1_VAE.pth
            diffusion_pytorch_model*.safetensors  (single model)

    Args:
        checkpoint_dir: Path to Wan checkpoint directory
        output_dir: Path to output MLX model directory
        dtype: Target dtype
        model_version: "2.1", "2.2", or "auto" (detect from directory)
        quantize: Whether to quantize the transformer weights
        bits: Quantization bits (4 or 8)
        group_size: Quantization group size (32, 64, or 128)
    """
    import json

    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "float16": mx.float16,
        "float32": mx.float32,
        "bfloat16": mx.bfloat16,
    }
    target_dtype = dtype_map.get(dtype, mx.bfloat16)

    # Auto-detect version
    if model_version == "auto":
        if (checkpoint_dir / "low_noise_model").exists():
            model_version = "2.2"
        elif (checkpoint_dir / "Wan2.2_VAE.pth").exists():
            model_version = "2.2"
        else:
            model_version = "2.1"
        print(f"Auto-detected Wan{model_version} checkpoint")

    is_dual = (checkpoint_dir / "low_noise_model").exists()

    # Detect S2V: released as a *single* transformer split across 4 safetensors
    # shards, alongside diffusion_pytorch_model.safetensors.index.json and a
    # config.json with model_type=="s2v".
    is_s2v = False
    src_cfg_path = checkpoint_dir / "config.json"
    if src_cfg_path.exists() and not is_dual:
        try:
            with open(src_cfg_path) as _f:
                _src_cfg = json.load(_f)
            is_s2v = _src_cfg.get("model_type") == "s2v"
        except Exception:
            pass

    if is_s2v:
        print("Converting Wan2.2 S2V transformer (single model, S2V-14B)...")
        weights = load_safetensors_weights(str(checkpoint_dir))
        weights = sanitize_wan_s2v_weights(weights)
        weights = {k: v.astype(target_dtype) for k, v in weights.items()}
        out_path = output_dir / "model.safetensors"
        mx.save_safetensors(str(out_path), weights)
        print(f"  Saved {len(weights)} weight tensors to {out_path}")
    elif is_dual:
        # Wan2.2: Convert dual transformer models
        low_noise_path = checkpoint_dir / "low_noise_model"
        if low_noise_path.exists():
            print("Converting low-noise transformer...")
            weights = load_safetensors_weights(str(low_noise_path))
            weights = sanitize_wan_transformer_weights(weights)
            weights = {k: v.astype(target_dtype) for k, v in weights.items()}
            out_path = output_dir / "low_noise_model.safetensors"
            mx.save_safetensors(str(out_path), weights)
            print(f"  Saved {len(weights)} weight tensors to {out_path}")

        high_noise_path = checkpoint_dir / "high_noise_model"
        if high_noise_path.exists():
            print("Converting high-noise transformer...")
            weights = load_safetensors_weights(str(high_noise_path))
            weights = sanitize_wan_transformer_weights(weights)
            weights = {k: v.astype(target_dtype) for k, v in weights.items()}
            out_path = output_dir / "high_noise_model.safetensors"
            mx.save_safetensors(str(out_path), weights)
            print(f"  Saved {len(weights)} weight tensors to {out_path}")
    else:
        # Wan2.1: Convert single transformer model
        # Try safetensors in the checkpoint dir itself
        print("Converting transformer (single model)...")
        weights = load_safetensors_weights(str(checkpoint_dir))
        if not weights:
            # Fallback: look for .pth files
            for pth in sorted(checkpoint_dir.glob("*.pth")):
                if "t5" not in pth.name.lower() and "vae" not in pth.name.lower():
                    print(f"  Loading from {pth.name}...")
                    weights = load_torch_weights(str(pth))
                    break
        if weights:
            weights = sanitize_wan_transformer_weights(weights)
            weights = {k: v.astype(target_dtype) for k, v in weights.items()}
            out_path = output_dir / "model.safetensors"
            mx.save_safetensors(str(out_path), weights)
            print(f"  Saved {len(weights)} weight tensors to {out_path}")
        else:
            print("  Warning: No transformer weights found!")

    # Save config — detect model size from source config.json or transformer weights
    from mlx_video.models.wan_2.config import WanModelConfig

    def _detect_config():
        """Detect config from source config.json or transformer weight shapes."""
        if is_s2v:
            return WanModelConfig.wan22_s2v_14b()
        if is_dual:
            # Check source config.json for model_type (I2V vs T2V)
            src_cfg_path = checkpoint_dir / "high_noise_model" / "config.json"
            if src_cfg_path.exists():
                with open(src_cfg_path) as f:
                    src_config = json.load(f)
                src_model_type = src_config.get("model_type", "t2v")
                if src_model_type == "i2v" or src_config.get("in_dim") == 36:
                    return WanModelConfig.wan22_i2v_14b()
            return WanModelConfig.wan22_t2v_14b()

        # Try reading source config.json first (most reliable)
        src_cfg_path = checkpoint_dir / "config.json"
        src_config = None
        if src_cfg_path.exists():
            with open(src_cfg_path) as f:
                src_config = json.load(f)

        if src_config and "dim" in src_config:
            src_dim = src_config.get("dim", 5120)
            src_in_dim = src_config.get("in_dim", 16)
            src_out_dim = src_config.get("out_dim", 16)
            src_ffn_dim = src_config.get("ffn_dim", 13824)
            src_num_heads = src_config.get("num_heads", 40)
            src_num_layers = src_config.get("num_layers", 40)
            src_model_type = src_config.get("model_type", "t2v")
            src_text_len = src_config.get("text_len", 512)

            print(
                f"  Source config: dim={src_dim}, layers={src_num_layers}, "
                f"heads={src_num_heads}, type={src_model_type}"
            )

            # Use preset for known TI2V 5B configuration
            if src_model_type == "ti2v" and src_dim == 3072:
                return WanModelConfig.wan22_ti2v_5b()

            is_22 = model_version == "2.2"

            # Wan2.2 uses different VAE with z_dim=48 and stride (4,16,16)
            vae_z = 48 if is_22 else 16
            vae_s = (4, 16, 16) if is_22 else (4, 8, 8)
            fps = 24 if is_22 else 16

            return WanModelConfig(
                model_type=src_model_type,
                model_version=model_version,
                dim=src_dim,
                ffn_dim=src_ffn_dim,
                in_dim=src_in_dim,
                out_dim=src_out_dim,
                num_heads=src_num_heads,
                num_layers=src_num_layers,
                text_len=src_text_len,
                vae_z_dim=vae_z,
                vae_stride=vae_s,
                dual_model=False,
                boundary=0.0,
                sample_shift=5.0,
                sample_steps=50,
                sample_guide_scale=5.0,
                sample_fps=fps,
            )

        # Fallback: detect from saved transformer weight shapes
        saved_model = output_dir / "model.safetensors"
        if saved_model.exists():
            det_weights = mx.load(str(saved_model))
            dim = None
            for k, v in det_weights.items():
                if "patch_embedding_proj.weight" in k:
                    dim = v.shape[0]
                    break
            del det_weights
            if dim is not None and dim <= 2048:
                print(f"  Auto-detected 1.3B model (dim={dim})")
                return WanModelConfig.wan21_t2v_1_3b()

        return WanModelConfig.wan21_t2v_14b()

    config = _detect_config()
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"  Saved config to {config_path}")

    # Convert T5 encoder
    t5_path = checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth"
    if t5_path.exists():
        print("Converting T5 encoder...")
        weights = load_torch_weights(str(t5_path))
        weights = sanitize_wan_t5_weights(weights)
        weights = {k: v.astype(target_dtype) for k, v in weights.items()}
        out_path = output_dir / "t5_encoder.safetensors"
        mx.save_safetensors(str(out_path), weights)
        print(f"  Saved {len(weights)} weight tensors to {out_path}")

    # Convert VAE (check both naming conventions)
    vae_path = checkpoint_dir / "Wan2.1_VAE.pth"
    is_wan22_vae = False
    if not vae_path.exists():
        vae_path = checkpoint_dir / "Wan2.2_VAE.pth"
        is_wan22_vae = True
    if vae_path.exists():
        print(f"Converting VAE ({'Wan2.2' if is_wan22_vae else 'Wan2.1'})...")
        weights = load_torch_weights(str(vae_path))
        if is_wan22_vae:
            from mlx_video.models.wan_2.vae22 import sanitize_wan22_vae_weights

            include_encoder = config.model_type in ("ti2v", "i2v")
            weights = sanitize_wan22_vae_weights(
                weights, include_encoder=include_encoder
            )
        else:
            weights = sanitize_wan_vae_weights(weights)
        # Always save VAE in float32 — official Wan2.2 runs VAE decode in
        # float32 (dtype=torch.float).  Saving in bfloat16 loses precision
        # that cannot be recovered by upcasting at load time.
        weights = {k: v.astype(mx.float32) for k, v in weights.items()}
        out_path = output_dir / "vae.safetensors"
        mx.save_safetensors(str(out_path), weights)
        print(f"  Saved {len(weights)} weight tensors to {out_path} (float32)")

    # Quantize transformer weights if requested
    if quantize:
        print(
            f"\nQuantizing transformer weights ({bits}-bit, group_size={group_size})..."
        )
        _quantize_saved_model(output_dir, config, is_dual, bits, group_size)

    print(f"\nConversion complete! Output: {output_dir}")


def _quantize_predicate(path: str, module) -> bool:
    """Return True for layers that should be quantized.

    Targets heavyweight Linear layers in attention and FFN blocks.
    Skips embeddings, norms, head, and modulation (small, precision-sensitive).
    """
    if not hasattr(module, "to_quantized"):
        return False
    # Quantize attention Q/K/V/O and FFN fc1/fc2
    quantize_patterns = (
        ".self_attn.q",
        ".self_attn.k",
        ".self_attn.v",
        ".self_attn.o",
        ".cross_attn.q",
        ".cross_attn.k",
        ".cross_attn.v",
        ".cross_attn.o",
        ".ffn.fc1",
        ".ffn.fc2",
    )
    return any(path.endswith(p) for p in quantize_patterns)


def _quantize_saved_model(
    output_dir: Path,
    config,
    is_dual: bool,
    bits: int,
    group_size: int,
    source_dir: Path = None,
):
    """Load saved bf16 model, quantize, and re-save.

    Args:
        output_dir: Directory to write quantized weights to.
        config: WanModelConfig for creating the model.
        is_dual: Whether this is a dual-expert model.
        bits: Quantization bits.
        group_size: Quantization group size.
        source_dir: Directory to read bf16 weights from. Defaults to output_dir.
    """
    import json

    import mlx.nn as nn

    from mlx_video.models.wan_2.wan_2 import WanModel

    if source_dir is None:
        source_dir = output_dir

    model_names = []
    if is_dual:
        for name in ["low_noise_model.safetensors", "high_noise_model.safetensors"]:
            if (source_dir / name).exists():
                model_names.append(name)
    else:
        if (source_dir / "model.safetensors").exists():
            model_names.append("model.safetensors")

    for name in model_names:
        print(f"  Quantizing {name}...")
        model = WanModel(config)
        weights = mx.load(str(source_dir / name))
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.parameters())
        del weights
        gc.collect()
        mx.clear_cache()

        # Apply quantization to targeted layers
        nn.quantize(
            model,
            group_size=group_size,
            bits=bits,
            class_predicate=lambda path, m: _quantize_predicate(path, m),
        )

        # Save quantized weights
        weights_dict = dict(mlx.utils.tree_flatten(model.parameters()))

        # Validate: check for NaN/Inf in bias tensors (corruption canary)
        bad_keys = []
        for k, v in weights_dict.items():
            if k.endswith(".bias") and not k.endswith(".biases"):
                mx.eval(v)
                if mx.any(mx.isnan(v)).item() or mx.any(mx.isinf(v)).item():
                    bad_keys.append(k)
        if bad_keys:
            raise RuntimeError(
                f"Quantization produced corrupted weights in {model_path.name}: "
                f"{len(bad_keys)} bias tensors contain NaN/Inf "
                f"(e.g. {bad_keys[0]}). Try re-running with more available memory."
            )

        mx.save_safetensors(str(output_dir / name), weights_dict)
        n_quantized = sum(1 for k in weights_dict if ".scales" in k)
        print(f"    {n_quantized} layers quantized, {len(weights_dict)} tensors saved")

        # Free model before processing next file
        del model, weights_dict
        gc.collect()
        mx.clear_cache()

    # Update config.json with quantization metadata
    config_path = output_dir / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)
    cfg["quantization"] = {
        "group_size": group_size,
        "bits": bits,
    }
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Updated config.json with quantization metadata")


def quantize_mlx_model(
    mlx_model_dir: str,
    output_dir: str,
    bits: int = 4,
    group_size: int = 64,
):
    """Quantize an already-converted MLX model (skips PyTorch conversion).

    Args:
        mlx_model_dir: Path to existing MLX model directory (bf16/fp16).
        output_dir: Path to output quantized model directory.
        bits: Quantization bits (4 or 8).
        group_size: Quantization group size (32, 64, or 128).
    """
    import json
    import shutil

    src = Path(mlx_model_dir)
    dst = Path(output_dir)

    config_path = src / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found in {src}")

    with open(config_path) as f:
        cfg = json.load(f)

    if cfg.get("quantization"):
        raise ValueError(
            f"Model at {src} is already quantized "
            f"({cfg['quantization']['bits']}-bit). Use a bf16/fp16 source."
        )

    # Detect dual vs single expert
    is_dual = (src / "low_noise_model.safetensors").exists() and (
        src / "high_noise_model.safetensors"
    ).exists()

    # Build model config
    from mlx_video.models.wan_2.config import WanModelConfig

    config_dict = {
        k: v for k, v in cfg.items() if k in WanModelConfig.__dataclass_fields__
    }
    for key in ("patch_size", "vae_stride", "window_size", "sample_guide_scale"):
        if key in config_dict and isinstance(config_dict[key], list):
            config_dict[key] = tuple(config_dict[key])
    config = WanModelConfig(**config_dict)

    # Copy non-transformer files to output dir (skip large model weights)
    transformer_files = {
        "low_noise_model.safetensors",
        "high_noise_model.safetensors",
        "model.safetensors",
    }
    if dst.resolve() != src.resolve():
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file() and f.name not in transformer_files:
                shutil.copy2(f, dst / f.name)
        print(f"Copied non-transformer files from {src} to {dst}")

    print(f"Quantizing transformer weights ({bits}-bit, group_size={group_size})...")
    _quantize_saved_model(dst, config, is_dual, bits, group_size, source_dir=src)

    print(f"\nQuantization complete! Output: {dst}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert Wan model to MLX format")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to Wan checkpoint directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="wan_mlx_model",
        help="Output path for MLX model",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "float32", "bfloat16"],
        default="bfloat16",
        help="Target dtype",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        choices=["2.1", "2.2", "auto"],
        default="auto",
        help="Wan model version (auto-detect by default)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize transformer weights for faster inference",
    )
    parser.add_argument(
        "--quantize-only",
        action="store_true",
        help="Quantize an already-converted MLX model (skips PyTorch conversion)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        choices=[4, 8],
        default=4,
        help="Quantization bits (default: 4)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        choices=[32, 64, 128],
        default=64,
        help="Quantization group size (default: 64)",
    )
    args = parser.parse_args()

    if args.quantize_only:
        quantize_mlx_model(
            args.checkpoint_dir,
            args.output_dir,
            bits=args.bits,
            group_size=args.group_size,
        )
    else:
        convert_wan_checkpoint(
            args.checkpoint_dir,
            args.output_dir,
            args.dtype,
            args.model_version,
            quantize=args.quantize,
            bits=args.bits,
            group_size=args.group_size,
        )
