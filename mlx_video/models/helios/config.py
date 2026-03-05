from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

from mlx_video.models.ltx.config import BaseModelConfig


@dataclass
class HeliosModelConfig(BaseModelConfig):
    """Configuration for Helios video generation models."""

    # Transformer architecture (identical to Wan 14B)
    dim: int = 5120
    ffn_dim: int = 13824
    num_heads: int = 40
    num_layers: int = 40
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    in_dim: int = 16
    out_dim: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    text_len: int = 512
    eps: float = 1e-6
    qk_norm: bool = True
    cross_attn_norm: bool = True

    # RoPE
    rope_dim: Tuple[int, int, int] = (44, 42, 42)
    rope_theta: float = 10000.0

    # Helios-specific: multi-scale history memory
    history_sizes: List[int] = field(default_factory=lambda: [16, 2, 1])
    num_latent_frames_per_chunk: int = 9
    has_multi_term_memory_patch: bool = True
    zero_history_timestep: bool = True

    # VAE (identical to Wan — AutoencoderKLWan)
    vae_stride: Tuple[int, int, int] = (4, 8, 8)
    vae_z_dim: int = 16

    # T5 text encoder (identical to Wan — UMT5-XXL)
    t5_vocab_size: int = 256384
    t5_dim: int = 4096
    t5_dim_attn: int = 4096
    t5_dim_ffn: int = 10240
    t5_num_heads: int = 64
    t5_num_layers: int = 24
    t5_num_buckets: int = 32

    # Scheduler
    num_train_timesteps: int = 1000
    shift: float = 1.0
    stages: int = 3
    stage_range: List[float] = field(default_factory=lambda: [0, 1 / 3, 2 / 3, 1])
    gamma: float = 1 / 3

    # Inference defaults
    sample_fps: int = 24
    frame_num: int = 99

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads

    @classmethod
    def helios_distilled(cls) -> "HeliosModelConfig":
        """Helios-Distilled: x0-prediction, no CFG, DMD scheduler, 2-3 steps."""
        return cls(
            shift=1.0,
        )
