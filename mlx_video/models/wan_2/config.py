from dataclasses import dataclass
from typing import Tuple, Union

from mlx_video.models.ltx_2.config import BaseModelConfig


@dataclass
class WanModelConfig(BaseModelConfig):
    """Configuration for Wan T2V models (supports both 2.1 and 2.2)."""

    model_type: str = "t2v"
    model_version: str = "2.2"
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    text_len: int = 512
    in_dim: int = 16
    dim: int = 5120
    ffn_dim: int = 13824
    freq_dim: int = 256
    text_dim: int = 4096
    out_dim: int = 16
    num_heads: int = 40
    num_layers: int = 40
    window_size: Tuple[int, int] = (-1, -1)
    qk_norm: bool = True
    cross_attn_norm: bool = True
    eps: float = 1e-6

    # -------- Wan 2.2 S2V-specific (unused for T2V/I2V/TI2V) --------
    audio_dim: int = 0  # wav2vec2 feature dim (1024 for xls-r), 0 disables audio
    num_audio_token: int = 4  # tokens per video frame from CausalAudioEncoder
    audio_inject_layers: Tuple[int, ...] = ()  # transformer blocks to inject audio
    cond_dim: int = 0  # pose/overlay conditioning channels (16 for S2V), 0 disables
    enable_adain: bool = False  # AdaLayerNorm on audio injector inputs
    adain_mode: str = "attn_norm"  # "attn_norm" is the released S2V setting
    enable_framepack: bool = False  # motion-history framepack module
    framepack_drop_mode: str = "padd"  # "padd" (released) or "drop"
    enable_motioner: bool = False  # MotionerTransformers (mutually exclusive w/ framepack)
    motion_token_num: int = 1024  # only used if enable_motioner=True
    trainable_token_pos_emb: bool = False  # only used if enable_motioner=True
    add_last_motion: bool = True
    zero_timestep: bool = False  # ref/motion tokens use t=0 modulation
    zero_init: bool = False  # zero-init output projections at construction

    # VAE
    vae_stride: Tuple[int, int, int] = (4, 8, 8)
    vae_z_dim: int = 16

    # Inference
    dual_model: bool = True
    boundary: float = 0.875
    sample_shift: float = 12.0
    sample_steps: int = 40
    sample_guide_scale: Union[float, Tuple[float, float]] = (3.0, 4.0)
    num_train_timesteps: int = 1000
    sample_fps: int = 16
    frame_num: int = 81
    sample_neg_prompt: str = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
        "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
        "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
        "杂乱的背景，三条腿，背景人很多，倒着走"
    )

    # Resolution constraints
    max_area: int = 0  # 0 = no limit; e.g. 704*1280 for TI2V-5B
    t5_vocab_size: int = 256384
    t5_dim: int = 4096
    t5_dim_attn: int = 4096
    t5_dim_ffn: int = 10240
    t5_num_heads: int = 64
    t5_num_layers: int = 24
    t5_num_buckets: int = 32

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads

    @classmethod
    def wan21_t2v_14b(cls) -> "WanModelConfig":
        """Wan2.1 T2V 14B: single model, 40 layers, dim=5120."""
        return cls(
            model_version="2.1",
            dual_model=False,
            boundary=0.0,
            sample_shift=5.0,
            sample_steps=50,
            sample_guide_scale=5.0,
        )

    @classmethod
    def wan21_t2v_1_3b(cls) -> "WanModelConfig":
        """Wan2.1 T2V 1.3B: single model, 30 layers, dim=1536."""
        return cls(
            model_version="2.1",
            dim=1536,
            ffn_dim=8960,
            num_heads=12,
            num_layers=30,
            dual_model=False,
            boundary=0.0,
            sample_shift=5.0,
            sample_steps=50,
            sample_guide_scale=5.0,
        )

    @classmethod
    def wan22_t2v_14b(cls) -> "WanModelConfig":
        """Wan2.2 T2V 14B: dual model, 40 layers, dim=5120 (default)."""
        return cls()

    @classmethod
    def wan22_i2v_14b(cls) -> "WanModelConfig":
        """Wan2.2 I2V 14B: dual model, image-to-video, 40 layers, dim=5120."""
        return cls(
            model_type="i2v",
            in_dim=36,
            out_dim=16,
            dual_model=True,
            boundary=0.900,
            sample_shift=5.0,
            sample_guide_scale=(3.5, 3.5),
            max_area=704 * 1280,
        )

    @classmethod
    def wan22_s2v_14b(cls) -> "WanModelConfig":
        """Wan2.2 Speech-to-Video 14B (audio-driven talking-head).

        Single-model (not MoE). Adds wav2vec2 audio conditioning via
        CausalAudioEncoder + AudioInjector cross-attention at 12 blocks,
        plus reference-image and framepack motion-history tokens appended
        to the transformer sequence.
        """
        return cls(
            model_type="s2v",
            model_version="2.2",
            dim=5120,
            ffn_dim=13824,
            in_dim=16,
            out_dim=16,
            num_heads=40,
            num_layers=40,
            # S2V-specific
            audio_dim=1024,
            num_audio_token=4,
            audio_inject_layers=(0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39),
            cond_dim=16,
            enable_adain=True,
            adain_mode="attn_norm",
            enable_framepack=True,
            framepack_drop_mode="padd",
            enable_motioner=False,
            add_last_motion=True,
            zero_timestep=True,
            zero_init=True,
            # Inference (single-model, not MoE)
            dual_model=False,
            boundary=0.0,
            sample_shift=5.0,
            sample_steps=40,
            sample_guide_scale=(4.5, 4.5),  # per official S2V pipeline
            max_area=704 * 1280,
        )

    @classmethod
    def wan22_ti2v_5b(cls) -> "WanModelConfig":
        """Wan2.2 TI2V 5B: text+image to video, 30 layers, dim=3072."""
        return cls(
            model_type="ti2v",
            dim=3072,
            ffn_dim=14336,
            in_dim=48,
            out_dim=48,
            num_heads=24,
            num_layers=30,
            vae_z_dim=48,
            vae_stride=(4, 16, 16),
            dual_model=False,
            boundary=0.0,
            sample_shift=5.0,
            sample_steps=40,
            sample_guide_scale=5.0,
            sample_fps=24,
            max_area=704 * 1280,
        )
