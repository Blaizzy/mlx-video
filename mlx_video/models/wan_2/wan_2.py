import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .attention import WanLayerNorm, _linear_dtype
from .config import WanModelConfig
from .rope import rope_params, rope_precompute_cos_sin, rope_precompute_cos_sin_segments
from .transformer import WanAttentionBlock


def sinusoidal_embedding_1d(dim: int, position: mx.array) -> mx.array:
    """Compute sinusoidal positional embeddings.

    Args:
        dim: Embedding dimension (must be even).
        position: Tensor of positions — 1D [L] or 2D [B, L].

    Returns:
        Embeddings of shape [L, dim] or [B, L, dim].
    """
    assert dim % 2 == 0
    half = dim // 2
    pos = position.astype(mx.float32)
    inv_freq = mx.power(10000.0, -mx.arange(half).astype(mx.float32) / half)
    sinusoid = pos[..., None] * inv_freq  # [..., half]
    return mx.concatenate([mx.cos(sinusoid), mx.sin(sinusoid)], axis=-1)


class Head(nn.Module):
    """Output projection head with learned modulation."""

    def __init__(self, dim: int, out_dim: int, patch_size: tuple, eps: float = 1e-6):
        super().__init__()
        self.out_dim = out_dim
        self.patch_size = patch_size
        proj_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, proj_dim)
        self.modulation = (mx.random.normal((1, 2, dim)) * (dim**-0.5)).astype(
            mx.float32
        )

    def __call__(self, x: mx.array, e: mx.array) -> mx.array:
        """
        Args:
            x: [B, L, dim]
            e: [B, dim] or [B, 1, dim] (broadcast) or [B, L, dim] (per-token)
        """
        if e.ndim == 2:
            e = e[:, None, :]  # [B, 1, dim]
        # Compute modulation in float32 (matching reference's autocast(float32))
        mod = self.modulation[:, None, :, :] + e[:, :, None, :]  # float32
        e0 = mod[:, :, 0, :]  # [B, L_e, dim] shift
        e1 = mod[:, :, 1, :]  # [B, L_e, dim] scale
        x_norm = self.norm(x)
        x_mod = x_norm * (1 + e1) + e0
        return self.head(x_mod)


class WanModel(nn.Module):
    """Wan2.2 diffusion backbone for text-to-video generation."""

    def __init__(self, config: WanModelConfig):
        super().__init__()
        self.config = config
        dim = config.dim
        self.dim = dim
        self.num_heads = config.num_heads
        self.out_dim = config.out_dim
        self.patch_size = config.patch_size
        self.text_len = config.text_len
        self.freq_dim = config.freq_dim

        # Patch embedding: Conv3d implemented as a reshaped linear
        # For kernel (1,2,2) and stride (1,2,2): reshape input then linear
        patch_dim = config.in_dim * math.prod(config.patch_size)
        self.patch_embedding_proj = nn.Linear(patch_dim, dim)
        self._patch_size = config.patch_size

        # Text embedding MLP
        self.text_embedding_0 = nn.Linear(config.text_dim, dim)
        self.text_embedding_act = nn.GELU(approx="tanh")
        self.text_embedding_1 = nn.Linear(dim, dim)

        # Time embedding MLP
        self.time_embedding_0 = nn.Linear(config.freq_dim, dim)
        self.time_embedding_act = nn.SiLU()
        self.time_embedding_1 = nn.Linear(dim, dim)

        # Time projection for modulation (6x dim)
        self.time_projection_act = nn.SiLU()
        self.time_projection = nn.Linear(dim, dim * 6)

        # Transformer blocks
        self.blocks = [
            WanAttentionBlock(
                dim=dim,
                ffn_dim=config.ffn_dim,
                num_heads=config.num_heads,
                window_size=config.window_size,
                qk_norm=config.qk_norm,
                cross_attn_norm=config.cross_attn_norm,
                eps=config.eps,
            )
            for _ in range(config.num_layers)
        ]

        # Output head
        self.head = Head(dim, config.out_dim, config.patch_size, config.eps)

        # Precompute RoPE frequencies — three separate tables concatenated.
        # Reference computes three rope_params with different dim normalizations
        # so each axis (temporal/height/width) gets its own full frequency range.
        d = dim // config.num_heads
        self.freqs = mx.concatenate(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            axis=1,
        )

        # Precompute sinusoidal inv_freq for time embedding.
        half = config.freq_dim // 2
        self._inv_freq = mx.array(
            np.power(10000.0, -np.arange(half, dtype=np.float64) / half).astype(
                np.float32
            )
        )

    def _patchify(self, x: mx.array) -> tuple:
        """Convert video tensor to patch embeddings.

        Args:
            x: Video latent [C, F, H, W]

        Returns:
            (patches, grid_size): patches [1, L, dim], grid_size (F', H', W')
        """
        c, f, h, w = x.shape
        pt, ph, pw = self._patch_size

        f_out = f // pt
        h_out = h // ph
        w_out = w // pw

        # Reshape: [C, F, H, W] -> [F', H', W', C, pt, ph, pw] -> [F'*H'*W', C*pt*ph*pw]
        # Order must be [C, pt, ph, pw] (C slowest) to match Conv3d weight layout
        x = x.reshape(c, f_out, pt, h_out, ph, w_out, pw)
        x = x.transpose(1, 3, 5, 0, 2, 4, 6)  # [F', H', W', C, pt, ph, pw]
        x = x.reshape(f_out * h_out * w_out, -1)  # [L, C*pt*ph*pw]

        # Project and cast to model dtype to prevent float32 cascade from input latents
        patches = self.patch_embedding_proj(x)  # [L, dim]
        patches = patches.astype(_linear_dtype(self.patch_embedding_proj))
        patches = patches[None, :, :]  # [1, L, dim]

        return patches, (f_out, h_out, w_out)

    def unpatchify(self, x: mx.array, grid_sizes: list) -> list:
        """Reconstruct video from patch embeddings.

        Args:
            x: [B, L, out_dim * prod(patch_size)]
            grid_sizes: List of (F', H', W') per batch element

        Returns:
            List of tensors [C, F, H, W]
        """
        c = self.out_dim
        pt, ph, pw = self.patch_size
        out = []
        for i, (f, h, w) in enumerate(grid_sizes):
            seq_len = f * h * w
            u = x[i, :seq_len]  # [L, out_dim * pt * ph * pw]
            u = u.reshape(f, h, w, pt, ph, pw, c)
            # Rearrange: [F', H', W', pt, ph, pw, C] -> [C, F'*pt, H'*ph, W'*pw]
            u = u.transpose(6, 0, 3, 1, 4, 2, 5)  # [C, F', pt, H', ph, W', pw]
            u = u.reshape(c, f * pt, h * ph, w * pw)
            out.append(u)
        return out

    def embed_text(self, context: list) -> mx.array:
        """Precompute text embeddings (call once, reuse across steps).

        Args:
            context: List of text embeddings [L_text, text_dim]

        Returns:
            Embedded context [B, text_len, dim] in model dtype
        """
        model_dtype = _linear_dtype(self.patch_embedding_proj)
        context_padded = []
        for ctx in context:
            pad_len = self.text_len - ctx.shape[0]
            if pad_len > 0:
                ctx = mx.concatenate(
                    [ctx, mx.zeros((pad_len, ctx.shape[1]), dtype=ctx.dtype)],
                    axis=0,
                )
            context_padded.append(ctx)
        context_batch = mx.stack(context_padded)  # [B, text_len, text_dim]
        context_batch = self.text_embedding_1(
            self.text_embedding_act(self.text_embedding_0(context_batch))
        )
        return context_batch.astype(model_dtype)

    def prepare_cross_kv(self, context: mx.array) -> list:
        """Pre-compute cross-attention K/V for all blocks.

        Call once before the diffusion loop to cache K/V projections,
        eliminating redundant computation at each denoising step.

        Args:
            context: Pre-embedded text [B, text_len, dim]

        Returns:
            List of (k, v) tuples, one per block
        """
        kv_caches = []
        for block in self.blocks:
            kv_caches.append(block.cross_attn.prepare_kv(context))
        return kv_caches

    def prepare_rope(self, grid_sizes: list) -> tuple:
        """Pre-compute RoPE cos/sin for constant grid sizes.

        Call once before the diffusion loop when grid sizes don't change
        across steps. Eliminates per-step broadcast/concat overhead.

        Args:
            grid_sizes: List of (F, H, W) tuples per batch element

        Returns:
            (cos_f, sin_f) precomputed frequency tensors
        """
        w_dtype = _linear_dtype(self.patch_embedding_proj)
        return rope_precompute_cos_sin(grid_sizes, self.freqs, dtype=w_dtype)

    def __call__(
        self,
        x_list: list,
        t: mx.array,
        context: list | mx.array,
        seq_len: int,
        cross_kv_caches: list | None = None,
        y: list | None = None,
        rope_cos_sin: tuple | None = None,
    ) -> list:
        """Forward pass.

        Args:
            x_list: List of video latent tensors [C, F, H, W]
            t: Timestep tensor [B]
            context: List of raw text embeddings, OR pre-embedded tensor
                     from embed_text() [B, text_len, dim]
            seq_len: Maximum sequence length for padding
            cross_kv_caches: Optional list of (k, v) tuples from
                             prepare_cross_kv(), one per block.
            y: Optional list of conditioning tensors for I2V [C_y, F, H, W].
               Channel-concatenated with x before patchify.
            rope_cos_sin: Optional precomputed (cos, sin) from prepare_rope().

        Returns:
            List of denoised tensors [C, F, H, W]
        """
        # Detect identical inputs (CFG B=2) to avoid duplicate patchify work.
        # Check BEFORE I2V concat since concat creates new array objects.
        batch_size = len(x_list)
        all_same = batch_size > 1 and all(
            x_list[i] is x_list[0] for i in range(1, batch_size)
        )
        if all_same and y is not None:
            all_same = all(y[i] is y[0] for i in range(1, len(y)))

        # I2V: channel-concatenate conditioning y with noise x
        if y is not None:
            x_list = [mx.concatenate([u, v], axis=0) for u, v in zip(x_list, y)]

        if all_same:
            # Patchify once and broadcast — saves a Linear projection per step
            p, gs = self._patchify(x_list[0])  # [1, L, dim]
            grid_sizes = [gs] * batch_size
            seq_lens_list = [p.shape[1]] * batch_size
            # Pad and broadcast
            if p.shape[1] < seq_len:
                p = mx.concatenate(
                    [p, mx.zeros((1, seq_len - p.shape[1], self.dim), dtype=p.dtype)],
                    axis=1,
                )
            x = mx.broadcast_to(p, (batch_size,) + p.shape[1:])
        else:
            patches = []
            grid_sizes = []
            seq_lens_list = []
            for vid in x_list:
                p, gs = self._patchify(vid)  # [1, L, dim]
                patches.append(p)
                grid_sizes.append(gs)
                seq_lens_list.append(p.shape[1])
            x = mx.concatenate(
                [
                    (
                        mx.concatenate(
                            [
                                p,
                                mx.zeros(
                                    (1, seq_len - p.shape[1], self.dim), dtype=p.dtype
                                ),
                            ],
                            axis=1,
                        )
                        if p.shape[1] < seq_len
                        else p
                    )
                    for p in patches
                ],
                axis=0,
            )  # [B, seq_len, dim]

        # Time embedding: sinusoidal from precomputed inv_freq.
        # inv_freq was computed in float64 for precision, stored as float32.
        # With integer timesteps (matching reference), float32 sin/cos is fine.
        if t.ndim == 0:
            t = t[None]

        sinusoid = t[..., None].astype(mx.float32) * self._inv_freq
        sin_emb = mx.concatenate([mx.cos(sinusoid), mx.sin(sinusoid)], axis=-1)

        if t.ndim == 1:
            # Standard T2V: scalar timestep per batch element [B]
            e = self.time_embedding_1(
                self.time_embedding_act(self.time_embedding_0(sin_emb))
            )  # [B, dim]
            e0 = self.time_projection(self.time_projection_act(e))  # [B, dim*6]
            e0 = e0.reshape(batch_size, 1, 6, self.dim)
        else:
            # I2V: per-token timesteps [B, L]
            e = self.time_embedding_1(
                self.time_embedding_act(self.time_embedding_0(sin_emb))
            )  # [B, L, dim]
            e0 = self.time_projection(self.time_projection_act(e))  # [B, L, dim*6]
            e0 = e0.reshape(batch_size, -1, 6, self.dim)

        # Text embedding: skip MLP if context is already embedded (mx.array)
        if isinstance(context, mx.array):
            # Pre-embedded: expand to batch size if needed
            context_batch = context
            if context_batch.shape[0] == 1 and batch_size > 1:
                context_batch = mx.broadcast_to(
                    context_batch, (batch_size,) + context_batch.shape[1:]
                )
        else:
            context_batch = self.embed_text(context)

        # Pre-compute attention mask from seq_lens (constant across all blocks)
        attn_mask = None
        w_dtype = _linear_dtype(self.patch_embedding_proj)
        if any(sl < seq_len for sl in seq_lens_list):
            attn_mask = mx.zeros((batch_size, 1, 1, seq_len), dtype=w_dtype)
            for i, sl in enumerate(seq_lens_list):
                attn_mask[i, :, :, sl:] = -1e9

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens_list,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context_batch,
            context_lens=None,
            rope_cos_sin=rope_cos_sin,
            attn_mask=attn_mask,
        )

        # Run transformer blocks
        for i, block in enumerate(self.blocks):
            kv = cross_kv_caches[i] if cross_kv_caches is not None else None
            x = block(x, cross_kv_cache=kv, **kwargs)

        # Output head
        x = self.head(x, e)

        # Unpatchify
        outputs = self.unpatchify(x, grid_sizes)
        return [u.astype(mx.float32) for u in outputs]


class WanS2VModel(WanModel):
    """Wan 2.2 Speech-to-Video model (Phase 2 — real forward).

    Extends :class:`WanModel` with the S2V-specific parameter set:
      * ``casual_audio_encoder`` (wav2vec2 layer-sum + MotionEncoder_tc)
      * ``audio_injector.injector[K]`` cross-attention + AdaLN sub-layers
      * ``frame_packer`` motion-history projections
      * ``cond_encoder`` pose/overlay projection
      * ``trainable_cond_mask`` 3-way (noise/ref/motion) mask embedding

    Forward-pass concept (design doc §3):
      1. Patchify denoise tokens → (B, F*N, D), grid = (F, H_lat/2, W_lat/2)
      2. Patchify ref image → (B, N, D), append; assign segment id = 1.
      3. FramePack motion history (if enabled) → append 3 buckets; segment id = 2.
      4. Build combined RoPE frequencies:
            noise:  temporal position [0..F-1], height/width per patch
            ref:    temporal position 30 (per design; single "future" slot)
            motion: negative temporal positions (packed by bucket)
      5. Standard time embedding — noise uses ``t``; ref/motion use 0 iff
         ``zero_timestep=True``. Encoded as per-token e vector.
      6. Run 40 blocks; after each block ``k`` where the injector has an entry,
         call ``audio_injector.inject(...)`` which residual-adds cross-attn
         audio→video on the *denoise slice only*.
      7. Head → unpatchify first F*N tokens → return epsilon predictions of
         shape [C_out, F, H, W] per batch element.

    Ref RoPE temporal index (verified against kijai ``rope_encode_comfy``,
    line 2703): ``t_start = max(30, F_video + 9)``. For F=21 (81-frame clip)
    this is 30, matching the design-doc value; for longer clips it grows.

    The current implementation deliberately SKIPS per-token RoPE reassignment
    for ref/motion — noise-slice RoPE only. This is a functional first cut;
    numerical parity with kijai will require reworking the ``rope_cos_sin``
    to take three separate temporal grids and concatenate.
    """

    def __init__(self, config: WanModelConfig):
        super().__init__(config)
        assert config.model_type == "s2v", (
            f"WanS2VModel requires config.model_type='s2v', got {config.model_type!r}"
        )

        # Local imports to avoid circular imports and to keep the T2V-only
        # code path from importing S2V modules it never uses.
        from .audio_encoder import CausalAudioEncoder
        from .s2v_utils import (
            AudioInjector,
            CondEncoderProj,
            FramePacker,
            TrainableCondMask,
        )

        dim = config.dim

        # Wav2vec2 feature encoder (weighted layer-sum + causal conv stack).
        self.casual_audio_encoder = CausalAudioEncoder(
            dim=config.audio_dim,
            num_layers=25,  # matches wav2vec2-large-xls-r hidden_states count
            out_dim=dim,
            num_token=config.num_audio_token,
            need_global=config.enable_adain,
        )

        # Audio cross-attention + AdaLN injectors at selected blocks.
        self.audio_injector = AudioInjector(
            dim=dim,
            num_heads=config.num_heads,
            inject_layers=config.audio_inject_layers,
            enable_adain=config.enable_adain,
            adain_dim=dim,
            qk_norm=config.qk_norm,
            eps=config.eps,
        )

        # Framepack motion-history projections (Conv3d params in raw layout).
        if config.enable_framepack:
            self.frame_packer = FramePacker(
                inner_dim=dim,
                in_channels=config.vae_z_dim,
                zip_frame_buckets=(1, 2, 16),
                drop_mode=config.framepack_drop_mode,
            )

        # Pose / overlay conditioning projection (Conv3d in PyTorch).
        if config.cond_dim > 0:
            self.cond_encoder = CondEncoderProj(
                out_ch=dim,
                in_ch=config.cond_dim,
                kernel=config.patch_size,
            )

        # 3-way (noise=0, ref=1, motion=2) token-id embedding.
        self.trainable_cond_mask = TrainableCondMask(num_embeddings=3, dim=dim)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _patchify_5d(self, x: mx.array):
        """Patchify a (C, F, H, W) tensor with the shared linear projection.

        Wrapper around :meth:`WanModel._patchify` that returns the same
        (patches[1, L, D], (F', H', W')) tuple.
        """
        return self._patchify(x)

    def _ref_patch_tokens(self, ref_image_latent: mx.array) -> tuple:
        """Patchify a single reference-image latent frame → tokens + seg embed.

        Matches kijai wanvideo/modules/model.py lines 2636-2650: ref latent is
        patch-embedded, gets ``cond_mask_weight[1]`` (segment 1), then when the
        full noise+ref+padding sequence is assembled kijai adds
        ``cond_mask_weight[0]`` to the entire sequence — so ref tokens end up
        with BOTH ``weight[0]`` and ``weight[1]`` added on top of the raw patch
        embedding. Noise tokens only get ``weight[0]``.

        In MLX we bake both segment vectors into the ref tokens here (noise
        tokens already receive ``segment_embedding(0)`` in ``__call__``). This
        makes the ref token project into the same video-space subspace as noise
        (via weight[0]) while retaining the segment-1 identity signal, which
        is what actually locks the generated face to the reference.

        Args:
            ref_image_latent: (C=16, 1, H_lat, W_lat) or (C, F_ref, H, W).

        Returns:
            (ref_tokens [1, L_ref, D], grid (F_ref', H', W'))
        """
        p, gs = self._patchify_5d(ref_image_latent)
        # Add BOTH the noise-baseline (seg 0) and ref-segment (seg 1) embeddings
        # to match kijai's post-concat behaviour (Phase 3 identity-preservation fix).
        seg0 = self.trainable_cond_mask.segment_embedding(
            0, p.shape[1], dtype=p.dtype
        )[None, :, :]
        seg1 = self.trainable_cond_mask.segment_embedding(
            1, p.shape[1], dtype=p.dtype
        )[None, :, :]
        return p + seg0.astype(p.dtype) + seg1.astype(p.dtype), gs

    def _motion_bucket_shapes(self, H_lat: int, W_lat: int) -> list:
        """Return list of (F_seg, H_seg, W_seg) per motion bucket after patchify.

        Matches FramePacker kernels: fine=(1,2,2), medium=(2,4,4), coarse=(4,8,8).
        Given noise-slice grid (H_lat, W_lat) — the un-patchified motion latent
        spatial dim is (H_lat * patch_h, W_lat * patch_w) since latent H/W is
        the same as the noise latent's — the buckets project to::

            fine   (1, H_lat, W_lat)          via (1,2,2)  → (1, H_lat/2, W_lat/2)  … wait
            no:
        Actually kijai calls ``rope_encode_comfy(1, lat_height, lat_width, ...)``
        with ``steps_h/w`` overridden per bucket, where ``lat_height/width`` is
        the un-patchified latent size. So the bucket H'/W' is::

            fine   : (H_lat, W_lat)    // (2,2) -> H_lat/2, W_lat/2
            medium : (H_lat, W_lat)    // (4,4) -> H_lat/4, W_lat/4
            coarse : (H_lat, W_lat)    // (8,8) -> H_lat/8, W_lat/8

        These MATCH the noise patch grid (H_noise = H_lat/2). So::

            H_noise = H_lat / 2, W_noise = W_lat / 2
            fine.h  = H_noise         = H_lat / 2
            medium.h = H_lat / 4      = H_noise / 2
            coarse.h = H_lat / 8      = H_noise / 4
        """
        # Motion buckets from the FramePacker output. Latent H/W == 2 * noise
        # patch H/W (since patch kernel is (1,2,2)); we're given noise-grid H/W
        # so H_lat = 2 * H_noise. The motion buckets' rope H/W:
        return [
            (1, H_lat, W_lat),          # fine   proj kernel (1,2,2) → 1 frame
            (1, H_lat // 2, W_lat // 2),  # medium proj_2x (2,4,4)   → 1 frame
            (4, H_lat // 4, W_lat // 4),  # coarse proj_4x (4,8,8)   → 4 frames
        ]

    def prepare_rope_s2v(
        self,
        noise_grid: tuple,
        ref_grid: tuple | None = None,
        motion_shapes: list | None = None,
        dtype: mx.Dtype | None = None,
    ) -> tuple:
        """Precompute (cos, sin) for the S2V multi-segment sequence.

        Segment order (verified against kijai ``forward``, model.py lines
        2637-2734):

            [noise (t=[0..F-1]),
             ref   (t=[max(30, F+9) + i for i in F_ref]),
             motion_fine   (t=[-1]),
             motion_medium (t=[-3]),
             motion_coarse (t=[-19..-16])]

        Args:
            noise_grid: (F, H, W) — noise latent patch grid.
            ref_grid:   (F_ref, H_ref, W_ref) or None.
            motion_shapes: list of (F_i, H_i, W_i) per bucket (fine, medium,
                coarse) or None. Use :meth:`_motion_bucket_shapes` if you have
                the raw latent H/W but not the bucket grids.
            dtype: output dtype for cos/sin.

        Returns:
            (cos_f, sin_f) each of shape (seq_total, 1, half_d).
        """
        from .attention import _linear_dtype as _dt
        if dtype is None:
            dtype = _dt(self.patch_embedding_proj)

        F, H, W = noise_grid
        segments: list = [{"t_indices": list(range(F)), "h": H, "w": W}]

        if ref_grid is not None:
            F_ref, H_ref, W_ref = ref_grid
            t_ref = max(30, F + 9)
            segments.append(
                {
                    "t_indices": [t_ref + i for i in range(F_ref)],
                    "h": H_ref,
                    "w": W_ref,
                }
            )

        if motion_shapes:
            # kijai concat order: [fine (t=-1), medium (t=-3), coarse (t=-19..-16)].
            motion_t_starts = [-1, -3, -19]
            for (F_m, H_m, W_m), t_start in zip(motion_shapes, motion_t_starts):
                t_indices = [t_start + i for i in range(F_m)]
                segments.append(
                    {"t_indices": t_indices, "h": H_m, "w": W_m}
                )

        return rope_precompute_cos_sin_segments(segments, self.freqs, dtype=dtype)

    def _time_embed_scalar(self, t: mx.array, batch_size: int) -> tuple:
        """Compute (e, e0): pre-projection ``e`` and block modulation ``e0``.

        Returns:
            e:  (B, D)        — used by ``head(x, e)``
            e0: (B, 1, 6, D)  — used by block modulation
        """
        sinusoid = t[..., None].astype(mx.float32) * self._inv_freq
        sin_emb = mx.concatenate([mx.cos(sinusoid), mx.sin(sinusoid)], axis=-1)
        e = self.time_embedding_1(
            self.time_embedding_act(self.time_embedding_0(sin_emb))
        )  # (B, D)
        e0 = self.time_projection(self.time_projection_act(e))  # (B, D*6)
        e0 = e0.reshape(batch_size, 1, 6, self.dim)
        return e, e0

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(
        self,
        x_list: list,
        t: mx.array,
        context,
        seq_len: int,
        audio_input: mx.array | None = None,
        ref_image_latent: mx.array | None = None,
        motion_history_latent: mx.array | None = None,
        cross_kv_caches: list | None = None,
        rope_cos_sin: tuple | None = None,
        y=None,
    ) -> list:
        """S2V forward pass.

        Args:
            x_list: list of denoise latents [C, F, H, W] per batch element.
            t: (B,) timesteps.
            context: text embeddings (list or (B, text_len, D)).
            seq_len: sequence-length hint (padding target for the noise slice).
                Ref/motion tokens are appended *after* padding.
            audio_input: (B, num_layers=25, audio_dim=1024, T_audio) wav2vec2 stack.
                If None, audio injection is skipped (silent fallback).
            ref_image_latent: (C, 1, H_lat, W_lat) or None. When provided,
                patchified and appended to every batch element.
            motion_history_latent: (B, C, F_motion, H_lat, W_lat) or None.
                Passed through the framepack module if enabled.
            cross_kv_caches, rope_cos_sin, y: passed through for parity with
                :class:`WanModel.__call__` but ``y`` (channel concat) is
                unused for S2V.

        Returns:
            list of (C_out, F, H, W) tensors — one epsilon prediction per batch.
        """
        from .attention import _linear_dtype as _dt
        w_dtype = _dt(self.patch_embedding_proj)
        batch_size = len(x_list)

        # ------------------------------------------------------------------
        # 1. Patchify denoise tokens (noise slice)
        # ------------------------------------------------------------------
        patches = []
        grid_sizes = []
        seq_lens_list = []
        for vid in x_list:
            p, gs = self._patchify_5d(vid)  # (1, L, D), (F, H, W)
            # Segment id = 0 for noise tokens.
            seg = self.trainable_cond_mask.segment_embedding(0, p.shape[1], dtype=p.dtype)
            p = p + seg[None, :, :].astype(p.dtype)
            patches.append(p)
            grid_sizes.append(gs)
            seq_lens_list.append(p.shape[1])

        F_grid, H_grid, W_grid = grid_sizes[0]
        F_video = F_grid
        N_per_frame = H_grid * W_grid
        seq_noise = F_video * N_per_frame  # tokens per batch in the noise slice
        assert seq_noise == seq_lens_list[0], (
            f"seq_noise {seq_noise} != patchified length {seq_lens_list[0]}"
        )

        # ------------------------------------------------------------------
        # 2. Reference-image tokens (optional)
        # ------------------------------------------------------------------
        ref_tokens = None
        ref_grid = None
        if ref_image_latent is not None:
            ref_tokens, ref_grid = self._ref_patch_tokens(ref_image_latent)
            # Broadcast to batch dim.
            if ref_tokens.shape[0] != batch_size:
                ref_tokens = mx.broadcast_to(
                    ref_tokens, (batch_size,) + ref_tokens.shape[1:]
                )

        # ------------------------------------------------------------------
        # 3. Framepack motion-history tokens (optional)
        # ------------------------------------------------------------------
        motion_token_list: list[mx.array] = []
        if motion_history_latent is not None and hasattr(self, "frame_packer"):
            # frame_packer.pack_motion_frames returns a list of (B, L_i, D)
            # tokens at three temporal scales.
            motion_token_list = self.frame_packer.pack_motion_frames(
                motion_history_latent
            )
            # Add motion segment embedding to each bucket, and broadcast to
            # batch_size (motion_history_latent is normally B=1 even under CFG).
            new_list = []
            for mt in motion_token_list:
                seg = self.trainable_cond_mask.segment_embedding(
                    2, mt.shape[1], dtype=mt.dtype
                )
                mt = mt + seg[None, :, :].astype(mt.dtype)
                if mt.shape[0] != batch_size:
                    mt = mx.broadcast_to(
                        mt, (batch_size,) + mt.shape[1:]
                    )
                new_list.append(mt)
            motion_token_list = new_list

        # ------------------------------------------------------------------
        # 4. Build the noise token slice (padded to seq_len) and concat ref/motion.
        # ------------------------------------------------------------------
        # Stack noise patches and pad to seq_len (padding rows are zero and get
        # masked out below by attn_mask on the noise slice).
        noise_padded = []
        for p in patches:
            if p.shape[1] < seq_len:
                pad = mx.zeros((1, seq_len - p.shape[1], self.dim), dtype=p.dtype)
                p = mx.concatenate([p, pad], axis=1)
            noise_padded.append(p)
        noise = mx.concatenate(noise_padded, axis=0)  # (B, seq_len, D)

        # Concat ref + motion tokens along the sequence axis.
        parts = [noise]
        if ref_tokens is not None:
            parts.append(ref_tokens.astype(noise.dtype))
        for mt in motion_token_list:
            parts.append(mt.astype(noise.dtype))
        x = mx.concatenate(parts, axis=1) if len(parts) > 1 else noise

        seq_orig = seq_noise  # tokens in the "noise" slice (= F*N)
        seq_total = x.shape[1]

        # ------------------------------------------------------------------
        # 5. Time embedding.  For zero_timestep=True, ref/motion get t=0.
        # ------------------------------------------------------------------
        if t.ndim == 0:
            t = t[None]
        e_pre, e_noise_mod = self._time_embed_scalar(t.astype(mx.float32), batch_size)
        # e_pre: (B, D) for head; e_noise_mod: (B, 1, 6, D) for blocks.

        if self.config.zero_timestep and (ref_tokens is not None or motion_token_list):
            # For ref/motion tokens, use t=0 modulation.
            zero_t = mx.zeros_like(t.astype(mx.float32))
            _, e_ref_mod = self._time_embed_scalar(zero_t, batch_size)
            # Build per-token modulation: (B, seq_total, 6, D) — noise slice uses
            # e_noise_mod broadcast over seq_orig, ref/motion slice uses e_ref_mod.
            e_block = mx.concatenate(
                [
                    mx.broadcast_to(e_noise_mod, (batch_size, seq_orig, 6, self.dim)),
                    mx.broadcast_to(
                        e_ref_mod, (batch_size, seq_total - seq_orig, 6, self.dim)
                    ),
                ],
                axis=1,
            )
        else:
            # Scalar-per-batch: broadcast in the block modulation add.
            e_block = e_noise_mod  # (B, 1, 6, D)

        # ------------------------------------------------------------------
        # 6. Text context (as usual)
        # ------------------------------------------------------------------
        if isinstance(context, mx.array):
            context_batch = context
            if context_batch.shape[0] == 1 and batch_size > 1:
                context_batch = mx.broadcast_to(
                    context_batch, (batch_size,) + context_batch.shape[1:]
                )
        else:
            context_batch = self.embed_text(context)

        # ------------------------------------------------------------------
        # 7. Attention mask — only mask the padding *inside* the noise slice.
        #     Ref/motion tokens are always valid.
        # ------------------------------------------------------------------
        attn_mask = None
        if any(sl < seq_len for sl in seq_lens_list) or seq_total > seq_len:
            attn_mask = mx.zeros((batch_size, 1, 1, seq_total), dtype=w_dtype)
            for i, sl in enumerate(seq_lens_list):
                if sl < seq_len:
                    attn_mask[i, :, :, sl:seq_len] = -1e9

        # ------------------------------------------------------------------
        # 8. Encode audio (once) into local + global tokens.
        # ------------------------------------------------------------------
        audio_emb = None
        audio_emb_global = None
        if audio_input is not None:
            enc_out = self.casual_audio_encoder(audio_input)
            if isinstance(enc_out, tuple):
                audio_emb, audio_emb_global = enc_out
            else:
                audio_emb, audio_emb_global = enc_out, None
            # Broadcast to batch dim if wav2vec was extracted for B=1.
            if audio_emb.shape[0] == 1 and batch_size > 1:
                audio_emb = mx.broadcast_to(
                    audio_emb, (batch_size,) + audio_emb.shape[1:]
                )
                if audio_emb_global is not None:
                    audio_emb_global = mx.broadcast_to(
                        audio_emb_global,
                        (batch_size,) + audio_emb_global.shape[1:],
                    )
            # Truncate/pad audio_emb along the F axis to match the video F.
            F_audio = audio_emb.shape[1]
            if F_audio < F_video:
                pad = mx.zeros(
                    (batch_size, F_video - F_audio, audio_emb.shape[2], self.dim),
                    dtype=audio_emb.dtype,
                )
                audio_emb = mx.concatenate([audio_emb, pad], axis=1)
                if audio_emb_global is not None:
                    pad_g = mx.zeros(
                        (batch_size, F_video - F_audio, 1, self.dim),
                        dtype=audio_emb_global.dtype,
                    )
                    audio_emb_global = mx.concatenate(
                        [audio_emb_global, pad_g], axis=1
                    )
            elif F_audio > F_video:
                audio_emb = audio_emb[:, :F_video]
                if audio_emb_global is not None:
                    audio_emb_global = audio_emb_global[:, :F_video]

        # ------------------------------------------------------------------
        # 9. RoPE for the full [noise, ref, motion] concatenated sequence.
        # ------------------------------------------------------------------
        # If the caller pre-computed cos/sin covering only the noise slice
        # (via WanModel.prepare_rope), we rebuild it here to cover ref+motion.
        # This matches kijai's rope_encode_comfy (model.py lines 2210, 2704):
        #   ref RoPE uses  t_start = max(30, F_video + 9)
        #   motion RoPE uses t_start = -1 / -3 / -19 for fine/medium/coarse.
        need_rebuild_rope = (
            rope_cos_sin is None
            or rope_cos_sin[0].shape[0] < seq_total
        )
        if need_rebuild_rope:
            motion_shapes = None
            if motion_token_list:
                # Derive per-bucket (F', H', W') from the latent H/W (= 2 * H_grid).
                H_lat = H_grid * self._patch_size[1]
                W_lat = W_grid * self._patch_size[2]
                motion_shapes = self._motion_bucket_shapes(H_lat, W_lat)
            rope_cos_sin = self.prepare_rope_s2v(
                noise_grid=(F_grid, H_grid, W_grid),
                ref_grid=ref_grid if ref_tokens is not None else None,
                motion_shapes=motion_shapes,
                dtype=w_dtype,
            )

        # Padding within the noise slice (if any) still needs masking, but the
        # multi-segment cos/sin already covers all real ref/motion tokens.
        kwargs = dict(
            e=e_block,
            seq_lens=[seq_total] * batch_size,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context_batch,
            context_lens=None,
            rope_cos_sin=rope_cos_sin,
            attn_mask=attn_mask,
        )
        for i, block in enumerate(self.blocks):
            kv = cross_kv_caches[i] if cross_kv_caches is not None else None
            x = block(x, cross_kv_cache=kv, **kwargs)
            # Post-block audio injection.
            if audio_emb is not None and i in self.audio_injector.injected_block_id:
                x = self.audio_injector.inject(
                    x,
                    block_idx=i,
                    audio_emb=audio_emb,
                    audio_emb_global=audio_emb_global,
                    F=F_video,
                    N=N_per_frame,
                    seq_orig=seq_orig,
                )

        # ------------------------------------------------------------------
        # 10. Head (only on the denoise slice) + unpatchify.
        # ------------------------------------------------------------------
        x_noise = x[:, :seq_orig, :]
        # Head modulation uses the pre-projection e (B, D).
        x_noise = self.head(x_noise, e_pre)

        outputs = self.unpatchify(x_noise, grid_sizes)
        return [u.astype(mx.float32) for u in outputs]
