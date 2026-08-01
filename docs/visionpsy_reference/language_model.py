import inspect
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _sdpa_supports_native_gqa() -> bool:

    try:
        params = inspect.signature(F.scaled_dot_product_attention).parameters
        return "enable_gqa" in params
    except (TypeError, ValueError):
        try:
            q = torch.zeros(1, 2, 1, 4)
            k = torch.zeros(1, 1, 1, 4)
            v = torch.zeros(1, 1, 1, 4)
            F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
            return True
        except TypeError:
            return False
        except Exception:
            return True

_SDPA_HAS_GQA = _sdpa_supports_native_gqa()

class RMSNorm(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(cfg.lm_hidden_dim))
        self.eps = cfg.lm_rms_eps
        self._normalized_shape = (cfg.lm_hidden_dim,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return F.rms_norm(x, self._normalized_shape, self.weight, self.eps)

class RotaryEmbedding(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        assert cfg.lm_hidden_dim % cfg.lm_n_heads == 0, "Hidden dimension must be divisible by number of heads"

        self.dim = cfg.lm_hidden_dim // cfg.lm_n_heads
        self.base = cfg.lm_re_base
        self.max_seq_len = cfg.lm_max_position_embeddings
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self.original_max_seq_len = cfg.lm_max_position_embeddings
        self.attention_scaling = cfg.lm_attn_scaling

        positions = torch.arange(self.max_seq_len, dtype=torch.float)
        freqs = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_cached = emb.cos() * self.attention_scaling
        sin_cached = emb.sin() * self.attention_scaling
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        if position_ids.dtype != torch.long:
            position_ids = position_ids.long()
        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]
        return cos, sin

    def extend_cache(self, new_max_seq_len: int) -> None:

        positions = torch.arange(
            new_max_seq_len, dtype=torch.float, device=self.cos_cached.device,
        )
        freqs = positions.unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.cos_cached = emb.cos() * self.attention_scaling
        self.sin_cached = emb.sin() * self.attention_scaling
        self.original_max_seq_len = new_max_seq_len

def apply_rotary_pos_embd(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    half = q.shape[-1] // 2
    cos_h = cos[..., :half]
    sin_h = sin[..., :half]

    q1 = q[..., :half]
    q2 = q[..., half:]
    k1 = k[..., :half]
    k2 = k[..., half:]

    q_embed = torch.cat(
        (q1 * cos_h - q2 * sin_h, q2 * cos_h + q1 * sin_h),
        dim=-1,
    )
    k_embed = torch.cat(
        (k1 * cos_h - k2 * sin_h, k2 * cos_h + k1 * sin_h),
        dim=-1,
    )
    return q_embed, k_embed

class LanguageModelGroupedQueryAttention(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.n_heads = cfg.lm_n_heads
        self.n_kv_heads = cfg.lm_n_kv_heads
        self.embd_dim = cfg.lm_hidden_dim
        self.dropout = cfg.lm_dropout

        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.embd_dim % self.n_heads == 0, "embd_dim must be divisible by num_heads"

        self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.head_dim = self.embd_dim // self.n_heads

        self.q_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        self.k_proj = nn.Linear(self.embd_dim, self.head_dim * self.n_kv_heads, bias=False)
        self.v_proj = nn.Linear(self.embd_dim, self.head_dim * self.n_kv_heads, bias=False)
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)

        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        self.sdpa = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.sdpa:
            print("Warning: scaled dot product attention not available, using standard attention in LM.")

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask=None, block_kv_cache=None, start_pos: int = 0) -> tuple[torch.Tensor, dict]:

        B, T_curr, C = x.size()

        q_curr = self.q_proj(x).view(B, T_curr, self.n_heads, self.head_dim).transpose(1, 2)
        k_curr = self.k_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v_curr = self.v_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k_rotated = apply_rotary_pos_embd(q_curr, k_curr, cos, sin)

        use_static_cache = (
            block_kv_cache is not None
            and block_kv_cache.get('cache_max_length') is not None
        )

        static_compile_decode = (
            use_static_cache
            and T_curr == 1
            and isinstance(start_pos, torch.Tensor)
        )
        sdpa_attn_mask_static: torch.Tensor | None = None

        if static_compile_decode:
            cache_max_length = block_kv_cache['cache_max_length']
            idx = start_pos.unsqueeze(0)
            block_kv_cache['key_cache'].index_copy_(2, idx, k_rotated)
            block_kv_cache['value_cache'].index_copy_(2, idx, v_curr)

            k = block_kv_cache['key_cache']
            v = block_kv_cache['value_cache']
            T_kv = cache_max_length

            arange = torch.arange(
                cache_max_length, device=q.device, dtype=torch.long,
            )
            sdpa_attn_mask_static = (arange <= start_pos).view(1, 1, 1, -1)
        elif use_static_cache:
            cache_max_length = block_kv_cache['cache_max_length']
            if block_kv_cache.get('key_cache') is None:
                cache_shape = (B, self.n_kv_heads, cache_max_length, self.head_dim)
                block_kv_cache['key_cache'] = torch.zeros(
                    cache_shape, dtype=k_rotated.dtype, device=k_rotated.device,
                )
                block_kv_cache['value_cache'] = torch.zeros(
                    cache_shape, dtype=v_curr.dtype, device=v_curr.device,
                )
            new_pos = start_pos + T_curr
            block_kv_cache['key_cache'][:, :, start_pos:new_pos] = k_rotated
            block_kv_cache['value_cache'][:, :, start_pos:new_pos] = v_curr
            k = block_kv_cache['key_cache'][:, :, :new_pos]
            v = block_kv_cache['value_cache'][:, :, :new_pos]
            T_kv = new_pos
        else:
            is_prefill = block_kv_cache is None
            if not is_prefill and block_kv_cache.get('key') is not None:
                k = block_kv_cache['key']
                v = block_kv_cache['value']
                k = torch.cat([k, k_rotated], dim=2)
                v = torch.cat([v, v_curr], dim=2)
                block_kv_cache['key'] = k
                block_kv_cache['value'] = v
            else:
                k = k_rotated
                v = v_curr
                block_kv_cache = {'key': k, 'value': v}

        if _SDPA_HAS_GQA and self.sdpa and x.device.type != 'mps':
            k_exp = k
            v_exp = v
        else:
            k_exp = k.repeat_interleave(self.n_kv_groups, dim=1)
            v_exp = v.repeat_interleave(self.n_kv_groups, dim=1)

        T_kv = k_exp.size(2)

        if self.sdpa and x.device.type != 'mps':
            if sdpa_attn_mask_static is not None:
                sdpa_attn_mask = sdpa_attn_mask_static
                sdpa_is_causal = False
            elif attention_mask is None:
                sdpa_attn_mask = None
                sdpa_is_causal = (T_curr == T_kv and T_curr > 1)
            else:
                if attention_mask.size(-1) < T_kv:
                    pad = attention_mask.new_ones(
                        attention_mask.size(0), T_kv - attention_mask.size(-1)
                    )
                    key_padding = torch.cat([attention_mask, pad], dim=-1).bool()
                else:
                    key_padding = attention_mask[:, :T_kv].bool()
                if T_curr == T_kv and T_curr > 1:
                    causal = torch.ones(
                        T_curr, T_kv, dtype=torch.bool, device=q.device,
                    ).tril()
                else:
                    causal = torch.ones(
                        T_curr, T_kv, dtype=torch.bool, device=q.device,
                    )
                sdpa_attn_mask = (
                    causal.unsqueeze(0).unsqueeze(0)
                    & key_padding.unsqueeze(1).unsqueeze(2)
                )
                sdpa_is_causal = False

            sdpa_kwargs = {
                "attn_mask": sdpa_attn_mask,
                "dropout_p": self.dropout if self.training else 0.0,
                "is_causal": sdpa_is_causal,
            }
            if _SDPA_HAS_GQA and k_exp.size(1) != q.size(1):
                sdpa_kwargs["enable_gqa"] = True

            y = torch.nn.functional.scaled_dot_product_attention(
                q, k_exp, v_exp,
                **sdpa_kwargs,
            )
        else:
            attn = torch.matmul(q, k_exp.transpose(2, 3)) / math.sqrt(self.head_dim)
            if T_curr == T_kv and T_curr > 1:
                causal_mask_val = torch.tril(
                    torch.ones(T_curr, T_curr, device=x.device, dtype=torch.bool),
                ).view(1, 1, T_curr, T_curr)
                attn = attn.masked_fill(~causal_mask_val, float('-inf'))

            if attention_mask is not None:
                additive = (
                    1.0 - attention_mask[:, :T_kv].unsqueeze(1).unsqueeze(2).float()
                ) * torch.finfo(q.dtype).min
                attn = attn + additive

            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v_exp
        y = y.transpose(1, 2).contiguous().view(B, T_curr, C)
        y = self.out_proj(y)
        y = self.resid_dropout(y)

        return y, block_kv_cache

class LanguageModelMLP(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.embd_dim = cfg.lm_hidden_dim
        self.inter_dim = cfg.lm_inter_dim

        self.activation_fn = F.silu
        self.gate_up_proj = nn.Linear(self.embd_dim, 2 * self.inter_dim, bias=False)
        self.down_proj = nn.Linear(self.inter_dim, self.embd_dim, bias=False)

        self._register_load_state_dict_pre_hook(self._fuse_legacy_gate_up_hook)

    @staticmethod
    def _fuse_legacy_gate_up_hook(
        state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):

        gate_key = f"{prefix}gate_proj.weight"
        up_key = f"{prefix}up_proj.weight"
        fused_key = f"{prefix}gate_up_proj.weight"
        if (
            gate_key in state_dict
            and up_key in state_dict
            and fused_key not in state_dict
        ):
            gate_w = state_dict.pop(gate_key)
            up_w = state_dict.pop(up_key)
            state_dict[fused_key] = torch.cat([gate_w, up_w], dim=0)

    def forward(self, x):

        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(self.activation_fn(gate) * up)

class LanguageModelBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = LanguageModelMLP(cfg)
        self.attn = LanguageModelGroupedQueryAttention(cfg)
        self.norm1 = RMSNorm(cfg)
        self.norm2 = RMSNorm(cfg)
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask: torch.Tensor=None, block_kv_cache: dict=None, start_pos: int = 0):

        res = x
        x = self.norm1(x)
        x, block_kv_cache = self.attn(x, cos, sin, attention_mask, block_kv_cache, start_pos=start_pos)
        x = res + x

        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = res + x

        return x, block_kv_cache

class LanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lm_use_tokens = cfg.lm_use_tokens
        self.lm_tie_weights = cfg.lm_tie_weights

        self.token_embedding = nn.Embedding(cfg.lm_vocab_size, cfg.lm_hidden_dim)
        self.rotary_embd = RotaryEmbedding(cfg)
        self.blocks = nn.ModuleList([
            LanguageModelBlock(cfg) for _ in range(cfg.lm_n_blocks)
        ])
        self.norm = RMSNorm(cfg)
        self.head = nn.Linear(cfg.lm_hidden_dim, cfg.lm_vocab_size, bias=False)
        if self.lm_tie_weights:
            self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor=None, kv_cache: list[dict]=None, start_pos: int=0):

        if self.lm_use_tokens:
            x = self.token_embedding(x)

        B, T_curr, _ = x.size()

        if isinstance(start_pos, torch.Tensor):
            offsets = torch.arange(0, T_curr, device=x.device, dtype=torch.long)
            current_position_ids = (offsets + start_pos).unsqueeze(0).expand(B, -1)
        else:
            current_position_ids = torch.arange(start_pos, start_pos + T_curr, device=x.device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary_embd(current_position_ids)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)

        for i, block in enumerate(self.blocks):
            x, kv_cache[i] = block(x, cos, sin, attention_mask, kv_cache[i], start_pos=start_pos)

        x = self.norm(x)

        if self.lm_use_tokens:
            x = self.head(x)

        return x, kv_cache

    @torch.inference_mode()
    def generate(self, inputs: torch.Tensor, max_new_tokens: int=20):

        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        generated_outputs = inputs.clone()

        prompt_output, kv_cache_list = self.forward(
            generated_outputs,
            attention_mask=None,
            kv_cache=None,
            start_pos=0
        )
        last_output = prompt_output[:, -1, :]

        for i in range(max_new_tokens):
            if self.lm_use_tokens:
                next_output = torch.argmax(last_output, dim=-1, keepdim=True)
            else:
                next_output = last_output.unsqueeze(1)

            generated_outputs = torch.cat((generated_outputs, next_output), dim=1)
            current_token_start_pos = generated_outputs.size(1) - 1

            if i == max_new_tokens - 1:
                break

            decode_step_output, kv_cache_list = self.forward(
                next_output,
                attention_mask=None,
                kv_cache=kv_cache_list,
                start_pos=current_token_start_pos
            )
            last_output = decode_step_output[:, -1, :]
        return generated_outputs

    @classmethod
    def from_pretrained(cls, cfg):
        from transformers import AutoConfig
        from huggingface_hub import hf_hub_download
        import safetensors
        import torch.nn.init as init
        import json
        from huggingface_hub.utils import EntryNotFoundError
        hf_config = AutoConfig.from_pretrained(cfg.lm_model_type)
        original_vocab_size = hf_config.vocab_size
        cfg.lm_hidden_dim = hf_config.hidden_size
        cfg.lm_inter_dim = hf_config.intermediate_size
        cfg.lm_rms_eps = hf_config.rms_norm_eps
        cfg.lm_re_base = hf_config.rope_theta
        cfg.lm_max_position_embeddings = hf_config.max_position_embeddings
        if hasattr(cfg, 'lm_vocab_size'):
            if cfg.lm_vocab_size < original_vocab_size:
                raise ValueError(f"Config vocab size ({cfg.lm_vocab_size}) is smaller than pretrained model vocab size ({original_vocab_size})")
        else:
            cfg.lm_vocab_size = original_vocab_size
        cfg.lm_n_heads = hf_config.num_attention_heads
        cfg.lm_n_kv_heads = hf_config.num_key_value_heads
        cfg.lm_dropout = hf_config.attention_dropout
        cfg.lm_n_blocks = hf_config.num_hidden_layers
        model = cls(cfg)
        try:
            index_path = hf_hub_download(repo_id=cfg.lm_model_type, filename="model.safetensors.index.json")
            with open(index_path, 'r') as f:
                index = json.load(f)
            safetensors_filenames = sorted(list(set(index['weight_map'].values())))
            safetensors_files = [hf_hub_download(repo_id=cfg.lm_model_type, filename=fn) for fn in safetensors_filenames]
        except EntryNotFoundError:
            safetensors_files = [hf_hub_download(repo_id=cfg.lm_model_type, filename="model.safetensors")]

        sd = model.state_dict()
        mapping = {
            'model.embed_tokens.weight': 'token_embedding.weight',
            'model.norm.weight': 'norm.weight'
        }
        for i in range(cfg.lm_n_blocks):
            layer_prefix = f'model.layers.{i}.'
            block_prefix = f'blocks.{i}.'
            mapping.update({
                f"{layer_prefix}self_attn.q_proj.weight": f"{block_prefix}attn.q_proj.weight",
                f"{layer_prefix}self_attn.k_proj.weight": f"{block_prefix}attn.k_proj.weight",
                f"{layer_prefix}self_attn.v_proj.weight": f"{block_prefix}attn.v_proj.weight",
                f"{layer_prefix}self_attn.o_proj.weight": f"{block_prefix}attn.out_proj.weight",
                f"{layer_prefix}mlp.down_proj.weight": f"{block_prefix}mlp.down_proj.weight",
                f"{layer_prefix}input_layernorm.weight": f"{block_prefix}norm1.weight",
                f"{layer_prefix}post_attention_layernorm.weight": f"{block_prefix}norm2.weight"
            })
        has_extended_embeddings = False
        loaded_keys = set()
        for safetensors_file in safetensors_files:
            with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
                for hf_key, our_key in mapping.items():
                    if our_key in loaded_keys:
                        continue
                    if hf_key in f.keys() and our_key in sd:
                        tensor = f.get_tensor(hf_key)
                        if hf_key == 'model.embed_tokens.weight' and tensor.shape[0] != sd[our_key].shape[0]:
                            has_extended_embeddings = True
                            print(f"Extending token embeddings from {tensor.shape} to {sd[our_key].shape}")
                            sd[our_key][:tensor.shape[0]].copy_(tensor)
                            std = 0.02
                            init.normal_(sd[our_key][tensor.shape[0]:], mean=0.0, std=std)
                            print(f"Initialized {sd[our_key].shape[0] - tensor.shape[0]} new token embeddings")
                            sd['head.weight'].copy_(sd[our_key])
                        elif tensor.shape == sd[our_key].shape:
                            sd[our_key].copy_(tensor)
                        else:
                            print(f"Shape mismatch for {hf_key} -> {our_key}: {tensor.shape} vs {sd[our_key].shape}")
                        loaded_keys.add(our_key)

        for hf_key, our_key in mapping.items():
            if our_key not in loaded_keys:
                if our_key in sd:
                    print(f"Warning: Key {our_key} not found in any safetensors file (HF key: {hf_key})")

        for i in range(cfg.lm_n_blocks):
            layer_prefix = f"model.layers.{i}."
            block_prefix = f"blocks.{i}."
            fused_param_key = f"{block_prefix}mlp.gate_up_proj.weight"
            if fused_param_key not in sd or fused_param_key in loaded_keys:
                continue
            gate_hf_key = f"{layer_prefix}mlp.gate_proj.weight"
            up_hf_key = f"{layer_prefix}mlp.up_proj.weight"
            gate_w = None
            up_w = None
            for safetensors_file in safetensors_files:
                with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
                    keys = f.keys()
                    if gate_w is None and gate_hf_key in keys:
                        gate_w = f.get_tensor(gate_hf_key)
                    if up_w is None and up_hf_key in keys:
                        up_w = f.get_tensor(up_hf_key)
                if gate_w is not None and up_w is not None:
                    break
            if gate_w is None or up_w is None:
                print(
                    f"Warning: gate_proj or up_proj missing for layer {i} "
                    f"(gate={gate_w is not None}, up={up_w is not None}); "
                    f"{fused_param_key} left at init values."
                )
                continue
            fused = torch.cat([gate_w, up_w], dim=0)
            if fused.shape != sd[fused_param_key].shape:
                print(
                    f"Shape mismatch for fused gate_up at layer {i}: "
                    f"{fused.shape} vs {sd[fused_param_key].shape}"
                )
                continue
            sd[fused_param_key].copy_(fused)
            loaded_keys.add(fused_param_key)

        model.load_state_dict(sd)
        if has_extended_embeddings and hasattr(model, 'head') and 'head.weight' in sd:
            lm_head_loaded = False
            for safetensors_file in safetensors_files:
                with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
                    if 'lm_head.weight' in f.keys():
                        lm_head = f.get_tensor('lm_head.weight')
                        if lm_head.shape[0] != sd['head.weight'].shape[0]:
                            print(f"Extending LM head from {lm_head.shape} to {sd['head.weight'].shape}")
                            sd['head.weight'][:lm_head.shape[0]].copy_(lm_head)
                            std = 0.02
                            init.normal_(sd['head.weight'][lm_head.shape[0]:], mean=0.0, std=std)
                            model.load_state_dict(sd)
                        lm_head_loaded = True
                        break
        if cfg.lm_tie_weights and hasattr(model, 'head') and hasattr(model, 'token_embedding'):
            model.head.weight = model.token_embedding.weight
        print(f"Successfully loaded {cfg.lm_model_type} weights from safetensors. Model has {sum(p.numel() for p in model.parameters()):,} parameters.")
        return model
