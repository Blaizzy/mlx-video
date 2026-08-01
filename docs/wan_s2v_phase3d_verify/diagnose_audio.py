"""Fast diagnostic: does audio_emb change when audio changes?"""
import os, sys, time
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import mlx.core as mx
import mlx.utils
import numpy as np
sys.path.insert(0, str(Path.home() / "mlx-video"))
from mlx_video.models.wan_2.audio_encoder import (
    extract_wav2vec_features, CausalAudioEncoder,
)
from mlx_video.models.wan_2.config import WanModelConfig

MODEL_DIR = Path.home() / "mlx-video" / "mlx-models" / "Wan2.2-S2V-14B-MLX-int4"
REAL_WAV = Path.home() / "movie" / "wang_wenchin" / "output" / "willy_hello_3s.wav"
SILENCE_WAV = Path("/tmp/silence.wav")
NUM_FRAMES = 49

def stats(name, arr):
    if arr is None: print(f"  {name}: None"); return
    a = np.asarray(arr.astype(mx.float32) if arr.dtype != mx.float32 else arr, dtype=np.float32)
    print(f"  {name}: shape={tuple(arr.shape)} mean|.|={np.abs(a).mean():.6f} "
          f"max|.|={np.abs(a).max():.6f} std={a.std():.6f}")

def main():
    print("=" * 60); print("AUDIO PATH DIAGNOSTIC"); print("=" * 60)
    F_video_latent = 1 + (NUM_FRAMES - 1) // 4
    print(f"num_pixel_frames={NUM_FRAMES}  F_video_latent={F_video_latent}")

    w2v_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    print(f"wav2vec2: {w2v_name}\n")

    print("[A] Extracting wav2vec2 features (real speech)...")
    t0 = time.time()
    feat_real = extract_wav2vec_features(str(REAL_WAV), F_video_latent, 4, w2v_name)
    mx.eval(feat_real)
    print(f"  took {time.time()-t0:.1f}s"); stats("feat_real", feat_real)

    print("\n[A] Extracting wav2vec2 features (silence)...")
    t0 = time.time()
    feat_silence = extract_wav2vec_features(str(SILENCE_WAV), F_video_latent, 4, w2v_name)
    mx.eval(feat_silence)
    print(f"  took {time.time()-t0:.1f}s"); stats("feat_silence", feat_silence)

    diff = mx.abs(feat_real.astype(mx.float32) - feat_silence.astype(mx.float32))
    print("\n[wav2vec output diff]:"); stats("|real - silence|", diff)

    print("\n[B] Loading full state dict then filtering audio encoder keys...")
    all_w = mx.load(str(MODEL_DIR / "model.safetensors"))
    prefix = "casual_audio_encoder."
    loaded = {k[len(prefix):]: v.astype(mx.float32) for k, v in all_w.items() if k.startswith(prefix)}
    print(f"  loaded {len(loaded)} tensors")
    if "weights" in loaded:
        stats("weights (layer weighting)", loaded["weights"])
    del all_w

    config = WanModelConfig.wan22_s2v_14b()
    enc = CausalAudioEncoder(dim=1024, num_layers=25, out_dim=config.dim,
                              num_token=config.num_audio_token, need_global=True)

    # Convert flat dict to nested and load.
    tree = mlx.utils.tree_unflatten(list(loaded.items()))
    enc.update(tree)
    mx.eval(enc.parameters())

    # Sanity check: print first-block weight norm from the module vs loaded dict.
    module_w = enc.encoder.conv1_local.conv.weight
    stats("module encoder.conv1_local.conv.weight (post-load)", module_w)
    stats("loaded encoder.conv1_local.conv.weight", loaded["encoder.conv1_local.conv.weight"])
    diff_w = mx.abs(module_w.astype(mx.float32) - loaded["encoder.conv1_local.conv.weight"])
    print(f"  weight-load diff mean = {float(mx.abs(diff_w).mean().item()):.6f}")

    print("\n[C] Encoding features...")
    local_r, glob_r = enc(feat_real)
    mx.eval([local_r, glob_r])
    local_s, glob_s = enc(feat_silence)
    mx.eval([local_s, glob_s])

    print("\n[audio_emb LOCAL]:"); stats("real", local_r); stats("silence", local_s)
    diff_l = mx.abs(local_r.astype(mx.float32) - local_s.astype(mx.float32))
    stats("|real - silence|", diff_l)

    print("\n[audio_emb GLOBAL]:"); stats("real", glob_r); stats("silence", glob_s)
    diff_g = mx.abs(glob_r.astype(mx.float32) - glob_s.astype(mx.float32))
    stats("|real - silence|", diff_g)

    print("\n" + "=" * 60)
    m_local = float(mx.abs(local_r - local_s).mean().item())
    m_global = float(mx.abs(glob_r - glob_s).mean().item())
    m_local_ref = float(mx.abs(local_r).mean().item())
    m_global_ref = float(mx.abs(glob_r).mean().item())
    rel_local = m_local / max(m_local_ref, 1e-9)
    rel_global = m_global / max(m_global_ref, 1e-9)
    print(f"local  diff mean = {m_local:.6f}  real mean = {m_local_ref:.6f}  rel = {rel_local:.3%}")
    print(f"global diff mean = {m_global:.6f}  real mean = {m_global_ref:.6f}  rel = {rel_global:.3%}")

    if m_local < 1e-6 and m_global < 1e-6:
        print("VERDICT: audio_emb IDENTICAL -> ENCODER BROKEN.")
    elif rel_local > 0.05 or rel_global > 0.05:
        print("VERDICT: audio_emb differs substantially -> encoder OK.")
    else:
        print("VERDICT: audio_emb differs only marginally.")

if __name__ == "__main__": main()
