#!/usr/bin/env python3
"""Merge a Wan 2.2 lightx2v LoRA into fp16 base weights, streaming.

Why merge before quantizing: mlx-video applies LoRA by DEQUANTIZING the int4
layers back to fp16, which undoes quantization entirely and blew swap to 21.4 GB
before reaching a single step. Baking the LoRA into the fp16 weights first means
the subsequent int4 quantization is self-contained and no runtime LoRA path runs.

Handles the three entry kinds in these files:
  <mod>.lora_down.weight / <mod>.lora_up.weight  -> W += scale * (up @ down)
  <mod>.diff_b                                   -> bias += diff_b
  <mod>.diff                                     -> W += diff        (norms etc.)

Key mapping: strip "diffusion_model.", remap ffn.0->ffn.fc1, ffn.2->ffn.fc2
(mlx-video's converter renames those; verified against the converted base).

Scale = alpha/rank when an .alpha entry exists, else 1.0.

Streaming: base header is reused verbatim (shapes/dtypes unchanged), tensors are
read, patched and written one at a time. LoRA factors stay resident (~5 GB fp32);
each up@down delta is computed at write time and freed — materializing all deltas
up front would need 57 GB fp32 and this machine has 48.

    merge_lora.py base.safetensors lora.safetensors out.safetensors
"""

import json
import struct
import sys

import numpy as np


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


DT = {"F16": np.float16, "F32": np.float32, "BF16": np.float16}


def load_all(path):
    hdr, base = read_header(path)
    out = {}
    with open(path, "rb") as f:
        for k, m in hdr.items():
            if k == "__metadata__":
                continue
            s, e = m["data_offsets"]
            f.seek(base + s)
            raw = f.read(e - s)
            if m["dtype"] == "BF16":  # widen bf16 -> f32 via bit shift
                u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
                arr = u.view(np.float32)
            else:
                arr = np.frombuffer(raw, dtype=DT[m["dtype"]])
            out[k] = arr.reshape(m["shape"]).astype(np.float32)
    return out


# mlx-video's converter flattens PyTorch Sequential indices; full table verified
# against the converted base header (fc1/fc2, _0/_1 suffixes, proj rename).
_FULL_RENAMES = {
    "patch_embedding": "patch_embedding_proj",
    "text_embedding.0": "text_embedding_0",
    "text_embedding.2": "text_embedding_1",
    "time_embedding.0": "time_embedding_0",
    "time_embedding.2": "time_embedding_1",
    "time_projection.1": "time_projection",
}


def canon(key):
    """LoRA module name -> base tensor prefix."""
    k = key
    for p in ("model.diffusion_model.", "diffusion_model.", "model."):
        if k.startswith(p):
            k = k[len(p):]
            break
    if k in _FULL_RENAMES:
        return _FULL_RENAMES[k]
    k = k.replace(".ffn.0.", ".ffn.fc1.").replace(".ffn.2.", ".ffn.fc2.")
    # module names end at the index, so the dotted replace above misses them
    if k.endswith(".ffn.0"):
        k = k[:-1] + "fc1"
    elif k.endswith(".ffn.2"):
        k = k[:-1] + "fc2"
    return k


def build_deltas(lora):
    """Return (direct, pairs).

    direct: {base_key: delta}          -- diff / diff_b, small, held as-is
    pairs:  {base_key: (down, up, scale)} -- up@down deferred to write time
    """
    mods, direct = {}, {}
    for k in lora:
        if k.endswith(".lora_down.weight"):
            mods.setdefault(canon(k[: -len(".lora_down.weight")]), {})["down"] = lora[k]
        elif k.endswith(".lora_up.weight"):
            mods.setdefault(canon(k[: -len(".lora_up.weight")]), {})["up"] = lora[k]
        elif k.endswith(".alpha"):
            mods.setdefault(canon(k[: -len(".alpha")]), {})["alpha"] = float(lora[k].reshape(-1)[0])
        elif k.endswith(".diff_b"):
            direct[canon(k[: -len(".diff_b")]) + ".bias"] = lora[k]
        elif k.endswith(".diff"):
            direct[canon(k[: -len(".diff")]) + ".weight"] = lora[k]

    pairs = {}
    for mod, p in mods.items():
        if "down" not in p or "up" not in p:
            continue
        rank = p["down"].shape[0]
        pairs[mod + ".weight"] = (p["down"], p["up"], p.get("alpha", float(rank)) / rank)
    return direct, pairs


def main(base_path, lora_path, out_path):
    print("loading LoRA...", flush=True)
    direct, pairs = build_deltas(load_all(lora_path))
    n_deltas = len(direct) + len(pairs)
    print(f"  {n_deltas} deltas (lazy)", flush=True)

    hdr, dstart = read_header(base_path)
    keys = [k for k in hdr if k != "__metadata__"]
    hjson = json.dumps({k: hdr[k] for k in keys}, separators=(",", ":")).encode()

    applied = skipped = 0
    with open(base_path, "rb") as fin, open(out_path, "wb") as fout:
        fout.write(struct.pack("<Q", len(hjson)))
        fout.write(hjson)
        # write in data_offsets order so the output matches its own header
        for i, k in enumerate(sorted(keys, key=lambda x: hdr[x]["data_offsets"][0])):
            m = hdr[k]
            s, e = m["data_offsets"]
            fin.seek(dstart + s)
            arr = np.frombuffer(fin.read(e - s), dtype=DT[m["dtype"]]).reshape(m["shape"])
            d = None
            if k in direct:
                d = direct[k]
            elif k in pairs:
                down, up, scale = pairs[k]
                d = (up @ down) * scale
            if d is not None:
                if d.shape != tuple(m["shape"]):
                    print(f"  SHAPE MISMATCH {k}: base{tuple(m['shape'])} vs delta{d.shape} -- skipped")
                    skipped += 1
                else:
                    arr = (arr.astype(np.float32) + d).astype(DT[m["dtype"]])
                    applied += 1
                del d
            fout.write(np.ascontiguousarray(arr).tobytes())
            if i % 300 == 0:
                print(f"  [{i}/{len(keys)}]", flush=True)

    print(f"DONE applied={applied} skipped={skipped} unused_deltas="
          f"{n_deltas-applied-skipped} -> {out_path}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
