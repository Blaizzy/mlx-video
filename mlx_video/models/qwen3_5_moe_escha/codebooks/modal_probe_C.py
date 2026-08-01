import modal

app = modal.App("escha-probe-C")

image = modal.Image.debian_slim().apt_install("build-essential").pip_install(
    "torch==2.9.0", "numpy", "safetensors", "huggingface_hub", "hf_transfer"
).run_commands(
    "HF_HUB_ENABLE_HF_TRANSFER=1 pip install "
    "https://huggingface.co/EschaLabs/escha-runtime-qwen3moe/resolve/main/sglang/"
    "escha-1.0.2%2Bqwen3moe-cp312-cp312-manylinux_2_28_x86_64.whl"
)


@app.function(image=image, gpu="T4", timeout=1200)
def probe_C():
    import escha
    import escha._C as C
    import torch
    import inspect
    import gc

    print("=== escha._C members ===")
    for name in sorted(dir(C)):
        if name.startswith('_'):
            continue
        obj = getattr(C, name)
        print(f"  {name}: {type(obj).__name__}")
        try:
            print(f"    sig: {inspect.signature(obj)}")
        except Exception as e:
            print(f"    sig-err: {e}")
        doc = getattr(obj, '__doc__', None)
        if doc:
            print(f"    doc: {doc[:400]}")

    print("\n=== escha top-level members ===")
    for name in sorted(dir(escha)):
        if name.startswith('_'):
            continue
        obj = getattr(escha, name)
        print(f"  {name}: {type(obj).__name__}")

    print("\n=== Try escha_init() ===")
    result = None
    try:
        result = C.escha_init()
        print(f"escha_init returned: {type(result)}")
        if isinstance(result, dict):
            for k, v in result.items():
                print(f"  [{k}]: {type(v).__name__}")
                if isinstance(v, torch.Tensor):
                    print(f"    shape={v.shape} dtype={v.dtype} device={v.device}")
        elif isinstance(result, torch.Tensor):
            print(f"  shape={result.shape} dtype={result.dtype} device={result.device}")
        elif isinstance(result, (tuple, list)):
            for i, item in enumerate(result):
                print(f"  [{i}]: {type(item).__name__}")
                if isinstance(item, torch.Tensor):
                    print(f"    shape={item.shape} dtype={item.dtype} device={item.device}")
        else:
            print(f"  value: {result}")
    except Exception as e:
        print(f"escha_init failed: {type(e).__name__}: {e}")

    print("\n=== escha_dequant signature deep dive ===")
    try:
        sig = inspect.signature(C.escha_dequant)
        print(f"escha_dequant params: {list(sig.parameters.keys())}")
        for name, p in sig.parameters.items():
            print(f"  {name}: default={p.default} annotation={p.annotation} kind={p.kind}")
    except Exception as e:
        print(f"couldn't inspect escha_dequant: {e}")
    doc = getattr(C.escha_dequant, '__doc__', None)
    if doc:
        print(f"  doc: {doc[:800]}")

    print("\n=== escha_lut_binary_gemv signature deep dive ===")
    try:
        sig = inspect.signature(C.escha_lut_binary_gemv)
        print(f"params: {list(sig.parameters.keys())}")
        for name, p in sig.parameters.items():
            print(f"  {name}: default={p.default} annotation={p.annotation} kind={p.kind}")
    except Exception as e:
        print(f"couldn't inspect: {e}")
    doc = getattr(C.escha_lut_binary_gemv, '__doc__', None)
    if doc:
        print(f"  doc: {doc[:800]}")

    print("\n=== escha_transform signature ===")
    for opname in ("escha_transform", "escha_transform_fp16", "escha_aqlm_gemv",
                   "escha_aqlm_fused_hmma", "escha_decgemv"):
        if not hasattr(C, opname):
            continue
        op = getattr(C, opname)
        try:
            sig = inspect.signature(op)
            print(f"{opname} params: {list(sig.parameters.keys())}")
        except Exception as e:
            print(f"{opname} sig-err: {e}")
        doc = getattr(op, '__doc__', None)
        if doc:
            print(f"  doc: {doc[:400]}")

    print("\n=== GC sweep after escha_init ===")
    import pickle
    results = {}
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.numel() >= 65536 * 8 and \
                    obj.dtype in (torch.float16, torch.bfloat16, torch.float32):
                key = f"gc_{id(obj)}"
                print(f"  {key}: shape={tuple(obj.shape)} dtype={obj.dtype} device={obj.device}")
                arr = obj.detach().to(torch.float16).cpu().numpy()
                results[key] = arr
        except Exception:
            pass

    if results:
        return pickle.dumps(results)
    return b""


@app.local_entrypoint()
def main():
    data = probe_C.remote()
    if data:
        out = "/Users/kaede/mlx-video/mlx_video/models/qwen3_5_moe_escha/codebooks/escha_probe_dump.pkl"
        with open(out, 'wb') as f:
            f.write(data)
        print(f"Saved dump: {out} ({len(data)/1e6:.1f} MB)")
    else:
        print("No large tensors found; check stdout for API signatures instead")
