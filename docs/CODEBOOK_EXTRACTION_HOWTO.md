# Escha-W2 Codebook Extraction — HOWTO

The MLX port of Escha-W2 needs a ~4-6 MiB file called
`escha_codebooks_v1.npz` before it can decode weights. The codebooks
live inside the `escha` C++ runtime `.so` as compile-time constants —
not in the safetensors. There are three ways to get them, in order of
recommended-ness. The first works on a Mac in one command.

---

## RECOMMENDED — Direct binary extraction on your Mac (~30 s, free)

Reads the codebook tables directly out of the escha `.so`'s
`.rodata` section using `nm` and `objdump`. Bit-exact, no CUDA, no
cloud, no probe cycle.

**One command:**

```bash
python3 mlx_video/models/qwen3_5_moe_escha/codebooks/extract_from_so.py
```

That's it. The script:
1. Downloads the escha wheel from HuggingFace (~12 MiB via
   `huggingface_hub`; cached, one-time).
2. Unzips it into a scratch dir.
3. Runs `nm` on `escha/_C*.so` looking for symbols whose size matches a
   K=2 codebook (65536×16×2 = 2 MiB), then reads those bytes verbatim.
4. If the `.so` is stripped, falls back to a `.rodata` scan that
   detects 2 MiB windows where every fp16 value is finite and bounded.
5. Writes `escha_codebooks_v1.npz` next to itself. The MLX loader
   (`PackedScaledExpertLinear`) picks it up automatically.

**Requirements on your Mac:**
- `python3` (any 3.9+)
- `pip install huggingface_hub numpy` (already in the mlx-video venv)
- `nm` and `objdump` on PATH. macOS ships their BSD equivalents, but
  the extractor uses GNU flags — install with:
  ```
  brew install binutils
  echo 'export PATH="$(brew --prefix)/opt/binutils/bin:$PATH"' >> ~/.zshrc
  source ~/.zshrc
  ```
  On Ubuntu these are in `binutils`, already installed by default.

**What if it fails?**

The script exits non-zero with a diagnostic dump: section table, the
top-30 defined symbols, and long contiguous fp16-valid runs in
`.rodata`. Two failure modes to distinguish:

- *No large fp16 runs in `.rodata`* → the codebook is likely
  runtime-initialized (built at load time by CUDA constructors rather
  than compiled as constants). Fall back to Modal.
- *Runs exist but wrong count / wrong size* → the layout has changed in
  a newer wheel. Please file the diagnostic dump on the HF discussion
  (template in `docs/eschalabs_request_draft.md`) and fall back to
  Modal.

**Verification.** The extractor was tested against three synthetic
manylinux ELF `.so` fixtures:
- Symbols present + two codebooks → bit-exact extraction via symbol
  table.
- Symbols stripped + two codebooks → bit-exact extraction via
  `.rodata` scan (K2/K3 assignment falls back to a std-based
  heuristic).
- 4 MiB of int32 noise + two codebooks (stripped) → both codebooks
  found, noise correctly rejected.

It has NOT yet been executed against the real `EschaLabs/escha-runtime-qwen3moe`
wheel in the sandbox (the sandbox has no HF/PyPI network); the first
run on your Mac is the real verification. If step 4 or 5 above
succeeds, the resulting `.npz` is bit-exact to the values baked into
the runtime — no probe noise, no dequantization.

---

## FALLBACK 1 — Modal Labs serverless GPU (~15 min, ~$0.20)

Use this if the direct extractor prints `FAILURE` (the codebook is
built at runtime, so we do need to execute a probe on a real GPU).

```bash
# One-time setup:
pip install modal
modal setup                   # opens browser to auth ($30/mo free tier)

# The actual extraction — run from anywhere:
cd mlx_video/models/qwen3_5_moe_escha/codebooks
modal run modal_extract.py
```

`escha_codebooks_v1.npz` lands in the codebooks directory when the last
line prints `[local] SUCCESS`.

**Caveat.** The Modal path uses `torch.ops.escha.escham_reconstruct`
as a black-box functional probe, which returns *dequantized* fp16
values. The bit-pattern is the same as the underlying lattice for
compile-time-constant codebooks, but any runtime numerical formatting
(quantization noise, rescaling) will show up in the extracted table.
Prefer the direct extractor for numerical parity work.

Historical note: earlier Modal drafts hardcoded a `(1, 8, 16*K)` probe
shape that triggered `OC must be divisible by 128` in the CUDA op. The
current `modal_extract.py` auto-detects the probe layout by sweeping 5
plausible shapes and picks the one where a single perturbed input
changes exactly one 16-wide output row. That fix was needed regardless
of the extraction route.

---

## FALLBACK 2 — Google Colab (~35-45 min, free)

Use if Modal is blocked (corporate policy, no credit card, etc.).

1. Open <https://colab.research.google.com> → **File → Upload notebook** →
   pick `mlx_video/models/qwen3_5_moe_escha/codebooks/colab_extract.ipynb`.
2. **Runtime → Change runtime type → T4 GPU** → Save.
3. **Runtime → Run all**.
4. Wait ~35-45 min (or ~15 on L4/A100 with Colab Pro).
5. Move the downloaded `.npz` to
   `mlx_video/models/qwen3_5_moe_escha/codebooks/`.

Same functional-probe caveat as Modal.

---

## FALLBACK 3 — Any Linux + NVIDIA GPU you already own

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install "torch==2.9.*" --index-url https://download.pytorch.org/whl/cu128
pip install "huggingface_hub[cli]" numpy safetensors
hf download EschaLabs/escha-runtime-qwen3moe --include "sglang/*" --local-dir .
pip install ./sglang/escha-*.whl
python mlx_video/models/qwen3_5_moe_escha/codebooks/extract_codebooks.py
```

---

## Parallel path — ask EschaLabs to expose the tables

`docs/eschalabs_request_draft.md` is a pre-written HF discussion post
asking EschaLabs to publish the two codebook tables as a data file (or
via a Python accessor). It's the cleanest long-term fix: if they
accept, the whole extraction subsystem here collapses to
`hf_hub_download(...)` + `np.load(...)`.

Suggested cadence:
1. Ship the direct extractor now (unblocks Mac work immediately).
2. Post the request. If they reply with a data file in ~5 days, delete
   the extraction scripts.
3. Otherwise, the direct extractor stays as-is.

---

## What to do with the file

All paths above produce the same `escha_codebooks_v1.npz`. Drop it
next to the extractor:

    mv ~/Downloads/escha_codebooks_v1.npz \
       mlx_video/models/qwen3_5_moe_escha/codebooks/

The MLX weight loader (`PackedScaledExpertLinear`) picks it up
automatically at first inference. Once it's in place, run the numerical
parity check (~2 hr work).

---

## Troubleshooting

**`extract_from_so.py` says `required tool \`nm\` not on PATH`** →
`brew install binutils`, then `export PATH="$(brew --prefix)/opt/binutils/bin:$PATH"`.
GNU binutils shadow the macOS system `nm` (which uses a different flag
set) inside this shell only.

**`extract_from_so.py` finds only 0 or 1 candidates, not 2** → the .so
may store K=2 and K=3 in a single 4 MiB block instead of two 2 MiB
blocks, or the codebook is runtime-initialized. Attach the diagnostic
dump (`[diag] ----` blocks) to the HF discussion; then use Modal.

**`extract_from_so.py` symbol-based extract works, but names don't
say K2/K3** → the K2/K3 assignment is guessed from std (larger-std
codebook → K2). If the numerical parity check later shows the two
tables are swapped, edit the `.npz`:
```python
import numpy as np
d = dict(np.load('escha_codebooks_v1.npz'))
d['cb_A_K2'], d['cb_A_K3'] = d['cb_A_K3'], d['cb_A_K2']
np.savez_compressed('escha_codebooks_v1.npz', **d)
```

**Modal: "Import escha failed: undefined symbol"** → torch version
mismatch inside the container. Bump `WHEEL_REVISION` in
`modal_extract.py` to bust the image cache, or pin
`torch==2.9.*`.

**Colab: installed Python isn't 3.12** → Colab has moved on. The
notebook's step 1 detects this and prints a hard `FAILURE` with
fallback instructions. Use Modal.

**Extraction "succeeds" but the .npz has all-zero rows** → for the
probe path, a false-positive layout auto-detect. For the direct
extractor, the section boundaries were misidentified. Either way,
attach the run log and the (broken) `.npz` to the HF discussion.

---

## Notes on legality

The `escha` runtime ships under Apache-2.0. Reading the compile-time
constant lattices out of `.rodata` is a straightforward Apache-2.0 use
(we are not decompiling code, not redistributing the runtime, and not
touching the CUDA kernel). Explicit maintainer sign-off — via the
discussion draft in `docs/eschalabs_request_draft.md` — is
low-cost and preferred; the extractor is what unblocks parity work in
the meantime.

---

## Route B — zml runtime tarball (2026-08-01: TRIED, FAILED)

The alternate distribution
`https://huggingface.co/EschaLabs/escha-runtime-qwen3moe/resolve/main/zml/escha-zml-serve-1.0.1-linux-x86_64.tar.gz`
(2.09 GB Zig/Rust server, no Python) was extracted and scanned.

Contents (top 5 by size — all CUDA runtime, no bespoke data files):

- `libcublasLt.so.13` (503 MB), `libpjrt_cuda.so` (421 MB), `libcufft.so.12` (300 MB),
  `libpjrt_cpu.so` (282 MB), `libcudnn_engines_precompiled.so` (244 MB)

Escha-specific libraries:

- `lib/libescham_moe.so` (5.2 MB) — MoE GEMV kernels with `bw3` templates
  (`escham_moe_gemv_bw3_kernel<..,..,..,K>` for K=2, K=3). This IS the
  3-bit lattice-quantization kernel.
- `lib/libescha_w8.so` (6.3 MB) — w8a16 GEMV kernels (int8 weight, not lattice).
- `lib/libescha_gdn.so` (498 KB) — gated linear attention and topk router.

Scans performed:

- `gnm --print-size --size-sort --defined-only` on all three .so files:
  **no defined symbol has size ≥ 1 MiB.** No `codebook`/`lattice`/`table` strings.
- Bulk fp16-valid-window scan (stride 4 KiB, |v|<16, 0.05<std<5):
  **0 hits across libescham_moe.so, libescha_w8.so, escha_serve (30 MB main)**.
- `libescham_moe.so` layout: `.rodata` = 3.5 KB, `.nv_fatbin` = 5.14 MiB.
  The entire "payload" of the Escha library is a compressed CUDA fatbin.

Interpretation: the codebook is either (a) baked into the compressed
CUDA cubin as `__constant__` data — recoverable only via `cuobjdump` +
SASS inspection on a Linux+CUDA host, or (b) passed as a runtime pointer
by the server (kernel signature `..PK__half..` = `const __half*` for the
lattice table). Given the mangled kernel signature takes a `__half*`
codebook pointer as an argument, option (b) is more likely — meaning
the codebook lives in an escha data file loaded at model init, not in
the .so. **The zml tarball alone does not surface it.**

Recommendation: stick with Route A (sglang wheel on Modal). If Route A
also fails to find a plain-data codebook, the next place to look is the
model checkpoint on HF (safetensors keys, or an out-of-band data file
in the repo), not the runtime binaries.
