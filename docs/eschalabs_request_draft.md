# Draft HF discussion post — request for exposed codebook tensors

Post this at: <https://huggingface.co/EschaLabs/escha-runtime-qwen3moe/discussions>

Subject and body below; edit before posting.

---

**Subject:** Could you expose the AQLM codebook tables as a data file (or a Python accessor)?

Hi Escha team,

I'm working on a native Apple Silicon (MLX) port of
`Qwen3.6-35B-A3B-Escha-W2` — so far the model loader, MoE routing, RoPE,
attention, and residual‑paths all pass numerical parity against a
reference. The only remaining blocker is the AQLM codebook tables
(65536×16 fp16 per residual code, K=2 and K=3), which today live inside
`escha._C.so` as compile‑time constants and are only reachable through
the CUDA op `torch.ops.escha.escham_reconstruct`.

Since the `.so` is `manylinux_2_28_x86_64` + CUDA‑linked, we cannot load
the runtime on a Mac to probe it, and using a Linux+GPU host or Modal for
a few‑minute op‑probe just to fetch two 2 MiB tables feels heavy.
Directly reading the `.rodata` bytes is the natural way, but I want to
check with you first before shipping any bit‑level extractor — both to
respect your engineering intent and to avoid drift if the tables move in
a future wheel.

**Would you consider one of these lightweight options?**

1. Ship the two lattices as a bundled data file (`.safetensors` or
   `.npz`) inside the wheel or alongside it — `cb_A_K2` and `cb_A_K3`,
   each `(65536, 16) fp16`, ~2 MiB apiece. The wheel is already ~12 MiB
   so this is a tiny addition, and it turns the mlx / mlc / rust‑burn /
   any‑non‑CUDA ports into a one‑line load.
2. Expose a Python accessor such as `escha.dump_codebooks() -> dict[str, np.ndarray]`
   that dumps the internal tables directly. Zero binary‑format
   commitment on your side.
3. Register the tables as PyTorch buffers on a `torch.classes.escha.Codebooks`
   TorchBind class — same effect, natural for downstream `state_dict`
   users.
4. Confirm on this thread that reading the tables directly from
   `.rodata` is acceptable (an "Apache‑2.0 says you can" is fine), and I
   can ship a tiny binary‑extractor in the MLX port's tree.

**What I'm not asking for:** the CUDA kernel, the compressed weights
themselves, or any internals beyond the two lattices. Everything else
is already reproducible from your public schema and the safetensors on
Hugging Face.

**Context on why this matters:** MLX is Apple's tensor library that
plugs into MPS on M‑series Macs — a 35B‑MoE that decodes at 20–40 tok/s
on a laptop is a big deal for the OSS community. Being able to point at
the Escha codebooks and say "and here's how you drop them in" would
make Escha‑W2 the first serious AQLM MoE with a working non‑CUDA port.

Happy to open a PR to your repo with the Python accessor (option 2) if
that's the least intrusive path. Also happy to link the MLX port back
once it lands.

Thanks!

— [your name / handle]
[link to your MLX port branch, if you want to include one]

---

**Notes for the poster (delete before posting):**
- Escha's `README.md` says Apache‑2.0 and treats the runtime as an
  inference product. Direct bit‑extraction of a compile‑time constant
  array is a straightforward Apache‑2.0 use, but explicit sign‑off is
  low‑cost and burns no bridges.
- The maintainer handle is `yzhwang` on HF — no need to @-mention;
  they auto‑subscribe to discussions on their repo.
- If they don't reply within ~5 business days, the direct
  `extract_from_so.py` path is what actually ships.
- If they reply with option 1 or 2, delete `extract_from_so.py`,
  `modal_extract.py`, and the Colab notebook — the whole extraction
  subsystem collapses to `hf_hub_download(...)` + `np.load(...)`.
