"""Route A: bit-exact codebook extraction directly from the escha `.so`.

RUNS ON macOS OR LINUX. NO GPU. NO CUDA. NO CLOUD.

One command:

    python3 mlx_video/models/qwen3_5_moe_escha/codebooks/extract_from_so.py

What it does — in order of preference; the first one that succeeds wins.

  1. Downloads the escha wheel from HuggingFace (~12 MiB, requires
     `pip install huggingface_hub`) if not already cached.
  2. Unzips the wheel and lists all shipped data files. If EschaLabs
     ever bundles a `.npz`/`.safetensors`/`.pt` codebook table inside
     the wheel, this step finds it and we're done in <1 s.
  3. Reads all Python glue files, greps for any codebook accessor
     (`escha.dump_codebooks`, `torch.ops.escha.*`) that would let us
     bypass the .so scan entirely.
  4. Uses the ELF symbol table (`nm`) to find defined symbols whose
     size == 65536 * 16 * 2 = 2 MiB. That is the exact byte-size of
     a K=2 or K=3 codebook. If two such symbols exist, extract both
     bit-exact.
  5. If symbols are stripped, dumps `.rodata` (via `objdump`) and
     scans it for 2 MiB windows where every fp16 value is finite and
     bounded — the signature of a codebook lattice.

The extracted codebook is written next to this script as
`escha_codebooks_v1.npz` with keys `cb_A_K2` and `cb_A_K3`
(each 65536 x 16 float16). The MLX loader picks it up automatically.

If neither step 4 nor step 5 finds two clean 2 MiB fp16 blocks, the
script prints a diagnostic dump (section table, top-20 largest
symbols, .rodata size, byte-histogram of large sections) and points
at the Modal fallback in `modal_extract.py`.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

CODEBOOK_ROWS = 65536
CODEBOOK_DIM = 16
CODEBOOK_BYTES = CODEBOOK_ROWS * CODEBOOK_DIM * 2  # fp16 -> 2 MiB per K

WHEEL_REPO = "EschaLabs/escha-runtime-qwen3moe"
WHEEL_SUBPATH = "sglang/escha-1.0.2+qwen3moe-cp312-cp312-manylinux_2_28_x86_64.whl"

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = SCRIPT_DIR / "escha_codebooks_v1.npz"
CACHE_DIR = SCRIPT_DIR / ".escha_wheel_cache"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(msg, flush=True)


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _require_tool(cmd: str) -> None:
    if not _have(cmd):
        raise SystemExit(
            f"[fatal] required tool `{cmd}` not on PATH. "
            f"On macOS: `brew install binutils` and add "
            f"`$(brew --prefix)/opt/binutils/bin` to PATH. "
            f"On Ubuntu: `sudo apt install binutils`."
        )


# ---------------------------------------------------------------------------
# stage 0 — get the .so
# ---------------------------------------------------------------------------

def download_wheel(force: bool = False) -> Path:
    """Fetch the escha wheel from HuggingFace via huggingface_hub."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    wheel_dest = CACHE_DIR / Path(WHEEL_SUBPATH).name
    if wheel_dest.exists() and not force:
        _log(f"[stage 0] wheel already cached: {wheel_dest}")
        return wheel_dest

    _log(f"[stage 0] downloading {WHEEL_SUBPATH} from {WHEEL_REPO}...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            "[fatal] huggingface_hub not installed. Run:\n"
            "    pip install huggingface_hub\n"
            "or supply the .so path yourself: `--so /path/to/escha/_C*.so`."
        )

    local = hf_hub_download(
        repo_id=WHEEL_REPO,
        filename=WHEEL_SUBPATH,
        cache_dir=str(CACHE_DIR / "hf_cache"),
    )
    shutil.copy(local, wheel_dest)
    _log(f"[stage 0] cached wheel: {wheel_dest}")
    return wheel_dest


def unpack_wheel(wheel_path: Path) -> Path:
    """Unzip a .whl into a fresh directory; return that directory."""
    out_dir = CACHE_DIR / (wheel_path.stem + "_unpacked")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    with zipfile.ZipFile(wheel_path) as zf:
        zf.extractall(out_dir)
    _log(f"[stage 0] unpacked wheel -> {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# stage 2 — bundled data-file check
# ---------------------------------------------------------------------------

def find_bundled_codebooks(unpacked: Path) -> Optional[Path]:
    """Look for any shipped .npz/.safetensors/.pt file that might be a
    pre-baked codebook table."""
    _log("[stage 2] scanning wheel contents for bundled data files...")
    hits = []
    for p in unpacked.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".npz", ".safetensors", ".pt", ".bin"}:
            size = p.stat().st_size
            _log(f"  data file: {p.relative_to(unpacked)} ({size/1024/1024:.2f} MiB)")
            hits.append(p)
    if not hits:
        _log("  no data files bundled — codebooks live in the .so.")
    return hits[0] if hits else None


# ---------------------------------------------------------------------------
# stage 3 — python-level introspection
# ---------------------------------------------------------------------------

CODEBOOK_HINT_RE = re.compile(
    r"codebook|lattice|lut|cb_a|cb_b|cb_c|dump_codebook|"
    r"escham?_(reconstruct|codebook|dump)",
    re.IGNORECASE,
)


def grep_python_glue(unpacked: Path) -> list[str]:
    _log("[stage 3] grep-ing Python glue for codebook access hints...")
    hits = []
    for py in unpacked.rglob("*.py"):
        try:
            text = py.read_text(errors="ignore")
        except OSError:
            continue
        for m in CODEBOOK_HINT_RE.finditer(text):
            hits.append(f"{py.relative_to(unpacked)}: ...{text[max(0,m.start()-40):m.end()+40]}...")
    seen = set()
    unique = []
    for h in hits:
        if h in seen:
            continue
        seen.add(h)
        unique.append(h)
    for h in unique[:20]:
        _log(f"  hint: {h}")
    if len(unique) > 20:
        _log(f"  ...and {len(unique) - 20} more hints (not shown)")
    if not unique:
        _log("  no hints found in .py files")
    return unique


# ---------------------------------------------------------------------------
# stage 4/5 — ELF analysis (symbol table + rodata scan)
# ---------------------------------------------------------------------------

def find_so_files(unpacked: Path) -> list[Path]:
    return sorted(p for p in unpacked.rglob("*.so"))


def enumerate_symbols(so_path: Path) -> list[tuple[int, int, str, str]]:
    """Return list of (address, size, section_letter, name) for all defined
    symbols with a nonzero size. Uses `nm --print-size`."""
    try:
        out = subprocess.check_output(
            ["nm", "--defined-only", "--print-size", "--radix=x", str(so_path)],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    rows = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            addr = int(parts[0], 16)
            size = int(parts[1], 16)
        except ValueError:
            continue
        rows.append((addr, size, parts[2], parts[3]))
    return rows


def get_section_data(so_path: Path, section_name: str) -> Optional[tuple[int, bytes]]:
    """Return (vma, raw_bytes) for a named ELF section, or None if absent."""
    try:
        out = subprocess.check_output(
            ["objdump", "-sj", section_name, str(so_path)],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    vma: Optional[int] = None
    chunks: list[bytes] = []
    for line in out.splitlines():
        if not line.startswith(" "):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            addr = int(parts[0], 16)
        except ValueError:
            continue
        if vma is None:
            vma = addr
        # objdump hex-dump: 4 groups of 4 hex bytes per row, then ascii
        for tok in parts[1:5]:
            if len(tok) == 8:
                try:
                    chunks.append(bytes.fromhex(tok))
                except ValueError:
                    pass
    if vma is None:
        return None
    return vma, b"".join(chunks)


def section_table(so_path: Path) -> list[tuple[str, int, int]]:
    """List (name, size, vma) for every ELF section."""
    try:
        out = subprocess.check_output(
            ["objdump", "-h", str(so_path)], text=True, stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    rows = []
    for line in out.splitlines():
        toks = line.split()
        if len(toks) < 4 or not toks[0].isdigit():
            continue
        # Format: idx name size vma lma off algn
        try:
            name = toks[1]
            size = int(toks[2], 16)
            vma = int(toks[3], 16)
        except (IndexError, ValueError):
            continue
        rows.append((name, size, vma))
    return rows


def extract_at_symbol(so_path: Path, addr: int, size: int, section_name: str):
    """Given a symbol's (addr, size), extract those bytes and reshape as
    (65536, 16) fp16 numpy array."""
    import numpy as np
    sec = get_section_data(so_path, section_name)
    if sec is None:
        return None
    sec_vma, sec_bytes = sec
    off = addr - sec_vma
    if off < 0 or off + size > len(sec_bytes):
        return None
    blk = sec_bytes[off:off+size]
    if len(blk) != CODEBOOK_BYTES:
        return None
    return np.frombuffer(blk, dtype=np.float16).copy().reshape(CODEBOOK_ROWS, CODEBOOK_DIM)


def _section_letter_to_name(letter: str) -> str:
    """nm section letter -> objdump section name (best guess)."""
    # r/R = .rodata, d/D = .data, b/B = .bss, t/T = .text
    return {"R": ".rodata", "r": ".rodata", "D": ".data", "d": ".data"}.get(letter, ".rodata")


SYMBOL_NAME_HINT_RE = re.compile(
    r"codebook|lattice|(?:^|_)lut(?:$|_)|(?:^|_)cb[_0-9AaBbCc]|"
    r"escham?_?(cb|codebook|table|const)",
    re.IGNORECASE,
)


def try_symbol_extraction(so_path: Path):
    """Find and extract by symbol name and/or size. Returns list of (label, array)."""
    import numpy as np
    _log(f"[stage 4] enumerating symbols in {so_path.name}...")
    syms = enumerate_symbols(so_path)
    if not syms:
        _log("  (no defined symbols — likely stripped)")
        return []
    # Sort by size, largest first
    syms.sort(key=lambda r: r[1], reverse=True)
    _log("  top 10 largest defined symbols:")
    for addr, size, sec, name in syms[:10]:
        marker = "  <-- matches codebook size" if size == CODEBOOK_BYTES else ""
        _log(f"    size={size:>10} ({size/1024/1024:5.2f} MiB) sec={sec} {name[:60]}{marker}")

    # Name-based candidates — take any symbol whose name matches the hint regex.
    named = [s for s in syms if SYMBOL_NAME_HINT_RE.search(s[3])]
    if named:
        _log(f"  {len(named)} symbols match codebook-name regex:")
        for addr, size, sec, name in named[:10]:
            _log(f"    size={size:>10} sec={sec} name={name[:80]}")

    # Exact-size matches
    exact = [s for s in syms if s[1] == CODEBOOK_BYTES]
    if exact:
        _log(f"  {len(exact)} symbols with exact codebook size (2 MiB)")

    # Union: try both name-based and size-based
    tried = set()
    hits = []
    for candidate_list, source_label in ((named, "name"), (exact, "size")):
        for addr, size, sec_letter, name in candidate_list:
            if (addr, size) in tried:
                continue
            tried.add((addr, size))
            if size % (CODEBOOK_DIM * 2) != 0 or size < CODEBOOK_BYTES // 4:
                # not fp16-16-wide-aligned or absurdly small
                continue
            section_name = _section_letter_to_name(sec_letter)
            # Adjust to exactly 2 MiB if the symbol is larger
            eff_size = CODEBOOK_BYTES if size >= CODEBOOK_BYTES else size
            arr_bytes = None
            for alt in (section_name, ".rodata", ".data.rel.ro", ".data", ".rodata.cst16"):
                sec = get_section_data(so_path, alt)
                if sec is None:
                    continue
                sec_vma, sec_data = sec
                off = addr - sec_vma
                if 0 <= off <= len(sec_data) - eff_size:
                    arr_bytes = sec_data[off:off+eff_size]
                    section_name = alt
                    break
            if arr_bytes is None:
                _log(f"    symbol {name} ({source_label}): could not locate bytes")
                continue
            if len(arr_bytes) != CODEBOOK_BYTES:
                _log(f"    symbol {name}: size {len(arr_bytes)} != 2 MiB, skipping")
                continue
            arr = np.frombuffer(arr_bytes, dtype=np.float16).copy().reshape(
                CODEBOOK_ROWS, CODEBOOK_DIM
            )
            a32 = arr.astype(np.float32)
            finite = float(np.isfinite(a32).mean())
            _log(
                f"    {name} ({source_label}, {section_name}): "
                f"finite={finite:.3f} min={a32.min():.3f} max={a32.max():.3f} "
                f"std={a32.std():.3f} first_row[:4]={arr[0][:4]}"
            )
            if finite > 0.99:
                hits.append((name, arr))
    return hits


def try_rodata_scan(so_path: Path):
    """Fallback: scan .rodata (+ .data.rel.ro, .data) for 2 MiB windows that
    decode cleanly as fp16 codebook lattices. Returns list of (label, array)."""
    import numpy as np
    _log("[stage 5] .rodata scan (fp16 codebook signature)...")
    results = []
    for sec_name in (".rodata", ".data.rel.ro", ".data"):
        sec = get_section_data(so_path, sec_name)
        if sec is None:
            continue
        vma, data = sec
        if len(data) < CODEBOOK_BYTES:
            continue
        _log(f"  scanning {sec_name} ({len(data)/1024/1024:.2f} MiB)")
        arr16 = np.frombuffer(data, dtype=np.float16)
        arr32 = arr16.astype(np.float32)
        # Per-value: finite AND |v| < 32
        valid = np.isfinite(arr32) & (np.abs(arr32) < 32)
        # Sliding sum via cumsum
        window_vals = CODEBOOK_BYTES // 2
        if len(valid) < window_vals:
            continue
        csum = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))
        n_win = len(valid) - window_vals + 1
        counts = csum[window_vals:] - csum[:n_win]
        frac_valid = counts / window_vals
        good_idx = np.where(frac_valid > 0.99)[0]
        _log(f"    {len(good_idx)} fp16-positions have >99% valid values in the next 2 MiB")
        # Greedy dedupe: pick non-overlapping windows, applying nz + std filters
        last_taken = -window_vals
        for i in good_idx:
            if i < last_taken + window_vals:
                continue
            off = int(i) * 2
            blk = data[off:off + CODEBOOK_BYTES]
            slice_arr = np.frombuffer(blk, dtype=np.float16)
            s32 = slice_arr.astype(np.float32)
            nz = float((slice_arr != 0).mean())
            std = float(s32.std())
            if nz < 0.5 or not (0.05 < std < 10):
                continue
            reshaped = slice_arr.copy().reshape(CODEBOOK_ROWS, CODEBOOK_DIM)
            label = f"{sec_name}+0x{off:x}"
            _log(f"    hit: {label} nz={nz:.3f} std={std:.3f} first_row[:4]={slice_arr[:4]}")
            results.append((label, reshaped))
            last_taken = int(i)
    if not results:
        _log("  no clean 2 MiB fp16 blocks in any rodata-like section")
    return results


# ---------------------------------------------------------------------------
# save / diagnose
# ---------------------------------------------------------------------------

def save_codebooks(candidates, out_path: Path) -> None:
    """Given a list of (label, array) candidates, pick two, save as .npz.

    Ordering heuristic: `cb_A_K2` gets the candidate with lower average
    magnitude (K=2 is the "coarse" codebook; K=3 residual tends to be
    tighter). If names have `K2`/`K3` in them, use those instead.
    """
    import numpy as np
    if len(candidates) < 2:
        raise RuntimeError(
            f"only {len(candidates)} codebook candidate(s) found; expected 2 "
            "(K=2 and K=3)."
        )

    named = {}
    for label, arr in candidates:
        low = label.lower()
        if "k2" in low or "cb_a" in low:
            named["K2"] = (label, arr)
        elif "k3" in low or "cb_b" in low or "cb_c" in low:
            named["K3"] = (label, arr)

    if "K2" in named and "K3" in named:
        cb_K2 = named["K2"][1]
        cb_K3 = named["K3"][1]
        note = f"picked by symbol name: K2={named['K2'][0]} K3={named['K3'][0]}"
    else:
        # Sort by std (coarser codebook has larger spread) — coarse = K2
        candidates_sorted = sorted(
            candidates,
            key=lambda x: float(x[1].astype(np.float32).std()),
            reverse=True,
        )
        cb_K2 = candidates_sorted[0][1]
        cb_K3 = candidates_sorted[1][1]
        note = (
            f"picked by std (K2=larger std): "
            f"K2={candidates_sorted[0][0]} K3={candidates_sorted[1][0]}"
        )
    _log(f"[save] {note}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, cb_A_K2=cb_K2, cb_A_K3=cb_K3)
    sz = out_path.stat().st_size / 1024 / 1024
    _log(f"[save] wrote {out_path} ({sz:.2f} MiB)")
    _log(
        f"[save] cb_A_K2 shape={cb_K2.shape} dtype={cb_K2.dtype} "
        f"first_row[:4]={cb_K2[0][:4]}"
    )
    _log(
        f"[save] cb_A_K3 shape={cb_K3.shape} dtype={cb_K3.dtype} "
        f"first_row[:4]={cb_K3[0][:4]}"
    )


def print_diagnostics(so_path: Path) -> None:
    import numpy as np
    _log("\n[diag] ---- section table (>= 1 KiB) ----")
    for name, size, vma in section_table(so_path):
        if size >= 1024:
            _log(f"  {name:<24} size={size:>10} ({size/1024/1024:6.2f} MiB) vma={vma:x}")
    _log("\n[diag] ---- top-30 defined symbols by size ----")
    syms = enumerate_symbols(so_path)
    syms.sort(key=lambda r: r[1], reverse=True)
    for addr, size, sec, name in syms[:30]:
        _log(f"  size={size:>10} ({size/1024/1024:5.2f} MiB) sec={sec} {name[:80]}")
    _log("\n[diag] ---- long contiguous fp16-valid runs in .rodata ----")
    sec = get_section_data(so_path, ".rodata")
    if sec:
        vma, data = sec
        arr16 = np.frombuffer(data, dtype=np.float16)
        a32 = arr16.astype(np.float32)
        valid = np.isfinite(a32) & (np.abs(a32) < 32)
        # find contiguous runs of valid
        change = np.diff(valid.astype(np.int8), prepend=0, append=0)
        starts = np.where(change == 1)[0]
        ends = np.where(change == -1)[0]
        # report runs >= 512 KiB
        for s, e in zip(starts, ends):
            byts = (e - s) * 2
            if byts >= 512 * 1024:
                arr = arr16[s:e]
                a = arr.astype(np.float32)
                _log(
                    f"  run: byte_off=0x{s*2:x} len={byts/1024/1024:6.2f} MiB "
                    f"nz={float((arr != 0).mean()):.3f} std={a.std():.3f}"
                )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--so", help="Path to escha .so (skip download).")
    ap.add_argument("--wheel", help="Path to escha .whl (skip download).")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output .npz path.")
    ap.add_argument("--redownload", action="store_true", help="Force re-download.")
    args = ap.parse_args()

    _require_tool("nm")
    _require_tool("objdump")

    try:
        import numpy  # noqa: F401
    except ImportError:
        raise SystemExit("[fatal] numpy required. `pip install numpy`.")

    # ------ stage 0: get the wheel & unpack ------
    if args.so:
        so_paths = [Path(args.so).resolve()]
        unpacked = None
    else:
        if args.wheel:
            wheel_path = Path(args.wheel).resolve()
        else:
            wheel_path = download_wheel(force=args.redownload)
        unpacked = unpack_wheel(wheel_path)
        so_paths = find_so_files(unpacked)
        if not so_paths:
            _log("[fatal] no .so found in wheel. Contents:")
            for p in unpacked.rglob("*"):
                if p.is_file():
                    _log(f"  {p.relative_to(unpacked)}")
            return 3

    # ------ stage 2/3: bundled data + python glue (informational) ------
    if unpacked is not None:
        find_bundled_codebooks(unpacked)
        grep_python_glue(unpacked)

    _log(f"\n[main] .so file(s): {[str(p) for p in so_paths]}")

    all_candidates = []
    for so in so_paths:
        _log(f"\n===== analyzing {so.name} ({so.stat().st_size/1024/1024:.2f} MiB) =====")
        # stage 4: symbol-based
        sym_hits = try_symbol_extraction(so)
        if sym_hits:
            all_candidates.extend((f"{so.name}::{n}", a) for n, a in sym_hits)

        # stage 5: .rodata scan (also run always — cross-check)
        scan_hits = try_rodata_scan(so)
        if scan_hits:
            all_candidates.extend((f"{so.name}::{lbl}", a) for lbl, a in scan_hits)

    # Deduplicate: two candidates that share the same first row are the same table.
    unique = []
    seen_fingerprints = set()
    for label, arr in all_candidates:
        import numpy as np
        fp = arr[0].tobytes()
        if fp in seen_fingerprints:
            continue
        seen_fingerprints.add(fp)
        unique.append((label, arr))

    _log(f"\n[main] {len(unique)} unique codebook candidate(s):")
    for label, arr in unique:
        import numpy as np
        _log(
            f"  {label}: shape={arr.shape} std={arr.astype(np.float32).std():.3f} "
            f"first_row[:4]={arr[0][:4]}"
        )

    if len(unique) >= 2:
        try:
            save_codebooks(unique, Path(args.out))
            _log("\n[SUCCESS] codebook extraction complete.")
            _log(
                "The MLX loader picks up escha_codebooks_v1.npz automatically at "
                "first inference. Nothing else to do."
            )
            return 0
        except Exception as e:
            _log(f"\n[fail] save_codebooks: {e}")

    # ---- diagnostics + fallback pointer ----
    _log(
        "\n[FAILURE] direct .so extraction did not find two clean codebook "
        "tables. Diagnostic dump follows; use it to file a HuggingFace "
        "discussion or fall back to the Modal path."
    )
    for so in so_paths:
        print_diagnostics(so)
    _log(
        "\n[next steps]\n"
        "  1. Modal fallback (~15 min, ~$0.20):\n"
        "     modal run mlx_video/models/qwen3_5_moe_escha/codebooks/modal_extract.py\n"
        "  2. File a diagnostic report at:\n"
        f"     https://huggingface.co/{WHEEL_REPO}/discussions\n"
        "     (see docs/eschalabs_request_draft.md for a pre-written template)."
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
