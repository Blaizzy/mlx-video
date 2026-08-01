---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- vision-language-model
- multimodal
- edge
- on-device
- efficient
- low-latency
- flash
- nanovlm
- vqa
- image-text-to-text
base_model:
- qvac/VisionPsy-Nano-460M
- lusxvr/nanoVLM-460M-8k
---

# VisionPsy-Nano-460M-Flash

**VisionPsy-Nano-460M-Flash** is the **efficiency-optimized** sibling of
[VisionPsy-Nano-460M](https://huggingface.co/qvac/VisionPsy-Nano-460M). It keeps the same ~460M
parameter nanoVLM architecture but processes each image with **far fewer visual tokens
(as few as 64 at its native 512x512, versus 256-1088 for peers)**. The result is a model that
responds **an order of magnitude faster** and with a **smaller memory footprint** on real phones,
while giving up only a small amount of accuracy.

If VisionPsy-Nano-460M is our **best-quality** ~0.5B model, **Flash is our best model for the
edge**: when time-to-first-token, battery, and RAM are the binding constraints, Flash is the one
you ship.

- **Fastest time-to-first-token (TTFT) in its class.** By collapsing the image to as few as 64
  visual tokens, Flash reaches the first token **~20-36x faster than nanoVLM-460M and SmolVLM2-500M**, and
  consistently faster than **LFM2.5-VL-450M** and **Qwen3.5-0.8B** on Pixel 9, Galaxy S23,
  Galaxy S25, and iPhone 15.
- **Small quality trade-off.** Retains **~99% of the full model's quality** (normalized score
  **61.4 vs 62.3**), and it still **beats the next best competitor (LFM2.5) on 13 of 17 benchmarks** (plus 1 tie).
- **Leaner on device.** ~40% lower peak memory than nanoVLM-460M and SmolVLM2-500M, and higher decode
  throughput than nanoVLM-460M, SmolVLM2-500M, and Qwen3.5-0.8B in most device/backend configurations.
- **Built for edge.** ~460M params, native 512x512 processing, deployable as a 4-bit GGUF that fits
  comfortably on a phone.

---

## Why Flash is fast: the visual-token lever

The dominant cost in a small VLM's prefill is the number of **visual tokens** the language model
has to attend to. Flash keeps the **same architecture** as the full model but adopts a more efficient
**image-processing policy**: it **preserves each image's native resolution** instead of upsampling it
before tiling, so smaller images produce fewer tiles and far fewer visual tokens. The savings hold across
resolutions — and against models like LFM2.5-VL-450M and Qwen3.5-0.8B, whose token counts balloon
on large images, the gap actually **widens** at high resolution.

**Visual tokens per image (fewer = faster prefill):**

| Input resolution | **Flash** | nanoVLM-460M | LFM2.5-VL-450M | SmolVLM2-500M | Qwen3.5-0.8B |
| --- | --- | --- | --- | --- | --- |
| 512x512 (native) | **64** | 1088 | 256 | 1088 | 256 |
| 512x1024 | **192** | 576 | 242 | 576 | 512 |
| 1024x662 | **320** | 832 | 228 | 832 | 672 |
| 2048x1024 | 576 | 576 | 2290 | 576 | 2048 |

At its native **512x512**, Flash uses **17x fewer** visual tokens than nanoVLM-460M and SmolVLM2-500M,
and **4x fewer** than LFM2.5-VL-450M and Qwen3.5-0.8B. At **2048x1024**, LFM2.5-VL-450M and
Qwen3.5-0.8B jump to **2,290** and **2,048** tokens while Flash stays at **576** — a ~4x advantage.

---

## On-device efficiency

All numbers below are measured **on real devices** — **Pixel 9, Samsung Galaxy S23, Samsung Galaxy
S25, and iPhone 15** — using **4-bit (Q4_0) GGUF** builds, reported as `CPU / GPU`. Lower is better
for TTFT and memory; higher is better for decode throughput. Bold marks the best value in each cell
group where Flash leads.

### Time to first token (TTFT, seconds) — the headline

TTFT is what a user actually feels when they point the camera and ask a question. This is where
Flash's token reduction pays off most.

**At 512x512 (native resolution):**

| Model | Vis tokens | Pixel 9 (CPU/GPU) | Galaxy S23 | Galaxy S25 | iPhone 15 |
| --- | --- | --- | --- | --- | --- |
| **VisionPsy-Nano-460M-Flash** | **64** | **6.1 / 16.4** | **6.1 / 5.9** | **2.7 / 2.6** | 5.9 / **0.3** |
| nanoVLM-460M (full) | 1088 | 153.0 / 138.3 | 119.0 / 116.7 | 67.0 / 58.8 | 70.3 / 10.9 |
| LFM2.5-VL-450M | 256 | 14.9 / 21.9 | 13.6 / 14.2 | 7.7 / 5.7 | 3.4 / 0.4 |
| SmolVLM2-500M | 1088 | 128.2 / 136.1 | 135.5 / 113.7 | 48.7 / 64.3 | 72.1 / 10.7 |
| Qwen3.5-0.8B | 256 | 21.4 / 25.7 | 22.0 / 19.9 | 8.4 / 10.4 | 5.0 / 0.7 |

**Takeaway:** Picking the best backend per phone, Flash reaches the first token **~19-23x faster than
nanoVLM-460M and SmolVLM2-500M** on Pixel 9, Galaxy S23, and Galaxy S25, and **up to ~36x** on
iPhone 15. It is also **faster than LFM2.5-VL-450M on all four devices** (\~1.3-2.4x) and faster than
Qwen3.5-0.8B across the board (\~2.3-3.5x). (On iPhone 15 CPU alone, LFM is slightly ahead; using the
iPhone GPU, Flash wins.)

![Time to first token across devices (512x512)](assets/ttft.png)

**At 2048x1024 (high-resolution / document pages):**

| Model | Vis tokens | Pixel 9 (CPU/GPU) | Galaxy S23 | Galaxy S25 | iPhone 15 |
| --- | --- | --- | --- | --- | --- |
| **VisionPsy-Nano-460M-Flash** | 576 | **51.5** / 156.0 | **51.5** / **58.8** | **21.7** / **24.7** | **33.9** / **2.8** |
| nanoVLM-460M (full) | 576 | 75.8 / 72.8 | 63.6 / 61.3 | 36.7 / 29.8 | 35.1 / 5.4 |
| LFM2.5-VL-450M | 2290 | 140.3 / 103.8 | 98.0 / 91.4 | 70.3 / 43.0 | 38.5 / 4.1 |
| SmolVLM2-500M | 576 | 62.1 / 72.5 | 70.9 / 59.9 | 29.2 / 33.8 | 36.7 / 5.4 |
| Qwen3.5-0.8B | 2048 | 311.4 / 325.2 | 265.2 / 248.9 | 127.5 / 137.3 | 64.8 / 14.8 |

**Takeaway:** At high resolution the token gap widens — LFM2.5-VL-450M and Qwen3.5-0.8B expand to
2,290 and 2,048 visual tokens while Flash holds at 576 — so Flash is the **fastest to first token on
7 of 8 device/backends**, roughly **~1.2-1.9x faster than nanoVLM-460M**, **~1.5-2x faster than
LFM2.5-VL-450M**, and **~5-6x faster than Qwen3.5-0.8B**. (The one exception is the Pixel 9 GPU,
where Flash's high-resolution prefill regresses.)

### Peak memory (RSS, MiB)

**At 512x512 (native resolution):**

| Model | Pixel 9 (CPU/GPU) | Galaxy S23 | Galaxy S25 | iPhone 15 |
| --- | --- | --- | --- | --- |
| **VisionPsy-Nano-460M-Flash** | 750 / 1692 | 746 / 1047 | 763 / 1066 | 292 / 285 |
| nanoVLM-460M (full) | 1236 / 2224 | 1232 / 1533 | 1251 / 1552 | 827 / 482 |
| LFM2.5-VL-450M | 501 / 1123 | 497 / 588 | 510 / 602 | 238 / 181 |
| SmolVLM2-500M | 1236 / 2224 | 1232 / 1534 | 1252 / 1552 | 780 / 482 |
| Qwen3.5-0.8B | 866 / 2243 | 853 / 942 | 878 / 972 | 903 / 836 |

**At 2048x1024 (high-resolution / document pages):**

| Model | Pixel 9 (CPU/GPU) | Galaxy S23 | Galaxy S25 | iPhone 15 |
| --- | --- | --- | --- | --- |
| **VisionPsy-Nano-460M-Flash** | 964 / 1915 | 960 / 1261 | 979 / 1279 | 567 / 399 |
| nanoVLM-460M (full) | 992 / 1949 | 988 / 1290 | 1005 / 1305 | 754 / 420 |
| LFM2.5-VL-450M | 535 / 1197 | 530 / 621 | 543 / 636 | 271 / 216 |
| SmolVLM2-500M | 993 / 1939 | 989 / 1290 | 1005 / 1306 | 719 / 420 |
| Qwen3.5-0.8B | 1080 / 2495 | 1066 / 1160 | 1081 / 1174 | 1153 / 941 |

At both resolutions Flash uses less peak memory than nanoVLM-460M and SmolVLM2-500M (~40% less than
the full-token models at 512x512), and less than Qwen3.5-0.8B in six of eight device/backend
configurations per resolution; Qwen keeps a small edge only on the Galaxy GPUs. LFM2.5-VL-450M
remains the lightest of the group.

### Decode throughput (tokens/s)

**At 512x512 (native resolution):**

| Model | Pixel 9 (CPU/GPU) | Galaxy S23 | Galaxy S25 | iPhone 15 |
| --- | --- | --- | --- | --- |
| **VisionPsy-Nano-460M-Flash** | 28.1 / 30.7 | 32.7 / 33.8 | 93.5 / 93.7 | 7.6 / 78.9 |
| nanoVLM-460M (full) | 15.6 / 27.9 | 23.2 / 17.8 | 47.0 / 51.9 | 8.3 / 52.6 |
| LFM2.5-VL-450M | 42.6 / 47.8 | 52.3 / 42.6 | 69.8 / 122.4 | 28.2 / 116.0 |
| SmolVLM2-500M | 18.2 / 28.1 | 18.5 / 18.3 | 64.8 / 49.6 | 11.4 / 53.6 |
| Qwen3.5-0.8B | 17.9 / 22.9 | 12.8 / 21.1 | 45.2 / 24.6 | 6.9 / 40.5 |

**At 2048x1024 (high-resolution / document pages):**

| Model | Pixel 9 (CPU/GPU) | Galaxy S23 | Galaxy S25 | iPhone 15 |
| --- | --- | --- | --- | --- |
| **VisionPsy-Nano-460M-Flash** | 23.3 / 13.3 | 27.7 / 23.3 | 89.7 / 66.2 | 9.8 / 71.7 |
| nanoVLM-460M (full) | 17.4 / 31.4 | 25.7 / 20.5 | 56.5 / 63.2 | 10.9 / 53.6 |
| LFM2.5-VL-450M | 26.3 / 48.9 | 38.5 / 35.0 | 39.7 / 76.3 | 28.9 / 123.4 |
| SmolVLM2-500M | 22.4 / 32.8 | 20.7 / 22.1 | 71.8 / 55.0 | 13.6 / 54.3 |
| Qwen3.5-0.8B | 12.6 / 20.6 | 15.3 / 17.0 | 36.8 / 32.5 | 7.0 / 24.4 |

Across the 16 device/backend configurations (two resolutions x four devices x CPU/GPU), Flash decodes
**faster than nanoVLM-460M and SmolVLM2-500M in 13 of 16**, and **faster than Qwen3.5-0.8B in 15 of 16**.
LFM2.5-VL-450M leads on sustained decode thanks to its hybrid backbone — but because Flash's TTFT is
so much lower, Flash still wins **end-to-end latency** for the short answers typical of on-device
VQA, where prefill dominates.

### Where Flash wins, in one line

Holds at both **512x512** and **2048x1024**:

- **vs nanoVLM-460M and SmolVLM2-500M:** wins on TTFT
  (~20-36x at 512x512, ~1.2-1.9x at 2048x1024) and memory across the board, and on decode throughput in
  **13 of 16** device/backend configurations.
- **vs Qwen3.5-0.8B:** wins on TTFT, on memory in **6 of 8** configurations per resolution (Qwen leads
  only on the Galaxy GPUs), and on decode throughput in **15 of 16** configurations.
- **vs LFM2.5-VL-450M:** wins decisively on **TTFT** (and the lead grows at high resolution, where
  LFM's token count balloons); LFM keeps an edge on **peak memory** and **decode throughput**.

*Methodology: latency, memory, and throughput measured on-device with 4-bit (Q4_0) GGUF builds
across Pixel 9, Galaxy S23, Galaxy S25, and iPhone 15 (CPU and GPU backends), mean over repeated
runs.*

---

## An Efficient Operating Point

Flash keeps most of the full model's quality. All scores are computed **in-house with a single
[VLMEvalKit](https://github.com/open-compass/VLMEvalKit) harness** so every model is scored
identically, using each benchmark's official metric (POPE = F1, MMVet = partial credit, MM-IFEval =
instruction-following accuracy, OCRBench = /1000, MME = Perception + Reasoning points).
LLM-as-judge scoring uses Qwen3.6-27B.

The changes used for these numbers are currently under review in VLMEvalKit:

- **VisionPsy model support**: [PR #1613](https://github.com/open-compass/VLMEvalKit/pull/1613)
- **Opt-in LLM-judge rescoring for open-ended VQA benchmarks**: [PR #1602](https://github.com/open-compass/VLMEvalKit/pull/1602)
- **MM-IFEval bug fix**: [PR #1601](https://github.com/open-compass/VLMEvalKit/pull/1601)
- **LLM-judge bug fix**: [PR #1611](https://github.com/open-compass/VLMEvalKit/pull/1611)

Until they are merged, you can reproduce the benchmarks by applying these PRs on top of
[VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

| Benchmark | Metric | **VisionPsy-Nano-460M-Flash** | VisionPsy-Nano-460M |
| --- | --- | --- | --- |
| MMStar | acc | 45.53 | 47.60 |
| MMBench | acc | 60.45 | 61.92 |
| RealWorldQA | acc | 58.43 | 60.00 |
| MME | P+R | 1619 | 1541 |
| SEEDBench | acc | 67.58 | 69.20 |
| POPE | F1 | 87.46 | 87.93 |
| MMMU | acc | 30.67 | 31.44 |
| MathVista | acc | 47.80 | 48.90 |
| AI2D | acc | 66.39 | 66.48 |
| ScienceQA | acc | 84.04 | 86.47 |
| OCRBench \* | /1000 | 738 / 699 | 757 / 726 |
| ChartQA \* | acc | 77.40 / 75.12 | 78.68 / 76.84 |
| TextVQA \* | acc | 75.58 / 67.44 | 79.34 / 70.98 |
| DocVQA \* | acc | 85.12 / 80.75 | 85.74 / 80.63 |
| InfoVQA \* | acc | 51.12 / 44.80 | 49.80 / 43.29 |
| MM-IFEval | IF-acc | 43.67 | 42.30 |
| MMVet | partial-credit | 31.10 | 32.34 |

VisionPsy-Nano-460M-Flash demonstrates robust performance on general VQA and reasoning, with gains
on MME and InfoVQA, while showing greater sensitivity on high-resolution OCR tasks.

- **Flash keeps ~99% of the full model's normalized score (61.4 vs 62.3)** while running an order of
  magnitude faster, and its normalized score still **edges out LFM2.5-VL-450M (59.6)**.
- **Flash still beats LFM2.5-VL-450M on 13 of 17 benchmarks** (plus 1 tie), and stays ahead of
  SmolVLM2-500M on **16 of 17**.
- The trade-off is concentrated in the OCR/document-heavy and fine-grained perception tasks
  (ChartQA, TextVQA, MMStar), where fewer visual tokens cost the most detail.

\* **ChartQA, TextVQA, DocVQA, InfoVQA, and OCRBench** are shown as **LLM-judge / strict**. On these
open-ended benchmarks, strict string/heuristic matching lowers every model's score, but an
**LLM-as-judge** (Qwen3.6-27B) is a more reliable evaluation of free-form answers (e.g., "12%" vs
"12 percent", paraphrases, units, formatting); see the main VisionPsy-Nano-460M card for details.

For the full-quality model and its detailed comparisons against larger models (FastVLM-0.5B,
Qwen3.5-0.8B, InternVL3.5-1B), see the
[VisionPsy-Nano-460M model card](https://huggingface.co/qvac/VisionPsy-Nano-460M).

---

## Model details

| Property | Value |
| --- | --- |
| Developed by | [Tether AI Research](https://tether.io/)<sup>*</sup> |
| Parameters | ~460M total |
| Built on | [VisionPsy-Nano-460M](https://huggingface.co/qvac/VisionPsy-Nano-460M) / [nanoVLM](https://github.com/huggingface/nanoVLM) (base: `lusxvr/nanoVLM-460M-8k`) |
| Vision encoder | SigLIP2 base, patch16, 512x512 |
| Language backbone | SmolLM2-360M (nanoVLM's LM) |
| Connector | pixel-shuffle MLP (64 image tokens / tile), same as the full model |
| Visual tokens | 64 at native 512x512 (vs 256-1088 for peers) |
| Context length | 8,192 tokens |
| Image handling | processed at native resolution (only upscaled below 512x512), then tiled |
| Precision | float32 (also shipped as GGUF for on-device) |

<sup>*</sup> References to Tether AI Research are references to Tether Data, S.A. de C.V.

---

## Usage

VisionPsy-Nano-460M-Flash loads directly with 🤗 Transformers via `trust_remote_code` — no extra
repo to clone. Requires Python ≥ 3.10, `transformers>=4.46` (tested with 5.13.1), and PyTorch ≥ 2.4
(CUDA recommended).

```python
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

repo = "qvac/VisionPsy-Nano-460M-Flash"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForImageTextToText.from_pretrained(
    repo,
    trust_remote_code=True,
    dtype="auto" if device == "cuda" else torch.float32,
).to(device).eval()
processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)

# apply_deploy_profile() enables torch.compile + CUDA graphs (fastest on GPU);
# use apply_eager_profile() for plain eager execution or CPU.
if device == "cuda":
    model.apply_deploy_profile(model.device)
else:
    model.apply_eager_profile()

# The processor applies the chat template and inserts image tokens for you —
# just pass the raw image(s) and a plain-text prompt.
image = Image.open("your_image.jpg").convert("RGB")
inputs = processor(images=image, text="What is in this image?", return_tensors="pt")
inputs = {
    k: (v.to(device) if torch.is_tensor(v) else v)
    for k, v in inputs.items()
    if v is not None
}
inputs.pop("pixel_values", None)

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=128, greedy=True)
print(processor.batch_decode(out, skip_special_tokens=True)[0].strip())
```

For on-device deployment, use the 4-bit (Q4_0) GGUF build with a llama.cpp-based runtime.

---

## Intended use

VisionPsy-Nano-460M-Flash is designed for **latency- and memory-constrained, on-device multimodal
applications** where responsiveness matters more than squeezing out the last accuracy point: live
camera Q&A, quick scene/document understanding, and lightweight visual instruction following on
phones and other edge hardware. If you can afford more compute and want maximum quality, use
[VisionPsy-Nano-460M](https://huggingface.co/qvac/VisionPsy-Nano-460M) instead. Because of its small
size, we recommend fine-tuning on your specific domain to maximize quality.

## Limitations

- Single-image by design: the model is trained and optimized for **one image per query**, so we
  recommend single-image inputs at inference; multi-image prompts are outside its intended use.
- As a compact model, it may occasionally hallucinate or miscount and is best suited to focused
  tasks rather than very dense documents or long multi-step math, where larger models have an edge.
- Primarily English; other languages are not officially supported yet.
- Not intended for safety-critical or high-stakes automated decisions.
- Efficiency numbers are measured with 4-bit GGUF builds on specific phones; absolute latency,
  memory, and throughput will vary with hardware, runtime, and quantization. Benchmark scores use a
  fixed in-house harness with an LLM judge (Qwen3.6-27B) and may differ from other reported setups.

## Acknowledgements

Built on the excellent open-source work of [nanoVLM](https://github.com/huggingface/nanoVLM),
[SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct), and
[SigLIP2](https://huggingface.co/google/siglip2-base-patch16-512).

## Citation

```bibtex
@misc{visionpsynanoflash2026,
  title  = {VisionPsy-Nano-460M-Flash: An Efficiency-Optimized Vision-Language Model for On-Device Inference},
  author = {Tether AI Research},
  year   = {2026},
  note   = {Hugging Face model card}
}
```

## Copyright

We will take appropriate actions in response to notices of copyright infringement. If you believe
your work has been used or copied in a manner that infringes upon your intellectual property rights,
please email data-apps@tether.io identifying and describing both the copyrighted work and alleged
infringing content.

## Licensing

This model, which was finetuned as described in [the blog post](https://huggingface.co/blog/qvac/visionpsy), is licensed by Tether Data,
S.A. de C.V. under the Apache 2.0 license. As described in the blog post, this model is a version of
the NanoVLM-460M-8k pre-trained model (https://huggingface.co/lusxvr/nanoVLM-460M-8k), which is made
available under the MIT license.

The FineVision dataset (https://huggingface.co/datasets/HuggingFaceM4/FineVision) is made available
under the CC-BY-4.0 (Creative Commons - Attribution 4.0) license. FineVision is an aggregation of a
number of public sources unified into a single corpus. Individual subsets within the collection may
inherit specific underlying terms from their original creators. As described in the blog post, a
subset of the FineVision dataset was used as a part of finetuning the model.

The NVIDIA Nemotron-Image-Training-v3 dataset
(https://huggingface.co/datasets/nvidia/Nemotron-Image-Training-v3) is made available under the
CC-BY-4.0 (Creative Commons - Attribution 4.0). The mPLUG TinyChartData dataset
(https://huggingface.co/datasets/mPLUG/TinyChartData) is made available under the Apache 2.0
license. The TabMWP dataset (https://promptpg.github.io/) is made available under the CC BY-NC-SA
4.0 (Creative-Commons-Attribution-NonCommercial-ShareAlike 4.0). The PopVQA dataset
(https://huggingface.co/datasets/idoco/PopVQA) is made available under the MIT license. The
InfoSeek dataset (https://github.com/open-vision-language/infoseek) is made available under the
Apache 2.0 license. The MMKU-Bench dataset (https://huggingface.co/datasets/baochenfu/MMKU-Bench)
is made available under the Apache 2.0 license. The VisionFoundry-10K dataset
(https://huggingface.co/datasets/zlab-princeton/VisionFoundry-10K) is made available under the
Apache 2.0 license. The PKU-SafeRLHF-V dataset
(https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF-V) is made available under the CC-BY-NC
4.0 (Attribution-NonCommercial 4.0 International). As described in the blog post, the NVIDIA
Nemotron-Image-Training-v3, mPLUG TinyChartData, TabMWP, PopVQA, InfoSeek, MMKU-Bench,
VisionFoundry-10K and PKU-SafeRLHF-V datasets were used as a part of finetuning the model.
