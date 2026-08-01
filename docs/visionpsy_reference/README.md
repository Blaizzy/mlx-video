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
- nanovlm
- vqa
- image-text-to-text
base_model:
- lusxvr/nanoVLM-460M-8k
---

# VisionPsy-Nano-460M

**VisionPsy-Nano-460M** is a compact (~460M parameter) vision-language model built for on-device and
edge deployment. It is built on top of [nanoVLM](https://github.com/huggingface/nanoVLM). Despite
its size, it **compares favorably on 16 of 17 public benchmarks** and matches or beats models
up to ~2x larger on knowledge reasoning, visual instruction following, and hallucination
robustness.

- **Best-in-class** among ~0.5B VLMs: **compares favorably** to LFM2.5-VL-450M,
  SmolVLM2-500M, and nanoVLM-460M-8k on **16 of 17** benchmarks.
- **Punches above its weight**: beats the larger **FastVLM-0.5B (~0.75B)**, **Qwen3.5-0.8B**, and
  **InternVL3.5-1B** outright on ScienceQA, MM-IFEval, and POPE, and on many fine-grained
  reasoning/perception capabilities.
- **Efficient architecture**: built on nanoVLM (SigLIP2 vision encoder + SmolLM2 language
  backbone), ~460M params total, 8K token context, native 512x512 image processing.

---

## Where VisionPsy-Nano-460M leads

To showcase where the model is strongest, we group the 17 public benchmarks into four core
capability areas — **Document Understanding & OCR, Visual Perception, Reasoning & Knowledge, and
Instruction Following & Reliability**. **VisionPsy-Nano-460M leads its ~0.5B cohort in every one of
these areas.** Scores are normalized to 0-100 for cross-benchmark comparison (MME /2800, OCRBench
/1000; percentages otherwise).

| Capability area | Benchmarks | VisionPsy-Nano-460M | LFM2.5-VL-450M | SmolVLM2-500M | nanoVLM-460M-8k |
| --- | --- | --- | --- | --- | --- |
| Document Understanding & OCR | OCRBench, DocVQA, ChartQA, InfoVQA, TextVQA | **73.9** | 71.6 | 62.4 | 69.5 |
| Visual Perception | MME, SEEDBench, MMBench, RealWorldQA | **61.5** | 58.8 | 53.9 | 55.2 |
| Reasoning & Knowledge | ScienceQA, AI2D, MMStar, MMMU, MathVista, MMVet | **52.2** | 48.6 | 45.1 | 43.7 |
| Instruction Following & Reliability | MM-IFEval, POPE | **65.1** | 64.3 | 46.9 | 51.4 |

![Category averages](assets/category_summary.png)

Per-benchmark detail within each capability area:

![Performance by capability](assets/category_performance.png)

---

## Model details

| Property | Value |
| --- | --- |
| Developed by | [Tether AI Research](https://tether.io/)<sup>*</sup> |
| Parameters | ~460M total |
| Built on | [nanoVLM](https://github.com/huggingface/nanoVLM) (base: `lusxvr/nanoVLM-460M-8k`) |
| Vision encoder | SigLIP2 base, patch16, 512x512 |
| Language backbone | SmolLM2-360M (nanoVLM's LM) |
| Connector | pixel-shuffle MLP (64 image tokens / tile) |
| Context length | 8,192 tokens |
| Image handling | native 512x512, tiling up to 2048px |
| Precision | float32 |

<sup>*</sup> References to Tether AI Research are references to Tether Data, S.A. de C.V.

---

## Benchmarks

All numbers below are computed **in-house with a single [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
harness** so every model is scored identically, using each benchmark's **official metric**:
accuracy for most, POPE = F1, MMVet = partial credit, MM-IFEval = instruction-following accuracy,
**MME = Perception + Reasoning score**, **OCRBench = Final Score (out of 1000)**. Each model is
evaluated on its own full sample set. **All LLM-as-judge scoring uses Qwen3.6-27B** as the judge.

The changes used for these numbers are currently under review in VLMEvalKit:

- **VisionPsy model support**: [PR #1613](https://github.com/open-compass/VLMEvalKit/pull/1613)
- **Opt-in LLM-judge rescoring for open-ended VQA benchmarks**: [PR #1602](https://github.com/open-compass/VLMEvalKit/pull/1602)
- **MM-IFEval bug fix**: [PR #1601](https://github.com/open-compass/VLMEvalKit/pull/1601)
- **LLM-judge bug fix**: [PR #1611](https://github.com/open-compass/VLMEvalKit/pull/1611)

Until they are merged, you can reproduce the benchmarks by applying these PRs on top of
[VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

### Similar-size models (~0.45-0.5B)

Bold = best score in the row across the similar-size cohort. Sorted by VisionPsy-Nano's lead. Scores are
% unless noted (MME = P+R points, OCRBench = /1000). Starred (\*) rows show **LLM-judge / strict**.

| Benchmark | Metric | VisionPsy-Nano-460M | LFM2.5-VL-450M | SmolVLM2-500M | nanoVLM-460M-8k |
| --- | --- | --- | --- | --- | --- |
| MME | P+R | **1541** | 1453 | 1455 | 1442 |
| OCRBench \* | /1000 | **757 / 726** | 710 / 679 | 619 / 611 | 745 / 710 |
| ScienceQA | acc | **86.5** | 77.7 | 76.3 | 76.8 |
| MathVista | acc | **48.9** | 42.2 | 38.5 | 33.3 |
| MMBench | acc | **61.9** | 56.8 | 51.4 | 53.1 |
| MMStar | acc | **47.6** | 42.7 | 38.3 | 37.0 |
| AI2D | acc | **66.5** | 62.2 | 57.3 | 55.3 |
| DocVQA \* | acc | **85.7 / 80.6** | 82.7 / 78.3 | 74.7 / 68.6 | 82.3 / 77.3 |
| ChartQA \* | acc | **78.7 / 76.8** | 76.6 / 75.0 | 64.9 / 60.9 | 70.4 / 65.9 |
| SEEDBench | acc | **69.2** | 67.5 | 62.1 | 64.3 |
| POPE | F1 | **87.9** | 86.5 | 82.7 | 82.7 |
| RealWorldQA | acc | **60.0** | 59.0 | 50.1 | 51.8 |
| InfoVQA \* | acc | **49.8 / 43.3** | 48.9 / 43.3 | 39.0 / 27.6 | 46.0 / 35.0 |
| TextVQA \* | acc | **79.3 / 71.0** | 78.8 / 69.8 | 71.3 / 60.6 | 74.5 / 64.8 |
| MM-IFEval | IF-acc | **42.3** | 42.0 | 11.1 | 20.2 |
| MMMU | acc | 31.4 | 30.7 | **31.6** | 31.3 |
| MMVet | partial-credit | 32.3 | **36.0** | 28.6 | 28.4 |

**VisionPsy-Nano-460M compares favorably on 16 of 17 benchmarks**, with MMVet the only clear
exception.

\* **ChartQA, TextVQA, DocVQA, InfoVQA, and OCRBench** are shown as **LLM-judge / strict**. On these
open-ended benchmarks, strict string/heuristic matching lowers every model's score, but an
**LLM-as-judge** (Qwen3.6-27B) is a more reliable evaluation of free-form answers (e.g., "12%" vs
"12 percent", paraphrases, units, formatting). Full analysis in the [blog post](https://huggingface.co/blog/qvac/visionpsy).

### Beating larger models (0.75-1B)

At roughly **half to two-thirds the parameters**, VisionPsy-Nano-460M still **beats FastVLM-0.5B (~0.75B),
Qwen3.5-0.8B, and InternVL3.5-1B outright on 3 full benchmarks**, and outperforms all three on many
specific reasoning/perception capabilities even where the larger models win on aggregate. Bold =
VisionPsy-Nano leads all three larger models.

| Scope | Capability | Benchmark (subset) | N | VisionPsy-Nano-460M | FastVLM-0.5B (759M) | Qwen3.5-0.8B (873M) | InternVL3.5-1B (1061M) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full benchmark | Science & diagram reasoning | ScienceQA | 2017 | **86.5** | 84.7 | 71.6 | 79.7 |
| Full benchmark | Visual instruction following | MM-IFEval | 400 | **42.3** | 21.1 | 36.9 | 33.6 |
| Full benchmark | Hallucination robustness | POPE (F1) | 5127 | **87.9** | 86.5 | 87.3 | 86.0 |
| Subcategory | Spatial position reasoning | MME (position) | 60 | **85.0** | 75.0 | 63.3 | 73.3 |
| Subcategory | Cross-instance relation reasoning | MMStar (L2) | 62 | **66.1** | 56.5 | 59.7 | 53.2 |
| Subcategory | Nature relation reasoning | MMBench (L1) | 62 | **51.6** | 46.8 | 41.9 | 45.2 |
| Subcategory | Irregular text recognition | OCRBench | 50 | **92.0** | 80.0 | 88.0 | 82.0 |
| Subcategory | Object localization | MMStar (L2) | 40 | **57.5** | 45.0 | 50.0 | 55.0 |
| Subcategory | Attribute recognition | MMBench (L1) | 60 | **86.7** | 85.0 | 83.3 | 83.3 |
| Subcategory | Instance interaction | SEEDBench | 97 | **70.1** | 67.0 | 69.1 | 69.1 |

*The 3 "full benchmark" rows are outright wins over all three larger models. The "subcategory" rows
are fine-grained capabilities (N >= 40) from suites where the larger models may lead on aggregate —
the point being that a 460M model can still beat models up to ~2x its size on these specific skills.
All scores are accuracy except POPE (F1).*

---

## Usage

VisionPsy-Nano-460M loads directly with 🤗 Transformers via `trust_remote_code` — no extra repo to
clone. Requires Python ≥ 3.10, `transformers>=4.46` (tested with 5.13.1), and PyTorch ≥ 2.4 (CUDA
recommended).

```python
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

repo = "qvac/VisionPsy-Nano-460M"
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

---

## Intended use

VisionPsy-Nano-460M targets latency- and memory-constrained, on-device multimodal applications:
visual question answering, document/chart/diagram understanding, scene-text reading, and
lightweight visual instruction following. Because of its small size, we recommend fine-tuning
on your specific domain to maximize quality.

## Limitations

- Single-image by design: the model is trained and optimized for **one image per query**, so we
  recommend single-image inputs at inference; multi-image prompts are outside its intended use.
- As a compact model, it may occasionally hallucinate or miscount and is best suited to focused
  tasks rather than very dense documents or long multi-step math, where larger models have an edge.
- Primarily English; other languages are not officially supported yet.
- Not intended for safety-critical or high-stakes automated decisions.
- Benchmark scores are produced with a fixed in-house harness and an LLM judge (Qwen3.6-27B);
  absolute numbers may differ from other reported setups.

## Acknowledgements

Built on the excellent open-source work of [nanoVLM](https://github.com/huggingface/nanoVLM),
[SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct), and
[SigLIP2](https://huggingface.co/google/siglip2-base-patch16-512).

## Citation

```bibtex
@misc{visionpsynano2026,
  title  = {VisionPsy-Nano-460M: A Compact Vision-Language Model for On-Device Inference},
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
