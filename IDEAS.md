# scorer-ocrs-dataset — measuring synthetic-dataset "richness" for OCR finetuning

**Goal.** Compare candidate synthetic handwriting datasets *without training*, to predict which
will transfer best zero-shot to the real target (Matan). Motivation: exp28 — own-test 1.4% CER
but Matan 55% (worse than the 40% fonts ceiling); the 3-style connected corpus mastered itself
and missed the target. We need a compass that costs minutes, not L4-hours.

## Core principle

**Diversity alone is not the goal — coverage of the target is.** Random noise is maximally
diverse and useless. What predicts zero-shot transfer is whether the synthetic distribution
*covers the region where the target lives*. So we always report two families of numbers:

1. **Distance/coverage w.r.t. target** (Matan) — where is the synthetic cloud?
2. **Intrinsic spread** — how wide is the synthetic cloud itself?

## Metrics in the scorecard (`compare_datasets.py`)

All computed on embeddings of line-image patches from a **frozen** vision encoder
(no training on any compared dataset → unbiased):

| metric | what it answers | direction |
|---|---|---|
| **KID → target** (unbiased MMD², poly kernel) | how far is this dataset's distribution from Matan | lower = better |
| **Coverage of target** (Naeem et al., kNN manifold) | fraction of Matan samples that have a synthetic neighbor nearby — "is every real style represented?" | higher = better |
| **Density** | how concentrated synthetic samples are on the real manifold | higher = better |
| **Vendi score** (eigenvalues of similarity kernel) | *effective number of distinct samples* — intrinsic diversity | higher = richer |
| **mean pairwise cosine dist** | crude intrinsic spread | higher = wider |
| **proxy A-distance** (logistic reg synth-vs-Matan, 5-fold CV) | how easily a classifier separates this dataset from Matan; 2·(2·acc−1); 0 = indistinguishable, 2 = trivially separable | lower = better |

Diagnostic extras (`--stats`): per-feature Wasserstein distance to Matan for stroke width,
ink density, slant proxy, components-per-width — tells you *which axis* lacks variety.

## Validation protocol (we have ground truth!)

Past experiments give known zero-shot Matan CERs. A trustworthy metric must reproduce this
ranking with no training:

| dataset | known Matan CER | expected scorecard rank |
|---|---|---|
| `matan_parquet/train` (real, same writers) | — | gold: best on everything |
| `human_parquet` (SCE real handwriting) | — | close to Matan (real ink) |
| `heb_bigram_composed_250k` (17-style bank composed; exp26) | ~38.6–40% | mid |
| `syntetic_modern_parquet` (font-rendered; exp20/21 track) | ~40–47% | mid |
| `heb_connected_composed_3track` (3-style connected; exp28) | ~55% | worst of the synthetics |

If KID/coverage/A-distance rank these correctly → compass validated; evaluate every future
dataset idea in minutes on the 2080. If not → the encoder's blind spots dominate; try the
second encoder before trusting anything.

## Encoder choices

- `mobilenetv2_100.ra_in1k` (timm, cached locally) — generic ImageNet texture features.
- `vit_small_patch14_dinov2.lvd142m` (DINOv2-S) — much stronger self-supervised features,
  needs an ~85 MB HF download (Xet outage permitting).
- Deliberately **not** using our own TrOCR encoders as primary: they were trained on some of
  the compared datasets → biased. (Can be added later as a third view.)
- Guard against blind spots by cross-checking two encoders.

## Patching scheme

Line images vary in width; encoders want fixed input. Each line: grayscale→RGB, resize to
encoder input height keeping aspect, slice up to 3 evenly-spaced square windows, drop
near-empty ones (<1% ink). Each window = one sample → captures local handwriting texture
(stroke style, spacing), which is exactly what transfers or doesn't.

## Caveats (honesty box)

- These metrics **correlate** with transfer; they don't guarantee it. An encoder that can't
  see a difference makes the metric blind to it.
- KID/coverage need a few thousand samples per dataset; below ~1k they get noisy.
- Comparing *content-matched* datasets (same sentences, different rendering) is cleanest;
  content differences leak into embeddings a little (letter frequencies etc.).
- Vendi on patches counts texture variety, not writer variety per se; interpret alongside
  coverage rather than alone.
