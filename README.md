# STELLAR — Spectral Type Estimation via Language Learning and Astronomical Reasoning

> A hybrid stellar classifier that combines deterministic astrophysical rules (Hard Computing) with a fine-tuned LLM (AstroSage-8B) to assign MK spectral types and generate natural-language stellar descriptions from Gaia DR3 data.

---

## What it does

Given a star's Gaia DR3 spectral data, STELLAR:

1. Runs a **4-agent Hard Computing (HC) pipeline** that deterministically extracts astrometric, photometric, spectral-line, and binary-star features — producing a structured JSON contract per star.
2. Feeds that contract to **AstroSage-8B** (a Llama 3.1-based LLM fine-tuned on astronomy literature) via a RAG-augmented prompt, which outputs:
   - An MK spectral classification (letter + luminosity class + population)
   - A natural-language stellar description

The HC contract acts as a **hard anchor**: if the LLM drifts from the deterministic classification, the system overrides the prediction — correcting a known G-type prior bias in the base model.

---

## Results (V7 — Final)

| Metric | Value |
|---|---|
| Accuracy (vs SIMBAD) | **0.7951** |
| Cohen's κ | **0.7529** |
| Macro F1 | **0.6936** |
| Near-miss accuracy (d ≤ 1 MK step) | **0.998** |
| Mean \|ΔTeff\| vs PASTEL | **212 K** |
| Mean \|Δlog g\| vs PASTEL | **0.46 dex** |
| BERTScore F1 (descriptions) | **0.866** |
| Bootstrap 95% CI on error | [0.141, 0.205] |

Corpus: **498 stars** from Gaia DR3 · 7 prompt versions iterated (V1–V7) · Validated against SIMBAD (classification) and PASTEL (physical parameters).

---

## Architecture

```
Gaia DR3 spectrum
       │
       ▼
┌──────────────────────────────────┐
│     Hard Computing (HC-2.0)      │
│  AstrometryAgent  → M_G, V_tan   │
│  ContinuumAgent   → Teff, log g  │
│  LineAgent        → EW(CaII, Hα) │
│  BinaryDetector   → RUWE, NSS    │
│                                  │
│  Output: JSON contract + letter  │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│   RAG  (all-MiniLM-L6-v2)        │
│   knowledge_base/ → top-3 chunks │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│   AstroSage-8B (Llama 3.1 8B)    │
│   MK type + stellar description  │
└──────────────────────────────────┘
```

---

## Tech stack

- **LLM:** [`AstroMLab/AstroSage-8B`](https://huggingface.co/AstroMLab/AstroSage-8B) — Llama 3.1 8B fine-tuned on 300k+ astronomy papers
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **HC pipeline:** Pure Python · NumPy · Astropy · `astroquery` (SIMBAD, Gaia)
- **Inference:** HuggingFace `transformers` · SLURM · 2× NVIDIA Titan RTX 24 GB
- **Validation:** `scikit-learn` · `scipy` · `bert-score` · `rouge-score`
- **Data:** [Gaia DR3](https://gea.esac.esa.int/archive/) · [PASTEL](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=B/pastel)

---

## Repository structure

```
├── cluster/
│   ├── scripts/            # Production SC scripts (V7)
│   │   ├── corpus_builder_v7.py
│   │   ├── inference_manager_v7.py
│   │   ├── system_prompt_v7.py
│   │   ├── validator_v7.py
│   │   └── pre_encode_queries.py
│   ├── knowledge_base/     # RAG knowledge base (5 MD guides)
│   ├── SLURM/              # Job scripts for HPC cluster
│   ├── Data/               # Gaia DR3 sample, HC contracts, ground truth
│   └── outputs/            # Validation results V5–V7
├── scripts/
│   ├── Hard_Computing/     # HC pipeline (4 agents)
│   └── Soft_Computing/     # Early SC versions (V1–V4)
├── Queries/                # ADQL queries and Gaia data fetching
├── Context/                # Agent specifications and design docs
└── reporte/                # LaTeX technical report (~33 pages)
```

---

## Reproduce

### 1. Build the HC contracts

```bash
conda activate prometheus   # or any env with astropy, astroquery, numpy
python scripts/Hard_Computing/hc_pipeline_orchestrator_v2.py \
    --catalog cluster/Data/catalog.csv \
    --output  cluster/Data/hc_contracts.json
```

### 2. Build the inference corpus

```bash
python cluster/scripts/corpus_builder_v7.py
```

### 3. Pre-encode RAG queries

```bash
python cluster/scripts/pre_encode_queries.py
```

### 4. Run inference on the cluster

```bash
sbatch cluster/SLURM/run_inference_v7.slurm
```

### 5. Validate

```bash
sbatch cluster/SLURM/run_validation_v7.slurm
```

Results land in `cluster/outputs/validation_v7/`.

> **Model weights** are not included. Download from HuggingFace:
> ```bash
> huggingface-cli download AstroMLab/AstroSage-8B --local-dir models/AstroSage-8B
> ```

---

## Key findings

- The **HC anchor mechanism** is the single most impactful design decision: without it, AstroSage-8B over-predicts G-type stars due to its training distribution. The anchor raises accuracy from ~0.58 (V1, no anchor) to **0.7951** (V7).
- **Near-miss rate of 99.8%** — when the model is wrong, it is almost always off by only one MK step (e.g., K→G, A→F), never catastrophically wrong.
- **RAG impact is class-dependent**: retrieval helps F-type stars (hit rate 75%) but barely affects G, K, or M types, which the model handles confidently from HC features alone.
- **B↔A boundary** is the hardest classification frontier (Teff ≈ 10 000 K), accounting for the majority of hard errors.

---

## Authors

**César Miguel Aguirre Calzadilla**
Internship project · CIMAT / Instituto de Astronomía UNAM · 2026

| | |
|---|---|
| Advisor (CIMAT) | Víctor Muñiz Sánchez |
| Advisor (UNAM) | José Antonio de Diego Onsurbe |
