# CheckCardioNet

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CLI](https://img.shields.io/badge/interface-CLI-green.svg)](#usage)
[![Status](https://img.shields.io/badge/status-beta-orange.svg)](#)

> **Predict ICI-driven cardiovascular acceleration risk and generate individualized ICI recommendations from a tumor checkpoint expression profile.**

CheckCardioNet is a command-line prediction tool for cardio-oncology. Given a patient's tumor checkpoint expression profile (z-score), cancer type, and baseline cardiovascular susceptibility, it returns:

- A **CVD-acceleration score** for any candidate ICI drug (PD-1, PD-L1, CTLA-4, LAG-3, TIGIT, TIM-3, CD47, …)
- An **individualized ICI recommendation** — primary drug + ranked alternatives + dual-benefit candidates
- A per-target decomposition (expression weight × MR causal beta × bidirectionality)
- A textual recommendation including cardiac-protection suggestions

The tool ships with **pre-trained artifacts** (Mendelian-randomization causal effects, bidirectional scores, dual-benefit drug atlas) bundled inside the package — **no external data download is required**.

---

## Installation

CheckCardioNet is distributed via GitHub. Install directly from the repository:

```bash
pip install git+https://github.com/liuxudr/CheckCardioNet.git
```

Or clone first for source access:

```bash
git clone https://github.com/liuxudr/CheckCardioNet.git
cd CheckCardioNet
pip install .
```

Verify install:

```bash
checkcardionet --help
checkcardionet list-drugs
```

---

## Quick start

### 1. List supported ICI drugs

```bash
checkcardionet list-drugs
```

### 2. Single patient — single drug

```bash
checkcardionet score-patient \
    --cancer-type NSCLC \
    --cvd 0.6 \
    --drug pembrolizumab \
    --expr "PDCD1=2.5,CD274=1.8,CD47=0.5"
```

### 3. Single patient — compare all candidate ICIs

```bash
checkcardionet score-patient \
    --cancer-type SKCM \
    --cvd 0.75 \
    --expr "PDCD1=2.5,CTLA4=1.5,CD47=0.8" \
    --output-json results/patient.json
```

### 4. Expression profile from file

```bash
checkcardionet score-patient \
    --cancer-type AML \
    --cvd 0.7 \
    --expr-file expression.csv \
    --candidates "magrolimab,nivolumab,pembrolizumab"
```

`expression.csv` minimal format:

```csv
gene,value
CD47,3.0
SIRPA,1.5
MERTK,1.0
PDCD1,1.2
```

### 5. Cohort batch prediction

```bash
checkcardionet score-cohort cohort.csv -o predictions.csv
```

`cohort.csv` schema (one row per patient):

```csv
patient_id,cancer_type,cvd_susceptibility,PDCD1,CD274,CD47,CTLA4,SIRPA
P001,NSCLC,0.20,2.0,1.5,0.5,0.8,0.4
P002,SKCM,0.75,2.5,2.0,1.0,1.2,0.7
P003,AML,0.70,0.5,0.4,3.0,0.3,1.5
```

Any column whose name matches a known checkpoint gene is automatically used as expression input. Required columns: `patient_id`, `cancer_type`, `cvd_susceptibility`.

---

## CLI reference

| Command | Purpose |
|---|---|
| `checkcardionet list-drugs`     | Print supported ICI drugs, targets, CVD-risk priors |
| `checkcardionet score-patient`  | Predict for one patient (single drug or all candidates) |
| `checkcardionet score-cohort`   | Batch CSV input → ranked drug × patient predictions |
| `checkcardionet --help`         | Top-level help |
| `checkcardionet <cmd> --help`   | Per-command help |

### `score-patient` flags

| Flag | Type | Notes |
|---|---|---|
| `--cancer-type, -c`        | str   | TCGA code (NSCLC, SKCM, BLCA, KIRC, HCC, AML, MDS, …) |
| `--cvd-susceptibility, --cvd` | float | 0–1 baseline CVD risk |
| `--expr, -e`               | str   | Inline `'GENE=val,GENE=val,...'` |
| `--expr-file, -f`          | path  | Two-column CSV/TSV (gene, value) |
| `--drug, -d`               | str   | Score a single drug only |
| `--candidates`             | str   | Comma-separated candidate ICIs |
| `--output-json, -o`        | path  | Save JSON result |

### `score-cohort` flags

| Flag | Type | Notes |
|---|---|---|
| `input_csv` (positional)   | path  | Cohort CSV |
| `--output, -o`             | path  | Output predictions CSV (default `predictions.csv`) |
| `--candidates`             | str   | Comma-separated candidate ICIs |

---

## What's inside

```
checkcardionet/
├── cli.py                              CLI entry point
├── configs/
│   └── checkpoint_panel.yaml           52 immune-checkpoint genes (5 categories)
├── data/
│   ├── preprocess.py                   Config / pretrained loaders
│   └── pretrained/                     ← bundled, no download required
│       ├── mr_results.parquet              MR causal effects (gene → CVD beta)
│       ├── bidirectional_scores.parquet    BDS bidirectionality scores
│       └── dual_benefit_atlas.parquet      Dual-benefit drug atlas
└── scoring/
    ├── cvd_acceleration_score.py       ICI-driven CVD acceleration scorer
    └── integrated_recommendation.py    Multi-drug ranking + text recommendation
```

---

## How it works

```
Input              ┌── tumor checkpoint expression profile (z-score)
                   ├── cancer type
                   └── baseline CVD susceptibility (0..1)
                                    │
                                    ▼
                   ┌────────────────────────────────────────┐
                   │  ICIDrivenCVDAccelerationScore         │
                   │                                        │
                   │  per target c of drug d:               │
                   │    expr_weight       (from --expr)     │
                   │  × MR_β               (pretrained)     │
                   │  × bidirectionality   (pretrained)     │
                   │  × cvd_susceptibility (input)          │
                   │                                        │
                   │  Σ  →  accel_score ∈ [-1, +1]          │
                   └──────────────┬─────────────────────────┘
                                  ▼
                   ┌────────────────────────────────────────┐
                   │  IntegratedRecommendationSystem        │
                   │                                        │
                   │  per candidate drug:                   │
                   │    onco_benefit ← cancer-type ORR prior│
                   │    net_benefit  = onco − 0.5·max(0,a)  │
                   │    category     ← dual-benefit atlas   │
                   │                                        │
                   │  Rank → primary + alternates + warnings│
                   └────────────────────────────────────────┘
```

The MR effect sizes, BDS scores, and dual-benefit atlas were derived from a published cross-disease analysis pipeline (TCGA pan-cancer + GTEx eQTL + GWAS Catalog CVD outcomes + CMap drug signatures). Only their **frozen results** are shipped — the upstream pipeline is not required to use the tool.

---

## Output

### Text recommendation

```
== Individualized ICI Recommendation ==
Cancer type: SKCM | Baseline CVD risk: high (0.75)

Primary recommendation: nivolumab
  Tumor benefit (est.):   40%
  CVD acceleration score: +0.218
  Net benefit:            +0.236
  Category:               Standard ICI (low CVD concern)

[!] Cardiac-protection suggestions:
  - Co-administer statin +/- aspirin (plaque stabilization)
  - Baseline echocardiogram, repeat every 3 months
  - If CVD event risk is very high, consider switching to magrolimab (dual-benefit)

Dual-benefit alternatives: magrolimab
```

### JSON output (single patient)

```json
{
  "patient_cancer": "SKCM",
  "cvd_susceptibility": 0.75,
  "primary_recommendation": "nivolumab",
  "ranked_drugs": [
    {"drug": "nivolumab", "onco_benefit": 0.40, "cvd_accel_score": 0.218,
     "net_benefit": 0.236, "category": "Standard ICI (low CVD concern)"},
    ...
  ],
  "recommendation": "..."
}
```

### Cohort CSV

One row per (patient × drug) with `primary_recommendation` flag = ✓ on the patient's top pick.

---

## Limitations

- **Research use only.** Not a medical device. See `LICENSE` for the full disclaimer.
- The CVD-acceleration model is calibrated on cross-sectional pharmacovigilance + MR data; **prospective cohort recalibration is recommended** before any clinical decision support use.
- Drugs not in the supported list (`list-drugs`) are not scored; you can extend `ICI_CVD_PRIOR` in `cvd_acceleration_score.py` to add new agents.
- If the user does not provide an expression profile, the scoring falls back to the drug-level CVD prior — directional, but coarse.

---

## Citing

```bibtex
@software{checkcardionet_2026,
  title  = {CheckCardioNet: Predicting ICI-driven cardiovascular acceleration risk
            and individualizing ICI recommendations},
  author = {{CheckCardioNet Authors}},
  year   = {2026},
  url    = {https://github.com/liuxudr/CheckCardioNet},
  version= {0.1.0}
}
```

A `CITATION.cff` is included.

---

## License

[MIT](LICENSE) © CheckCardioNet Authors
