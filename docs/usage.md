# CheckCardioNet — Detailed Usage

CheckCardioNet is a **prediction tool**. Given a patient's tumor checkpoint expression profile, cancer type, and baseline CVD susceptibility, it estimates the cardiovascular acceleration risk of each candidate ICI drug and produces an individualized treatment recommendation.

The model artifacts (Mendelian-randomization causal effects, bidirectionality scores, dual-benefit drug atlas) are pre-computed and **bundled inside the package** — no upstream pipeline or external download is needed.

---

## 1. Installation

CheckCardioNet is distributed via GitHub.

```bash
pip install git+https://github.com/liuxudr/CheckCardioNet.git
```

Or for source:

```bash
git clone https://github.com/liuxudr/CheckCardioNet.git
cd CheckCardioNet
pip install .
```

Verify:

```bash
checkcardionet --help
checkcardionet list-drugs
```

---

## 2. Inputs

### 2.1 Checkpoint expression profile

Z-scored expression values (range roughly −3..+3) for the immune-checkpoint genes you have data for. Genes you don't have can be omitted — the scorer falls back to drug-level priors for those targets.

Two ways to supply the profile:

**Inline (good for ≤ 10 genes):**

```bash
--expr "PDCD1=2.5,CD274=1.8,CD47=0.5,CTLA4=0.9"
```

**From file (recommended for ≥ 10 genes):**

```bash
--expr-file expression.csv
```

`expression.csv` accepted shapes:

```csv
gene,value             # OR  gene,expression  OR  gene,expr
PDCD1,2.5
CD274,1.8
```

If both `--expr` and `--expr-file` are given, the file is loaded first and inline values overwrite per gene.

### 2.2 Cancer type

Use TCGA / common oncology codes:

| Code | Cancer |
|---|---|
| NSCLC, LUAD, LUSC | Non-small-cell lung cancer |
| SKCM      | Melanoma |
| BLCA      | Bladder |
| KIRC      | Renal cell |
| HCC, LIHC | Liver |
| HNSC      | Head & neck |
| AML       | Acute myeloid leukemia |
| MDS       | Myelodysplastic syndrome |

Any code not in the prior table falls back to `default` ORR.

### 2.3 CVD susceptibility

Single number ∈ [0, 1]:

| Range | Meaning |
|---|---|
| 0.0–0.3 | Low — no major CVD history, normal lipids/BP |
| 0.3–0.6 | Moderate — controlled hypertension or dyslipidemia |
| 0.6–1.0 | High — prior MI / stroke / heart failure / uncontrolled risk factors |

This is a **clinical input**; the tool does not infer it from the expression profile.

---

## 3. Commands

### 3.1 `list-drugs`

```bash
checkcardionet list-drugs
```

Prints supported ICI drugs, their molecular targets, and a literature-derived CVD-risk prior. Drugs with negative prior (e.g. magrolimab) are predicted CVD-protective.

### 3.2 `score-patient` (single drug)

```bash
checkcardionet score-patient \
    --cancer-type SKCM \
    --cvd 0.6 \
    --drug pembrolizumab \
    --expr-file expr.csv \
    --output-json result.json
```

Output: a panel with the drug's accel score and a per-target decomposition.

### 3.3 `score-patient` (compare all candidates)

Omit `--drug` to compare every supported ICI:

```bash
checkcardionet score-patient \
    --cancer-type SKCM \
    --cvd 0.75 \
    --expr "PDCD1=2.5,CTLA4=1.5"
```

Or restrict to a subset:

```bash
--candidates "pembrolizumab,nivolumab,magrolimab"
```

Output: textual recommendation (primary + cardiac-protection suggestions + dual-benefit alternatives) plus a ranked drug table.

### 3.4 `score-cohort`

```bash
checkcardionet score-cohort cohort.csv -o predictions.csv
```

Required `cohort.csv` columns: `patient_id`, `cancer_type`, `cvd_susceptibility`. Any additional column whose name matches a known checkpoint gene (e.g. `PDCD1`, `CD274`, `CD47`) is automatically used as the patient's expression value.

Output: long-form CSV — one row per (patient × candidate drug) with `onco_benefit`, `cvd_accel_score`, `net_benefit`, `category`, and a ✓ in `primary_recommendation` for each patient's top pick.

---

## 4. Scoring formula

```
For each target gene g of drug d:

  expr_weight       = clip((z_g + 3) / 6, 0, 1)
  MR_beta_g         = pretrained median IVW β (gene → CVD); 0 if absent
  BDS_g             = pretrained bidirectionality score
  component_g       = expr_weight × |MR_beta_g|
                       × (1 + 0.3 × |BDS_g|)
                       × cvd_susceptibility
  if MR_beta_g < 0:  component_g = -component_g     # CVD-protective target

  accel_score(d)    = sum_g component_g, clipped to [-1, +1]
```

```
For each candidate drug d (recommendation pass):

  onco_benefit(d, cancer)  = pretrained ORR prior
  net_benefit(d)           = onco_benefit - 0.5 × max(0, accel_score(d))
  category(d)              = lookup in dual-benefit atlas;
                             else qualitative bin from (onco, accel)
  recommendation_priority  = onco_benefit - 0.5 × max(0, accel_score(d))

Primary recommendation = drug with highest recommendation_priority.
```

---

## 5. Bundled pre-trained artifacts

| File | Content | Source |
|---|---|---|
| `data/pretrained/mr_results.parquet`         | gene → CVD MR causal effects (IVW β) | Two-sample MR on GTEx eQTL × CVD GWAS |
| `data/pretrained/bidirectional_scores.parquet` | per-gene BDS across tumor + CVD | Cross-disease co-expression network analysis |
| `data/pretrained/dual_benefit_atlas.parquet` | drug → (onco signal, CVD signal, tier) | CMap signature reversal + curated cardio-oncology drug list |

These artifacts are frozen results from a published cross-disease pipeline. The pipeline itself is not shipped in this prediction tool.

---

## 6. Limitations

- Research use only. **Not a medical device.** See `LICENSE`.
- Local cohort recalibration is recommended for clinical decision support.
- Drug not in `list-drugs` ⇒ not scored. Extend `ICI_CVD_PRIOR` in source to add new agents.
- Without an expression profile, scoring relies on drug-level CVD prior alone — directional but coarse.
