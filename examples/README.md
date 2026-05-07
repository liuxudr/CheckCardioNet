# Examples

Synthetic inputs for trying CheckCardioNet end-to-end. None of these files contain real patient data.

## Files

| File | Description |
|---|---|
| `expression.csv` | A 10-gene checkpoint expression profile (z-score) for one patient. Two-column `gene,value` format accepted by `--expr-file`. |
| `cohort.csv`     | A 5-patient cohort table. Columns: `patient_id`, `cancer_type`, `cvd_susceptibility`, plus checkpoint expression columns. |

## Try it

```bash
# 1. List supported ICI drugs
checkcardionet list-drugs

# 2. Single patient — compare all candidate ICIs (uses inline expression)
checkcardionet score-patient \
    --cancer-type SKCM \
    --cvd 0.75 \
    --expr "PDCD1=2.5,CTLA4=1.5,CD47=0.8"

# 3. Single patient — read expression profile from file
checkcardionet score-patient \
    --cancer-type NSCLC --cvd 0.6 \
    --expr-file examples/expression.csv \
    --output-json patient.json

# 4. Single patient — score one specific drug
checkcardionet score-patient \
    --cancer-type AML --cvd 0.7 \
    --drug magrolimab \
    --expr-file examples/expression.csv

# 5. Cohort batch prediction
checkcardionet score-cohort examples/cohort.csv -o predictions.csv
```

## What you should see

- `score-patient` (multi-drug) prints a Chinese textual recommendation plus a ranked drug table (drug × onco_benefit × CVD accel × net benefit × category).
- `score-patient --drug X` prints a per-target decomposition (expression weight × MR β × BDS × component score).
- `score-cohort` produces a long CSV with one row per (patient × drug); the patient's primary recommendation is flagged with ✓.
