# Diabetes Risk Prediction

Screening models for diabetes risk, built twice on two verified public datasets: a small clinical
cohort (Pima, 768 patients) and a large population survey (CDC BRFSS 2015, 253,680 adults). The focus
is on **honest evaluation**: leak-free pipelines, repeated and nested cross-validation, a locked test
set, calibrated probabilities, a pre-registered screening threshold and subgroup checks.

## Results

| | Pima (clinical) | CDC BRFSS (survey) |
|---|---|---|
| Final model | Logistic regression + age/BMI splines | Tuned LightGBM |
| ROC-AUC | **0.849 ± 0.026** (50-fold CV) | **0.832** (test set, 95% CI 0.827–0.836) |
| PR-AUC | 0.732 (base rate 0.35) | 0.434 (base rate 0.14) |
| Brier score | 0.152 | 0.0965 |
| Calibration error (ECE) | 0.035 | 0.0025 |
| Recall at screening threshold | 84% on unseen folds | 85.2% on the test set |
| Precision at that threshold | 58% | 28% |

### How much the rebuild improved the original notebook (Pima)
The original notebook reported **0.828 AUC** from one random split. Re-tested without leakage across 50
folds, that approach scores **0.817** (single splits range from 0.751 to 0.870).

| Change | ROC-AUC | Gain |
|---|---|---|
| Original model, re-tested leak-free | 0.817 | – |
| Regularised logistic regression | 0.829 | +0.012 |
| Zeros treated as missing, imputed inside folds | 0.835 | +0.006 |
| Cubic splines for age and BMI | 0.846 | +0.011 |
| Regularisation tuned with nested CV | **0.849** | +0.003 |

Total **+0.032 AUC** (p = 0.001, corrected resampled t-test), PR-AUC +0.050, and calibration error down
from 0.117 to 0.035. Tested and rejected, with numbers: missing-value indicators, clinical threshold
flags, SMOTE, class weights, SVM, k-NN, random forest, XGBoost, LightGBM, soft voting and stacking.

Full write-up of every step, why it was used and what it changed: **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**.

<p float="left">
<img src="reports/figures/pima_ablation.png" width="49%"/>
<img src="reports/figures/cdc_subgroups.png" width="49%"/>
</p>

## Data
Both datasets are downloaded from their official sources and verified by checksum before use. Details,
provenance, quality checks and known issues are in **[docs/DATA_CARD.md](docs/DATA_CARD.md)**.

| Dataset | Source | Rows | Quality checks |
|---|---|---|---|
| Pima Indians Diabetes | OpenML 37 (originally NIDDK via UCI) | 768 | 10/10 passed; 652 impossible zeros recoded as missing |
| CDC Diabetes Health Indicators | UCI 891 (BRFSS 2015) | 253,680 | 29/29 passed; 9.5% duplicate rows grouped across the split |

Intended use, limitations and subgroup risks: **[docs/MODEL_CARD.md](docs/MODEL_CARD.md)**.

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
make setup     # pinned versions from requirements.txt
make data      # download, verify checksums, run data-quality checks
make pima      # ~20 min on 4 cores
make cdc       # ~25 min on 4 cores
make test      # leakage, validation and metric unit tests
```
Outputs: tables in `reports/pima/`, `reports/cdc/`, `reports/data_quality/`; charts in `reports/figures/`.
All randomness uses seed 42.

## Project structure
```
src/diabetes_risk/
  config.py        paths, seeds, dataset checksums, codebook ranges, screening target
  data.py          checksum-verified loading; duplicate grouping
  validation.py    data-quality checks (blocking and informational)
  features.py      feature engineering as scikit-learn transformers
  evaluation.py    metrics, calibration error, corrected t-test, threshold selection, bootstrap CIs
  plots.py         shared chart style
scripts/
  01_validate_data.py      step 1: data quality
  02_pima_experiments.py   step 2: Pima ablation, models, imbalance, calibration, threshold, explainability
  03_cdc_experiments.py    step 3: CDC tuning, ablation, locked test, subgroups, SHAP
tests/test_project.py      9 tests: leakage, duplicates, validation, metrics
docs/                      data card, methodology, model card
reports/                   generated tables and figures
```

## Limitations
Self-reported survey labels, a single historical clinical cohort, and weaker performance for some
subgroups (see the model card). These models are for research and screening prioritisation only,
never for diagnosis.
