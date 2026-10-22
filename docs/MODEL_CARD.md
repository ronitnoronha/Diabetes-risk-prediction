# Model card

## Intended use
* **Purpose:** rank people by their risk of diabetes so the highest-risk can be offered a confirmatory
  blood test (HbA1c or fasting glucose).
* **Not for:** diagnosing diabetes, setting insurance prices, or any decision without a confirmatory test.
* **Users:** analysts and researchers. This is a portfolio project; neither model has been clinically validated.

## Model 1: Pima clinical model

| | |
|---|---|
| Algorithm | L2-regularised logistic regression; cubic splines (4 knots) for Age and BMI; median imputation of impossible zeros; standardised inputs; regularisation chosen by nested CV |
| Inputs | Pregnancies, OGTT glucose, diastolic BP, skin-fold, insulin, BMI, pedigree function, age |
| Training data | 768 women of Pima heritage (NIDDK, OpenML 37) |
| Performance | ROC-AUC **0.849 ± 0.026**, PR-AUC 0.732, Brier 0.152, ECE 0.035 (50-fold repeated CV) |
| Screening threshold | ≈ 0.26 → recall 84%, precision 58%, specificity 68% on unseen folds |
| Main drivers | Glucose (odds ratio 3.1 per SD), BMI and age (non-linear), family-history score |

## Model 2: CDC survey model

| | |
|---|---|
| Algorithm | LightGBM gradient-boosted trees (584 trees, 26 leaves, learning rate 0.021, min 244 samples per leaf, L2 1.12, row 0.81 / column 0.53 subsampling) |
| Inputs | 21 BRFSS survey answers + 3 engineered counts; BMI clipped to [15, 70] |
| Training data | 202,944 BRFSS 2015 respondents (UCI 891); duplicates grouped away from the test set |
| Performance (locked test set, 50,736 people) | ROC-AUC **0.832** (95% CI 0.827–0.836), PR-AUC 0.434, Brier 0.0965, ECE 0.0025 |
| Screening threshold | 0.110 → recall 85.2%, precision 28.3%, specificity 65.1%; flags 419 per 1,000 |
| Main drivers | Self-rated general health, BMI, age, number of cardiometabolic conditions, high blood pressure |

## Limitations and risks
* **Population shift.** Pima: one community of women. CDC: US adults who answered every question in
  2015. Neither should be used elsewhere without re-validation.
* **Label quality.** CDC labels are self-reported and merge prediabetes with diabetes; undiagnosed
  diabetes counts as negative.
* **Subgroup gaps.** On the CDC model, a single threshold catches only 50% of cases aged 18–44 and the
  ranking is weaker for adults 65+ (AUC 0.77) and incomes under $20k (AUC 0.78). Use group-specific
  thresholds and monitor these groups.
* **Proxies.** Features such as "had a cholesterol check" reflect access to healthcare, not biology.
  The model can under-rate people with poor access to care.
* **Not causal.** Coefficients and SHAP values describe associations in the training data.

## Maintenance
Re-check calibration and subgroup recall before any reuse; refit on newer BRFSS years, which use the
same questions, to measure drift.
