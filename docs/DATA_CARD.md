# Data card

Two public datasets are used. They answer different questions and have different features, so they are
modelled separately and never merged.

| | Pima Indians Diabetes | CDC Diabetes Health Indicators |
|---|---|---|
| Question | Will this patient develop diabetes, given clinical measurements? | Does this adult report diabetes or prediabetes, given survey answers? |
| Rows | 768 women | 253,680 adults |
| Features | 8 clinical measurements | 21 survey answers |
| Positive class | 268 (34.9%) | 35,346 (13.9%) |
| Source used | OpenML, data id 37, version 1 | UCI Machine Learning Repository, id 891 |
| Original collector | US National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) | US Centers for Disease Control and Prevention (CDC), BRFSS 2015 |
| Licence | Listed as "Public" on OpenML | Underlying BRFSS data is US federal public data. UCI distributes its datasets under CC BY 4.0; confirm on the dataset page before reuse |
| Integrity check | MD5 matches the checksum published by OpenML; SHA-256 pinned | SHA-256 pinned in `config.py` |

Both raw files are downloaded by `scripts/01_validate_data.py`, which refuses to continue if a checksum
does not match. The raw files are not committed to git.

---

## 1. Pima Indians Diabetes

**Provenance.** Collected by NIDDK among women of Pima heritage living near Phoenix, Arizona, and
donated to the UCI repository by Vincent Sigillito (Johns Hopkins APL) in 1990. First used in
Smith et al. (1988), *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus*,
Proceedings of the Symposium on Computer Applications and Medical Care, 261–265. UCI has since retired
its copy; OpenML (data id 37) hosts the same 768 rows with a published checksum.

**Population.** Women aged 21 or older of Pima heritage. The outcome is a diabetes diagnosis by WHO
criteria (2-hour post-load plasma glucose ≥ 200 mg/dL) at a later examination.

| Column | Meaning | Unit |
|---|---|---|
| Pregnancies | Number of times pregnant | count |
| Glucose | Plasma glucose 2 hours into an oral glucose tolerance test | mg/dL |
| BloodPressure | Diastolic blood pressure | mmHg |
| SkinThickness | Triceps skin-fold thickness | mm |
| Insulin | 2-hour serum insulin | μU/mL |
| BMI | Body mass index | kg/m² |
| DiabetesPedigreeFunction | Family-history score from the original authors | score |
| Age | Age | years |
| Outcome | 1 = diabetes | – |

**Quality checks** (`reports/data_quality/pima_checks.csv`): 10 of 10 passed.

| Finding | Count | How it is handled |
|---|---|---|
| Glucose recorded as 0 | 5 (0.7%) | Treated as missing, imputed with the training-fold median |
| Blood pressure recorded as 0 | 35 (4.6%) | Same |
| Skin thickness recorded as 0 | 227 (29.6%) | Same |
| Insulin recorded as 0 | 374 (48.7%) | Same |
| BMI recorded as 0 | 11 (1.4%) | Same |
| Duplicate rows | 0 | – |
| Null cells | 0 | The file stores missing values as 0, not as blanks |

A zero is impossible for these five measurements in a living person, so a zero means "not recorded".
Imputation happens inside each cross-validation fold so the test rows never influence the fill values.

**Limitations.** 768 rows is small: a single 80/20 split gives a test set of 154 patients, and the
test score moves by ±0.06 AUC depending on which patients land in it. The cohort is one community,
all women, measured decades ago, so results do not transfer to other populations without
re-validation. The data comes from an Indigenous community; use it for method development, not for
claims about that community.

---

## 2. CDC Diabetes Health Indicators (BRFSS 2015)

**Provenance.** The Behavioral Risk Factor Surveillance System (BRFSS) is the CDC's annual telephone
health survey. Alex Teboul cleaned the 2015 survey into a 21-feature, 253,680-row file published on
Kaggle; UCI republished it as dataset 891 (DOI 10.24432/C53919). This project downloads it from UCI.

**Metadata inconsistencies found on the UCI page** (documented so nobody is misled):
* The abstract mentions 35 features and a three-class target. The file has 21 features and a binary
  target (`Diabetes_binary`: 0 = no diabetes, 1 = prediabetes or diabetes).
* The summary links to the BRFSS **2014** annual data page. The file matches the Kaggle
  **2015** release (same 253,680 rows and columns).

**Population.** US adults aged 18+ who answered the 2015 BRFSS. The curator removed respondents with
missing or "don't know / refused" answers, so the file has no missing values but is not a random
sample of all respondents.

| Column | Meaning | Values |
|---|---|---|
| Diabetes_binary | Ever told you have diabetes or prediabetes | 0 / 1 |
| HighBP, HighChol | Ever told you have high blood pressure / high cholesterol | 0 / 1 |
| CholCheck | Cholesterol checked in the past 5 years | 0 / 1 |
| BMI | Body mass index | 12–98 |
| Smoker | Smoked at least 100 cigarettes in your life | 0 / 1 |
| Stroke, HeartDiseaseorAttack | Ever had a stroke / coronary heart disease or heart attack | 0 / 1 |
| PhysActivity | Physical activity in the past 30 days, outside work | 0 / 1 |
| Fruits, Veggies | Fruit / vegetables at least once a day | 0 / 1 |
| HvyAlcoholConsump | Heavy drinking (men > 14, women > 7 drinks a week) | 0 / 1 |
| AnyHealthcare, NoDocbcCost | Has health coverage / skipped a doctor because of cost | 0 / 1 |
| GenHlth | Self-rated general health | 1 excellent – 5 poor |
| MentHlth, PhysHlth | Days of poor mental / physical health in the past 30 | 0–30 |
| DiffWalk | Serious difficulty walking or climbing stairs | 0 / 1 |
| Sex | 0 = female, 1 = male | 0 / 1 |
| Age | 13 five-year bands | 1 = 18–24 … 13 = 80+ |
| Education | 1 = never attended school … 6 = college graduate | 1–6 |
| Income | 1 = < $10k … 8 = ≥ $75k | 1–8 |

**Quality checks** (`reports/data_quality/cdc_checks.csv`): 29 of 29 passed. Every column holds only
values allowed by the BRFSS codebook; there are no nulls; IDs are unique; row and column counts match
the source.

| Finding | Count | How it is handled |
|---|---|---|
| Exact duplicate rows (all 21 answers + label) | 24,206 (9.5%) | Kept. With mostly yes/no questions, different people can give identical answers, so duplicates are plausible. Each set of identical rows is placed entirely in train or entirely in test (grouped split), so the test score is not inflated by memorised copies |
| Identical answers with different labels | 1,566 patterns | Kept. This is irreducible noise: the survey cannot tell these people apart, which caps achievable accuracy |
| Implausible BMI (< 15 or > 70) | 652 (0.26%) | Clipped to [15, 70] in the feature step |

**Limitations.**
* Everything is **self-reported**, including the diagnosis. Undiagnosed diabetes is counted as
  "no diabetes", so the model learns who has been *told* they have diabetes.
* The positive class merges **prediabetes and diabetes**.
* The survey is cross-sectional: answers such as "difficulty walking" may be consequences of
  diabetes, not causes. The model is for **screening** (who should get a blood test), not diagnosis or
  causal claims.
* BRFSS survey weights are not included in the file, so results describe the respondents, not the
  US population.
* Sex, income and education are sensitive attributes. Performance is reported by subgroup in
  `reports/cdc/subgroups.csv`.
