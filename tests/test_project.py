import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import diabetes_risk  # noqa: F401
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold

from diabetes_risk.data import duplicate_groups, load_cdc, load_pima
from diabetes_risk.evaluation import at_threshold, corrected_ttest, expected_calibration_error, threshold_for_recall
from diabetes_risk.features import CdcFeatures, PimaClinicalFeatures
from diabetes_risk.validation import DataQualityError, validate_cdc, validate_pima


@pytest.fixture(scope="module")
def pima():
    return load_pima()


@pytest.fixture(scope="module")
def cdc():
    return load_cdc()


def test_pima_zeros_become_missing(pima):
    X, y = pima
    assert X["Insulin"].isna().sum() == 374
    assert (X[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] == 0).sum().sum() == 0
    assert y.mean() == pytest.approx(268 / 768)


def test_validation_passes_on_real_data(pima, cdc):
    X_raw, y = load_pima(raw=True)
    assert validate_pima(X_raw, y).passed.all()
    assert validate_cdc(cdc[2]).passed.all()


def test_validation_catches_bad_codes(cdc):
    bad = cdc[2].copy()
    bad.loc[0, "GenHlth"] = 9
    with pytest.raises(DataQualityError):
        validate_cdc(bad)


def test_duplicates_never_cross_the_split(cdc):
    X, y, _ = cdc
    g = duplicate_groups(X, y)
    tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, g))
    assert not set(g[tr]) & set(g[te])
    train_rows = set(map(tuple, X.iloc[tr].assign(y=y[tr]).to_numpy()))
    assert not any(tuple(r) in train_rows for r in X.iloc[te[:2000]].assign(y=y[te[:2000]]).to_numpy())


def test_imputer_learns_only_from_training_rows(pima):
    X, _ = pima
    train, test = X.iloc[:500], X.iloc[500:].copy()
    test["Insulin"] = 10_000.0  # extreme values in the test rows must not move the fitted medians
    imp = SimpleImputer(strategy="median").fit(train)
    assert imp.statistics_[list(X.columns).index("Insulin")] == pytest.approx(train["Insulin"].median())


def test_feature_transformers_add_columns(pima, cdc):
    X, _ = pima
    Z = PimaClinicalFeatures().fit(X).transform(X)
    assert {"Glucose_x_BMI", "Impaired_glucose", "Insulin_resistance"} <= set(Z.columns)
    assert Z["Impaired_glucose"].isna().sum() == X["Glucose"].isna().sum()
    C = CdcFeatures().fit(cdc[0]).transform(cdc[0].head(100))
    assert C["BMI"].between(15, 70).all() and "Cardiometabolic_count" in C


def test_threshold_reaches_target_recall():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.3 + rng.normal(0.35, 0.2, 2000), 0, 1)
    t = threshold_for_recall(y, p, 0.85)
    assert at_threshold(y, p, t)["recall"] >= 0.85


def test_calibration_error_is_zero_for_perfect_probabilities():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=200_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01


def test_corrected_ttest_is_more_conservative_than_naive():
    a = np.array([0.84, 0.85, 0.83, 0.86, 0.84] * 10)
    b = a - 0.005 + np.tile([0.002, -0.002, 0.001, -0.001, 0.0], 10)
    _, _, p = corrected_ttest(a, b, 614, 154)
    from scipy import stats
    assert p > stats.ttest_rel(a, b).pvalue
