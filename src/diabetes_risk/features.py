"""Feature engineering as scikit-learn transformers, so it is fitted inside each CV fold."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PimaClinicalFeatures(BaseEstimator, TransformerMixin):
    """Clinically motivated features for the Pima data.

    Thresholds follow ADA / WHO definitions: 2-hour OGTT glucose ≥ 140 mg/dL (impaired tolerance),
    BMI ≥ 30 (obesity), diastolic BP ≥ 80 mmHg (stage 1 hypertension, ACC/AHA 2017).
    HOMA-like insulin resistance uses glucose × insulin, which is only defined when insulin was measured.
    """

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        X["Glucose_x_BMI"] = X["Glucose"] * X["BMI"] / 100
        X["Glucose_x_Age"] = X["Glucose"] * X["Age"] / 100
        X["Insulin_resistance"] = X["Glucose"] * X["Insulin"] / 405  # HOMA-IR style index
        X["Impaired_glucose"] = (X["Glucose"] >= 140).astype(float)
        X["Obese"] = (X["BMI"] >= 30).astype(float)
        X["High_diastolic_BP"] = (X["BloodPressure"] >= 80).astype(float)
        X["Log_pedigree"] = np.log(X["DiabetesPedigreeFunction"])
        # keep NaN where an input was missing so the imputer (and its indicator) sees it
        for col, src in [("Impaired_glucose", "Glucose"), ("Obese", "BMI"), ("High_diastolic_BP", "BloodPressure")]:
            X.loc[X[src].isna(), col] = np.nan
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.feature_names_in_) + [
            "Glucose_x_BMI", "Glucose_x_Age", "Insulin_resistance", "Impaired_glucose",
            "Obese", "High_diastolic_BP", "Log_pedigree"])


class NotebookFeatures(BaseEstimator, TransformerMixin):
    """The 10 features from the original notebook, kept to reproduce and compare it."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        X["Glucose_BMI"] = X.Glucose * X.BMI / 100
        X["Age_Glucose"] = X.Age * X.Glucose / 100
        X["BP_BMI"] = X.BloodPressure * X.BMI / 100
        X["Metabolic_Risk"] = X.Glucose * .4 + X.BMI * .3 + X.Age * .2 + X.BloodPressure * .1
        X["Glucose_squared"] = X.Glucose ** 2
        X["BMI_squared"] = X.BMI ** 2
        X["Age_squared"] = X.Age ** 2
        X["High_Glucose"] = (X.Glucose > 140).astype(float)
        X["Obese"] = (X.BMI > 30).astype(float)
        X["High_BP"] = (X.BloodPressure > 80).astype(float)
        return X


class CdcFeatures(BaseEstimator, TransformerMixin):
    """Count-style health burden features for the CDC survey data."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        X["BMI"] = X["BMI"].clip(15, 70)
        X["Cardiometabolic_count"] = X[["HighBP", "HighChol", "Stroke", "HeartDiseaseorAttack"]].sum(axis=1)
        X["Unhealthy_days"] = (X["MentHlth"] + X["PhysHlth"]).clip(upper=30)
        X["Healthy_habits"] = X["PhysActivity"] + X["Fruits"] + X["Veggies"] - X["Smoker"] - X["HvyAlcoholConsump"]
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.feature_names_in_) + ["Cardiometabolic_count", "Unhealthy_days", "Healthy_habits"])
