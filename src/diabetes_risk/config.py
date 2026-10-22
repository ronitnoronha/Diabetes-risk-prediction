"""Paths, seeds and dataset provenance for the whole project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"

SEED = 42

# Every raw file is pinned to a checksum so a silently changed download is caught.
DATASETS = {
    "pima": {
        "file": "pima_openml37.arff",
        "url": "https://www.openml.org/data/download/37/dataset_37_diabetes.arff",
        "md5": "3cbaa3e54586aa88cf6aacb4033e4470",  # published by OpenML for data id 37
        "sha256": "4eddd5b2b64679e8888348e306520a393d6a28e1ddc9643cfb76fc5d912d6d40",
    },
    "cdc": {
        "file": "cdc_diabetes_health_indicators_uci891.csv",
        "url": "https://archive.ics.uci.edu/static/public/891/data.csv",
        "sha256": "9f71fda9d4ae5f4878c99b9233b6a16accfa9a17c194116a6b78100540934964",
    },
}

# Pima: these measurements cannot be 0 in a living person, so 0 means "not recorded".
PIMA_ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
PIMA_FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                 "BMI", "DiabetesPedigreeFunction", "Age"]

# CDC: allowed values from the BRFSS 2015 codebook (as published with UCI dataset 891).
CDC_TARGET = "Diabetes_binary"
CDC_BINARY = ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
              "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
              "NoDocbcCost", "DiffWalk", "Sex"]
CDC_RANGES = {"BMI": (12, 98), "GenHlth": (1, 5), "MentHlth": (0, 30), "PhysHlth": (0, 30),
              "Age": (1, 13), "Education": (1, 6), "Income": (1, 8)}
CDC_ORDINAL = ["GenHlth", "Age", "Education", "Income"]
CDC_FEATURES = CDC_BINARY[:-1] + ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Sex",
                                   "Age", "Education", "Income"]

# Screening goal fixed before modelling: catch at least 85% of people with diabetes.
TARGET_RECALL = 0.85
