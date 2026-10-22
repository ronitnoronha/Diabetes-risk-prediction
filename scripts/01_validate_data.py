"""Step 1: download, verify checksums and run data-quality checks on both datasets."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diabetes_risk.config import REPORTS
from diabetes_risk.data import load_cdc, load_pima
from diabetes_risk.validation import validate_cdc, validate_pima


def main():
    out = REPORTS / "data_quality"
    out.mkdir(parents=True, exist_ok=True)

    X_raw, y = load_pima(raw=True)
    pima = validate_pima(X_raw, y)
    pima.to_csv(out / "pima_checks.csv", index=False)
    pima_profile = X_raw.replace(0, float("nan")).describe().T.round(2)
    pima_profile.to_csv(out / "pima_profile.csv")

    _, _, df = load_cdc()
    cdc = validate_cdc(df)
    cdc.to_csv(out / "cdc_checks.csv", index=False)
    df.drop(columns=["ID"]).describe().T.round(3).to_csv(out / "cdc_profile.csv")
    prevalence = df.groupby("Age")["Diabetes_binary"].mean().round(3)

    summary = {
        "pima": {"rows": len(X_raw), "positives": int(y.sum()), "prevalence": round(float(y.mean()), 4),
                 "checks_passed": int(pima.passed.sum()), "checks_total": len(pima)},
        "cdc": {"rows": len(df), "positives": int(df.Diabetes_binary.sum()),
                "prevalence": round(float(df.Diabetes_binary.mean()), 4),
                "checks_passed": int(cdc.passed.sum()), "checks_total": len(cdc),
                "prevalence_by_age_band": prevalence.to_dict()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for name, table in [("Pima", pima), ("CDC", cdc)]:
        print(f"\n{name}: {int(table.passed.sum())}/{len(table)} checks passed")
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
