"""Data-quality checks. Each check returns a row for the report; hard failures raise."""
import numpy as np
import pandas as pd

from .config import CDC_BINARY, CDC_RANGES, CDC_TARGET, PIMA_ZERO_AS_MISSING


class DataQualityError(ValueError):
    pass


def _check(rows, name, passed, detail, hard=True):
    rows.append({"check": name, "passed": bool(passed), "detail": detail, "blocking": hard})
    if hard and not passed:
        raise DataQualityError(f"{name}: {detail}")


def validate_pima(X_raw, y):
    rows = []
    _check(rows, "Row count matches source (768)", len(X_raw) == 768, f"{len(X_raw)} rows")
    _check(rows, "No null cells", not X_raw.isna().any().any(), f"{int(X_raw.isna().sum().sum())} nulls")
    _check(rows, "Target is binary", set(np.unique(y)) <= {0, 1},
           f"positives {int(y.sum())} ({y.mean():.1%})")
    _check(rows, "No negative values", (X_raw >= 0).all().all(), "all values ≥ 0")
    dups = int(X_raw.assign(y=y).duplicated().sum())
    _check(rows, "No duplicate rows", dups == 0, f"{dups} duplicates")
    for col in PIMA_ZERO_AS_MISSING:
        n = int((X_raw[col] == 0).sum())
        _check(rows, f"{col}: zeros are missing values", True,
               f"{n} zeros ({n / len(X_raw):.1%}) recoded to missing", hard=False)
    return pd.DataFrame(rows)


def validate_cdc(df):
    rows = []
    _check(rows, "Row count matches source (253,680)", len(df) == 253_680, f"{len(df):,} rows")
    _check(rows, "Column count (ID + target + 21 features)", df.shape[1] == 23, f"{df.shape[1]} columns")
    _check(rows, "No null cells", not df.isna().any().any(), f"{int(df.isna().sum().sum())} nulls")
    _check(rows, "ID is unique", df["ID"].is_unique, "unique")
    _check(rows, "Target is binary", set(df[CDC_TARGET].unique()) <= {0, 1},
           f"positives {int(df[CDC_TARGET].sum()):,} ({df[CDC_TARGET].mean():.1%})")
    for col in CDC_BINARY:
        _check(rows, f"{col} in {{0,1}}", df[col].isin([0, 1]).all(), "codebook values only")
    for col, (lo, hi) in CDC_RANGES.items():
        ok = df[col].between(lo, hi).all() and (df[col] % 1 == 0).all()
        _check(rows, f"{col} integer in [{lo}, {hi}]", ok, f"observed {df[col].min()}–{df[col].max()}")
    feats = df.drop(columns=["ID"])
    dups = int(feats.duplicated().sum())
    _check(rows, "Duplicate rows (excluding ID)", True,
           f"{dups:,} ({dups / len(df):.1%}); kept, but grouped so copies never cross train/test",
           hard=False)
    fx = feats.drop(columns=[CDC_TARGET])
    g = feats.groupby(list(fx.columns))[CDC_TARGET].nunique()
    _check(rows, "Identical answers with different labels", True,
           f"{int((g > 1).sum()):,} answer patterns appear with both labels (irreducible noise)", hard=False)
    extreme = int(((df["BMI"] > 70) | (df["BMI"] < 15)).sum())
    _check(rows, "Implausible BMI (<15 or >70)", True,
           f"{extreme} rows ({extreme / len(df):.2%}); clipped to [15, 70] for linear models", hard=False)
    return pd.DataFrame(rows)
