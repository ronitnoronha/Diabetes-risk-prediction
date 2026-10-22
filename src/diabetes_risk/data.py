"""Download, verify and load the two datasets."""
import hashlib
import urllib.request

import numpy as np
import pandas as pd
from scipy.io import arff

from .config import CDC_FEATURES, CDC_TARGET, DATA_RAW, DATASETS, PIMA_FEATURES


def _digest(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name):
    """Download a raw file if it is missing, then verify its checksum."""
    spec = DATASETS[name]
    path = DATA_RAW / spec["file"]
    if not path.exists():
        DATA_RAW.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(spec["url"], path)
    actual = _digest(path, "sha256")
    if actual != spec["sha256"]:
        raise ValueError(f"{path.name}: sha256 {actual} does not match the pinned {spec['sha256']}")
    if "md5" in spec and _digest(path, "md5") != spec["md5"]:
        raise ValueError(f"{path.name}: md5 does not match the value published by the source")
    return path


def load_pima(raw=False):
    """Pima Indians Diabetes (OpenML 37). Returns X, y with readable column names.

    With raw=False, biologically impossible zeros become NaN so they are treated as missing.
    """
    data, _ = arff.loadarff(fetch("pima"))
    df = pd.DataFrame(data)
    df.columns = PIMA_FEATURES + ["Outcome"]
    y = (df.pop("Outcome").str.decode("utf-8") == "tested_positive").astype(int).to_numpy()
    X = df.astype(float)
    if not raw:
        from .config import PIMA_ZERO_AS_MISSING
        X[PIMA_ZERO_AS_MISSING] = X[PIMA_ZERO_AS_MISSING].replace(0, np.nan)
    return X, y


def load_cdc():
    """CDC Diabetes Health Indicators (UCI 891, BRFSS 2015). Returns X, y and the full frame."""
    df = pd.read_csv(fetch("cdc"))
    X = df[CDC_FEATURES].astype(float)
    y = df[CDC_TARGET].to_numpy().astype(int)
    return X, y, df


def duplicate_groups(X, y):
    """Group id per distinct (features, label) row, so identical rows never straddle a split."""
    key = pd.util.hash_pandas_object(pd.concat([X, pd.Series(y, index=X.index, name="y")], axis=1),
                                     index=False)
    return pd.factorize(key)[0]
