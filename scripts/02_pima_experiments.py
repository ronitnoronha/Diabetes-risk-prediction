"""Step 2: Pima experiments.

768 patients is small, so there is no single hold-out test set. Every number is the mean ± sd over
the same 50 folds (5-fold CV repeated 10 times), with all preprocessing fitted inside each fold.
Tuned models are scored with nested CV so the tuning never sees the fold it is scored on.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import diabetes_risk  # noqa: F401  (sets pandas output for transformers)
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imb_pipeline
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from diabetes_risk import plots
from diabetes_risk.config import FIGURES, PIMA_ZERO_AS_MISSING, REPORTS, SEED, TARGET_RECALL
from diabetes_risk.data import load_pima
from diabetes_risk.evaluation import at_threshold, corrected_ttest, scores, threshold_for_recall
from diabetes_risk.features import NotebookFeatures, PimaClinicalFeatures

OUT = REPORTS / "pima"
OUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
METRICS = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "brier": "neg_brier_score"}
NOTEBOOK_XGB = dict(colsample_bytree=0.9, learning_rate=0.1, max_depth=9, n_estimators=100,
                    subsample=0.9, eval_metric="logloss", random_state=SEED)

X, y = load_pima()               # zeros already recoded to NaN
X_raw, _ = load_pima(raw=True)   # original zeros, for the "before" rows
N_TRAIN, N_TEST = int(len(y) * 0.8), int(len(y) * 0.2)


def impute():
    return SimpleImputer(strategy="median", add_indicator=True)


def logreg(C=1.0, **kw):
    return LogisticRegression(C=C, max_iter=5000, **kw)


def shallow_xgb():
    # depth-2 trees, many small steps: high bias / low variance suits 768 rows; NaN handled natively
    return XGBClassifier(n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=3, eval_metric="logloss", random_state=SEED)


def splines(cols=("Age", "BMI")):
    # cubic splines let risk rise and then level off with age and BMI instead of forcing a straight line
    return ColumnTransformer([("spline", SplineTransformer(n_knots=4, degree=3), list(cols))],
                             remainder=StandardScaler(), verbose_feature_names_out=False)


def final_logreg(**kw):
    return make_pipeline(SimpleImputer(strategy="median"), splines(), logreg(**kw))


def tuned_logreg():
    grid = {"logisticregression__C": np.logspace(-3, 2, 11)}
    return GridSearchCV(final_logreg(), grid, scoring="roc_auc",
                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED))


def run(model, data=None):
    r = cross_validate(model, X if data is None else data, y, cv=CV, scoring=METRICS, n_jobs=-1)
    return {"roc_auc": r["test_roc_auc"], "pr_auc": r["test_pr_auc"], "brier": -r["test_brier"]}


def summarise(results):
    rows = []
    for name, r in results.items():
        rows.append({"model": name, **{f"{k}_mean": v.mean() for k, v in r.items()},
                     **{f"{k}_sd": v.std() for k, v in r.items()}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 1. The original notebook, as written
def original_notebook_split_sensitivity():
    """Reproduce the notebook (impute on all data, one 80/20 split, SMOTE, depth-9 XGBoost)."""
    Xi = X_raw.copy()
    Xi[PIMA_ZERO_AS_MISSING] = KNNImputer(missing_values=0, n_neighbors=5).fit_transform(Xi[PIMA_ZERO_AS_MISSING])
    aucs = []
    for seed in range(50):
        Xtr, Xte, ytr, yte = train_test_split(Xi, y, test_size=0.2, random_state=seed, stratify=y)
        fe = NotebookFeatures().fit(Xtr)
        sc = StandardScaler().fit(fe.transform(Xtr))
        Xb, yb = SMOTE(random_state=SEED).fit_resample(sc.transform(fe.transform(Xtr)), ytr)
        m = XGBClassifier(**NOTEBOOK_XGB).fit(Xb, yb)
        aucs.append(scores(yte, m.predict_proba(sc.transform(fe.transform(Xte)))[:, 1])["roc_auc"])
    return np.array(aucs)


# ---------------------------------------------------------------- 2. Ablation: one change at a time
def ablation():
    notebook_leak_free = imb_pipeline(KNNImputer(n_neighbors=5), NotebookFeatures(), StandardScaler(),
                                      SMOTE(random_state=SEED), XGBClassifier(**NOTEBOOK_XGB))
    steps = {
        "A. Notebook model, re-tested without leakage": (notebook_leak_free, X),
        "B. Logistic regression, zeros left as values": (make_pipeline(StandardScaler(), logreg()), X_raw),
        "C. + zeros treated as missing (median)": (make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), logreg()), X),
        "D. + missing-value indicators": (make_pipeline(impute(), StandardScaler(), logreg()), X),
        "E. + clinical threshold features": (make_pipeline(impute(), PimaClinicalFeatures(), StandardScaler(), logreg()), X),
        "F. + notebook's 10 engineered features": (make_pipeline(SimpleImputer(strategy="median"), NotebookFeatures(), StandardScaler(), logreg()), X),
        "G. Splines for age and BMI (instead of D–F)": (final_logreg(), X),
        "H. + tuned regularisation (nested CV)": (tuned_logreg(), X),
    }
    return {name: run(m, data) for name, (m, data) in steps.items()}


# ---------------------------------------------------------------- 3. Model families on the same folds
def model_comparison():
    base_tree = make_pipeline(impute(), PimaClinicalFeatures())
    models = {
        "Logistic regression (final)": final_logreg(),
        "Logistic regression, notebook features": make_pipeline(SimpleImputer(strategy="median"), NotebookFeatures(), StandardScaler(), logreg()),
        "Notebook model (deep XGBoost + SMOTE)": imb_pipeline(KNNImputer(n_neighbors=5), NotebookFeatures(), StandardScaler(), SMOTE(random_state=SEED), XGBClassifier(**NOTEBOOK_XGB)),
        "SVM (RBF kernel)": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVC(C=0.5, probability=True, random_state=SEED)),
        "k-nearest neighbours (k=25)": make_pipeline(impute(), StandardScaler(), KNeighborsClassifier(25)),
        "Random forest": make_pipeline(impute(), RandomForestClassifier(500, min_samples_leaf=5, max_features=0.5, random_state=SEED, n_jobs=1)),
        "XGBoost, shallow (native missing)": shallow_xgb(),
        "LightGBM, small leaves": make_pipeline(base_tree, LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=4, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=SEED)),
        "Soft vote: logistic + shallow XGBoost": VotingClassifier([("lr", final_logreg()), ("xgb", shallow_xgb())], voting="soft"),
        "Stacking: logistic + forest + XGBoost": StackingClassifier(
            [("lr", final_logreg()), ("rf", make_pipeline(impute(), RandomForestClassifier(200, min_samples_leaf=5, random_state=SEED, n_jobs=1))), ("xgb", shallow_xgb())],
            final_estimator=logreg(), cv=5),
    }
    return {name: run(m) for name, m in models.items()}


# ---------------------------------------------------------------- 4. Class imbalance strategies
def imbalance():
    return {
        "No re-weighting": run(final_logreg()),
        "Class weights (balanced)": run(final_logreg(class_weight="balanced")),
        "SMOTE inside each fold": run(imb_pipeline(SimpleImputer(strategy="median"), splines(), SMOTE(random_state=SEED), logreg())),
    }


# ---------------------------------------------------------------- 5. Calibration and screening threshold
def oof(model, seed=SEED):
    return cross_val_predict(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]


def nested_threshold(model, repeats=5):
    """Pick the threshold on inner out-of-fold predictions, then measure it on the untouched outer fold."""
    rows = []
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED)
    for tr, te in outer.split(X, y):
        inner = cross_val_predict(model, X.iloc[tr], y[tr], cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
        t = threshold_for_recall(y[tr], inner, TARGET_RECALL)
        p = model.fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        rows.append(at_threshold(y[te], p, t))
    return pd.DataFrame(rows)


def explain():
    model = final_logreg().fit(X, y)
    lr = model[-1]
    names = model[:-1].get_feature_names_out()
    coef = pd.DataFrame({"feature": names, "coef": lr.coef_[0]})
    # spline columns have no "per SD" meaning; their combined effect is drawn as a curve instead
    curves = {}
    ct = model[1]
    for col in ["Age", "BMI"]:
        grid = np.linspace(X[col].quantile(0.02), X[col].quantile(0.98), 100)
        mask = coef.feature.str.startswith(f"{col}_sp_")
        basis = ct.named_transformers_["spline"].transform(
            pd.DataFrame({c: (grid if c == col else np.full(100, X[c].median())) for c in ["Age", "BMI"]}))
        contrib = basis[[f for f in basis.columns if f.startswith(f"{col}_sp_")]].to_numpy() @ coef.loc[mask, "coef"].to_numpy()
        curves[col] = (grid, contrib - contrib[0])
    coef = coef[~coef.feature.str.contains("_sp_")].assign(odds_ratio_per_sd=lambda d: np.exp(d.coef))
    coef = coef.rename(columns={"coef": "coef_per_sd"})
    coef = coef.reindex(coef.coef_per_sd.abs().sort_values(ascending=False).index)
    perm = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        m = final_logreg().fit(X.iloc[tr], y[tr])
        r = permutation_importance(m, X.iloc[te], y[te], scoring="roc_auc", n_repeats=20, random_state=SEED)
        perm.append(r.importances_mean)
    perm = pd.DataFrame({"feature": X.columns, "auc_drop": np.mean(perm, axis=0)}).sort_values("auc_drop", ascending=False)
    return coef, perm, curves


def main():
    summary = {}

    split_aucs = original_notebook_split_sensitivity()
    summary["notebook_reported_auc"] = 0.828
    summary["notebook_method_50_splits"] = {"mean": split_aucs.mean(), "min": split_aucs.min(), "max": split_aucs.max(), "sd": split_aucs.std()}
    print(f"Notebook method over 50 random splits: {split_aucs.mean():.3f} (range {split_aucs.min():.3f}–{split_aucs.max():.3f})")

    abl = ablation()
    abl_df = summarise(abl)
    base = abl["A. Notebook model, re-tested without leakage"]["roc_auc"]
    abl_df["delta_auc_vs_A"] = abl_df.roc_auc_mean - base.mean()
    abl_df["p_value_vs_A"] = [corrected_ttest(r["roc_auc"], base, N_TRAIN, N_TEST)[2] for r in abl.values()]
    abl_df.to_csv(OUT / "ablation.csv", index=False)
    print("\nAblation\n", abl_df[["model", "roc_auc_mean", "roc_auc_sd", "pr_auc_mean", "brier_mean", "delta_auc_vs_A", "p_value_vs_A"]].round(4).to_string(index=False))

    cmp = model_comparison()
    cmp_df = summarise(cmp).sort_values("roc_auc_mean", ascending=False)
    ref = cmp["Logistic regression (final)"]["roc_auc"]
    cmp_df["p_value_vs_final"] = [corrected_ttest(cmp[m]["roc_auc"], ref, N_TRAIN, N_TEST)[2] for m in cmp_df.model]
    cmp_df.to_csv(OUT / "model_comparison.csv", index=False)
    print("\nModels\n", cmp_df[["model", "roc_auc_mean", "roc_auc_sd", "pr_auc_mean", "brier_mean", "p_value_vs_final"]].round(4).to_string(index=False))

    imb = summarise(imbalance())
    imb.to_csv(OUT / "imbalance.csv", index=False)
    print("\nImbalance\n", imb[["model", "roc_auc_mean", "pr_auc_mean", "brier_mean"]].round(4).to_string(index=False))

    # calibration: out-of-fold probabilities, before and after Platt scaling
    p_final = oof(final_logreg())
    p_platt = oof(CalibratedClassifierCV(final_logreg(), method="sigmoid", cv=5))
    p_nb = oof(imb_pipeline(KNNImputer(n_neighbors=5), NotebookFeatures(), StandardScaler(), SMOTE(random_state=SEED), XGBClassifier(**NOTEBOOK_XGB)))
    p_bal = oof(final_logreg(class_weight="balanced"))
    cal = pd.DataFrame([{"model": n, **scores(y, p)} for n, p in [
        ("Notebook model (SMOTE + deep XGBoost)", p_nb), ("Logistic, class-balanced", p_bal),
        ("Logistic (final)", p_final), ("Logistic + Platt scaling", p_platt)]])
    cal.to_csv(OUT / "calibration.csv", index=False)
    print("\nCalibration (out-of-fold)\n", cal.round(4).to_string(index=False))

    thr = nested_threshold(final_logreg())
    thr_nb = nested_threshold(imb_pipeline(KNNImputer(n_neighbors=5), NotebookFeatures(), StandardScaler(), SMOTE(random_state=SEED), XGBClassifier(**NOTEBOOK_XGB)))
    default = pd.DataFrame([at_threshold(y[te], final_logreg().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1], 0.5)
                            for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(X, y)])
    thr_table = pd.DataFrame({
        "Final model, threshold for 85% recall": thr.mean(numeric_only=True),
        "Final model, default 0.5": default.mean(numeric_only=True),
        "Notebook model, threshold for 85% recall": thr_nb.mean(numeric_only=True)}).T
    thr_table.to_csv(OUT / "threshold.csv")
    print("\nScreening threshold (outer folds)\n", thr_table[["threshold", "recall", "precision", "specificity", "accuracy", "flagged_per_1000"]].round(3).to_string())

    coef, perm, curves = explain()
    coef.to_csv(OUT / "odds_ratios.csv", index=False)
    perm.to_csv(OUT / "permutation_importance.csv", index=False)

    # ---- figures
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.8))
    labels = list(abl_df.model)
    plots.bar_with_error(ax, labels, abl_df.roc_auc_mean.to_numpy(), abl_df.roc_auc_sd.to_numpy(),
                         highlight={labels[-2], labels[-1]}, xlabel="ROC-AUC (mean ± sd, 50 folds)", xlim=(0.70, 0.90))
    ax.set_title("Pima: what each step added")
    fig.savefig(FIGURES / "pima_ablation.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    plots.bar_with_error(ax, list(cmp_df.model), cmp_df.roc_auc_mean.to_numpy(), cmp_df.roc_auc_sd.to_numpy(),
                         highlight={"Logistic regression (final)"}, xlabel="ROC-AUC (mean ± sd, 50 folds)", xlim=(0.70, 0.90))
    ax.set_title("Pima: model families on identical folds")
    fig.savefig(FIGURES / "pima_models.png"); plt.close(fig)

    plots.roc_pr({"Final logistic model": (y, p_final, plots.MAIN), "Notebook model": (y, p_nb, plots.ALT)},
                 FIGURES / "pima_roc_pr.png", " (out-of-fold)")
    plots.calibration({"Final logistic model": (y, p_final, plots.MAIN), "Notebook model (SMOTE)": (y, p_nb, plots.ALT)},
                      FIGURES / "pima_calibration.png", title="Pima: calibration (out-of-fold)")
    plots.threshold_tradeoff(y, p_final, float(thr.threshold.median()), FIGURES / "pima_threshold.png",
                             title="Pima: choosing the screening threshold")

    top = coef.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(top.feature, top.odds_ratio_per_sd, color=[plots.MAIN if v > 1 else plots.ACCENT for v in top.odds_ratio_per_sd])
    ax.axvline(1, color=plots.MUTED, lw=1)
    ax.set(xlabel="Odds ratio per 1 SD increase", title="Pima: what drives the final model")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIGURES / "pima_odds_ratios.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (col, (grid, c)) in zip(axes, curves.items()):
        ax.plot(grid, np.exp(c), color=plots.MAIN, lw=2.5)
        ax.axhline(1, color=plots.MUTED, lw=1)
        ax.set(xlabel=col + (" (years)" if col == "Age" else " (kg/m²)"), title=f"Effect of {col.lower() if col == 'Age' else col}")
    axes[0].set_ylabel("Odds vs lowest value shown")
    fig.suptitle("Pima: non-linear effects learned by the splines", fontweight="bold", y=1.04)
    fig.savefig(FIGURES / "pima_spline_effects.png"); plt.close(fig)

    summary.update({
        "ablation": abl_df.round(4).to_dict(orient="records"),
        "models": cmp_df.round(4).to_dict(orient="records"),
        "calibration": cal.round(4).to_dict(orient="records"),
        "threshold": thr_table.round(4).reset_index().to_dict(orient="records"),
    })
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
