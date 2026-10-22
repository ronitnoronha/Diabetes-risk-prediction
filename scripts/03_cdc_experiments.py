"""Step 3: CDC BRFSS 2015 experiments (253,680 adults).

Design
- 20% hold-out test set, locked until the end and scored once.
- Exact duplicate rows (9.5%) are grouped, so identical answers never sit on both sides of a split.
- Model choices use 5-fold grouped, stratified CV on the training set only.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import diabetes_risk  # noqa: F401
import numpy as np
import pandas as pd
from imblearn.pipeline import make_pipeline as imb_pipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import (RandomizedSearchCV, StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_predict, cross_validate)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from xgboost import XGBClassifier

from diabetes_risk import plots
from diabetes_risk.config import CDC_ORDINAL, FIGURES, REPORTS, SEED, TARGET_RECALL
from diabetes_risk.data import duplicate_groups, load_cdc
from diabetes_risk.evaluation import (at_threshold, bootstrap_ci, expected_calibration_error, scores,
                                      threshold_for_recall)
from diabetes_risk.features import CdcFeatures

OUT = REPORTS / "cdc"
OUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
METRICS = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "brier": "neg_brier_score"}

X_all, y_all, raw = load_cdc()
groups_all = duplicate_groups(X_all, y_all)
tr_idx, te_idx = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X_all, y_all, groups_all))
X, y, g = X_all.iloc[tr_idx].reset_index(drop=True), y_all[tr_idx], groups_all[tr_idx]
X_test, y_test = X_all.iloc[te_idx].reset_index(drop=True), y_all[te_idx]
CV = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, g))


def run(model, cv=CV):
    r = cross_validate(model, X, y, cv=cv, scoring=METRICS, n_jobs=-1 if not _is_parallel(model) else 1)
    return {"roc_auc": r["test_roc_auc"], "pr_auc": r["test_pr_auc"], "brier": -r["test_brier"]}


def _is_parallel(model):
    return any(isinstance(s, (LGBMClassifier, XGBClassifier, RandomForestClassifier)) for s in _steps(model))


def _steps(model):
    if hasattr(model, "steps"):
        return [s for _, s in model.steps]
    if hasattr(model, "estimators"):
        return [s for _, e in model.estimators for s in _steps(e)]
    return [model]


def summarise(results):
    return pd.DataFrame([{"model": n, **{f"{k}_mean": v.mean() for k, v in r.items()},
                          **{f"{k}_sd": v.std() for k, v in r.items()}} for n, r in results.items()])


# ---------------------------------------------------------------- models
def logreg_raw():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))


def logreg_engineered():
    # ordinal answers one-hot encoded (their effect is not linear), BMI as a spline (risk rises non-linearly)
    ct = ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CDC_ORDINAL),
        ("bmi_spline", SplineTransformer(n_knots=6, degree=3), ["BMI"]),
    ], remainder=StandardScaler(), verbose_feature_names_out=False)
    return make_pipeline(CdcFeatures(), ct, LogisticRegression(max_iter=5000, C=1.0))


def lgbm(**params):
    base = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=SEED, n_jobs=4)
    base.update(params)
    return LGBMClassifier(**base)


def xgb():
    return XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=10, tree_method="hist", eval_metric="logloss", random_state=SEED, n_jobs=4)


def forest():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=50, max_features=0.4, max_samples=0.3,
                                  random_state=SEED, n_jobs=4)


def tune_lgbm():
    space = {
        "lgbmclassifier__num_leaves": randint(8, 96),
        "lgbmclassifier__learning_rate": loguniform(0.01, 0.15),
        "lgbmclassifier__n_estimators": randint(200, 900),
        "lgbmclassifier__min_child_samples": randint(20, 400),
        "lgbmclassifier__reg_lambda": loguniform(1e-3, 30),
        "lgbmclassifier__subsample": uniform(0.6, 0.4),
        "lgbmclassifier__colsample_bytree": uniform(0.5, 0.5),
    }
    pipe = make_pipeline(CdcFeatures(), lgbm(subsample_freq=1, n_jobs=1))
    search = RandomizedSearchCV(pipe, space, n_iter=25, scoring="roc_auc", cv=CV[:3], n_jobs=4,
                                random_state=SEED, refit=False)
    search.fit(X, y)
    res = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    res.to_csv(OUT / "lgbm_tuning.csv", index=False)
    best = {k.replace("lgbmclassifier__", ""): v for k, v in search.best_params_.items()}
    return best, float(search.best_score_)


def main():
    summary = {"split": {"train_rows": len(y), "test_rows": len(y_test),
                         "train_prevalence": float(y.mean()), "test_prevalence": float(y_test.mean())}}
    print(summary["split"])

    # ---- 1. why the grouped split matters: a memorising model on random vs grouped folds
    memoriser = RandomForestClassifier(n_estimators=150, min_samples_leaf=1, max_samples=0.5, random_state=SEED, n_jobs=4)
    rnd = run(memoriser, cv=list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)))
    grp = run(memoriser)
    dup = pd.DataFrame([{"split": "Random folds (duplicates can cross)", **{k: v.mean() for k, v in rnd.items()}},
                        {"split": "Grouped folds (duplicates kept together)", **{k: v.mean() for k, v in grp.items()}}])
    dup.to_csv(OUT / "duplicate_leakage.csv", index=False)
    print("\nDuplicate leakage check\n", dup.round(4).to_string(index=False))

    # ---- 2. tuning
    best, best_cv = tune_lgbm()
    print("\nBest LightGBM params", best, round(best_cv, 4))
    summary["lgbm_best_params"] = best

    tuned = lambda: make_pipeline(CdcFeatures(), lgbm(subsample_freq=1, **best))

    # ---- 3. ablation ladder + model comparison (same grouped folds)
    gen_hlth_auc = [roc_auc_score(y[te], X.GenHlth.iloc[te]) for _, te in CV]
    results = {
        "A. One question: self-rated general health": {"roc_auc": np.array(gen_hlth_auc),
                                                       "pr_auc": np.array([average_precision_score(y[te], X.GenHlth.iloc[te]) for _, te in CV]),
                                                       "brier": np.full(5, np.nan)},
        "B. Logistic regression, raw features": run(logreg_raw()),
        "C. + one-hot answers, BMI spline, burden counts": run(logreg_engineered()),
        "D. LightGBM, default settings": run(make_pipeline(CdcFeatures(), lgbm())),
        "E. LightGBM, tuned (25-trial random search)": run(tuned()),
        "F. Soft vote: tuned LightGBM + logistic (C)": run(VotingClassifier([("lgbm", tuned()), ("lr", logreg_engineered())], voting="soft", weights=[2, 1])),
    }
    other = {
        "XGBoost (hist)": run(make_pipeline(CdcFeatures(), xgb())),
        "Random forest": run(make_pipeline(CdcFeatures(), forest())),
    }
    abl = summarise(results)
    abl["delta_auc_vs_B"] = abl.roc_auc_mean - abl.loc[abl.model.str.startswith("B."), "roc_auc_mean"].iloc[0]
    abl.to_csv(OUT / "ablation.csv", index=False)
    models = summarise({**{k: v for k, v in results.items() if not k.startswith("A.")}, **other}).sort_values("roc_auc_mean", ascending=False)
    models.to_csv(OUT / "model_comparison.csv", index=False)
    print("\nAblation\n", abl[["model", "roc_auc_mean", "roc_auc_sd", "pr_auc_mean", "brier_mean", "delta_auc_vs_B"]].round(4).to_string(index=False))
    print("\nModels\n", models[["model", "roc_auc_mean", "pr_auc_mean", "brier_mean"]].round(4).to_string(index=False))

    # final = best CV AUC among the tuned candidates; ties (< 0.001) go to the simpler model
    cand = abl[abl.model.str[:2].isin(["E.", "F."])].set_index("model").roc_auc_mean
    use_vote = cand.iloc[1] - cand.iloc[0] >= 0.001
    final_name = cand.index[1] if use_vote else cand.index[0]
    make_final = (lambda: VotingClassifier([("lgbm", tuned()), ("lr", logreg_engineered())], voting="soft", weights=[2, 1])) if use_vote else tuned
    summary["final_model"] = final_name
    print("\nFinal model:", final_name)

    # ---- 4. class imbalance
    imb = summarise({
        "No re-weighting": results["E. LightGBM, tuned (25-trial random search)"],
        "Class weights (balanced)": run(make_pipeline(CdcFeatures(), lgbm(subsample_freq=1, class_weight="balanced", **best))),
        "Random undersampling of majority": run(imb_pipeline(CdcFeatures(), RandomUnderSampler(random_state=SEED), lgbm(subsample_freq=1, **best))),
    })
    imb.to_csv(OUT / "imbalance.csv", index=False)
    print("\nImbalance\n", imb[["model", "roc_auc_mean", "pr_auc_mean", "brier_mean"]].round(4).to_string(index=False))

    # ---- 5. threshold on out-of-fold training predictions (test set untouched)
    oof = np.zeros(len(y))
    for tr, te in CV:
        oof[te] = make_final().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    oof_ece = expected_calibration_error(y, oof)
    calibrate = oof_ece > 0.01
    if calibrate:
        oof_cal = np.zeros(len(y))
        for tr, te in CV:
            m = CalibratedClassifierCV(make_final(), method="isotonic", cv=3).fit(X.iloc[tr], y[tr])
            oof_cal[te] = m.predict_proba(X.iloc[te])[:, 1]
        oof_used = oof_cal
    else:
        oof_used = oof
    t = threshold_for_recall(y, oof_used, TARGET_RECALL)
    summary["threshold"] = t
    summary["oof"] = {"uncalibrated": scores(y, oof), "used": scores(y, oof_used), "calibrated": bool(calibrate)}
    print(f"\nOOF ECE {oof_ece:.4f}; calibrate={calibrate}; threshold for {TARGET_RECALL:.0%} recall = {t:.4f}")

    # ---- 6. the one and only test-set evaluation
    final = make_final().fit(X, y)
    p_raw = final.predict_proba(X_test)[:, 1]
    if calibrate:
        p_test = CalibratedClassifierCV(make_final(), method="isotonic", cv=3).fit(X, y).predict_proba(X_test)[:, 1]
    else:
        p_test = p_raw
    lr_test = logreg_raw().fit(X, y).predict_proba(X_test)[:, 1]
    pd.DataFrame({"y": y_test, "p_final": p_test, "p_logreg_raw": lr_test}).to_csv(OUT / "test_predictions.csv.gz", index=False)
    test = {"model": final_name, **scores(y_test, p_test), "at_threshold": at_threshold(y_test, p_test, t),
            "default_0.5": at_threshold(y_test, p_test, 0.5),
            "ci95": {"roc_auc": bootstrap_ci(y_test, p_test, roc_auc_score),
                     "pr_auc": bootstrap_ci(y_test, p_test, average_precision_score),
                     "brier": bootstrap_ci(y_test, p_test, brier_score_loss)},
            "baseline_logreg_raw": scores(y_test, lr_test)}
    summary["test"] = test
    print("\nTEST", json.dumps(test, indent=1, default=float))

    # ---- 7. subgroups on the test set
    sub = X_test.assign(y=y_test, p=p_test)
    sub["Age band"] = pd.cut(sub.Age, [0, 5, 9, 13], labels=["18–44", "45–64", "65+"])
    sub["Income band"] = pd.cut(sub.Income, [0, 3, 6, 8], labels=["< $20k", "$20k–50k", "≥ $50k"])
    sub["Sex"] = sub.Sex.map({0: "Female", 1: "Male"})
    sub["Health coverage"] = sub.AnyHealthcare.map({0: "No coverage", 1: "Has coverage"})
    rows = []
    for col in ["Sex", "Age band", "Income band", "Health coverage"]:
        for level, d in sub.groupby(col, observed=True):
            th = at_threshold(d.y.to_numpy(), d.p.to_numpy(), t)
            rows.append({"group": col, "level": str(level), "n": len(d), "prevalence": d.y.mean(),
                         "roc_auc": roc_auc_score(d.y, d.p), "recall": th["recall"], "precision": th["precision"],
                         "flagged_per_1000": th["flagged_per_1000"]})
    subgroups = pd.DataFrame(rows)
    subgroups.to_csv(OUT / "subgroups.csv", index=False)
    print("\nSubgroups\n", subgroups.round(3).to_string(index=False))

    # ---- 8. SHAP on the LightGBM component
    import shap
    import matplotlib.pyplot as plt
    lgb_pipe = tuned().fit(X, y)
    sample = X_test.sample(4000, random_state=SEED)
    Z = lgb_pipe[:-1].transform(sample)
    sv = shap.TreeExplainer(lgb_pipe[-1]).shap_values(Z)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.DataFrame({"feature": Z.columns, "mean_abs_shap": np.abs(sv).mean(0)}).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(OUT / "shap_importance.csv", index=False)
    shap.summary_plot(sv, Z, show=False, max_display=15, plot_size=(8, 6))
    plt.title("CDC: SHAP values on 4,000 test respondents", fontweight="bold")
    plt.savefig(FIGURES / "cdc_shap_summary.png", bbox_inches="tight", dpi=160); plt.close()

    # ---- figures
    fig, ax = plt.subplots(figsize=(8, 3.6))
    plots.bar_with_error(ax, list(abl.model), abl.roc_auc_mean.to_numpy(), abl.roc_auc_sd.to_numpy(),
                         highlight={final_name}, xlabel="ROC-AUC (mean ± sd, 5 grouped folds)", xlim=(0.65, 0.86))
    ax.set_title("CDC: what each step added")
    fig.savefig(FIGURES / "cdc_ablation.png"); plt.close(fig)
    plots.roc_pr({final_name.split(": ")[-1].split(" (")[0]: (y_test, p_test, plots.MAIN),
                  "Logistic regression, raw": (y_test, lr_test, plots.ALT)}, FIGURES / "cdc_roc_pr.png", " (test set)")
    plots.calibration({"Final model": (y_test, p_test, plots.MAIN), "Before calibration": (y_test, p_raw, plots.ALT)} if calibrate
                      else {"Final model": (y_test, p_test, plots.MAIN)}, FIGURES / "cdc_calibration.png", title="CDC: calibration (test set)")
    plots.threshold_tradeoff(y_test, p_test, t, FIGURES / "cdc_threshold.png", title="CDC: screening threshold (test set)")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    lab = subgroups.group + ": " + subgroups.level
    ax.barh(lab[::-1], subgroups.roc_auc[::-1], color=plots.MAIN, height=0.6)
    ax.axvline(test["roc_auc"], color=plots.INK, ls=":", lw=1)
    for i, v in enumerate(subgroups.roc_auc[::-1]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set(xlim=(0.6, 0.9), xlabel="ROC-AUC on the test set", title="CDC: performance by subgroup")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIGURES / "cdc_subgroups.png"); plt.close(fig)

    summary.update({"ablation": abl.round(4).to_dict(orient="records"), "models": models.round(4).to_dict(orient="records"),
                    "imbalance": imb.round(4).to_dict(orient="records"), "duplicate_leakage": dup.round(4).to_dict(orient="records")})
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
