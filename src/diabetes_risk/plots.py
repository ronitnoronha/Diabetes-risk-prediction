"""Shared chart style so every figure in the report looks the same."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
MAIN, ALT, ACCENT = "#2563eb", "#9ca3af", "#ea580c"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "savefig.bbox": "tight", "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.grid": True, "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
})


def bar_with_error(ax, labels, means, sds, highlight=None, xlabel="", xlim=None):
    colors = [MAIN if (highlight and lab in highlight) else ALT for lab in labels]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, means, xerr=sds, color=colors, height=0.6, error_kw={"ecolor": MUTED, "capsize": 3})
    ax.set_yticks(y, labels)
    for yi, m in zip(y, means):
        ax.text(m + (sds.max() if len(sds) else 0) + 0.002, yi, f"{m:.3f}", va="center", fontsize=9, color=INK)
    ax.set_xlabel(xlabel)
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(axis="y", visible=False)


def roc_pr(curves, path, title_suffix=""):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
    for name, (y, p, color) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        a.plot(fpr, tpr, color=color, lw=2, label=name)
        pr, rc, _ = precision_recall_curve(y, p)
        b.plot(rc, pr, color=color, lw=2, label=name)
    y0 = next(iter(curves.values()))[0]
    a.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
    b.axhline(y0.mean(), ls="--", color=MUTED, lw=1)
    a.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC curve{title_suffix}")
    b.set(xlabel="Recall", ylabel="Precision", title=f"Precision–recall curve{title_suffix}")
    a.legend(loc="lower right"); b.legend(loc="upper right")
    fig.savefig(path); plt.close(fig)


def calibration(curves, path, bins=10, title="Calibration"):
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1, label="Perfect calibration")
    for name, (y, p, color) in curves.items():
        frac, mean = calibration_curve(y, p, n_bins=bins, strategy="quantile")
        ax.plot(mean, frac, marker="o", color=color, lw=2, label=name)
    ax.set(xlabel="Predicted probability", ylabel="Observed rate", title=title)
    ax.legend(loc="upper left")
    fig.savefig(path); plt.close(fig)


def threshold_tradeoff(y, p, chosen, path, title="Threshold trade-off"):
    ts = np.linspace(0.01, 0.95, 120)
    rec = [((p >= t) & (y == 1)).sum() / (y == 1).sum() for t in ts]
    # precision is undefined once nobody is flagged; leave a gap instead of drawing 0
    prec = [((p >= t) & (y == 1)).sum() / (p >= t).sum() if (p >= t).sum() >= 20 else np.nan for t in ts]
    spec = [((p < t) & (y == 0)).sum() / (y == 0).sum() for t in ts]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(ts, rec, color=MAIN, lw=2, label="Recall (diabetes caught)")
    ax.plot(ts, prec, color=ACCENT, lw=2, label="Precision")
    ax.plot(ts, spec, color=ALT, lw=2, label="Specificity")
    ax.axvline(chosen, color=INK, lw=1, ls=":")
    ax.text(chosen + 0.01, 0.05, f"chosen {chosen:.3f}", color=INK, fontsize=9)
    ax.set(xlabel="Decision threshold", ylabel="Rate", title=title, ylim=(0, 1.02))
    ax.legend(loc="center right")
    fig.savefig(path); plt.close(fig)
