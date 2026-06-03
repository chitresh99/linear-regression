from __future__ import annotations

import warnings
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

_BLUE  ="#3A88AB"
_PINK  = "#A43D7A"

_PALETTE = {
    "bg":      "#ffffff",
    "surface": "#ffffff",
    "border":  "#e0e0e0",
    "primary": _BLUE,
    "accent":  _PINK,
    "text":    "#1a1a2e",
    "muted":   "#555555",
    # derived tints (used for fills / bands)
    "blue_tint": "#d0e9f2",
    "pink_tint": "#f2d0e6",
}

_SCATTER_KW = dict(alpha=0.70, s=55, edgecolors=_PALETTE["border"], linewidth=0.5)
_HIST_KW    = dict(edgecolor=_PALETTE["border"], alpha=0.75, density=True)
_GRID_KW    = dict(alpha=0.35, linestyle="--", linewidth=0.6, color=_PALETTE["border"])
_TITLE_KW   = dict(fontsize=13, fontweight="bold", color=_PALETTE["text"],
                   fontfamily="monospace")
_LABEL_KW   = dict(fontsize=11, color=_PALETTE["muted"])


def apply_theme() -> None:
    """Apply the light two-color ML theme globally. Call once at notebook start."""
    mpl.rcParams.update({
        "figure.facecolor":  _PALETTE["bg"],
        "axes.facecolor":    _PALETTE["surface"],
        "axes.edgecolor":    _PALETTE["border"],
        "axes.labelcolor":   _PALETTE["muted"],
        "axes.titlecolor":   _PALETTE["text"],
        "xtick.color":       _PALETTE["muted"],
        "ytick.color":       _PALETTE["muted"],
        "grid.color":        _PALETTE["border"],
        "text.color":        _PALETTE["text"],
        "legend.facecolor":  _PALETTE["bg"],
        "legend.edgecolor":  _PALETTE["border"],
        "legend.labelcolor": _PALETTE["text"],
        "figure.dpi":        110,
        "savefig.dpi":       150,
        "savefig.bbox":      "tight",
        "savefig.facecolor": _PALETTE["bg"],
        "font.family":       "monospace",
    })


def _spine_style(ax) -> None:
    for spine in ax.spines.values():
        spine.set_edgecolor(_PALETTE["border"])
        spine.set_linewidth(0.8)


def _save(fig: Figure, path: Optional[str]) -> None:
    if path:
        fig.savefig(path)
        print(f"  saved {path}")


def plot_cost_history(
    cost_history: list[float],
    *,
    log_scale: bool = False,
    title: str = "Cost Function History",
    save_path: Optional[str] = None,
) -> Figure:
    if not cost_history:
        raise ValueError("cost_history is empty.")

    costs = np.asarray(cost_history, dtype=float)
    iters = np.arange(len(costs))

    fig, ax = plt.subplots(figsize=(10, 5))

    # Filled area under curve for visual weight
    ax.fill_between(iters, costs, alpha=0.12, color=_BLUE)
    ax.plot(iters, costs, color=_BLUE, linewidth=2.0, label="Train cost")

    # Mark best point in pink
    best_idx = int(np.argmin(costs))
    ax.scatter(best_idx, costs[best_idx], color=_PINK, zorder=5, s=90,
               label=f"Best  iter={best_idx}  val={costs[best_idx]:.4g}")
    ax.axvline(best_idx, color=_PINK, linewidth=0.8, linestyle=":", alpha=0.6)

    if log_scale and costs.min() > 0:
        ax.set_yscale("log")

    ax.set_xlabel("Iteration", **_LABEL_KW)
    ax.set_ylabel("Cost (MSE)", **_LABEL_KW)
    ax.set_title(title, **_TITLE_KW)
    ax.legend(fontsize=10)
    ax.grid(True, **_GRID_KW)
    _spine_style(ax)

    fig.tight_layout()
    _save(fig, save_path)
    return fig

def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    annotate_r2: bool = True,
    title: str = "Predictions vs Actual",
    save_path: Optional[str] = None,
) -> Figure:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(y_true, y_pred, color=_BLUE, **_SCATTER_KW, label="Samples")

    lo  = min(y_true.min(), y_pred.min())
    hi  = max(y_true.max(), y_pred.max())
    pad = (hi - lo) * 0.05
    diag = [lo - pad, hi + pad]
    ax.plot(diag, diag, color=_PINK, linewidth=1.8, linestyle="--", label="Perfect fit")
    ax.set_xlim(diag)
    ax.set_ylim(diag)

    if annotate_r2:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        ax.text(0.05, 0.93, f"R² = {r2:.4f}", transform=ax.transAxes,
                fontsize=11, color=_BLUE, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.35", facecolor=_PALETTE["blue_tint"],
                          edgecolor=_BLUE, alpha=0.9))

    ax.set_xlabel("Actual", **_LABEL_KW)
    ax.set_ylabel("Predicted", **_LABEL_KW)
    ax.set_title(title, **_TITLE_KW)
    ax.legend(fontsize=10)
    ax.grid(True, **_GRID_KW)
    _spine_style(ax)

    fig.tight_layout()
    _save(fig, save_path)
    return fig

def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "Residual Analysis",
    save_path: Optional[str] = None,
) -> Figure:
    y_true    = np.asarray(y_true).ravel()
    y_pred    = np.asarray(y_pred).ravel()
    residuals = y_true - y_pred
    std       = residuals.std()

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5))
    for ax in (ax_left, ax_right):
        _spine_style(ax)

    ax_left.scatter(y_pred, residuals, color=_BLUE, **_SCATTER_KW)
    ax_left.axhline(0,    color=_PINK, linewidth=1.8, linestyle="--")
    ax_left.axhline( std, color=_PINK, linewidth=0.9, linestyle=":",
                     label=f"±1σ  ({std:.3g})", alpha=0.7)
    ax_left.axhline(-std, color=_PINK, linewidth=0.9, linestyle=":", alpha=0.7)
    # shaded ±1σ band
    ax_left.fill_between(
        [y_pred.min(), y_pred.max()], -std, std,
        color=_PINK, alpha=0.06
    )
    ax_left.set_xlabel("Fitted values", **_LABEL_KW)
    ax_left.set_ylabel("Residuals", **_LABEL_KW)
    ax_left.set_title("Residuals vs Fitted", **_TITLE_KW)
    ax_left.legend(fontsize=9)
    ax_left.grid(True, **_GRID_KW)

    ax_right.hist(residuals, bins="auto", color=_BLUE, **_HIST_KW)
    ax_right.axvline(0, color=_PINK, linewidth=1.8, linestyle="--")
    ax_right.axvline(residuals.mean(), color=_PINK, linewidth=1.2,
                     linestyle="-.", alpha=0.8,
                     label=f"mean = {residuals.mean():.3g}")

    x = np.linspace(residuals.min(), residuals.max(), 300)
    from scipy.stats import norm as _norm
    ax_right.plot(x, _norm.pdf(x, residuals.mean(), std),
                  color=_PINK, linewidth=1.8, label="N(μ, σ)")

    ax_right.set_xlabel("Residual", **_LABEL_KW)
    ax_right.set_ylabel("Density", **_LABEL_KW)
    ax_right.set_title("Residual Distribution", **_TITLE_KW)
    ax_right.legend(fontsize=9)
    ax_right.grid(True, **_GRID_KW)

    fig.suptitle(title, **_TITLE_KW, y=1.02)
    fig.tight_layout()
    _save(fig, save_path)
    return fig

def plot_feature_importance(
    feature_names: list[str],
    coefficients: np.ndarray,
    *,
    top_n: Optional[int] = None,
    title: str = "Feature Importance",
    save_path: Optional[str] = None,
) -> Figure:
    coefficients = np.asarray(coefficients).ravel()
    if len(feature_names) != len(coefficients):
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but "
            f"coefficients has {len(coefficients)}."
        )

    order = np.argsort(np.abs(coefficients))[::-1]
    if top_n is not None:
        order = order[:top_n]

    names  = [feature_names[i] for i in order]
    coeffs = coefficients[order]
    colors = [_BLUE if c >= 0 else _PINK for c in coeffs]

    height = max(5, len(names) * 0.48)
    fig, ax = plt.subplots(figsize=(10, height))

    bars = ax.barh(range(len(names)), coeffs, color=colors,
                   edgecolor=_PALETTE["border"], linewidth=0.5, height=0.65)

    span = coeffs.max() - coeffs.min() if len(coeffs) > 1 else abs(coeffs[0]) or 1
    x_off = span * 0.012

    for bar, val in zip(bars, coeffs):
        ha = "left" if val >= 0 else "right"
        ax.text(
            val + (x_off if val >= 0 else -x_off),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3g}", va="center", ha=ha,
            fontsize=9, color=_PALETTE["text"], fontfamily="monospace",
        )

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10, color=_PALETTE["text"])
    ax.set_xlabel("Coefficient value", **_LABEL_KW)
    ax.set_title(title, **_TITLE_KW)
    ax.axvline(0, color=_PALETTE["muted"], linewidth=0.8)
    ax.grid(True, axis="x", **_GRID_KW)
    _spine_style(ax)

    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor=_BLUE, label="Positive"),
                 Patch(facecolor=_PINK, label="Negative")],
        fontsize=9, loc="lower right",
    )

    fig.tight_layout()
    _save(fig, save_path)
    return fig

_METRIC_META = [
    ("R² Score", "r2",   True ),   # ↑ higher = better
    ("RMSE",     "rmse", False),   # ↓ lower  = better
    ("MAE",      "mae",  False),
]

# Alternating blue/pink for multi-model bars
def _bar_color(i: int) -> str:
    return _BLUE if i % 2 == 0 else _PINK


def create_comparison_plot(
    models_info: list[dict],
    *,
    metrics: Optional[list[str]] = None,
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
) -> Figure:
    if not models_info:
        raise ValueError("models_info is empty.")

    active_meta = (
        [(lbl, key, hib) for lbl, key, hib in _METRIC_META if key in (metrics or [])]
        or _METRIC_META
    )

    n_metrics   = len(active_meta)
    model_names = [m["name"] for m in models_info]
    x           = np.arange(len(model_names))

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, (label, key, higher_better) in zip(axes, active_meta):
        _spine_style(ax)

        values = []
        for m in models_info:
            v = m["metrics"].get(key)
            if v is None:
                warnings.warn(f"Model '{m['name']}' is missing metric '{key}'.")
                v = float("nan")
            values.append(v)

        bar_colors = [_bar_color(i) for i in range(len(model_names))]
        bars = ax.bar(x, values, width=0.55, color=bar_colors,
                      edgecolor=_PALETTE["border"], linewidth=0.5)

        # Bold border on best bar
        valid = [(i, v) for i, v in enumerate(values) if not np.isnan(v)]
        if valid:
            best_i = max(valid, key=lambda iv: iv[1] if higher_better else -iv[1])[0]
            bars[best_i].set_edgecolor(_PALETTE["text"])
            bars[best_i].set_linewidth(2.0)

        # Value labels
        y_range = (max(values, default=1) - min(values, default=0)) or 0.01
        nudge   = y_range * 0.02
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + nudge,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=8, color=_PALETTE["text"], fontfamily="monospace")

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=35, ha="right",
                           fontsize=9, color=_PALETTE["text"])
        ax.set_title(f"{label}  {'↑' if higher_better else '↓'}", **_TITLE_KW)
        ax.grid(True, axis="y", **_GRID_KW)
        ax.set_ylim(bottom=0)

    fig.suptitle(title, **_TITLE_KW, y=1.03)
    fig.tight_layout()
    _save(fig, save_path)
    return fig