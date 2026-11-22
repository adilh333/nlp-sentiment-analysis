"""Charts for the register-level counts and the modelling sample."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fos_triage.config import FIGURES

NAVY = "#0B2545"
TEAL = "#1F7A8C"
GOLD = "#B0893E"
CRIMSON = "#9B3A4A"
SLATE = "#5C6B73"
GREEN = "#2F6F4E"
PALETTE = [NAVY, TEAL, GOLD, CRIMSON, GREEN, "#6E4C7A", SLATE]


def _style() -> None:
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
        }
    )


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    margin = z * math.sqrt((p * (1 - p) / n) + z**2 / (4 * n**2)) / den
    return centre - margin, centre + margin


def plot_sector_volumes(counts: pd.DataFrame, path=None) -> None:
    _style()
    sectors = counts[counts["sector"] != "All sectors"].copy()
    sectors["total"] = pd.to_numeric(sectors["total"], errors="coerce").fillna(0)
    pivot = sectors.pivot(index="year", columns="sector", values="total").fillna(0)
    # Stable, readable column order: largest recent sector first.
    order = pivot.iloc[-1].sort_values(ascending=False).index.tolist()
    pivot = pivot[order]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=PALETTE[: len(pivot.columns)], width=0.84)
    ax.set_title("Published ombudsman decisions by sector (2026 is to 21 November)")
    ax.set_xlabel("Decision year")
    ax.set_ylabel("Decisions")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    _save(fig, path or FIGURES / "sector_volumes.png")


def plot_uphold_rates(counts: pd.DataFrame, path=None) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    sectors = counts[counts["sector"] != "All sectors"].copy()
    # Claims management and funeral plans are too small to put on the same axis.
    big = sectors.groupby("sector")["total"].max()
    sectors = sectors[sectors["sector"].isin(big[big >= 500].index)]
    for color, (sector, group) in zip(PALETTE, sectors.groupby("sector")):
        group = group.copy()
        group["total"] = pd.to_numeric(group["total"], errors="coerce")
        group["upheld"] = pd.to_numeric(group["upheld"], errors="coerce")
        group = group[group["total"] > 0].dropna(subset=["upheld"]).sort_values("year")
        rate = group["upheld"] / group["total"]
        ax.plot(group["year"], rate, marker="o", label=sector, color=color, linewidth=2)
    overall = counts[counts["sector"] == "All sectors"].sort_values("year")
    rate = overall["upheld"] / overall["total"].replace(0, np.nan)
    ax.plot(overall["year"], rate, marker="o", label="All sectors", color="black", linewidth=2.4, linestyle="--")
    ax.set_ylim(0, 1)
    ax.set_title("Uphold rate of published decisions (2026 is to 21 November)")
    ax.set_xlabel("Decision year")
    ax.set_ylabel("Share upheld")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    _save(fig, path or FIGURES / "uphold_rate_by_sector.png")


def plot_themes(themes: pd.DataFrame, path=None) -> None:
    _style()
    labels = list(themes["label"].unique())
    n = len(labels)
    cols = 3
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.1 * rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, label, color in zip(axes, labels, PALETTE):
        group = themes[themes["label"] == label].sort_values("year")
        ax.fill_between(group["year"], group["total"], color=color, alpha=0.25)
        ax.plot(group["year"], group["total"], color=color, marker="o", linewidth=2)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("Decisions")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Keyword hits in the decisions register", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, path or FIGURES / "theme_volumes.png")


def plot_rate_bars(frame: pd.DataFrame, column: str, title: str, path, label_map: dict | None = None) -> None:
    _style()
    rows = []
    for key, group in frame.groupby(column):
        n = int(len(group))
        k = int(group["upheld"].sum())
        low, high = wilson_interval(k, n)
        label = label_map.get(key, key) if label_map else key
        rows.append({"label": label, "rate": k / n if n else np.nan, "low": low, "high": high, "n": n})
    table = pd.DataFrame(rows).sort_values("rate", ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(3.6, 0.48 * len(table) + 1.4)))
    y = np.arange(len(table))
    ax.barh(y, table["rate"], color=TEAL, height=0.72)
    xerr = np.vstack([table["rate"] - table["low"], table["high"] - table["rate"]])
    ax.errorbar(table["rate"], y, xerr=xerr, fmt="none", ecolor=NAVY, capsize=3, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{label}  (n={n})" for label, n in zip(table["label"], table["n"])])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Uphold rate, with 95% Wilson interval")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, path)


def plot_confusion(matrix: list, labels: list[str], title: str, path, cmap="Blues") -> None:
    _style()
    fig_w = max(5.2, 0.7 * len(labels) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.85))
    sns.heatmap(
        np.asarray(matrix),
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=False,
        linewidths=0.4,
        linecolor="white",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    _save(fig, path)


def _save(fig, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
