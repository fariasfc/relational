"""Standardised matplotlib helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


def save_curves(
    path: Path,
    series: dict[str, Sequence[float]],
    *,
    xlabel: str = "epoch",
    ylabel: str = "value",
    title: str = "",
    logy: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, vals in series.items():
        ax.plot(range(len(vals)), vals, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_bar(
    path: Path,
    labels: Sequence[str],
    means: Sequence[float],
    stds: Sequence[float],
    *,
    ylabel: str = "test MSE",
    title: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = range(len(labels))
    ax.bar(xs, means, yerr=stds, capsize=4)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_sigma_curves(
    path: Path,
    grid_x: Sequence[float],
    sigmas: dict[str, Sequence[float]],
    *,
    title: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, vals in sigmas.items():
        ax.plot(grid_x, vals, label=name)
    ax.set_xlabel("x")
    ax.set_ylabel("σ(q)")
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
