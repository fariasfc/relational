"""Datasets used by the experiments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms


# --------- MNIST as regression --------------------------------------------------


def mnist_regression(
    root: str | Path,
    n_train: int = 5_000,
    n_val: int = 1_000,
    seed: int = 0,
) -> Tuple[Subset, Subset]:
    """Return (train, val) MNIST subsets with float targets in [0, 9].

    Used by exp0 to mirror the PDF's ``version_15``/``version_16`` setup.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    full_train = datasets.MNIST(
        root=str(root), train=True, download=True, transform=transform
    )
    full_test = datasets.MNIST(
        root=str(root), train=False, download=True, transform=transform
    )

    g = torch.Generator().manual_seed(seed)
    train_idx = torch.randperm(len(full_train), generator=g)[:n_train].tolist()
    val_idx = torch.randperm(len(full_test), generator=g)[:n_val].tolist()

    train = Subset(full_train, train_idx)
    val = Subset(full_test, val_idx)
    return train, val


def mnist_loaders(
    train: Subset,
    val: Subset,
    batch_size: int = 128,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, generator=g, drop_last=True
    )
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def mnist_regression_with_test(
    root: str | Path,
    n_train: int = 5_000,
    n_val: int = 1_000,
    n_test: int = 1_000,
    seed: int = 0,
) -> Tuple[Subset, Subset, Subset]:
    """Like :func:`mnist_regression` but also returns a held-out test split.

    The 10 000-sample MNIST test set is split into:
      * ``val_set``  — first ``n_val``  shuffled indices (early-stopping / "best epoch" selection).
      * ``test_set`` — next  ``n_test`` shuffled indices (disjoint from val; final unbiased reporting).
    The ``train_set`` is drawn from the MNIST train split exactly as in
    :func:`mnist_regression`.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    full_train = datasets.MNIST(
        root=str(root), train=True, download=True, transform=transform
    )
    full_test = datasets.MNIST(
        root=str(root), train=False, download=True, transform=transform
    )

    g = torch.Generator().manual_seed(seed)
    train_idx = torch.randperm(len(full_train), generator=g)[:n_train].tolist()
    test_perm = torch.randperm(len(full_test), generator=g).tolist()
    if n_val + n_test > len(full_test):
        raise ValueError(
            f"n_val + n_test = {n_val + n_test} > len(MNIST test) = {len(full_test)}"
        )
    val_idx = test_perm[:n_val]
    test_idx = test_perm[n_val : n_val + n_test]

    return (
        Subset(full_train, train_idx),
        Subset(full_test, val_idx),
        Subset(full_test, test_idx),
    )


def mnist_loaders_with_test(
    train: Subset,
    val: Subset,
    test: Subset,
    batch_size: int = 128,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, generator=g, drop_last=True
    )
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# --------- 1D synthetic regression with non-uniform Lipschitz ------------------


def y_star(x: torch.Tensor) -> torch.Tensor:
    """``sin(x)`` on [0, π], ``5·sin(x)`` on (π, 2π]. Lipschitz contrast = 5×."""
    high = x > math.pi
    return torch.where(high, 5.0 * torch.sin(x), torch.sin(x))


def synthetic_1d(
    n_labeled: int = 50,
    n_unlabeled: int = 2_000,
    n_test: int = 1_000,
    shifted: bool = False,
    noise_std: float = 0.05,
    seed: int = 0,
) -> dict:
    """Generate the 1D regression task.

    If ``shifted``, labeled data only covers ``[0, π]`` while unlabeled covers
    ``[0, 2π]`` — the distribution-shift regime from the docx §5.4.
    """
    g = torch.Generator().manual_seed(seed)

    lo, hi = 0.0, 2.0 * math.pi
    if shifted:
        x_lab = torch.rand(n_labeled, 1, generator=g) * math.pi
    else:
        x_lab = torch.rand(n_labeled, 1, generator=g) * (hi - lo) + lo

    x_unl = torch.rand(n_unlabeled, 1, generator=g) * (hi - lo) + lo
    # Test always covers full domain so we measure generalization to the steep half.
    x_test = torch.linspace(lo, hi, n_test).unsqueeze(1)

    y_lab = y_star(x_lab) + noise_std * torch.randn(x_lab.shape, generator=g)
    y_test = y_star(x_test)

    return {
        "x_lab": x_lab,
        "y_lab": y_lab,
        "x_unl": x_unl,
        "x_test": x_test,
        "y_test": y_test,
    }


def labeled_loader(
    x: torch.Tensor, y: torch.Tensor, batch_size: int = 64, seed: int = 0
) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)


def unlabeled_loader(
    x: torch.Tensor, batch_size: int = 256, seed: int = 0
) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    ds = TensorDataset(x)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)
