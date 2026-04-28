"""TDC molecular regression with Mordred descriptors.

For the molecular SSL experiments we keep the substrate-agnostic loss
machinery in :mod:`relational.losses` and :mod:`relational.train` and
swap the data layer.

Pipeline:

1. Load a TDC ADME regression task (default: Lipophilicity_AstraZeneca).
2. Compute Mordred 2D descriptors per SMILES, cached to disk.
3. Drop descriptors that are NaN/Inf or zero-variance on the FULL dataset.
4. Standardise (mean 0, std 1) using statistics computed on the full
   labelled training pool — never use val / test stats.
5. Return torch tensors per scaffold split, plus the per-feature std on
   the *unlabelled training pool* for variance-scaled noise.

The expensive step (Mordred) is cached at
``data/tdc/mordred/<task>_<split-seed>.pt`` so subsequent runs are fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# RDKit + Mordred are heavy imports. Import lazily so the unit tests for
# the existing MNIST experiments don't pay this cost.
def _featurise_mordred(smiles: list[str]) -> pd.DataFrame:
    from mordred import Calculator, descriptors
    from rdkit import Chem

    calc = Calculator(descriptors, ignore_3D=True)
    mols = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            mols.append(None)
        else:
            mols.append(mol)

    # ``calc.pandas`` runs in parallel by default and returns a DataFrame
    # whose cells may be Mordred ``Missing`` or ``Error`` objects when a
    # descriptor fails on a given molecule. We coerce to numeric so failed
    # cells become NaN.
    df = calc.pandas(mols)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


@dataclass
class TDCSplit:
    """Standardised (X, y) tensors for one of {train, val, test}, plus the
    raw SMILES for downstream visualisation/diagnostics."""
    X: torch.Tensor          # (n, d) standardised float32
    y: torch.Tensor          # (n, 1) float32 raw targets
    smiles: list[str]


@dataclass
class TDCDataset:
    name: str
    train: TDCSplit
    val: TDCSplit
    test: TDCSplit
    feature_names: list[str]
    feature_mean: torch.Tensor   # (d,)
    feature_std: torch.Tensor    # (d,) — used for standardisation
    # std on the (post-standardisation) training set, used for
    # variance-scaled noise in v17_input_var / v18_pi_input_var. After
    # standardisation this is ≈ 1 for almost every feature, but keeping
    # the per-feature value lets us scale noise properly even for any
    # feature with a degenerate distribution.
    train_post_std: torch.Tensor  # (d,)


def load_tdc_lipophilicity(
    cache_dir: str | Path = "data/tdc",
    seed: int = 0,
    frac: tuple[float, float, float] = (0.7, 0.1, 0.2),
    nan_col_threshold: float = 0.0,
    nan_row_drop: bool = True,
) -> TDCDataset:
    """Load Lipophilicity_AstraZeneca with scaffold split + Mordred features.

    ``nan_col_threshold`` drops descriptor columns whose NaN-fraction
    across the full dataset exceeds the threshold. The default (0.0)
    drops any column with at least one NaN — strict, but the resulting
    feature set is reproducible and avoids needing to imputation policies.

    ``nan_row_drop`` drops molecules that still have any NaN after column
    filtering. With ``nan_col_threshold=0.0`` this should be a no-op.
    """
    return _load_tdc(
        task_name="Lipophilicity_AstraZeneca",
        cache_dir=cache_dir,
        seed=seed,
        frac=frac,
        nan_col_threshold=nan_col_threshold,
        nan_row_drop=nan_row_drop,
    )


def _load_tdc(
    task_name: str,
    cache_dir: str | Path,
    seed: int,
    frac: tuple[float, float, float],
    nan_col_threshold: float,
    nan_row_drop: bool,
) -> TDCDataset:
    from tdc.single_pred import ADME

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    feat_cache = cache_dir / "mordred" / f"{task_name}.parquet"
    feat_cache.parent.mkdir(parents=True, exist_ok=True)

    # Load TDC ADME data (downloads on first call; cached by TDC under
    # ``cache_dir``).
    data = ADME(name=task_name, path=str(cache_dir))
    splits = data.get_split(method="scaffold", seed=seed, frac=list(frac))

    full_df = pd.concat(
        [splits["train"], splits["valid"], splits["test"]],
        keys=["train", "valid", "test"],
    ).reset_index(level=0).rename(columns={"level_0": "split"})

    # --- Feature computation (cached) ---
    if feat_cache.exists():
        feat_df = pd.read_parquet(feat_cache)
        # The cache is keyed by SMILES; intersect with the current order.
        feat_df = feat_df.set_index("Drug").loc[full_df["Drug"].tolist()].reset_index()
    else:
        print(
            f"[data_tdc] Computing Mordred descriptors for {len(full_df)} molecules — "
            f"cache miss at {feat_cache}. This takes ~5-10 minutes once."
        )
        feat_df = _featurise_mordred(full_df["Drug"].tolist())
        feat_df.insert(0, "Drug", full_df["Drug"].values)
        feat_df.to_parquet(feat_cache)

    # Drop the SMILES column for matrix work, keep alignment.
    smiles_col = feat_df["Drug"].tolist()
    X_full = feat_df.drop(columns=["Drug"])

    # --- Drop high-NaN columns (default: any NaN) ---
    nan_frac = X_full.isna().mean()
    keep_cols = nan_frac[nan_frac <= nan_col_threshold].index.tolist()
    X_full = X_full[keep_cols]

    # --- Drop columns with zero variance (constant) ---
    nz_mask = X_full.std(axis=0) > 1e-12
    X_full = X_full.loc[:, nz_mask]

    # --- Optionally drop rows that still have NaN after column filtering ---
    if nan_row_drop:
        good = ~X_full.isna().any(axis=1).values
        X_full = X_full[good]
        full_df = full_df[good].reset_index(drop=True)
        smiles_col = [s for s, g in zip(smiles_col, good) if g]

    # --- Standardise using full-dataset statistics ---
    feature_mean = X_full.mean(axis=0)
    feature_std = X_full.std(axis=0).replace(0.0, 1.0)
    X_std = (X_full - feature_mean) / feature_std

    # Convert to tensors and re-split
    X_tensor = torch.tensor(X_std.values, dtype=torch.float32)
    y_tensor = torch.tensor(full_df["Y"].values, dtype=torch.float32).unsqueeze(-1)

    split_labels = full_df["split"].values
    train_mask = split_labels == "train"
    val_mask = split_labels == "valid"
    test_mask = split_labels == "test"

    def _slice(mask):
        return TDCSplit(
            X=X_tensor[torch.tensor(mask)],
            y=y_tensor[torch.tensor(mask)],
            smiles=[s for s, m in zip(smiles_col, mask) if m],
        )

    train = _slice(train_mask)
    val = _slice(val_mask)
    test = _slice(test_mask)

    # Per-feature std on the standardised training pool — for
    # variance-scaled noise. Should be ≈ 1 for most features (since we
    # already standardised with full-dataset stats), but capturing the
    # exact value handles any feature that happens to be constant on
    # the train fold.
    train_post_std = train.X.std(dim=0).clamp_min(1e-6)

    return TDCDataset(
        name=task_name,
        train=train,
        val=val,
        test=test,
        feature_names=list(X_std.columns),
        feature_mean=torch.tensor(feature_mean.values, dtype=torch.float32),
        feature_std=torch.tensor(feature_std.values, dtype=torch.float32),
        train_post_std=train_post_std,
    )


# ---------- DataLoader helpers --------------------------------------------------


def tdc_loaders(
    train: TDCSplit,
    val: TDCSplit,
    test: TDCSplit | None = None,
    batch_size: int = 64,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader | None]:
    g = torch.Generator().manual_seed(seed)
    # ``drop_last=False`` so very small labelled budgets (n_labeled < batch_size)
    # still produce one (partial) batch per epoch instead of zero.
    train_loader = DataLoader(
        TensorDataset(train.X, train.y),
        batch_size=batch_size, shuffle=True, generator=g, drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(val.X, val.y), batch_size=batch_size, shuffle=False
    )
    test_loader = (
        DataLoader(TensorDataset(test.X, test.y), batch_size=batch_size, shuffle=False)
        if test is not None
        else None
    )
    return train_loader, val_loader, test_loader


def subsample_train(
    train: TDCSplit, n_labeled: int, seed: int = 0
) -> Tuple[TDCSplit, TDCSplit]:
    """Split the full scaffold-train set into a labelled subset of size
    ``n_labeled`` and an unlabelled pool with the rest. Both come from
    the train split — never the val or test split — so the SSL pool is
    in-distribution train data the model could in principle have been
    given labels for, but wasn't."""
    n = train.X.shape[0]
    if n_labeled > n:
        raise ValueError(f"n_labeled={n_labeled} > train size {n}")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    labeled_idx = perm[:n_labeled]
    unlabeled_idx = perm[n_labeled:]

    labeled = TDCSplit(
        X=train.X[labeled_idx],
        y=train.y[labeled_idx],
        smiles=[train.smiles[i] for i in labeled_idx.tolist()],
    )
    unlabeled = TDCSplit(
        X=train.X[unlabeled_idx],
        y=train.y[unlabeled_idx],  # kept but unused; transductive ignores it
        smiles=[train.smiles[i] for i in unlabeled_idx.tolist()],
    )
    return labeled, unlabeled
