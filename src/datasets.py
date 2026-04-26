import os
import warnings

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# Expected CT volume shape produced by prepare_ct_volumes.py
CT_SHAPE = (128, 128, 128)


class LungSurvivalDataset(Dataset):
    """
    PyTorch Dataset for multimodal lung cancer survival prediction.

    Gene normalisation
    ------------------
    Column-wise z-score (per gene, across patients) is the correct choice
    because Cox models are sensitive to the relative ordering of risk scores
    across patients — keeping genes on a comparable scale matters.

    To avoid data leakage the dataset does NOT compute normalisation stats
    itself. Instead, callers must pass precomputed `gene_mean` and `gene_std`
    (computed on the TRAINING split only) via `set_gene_stats()` before the
    DataLoader starts iterating. train_lung_model.py handles this.

    CT normalisation
    ----------------
    prepare_ct_volumes.py already clips Hounsfield units and rescales to
    [0, 1]. This dataset loads the preprocessed volumes as-is and applies
    NO further normalisation. Any additional normalisation here would
    contradict the preprocessing pipeline and diverge from predict.py.
    """

    def __init__(self, csv_file: str, ct_dir: str, clinical_csv: str | None = None,
                 augment: bool = False):
        df = pd.read_csv(csv_file, low_memory=False)

        # ------------------------------------------------------------------
        # Deduplication safety net.
        # train_lung_model.py deduplicates the raw CSV and writes
        # multimodal_dataset_clean.csv before passing it here, so under
        # normal operation this call is a no-op. It guards against the
        # dataset being constructed directly from the unclean raw CSV.
        # ------------------------------------------------------------------
        before = len(df)
        df = df.drop_duplicates(subset="patient_id").reset_index(drop=True)
        if len(df) != before:
            warnings.warn(
                f"LungSurvivalDataset dropped {before - len(df)} duplicate "
                "patient rows. Pass 'multimodal_dataset_clean.csv' (written "
                "by train_lung_model.py) instead of the raw CSV.",
                UserWarning,
                stacklevel=2,
            )

        # ------------------------------------------------------------------
        # Gene columns
        # ------------------------------------------------------------------
        self.gene_cols = [c for c in df.columns if c.startswith("ENSG")]
        if len(self.gene_cols) == 0:
            raise ValueError(
                "No ENSG* gene columns found in the CSV. "
                "Check that build_multimodal_dataset.py ran correctly."
            )

        # Fill any residual NaNs with 0 (expression not measured -> absent)
        df[self.gene_cols] = df[self.gene_cols].fillna(0.0)

        # ------------------------------------------------------------------
        # Fix 2: Do NOT normalise genes here.
        # The original code normalised the full DataFrame in __init__, which
        # means val-set patient expression values contribute to the mean and
        # std used during training — a form of data leakage.
        #
        # Correct approach: compute stats on training rows only, then apply
        # to both splits. train_lung_model.py calls set_gene_stats() with
        # training-split stats before the DataLoader iterates.
        #
        # gene_mean / gene_std are initialised to None here; __getitem__
        # will raise a clear error if set_gene_stats() was never called.
        # ------------------------------------------------------------------
        self._gene_mean: np.ndarray | None = None
        self._gene_std:  np.ndarray | None = None

        # ------------------------------------------------------------------
        # Fix 3: Validate CT existence up front and drop patients without a
        # matching CT file. Raising FileNotFoundError inside __getitem__
        # crashes the entire DataLoader worker with no recovery.
        # ------------------------------------------------------------------
        self.ct_dir = ct_dir
        valid_mask  = df["patient_id"].apply(
            lambda pid: os.path.exists(os.path.join(ct_dir, f"{pid}.npy"))
        )
        n_missing = (~valid_mask).sum()
        if n_missing > 0:
            missing = df.loc[~valid_mask, "patient_id"].tolist()
            warnings.warn(
                f"{n_missing} patient(s) have no CT file and will be excluded: "
                f"{missing}",
                UserWarning,
                stacklevel=2,
            )
            df = df[valid_mask].reset_index(drop=True)

        if len(df) == 0:
            raise RuntimeError(
                f"No patients remain after CT validation. "
                f"Check that ct_dir='{ct_dir}' contains .npy files."
            )

        # Validate required survival columns
        for col in ("OS_months", "event"):
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV.")

        self.df      = df
        self.augment = augment   # True only for training Subset


        # ------------------------------------------------------------------
        # Clinical features (optional third modality)
        # If clinical_csv is provided, load it and align on patient_id.
        # Patients missing from the clinical CSV get a row of zeros — the
        # model can still train on CT + gene for those patients.
        # ------------------------------------------------------------------
        self._clinical_cols: list[str] = []
        self._clinical_mean: np.ndarray | None = None
        self._clinical_std:  np.ndarray | None = None
        self._clinical_df:   pd.DataFrame | None = None

        if clinical_csv is not None:
            if not os.path.exists(clinical_csv):
                raise FileNotFoundError(
                    f"clinical_csv not found: {clinical_csv}. "
                    "Run build_clinical_features.py first."
                )
            clin = pd.read_csv(clinical_csv)
            self._clinical_cols = [c for c in clin.columns if c != "patient_id"]
            # Align to dataset patients; fill missing with 0
            self._clinical_df = (
                self.df[["patient_id"]]
                .merge(clin, on="patient_id", how="left")
                .fillna(0.0)
                .reset_index(drop=True)
            )

    # ------------------------------------------------------------------
    # Gene normalisation API
    # Call this from train_lung_model.py AFTER computing stats on train
    # indices only, and BEFORE creating the DataLoader.
    # ------------------------------------------------------------------
    def set_gene_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Provide precomputed per-gene mean and std (from training split)."""
        self._gene_mean = mean.astype(np.float32)
        self._gene_std  = std.astype(np.float32)

    def get_raw_gene_matrix(self) -> np.ndarray:
        """
        Return the raw (un-normalised) gene matrix as a (n_patients, n_genes)
        float32 array. Used by train_lung_model.py to compute training stats.
        """
        return self.df[self.gene_cols].values.astype(np.float32)

    def set_clinical_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Provide precomputed per-feature mean and std (from training split)."""
        self._clinical_mean = mean.astype(np.float32)
        self._clinical_std  = std.astype(np.float32)

    def get_raw_clinical_matrix(self) -> np.ndarray | None:
        """Return raw clinical matrix (n_patients, n_features) or None."""
        if self._clinical_df is None or not self._clinical_cols:
            return None
        return self._clinical_df[self._clinical_cols].values.astype(np.float32)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row        = self.df.iloc[idx]
        patient_id = row["patient_id"]

        # ---- CT ---------------------------------------------------------
        ct_path = os.path.join(self.ct_dir, f"{patient_id}.npy")
        ct = np.load(ct_path).astype(np.float32)
        if ct.shape != CT_SHAPE:
                raise ValueError(
                    f"CT for patient {patient_id} has shape {ct.shape}, "
                    f"expected {CT_SHAPE}. Re-run prepare_ct_volumes.py."
                )
        # CT is already float32 in [0,1] from prepare_ct_volumes.py — no renorm needed.

        # ---- Training augmentation (applied only when self.augment=True) ----
        # With only 21 training volumes, augmentation is essential to prevent
        # the 3D CNN from memorising the training set.
        #
        # Augmentations used:
        #   - Random axis flips: left-right and superior-inferior mirrors are
        #     anatomically valid for lung CT (bilateral symmetry).
        #   - Random intensity shift: small additive noise (+/-0.05) simulates
        #     scanner variability in HU calibration.
        #
        # NOT used:
        #   - Rotation: 3D rotation requires expensive resampling and can
        #     introduce interpolation artefacts on a 128^3 grid.
        #   - Zoom/crop: changes apparent tumour size, misleading for survival.
        #   - Augmentation is NEVER applied at val or inference.
        if self.augment:
            # Random flip along each spatial axis independently (50% each)
            for axis in range(3):
                if np.random.rand() > 0.5:
                    ct = np.flip(ct, axis=axis).copy()  # .copy() removes neg stride
            # Random intensity shift in [-0.05, +0.05]
            shift = np.random.uniform(-0.05, 0.05)
            ct    = np.clip(ct + shift, 0.0, 1.0)

        # ---- Genes -----------------------------------------------------
        if self._gene_mean is None or self._gene_std is None:
            raise RuntimeError(
                "Gene normalisation stats have not been set. "
                "Call dataset.set_gene_stats(mean, std) before iterating."
            )

        genes = row[self.gene_cols].values.astype(np.float32)

        # Fix 6: Column-wise z-score using TRAINING stats only (no leakage).
        # ddof=0 is used here (population std) to be consistent with the
        # numpy-based stats computation in train_lung_model.py and predict.py.
        genes = (genes - self._gene_mean) / (self._gene_std + 1e-8)

        # ---- Clinical features (optional) --------------------------------
        if self._clinical_df is not None:
            if self._clinical_mean is None or self._clinical_std is None:
                raise RuntimeError(
                    "Clinical normalisation stats not set. "
                    "Call dataset.set_clinical_stats(mean, std) before iterating."
                )
            clin_row = self._clinical_df.iloc[idx][self._clinical_cols]
            clinical = clin_row.values.astype(np.float32)
            clinical = (clinical - self._clinical_mean) / (self._clinical_std + 1e-8)
        else:
            clinical = np.zeros(0, dtype=np.float32)

        # ---- Survival labels -------------------------------------------
        label = np.float32(row["OS_months"])
        event = np.float32(row["event"])

        return (ct, genes, clinical), (label, event)