import os
import numpy as np
import pandas as pd
import torch

# Import updated model class (renamed in train_lung_model.py)
from train_lung_model import MultimodalSurvivalModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# LOAD MODEL + DATA
# ======================
def load_model():
    # Load multimodal_dataset_ae.csv (2000 genes, written by train_lung_model.py).
    # Falls back to clean CSV if AE file is missing.
    if os.path.exists("data/multimodal_dataset_ae.csv"):
        csv_path = "data/multimodal_dataset_ae.csv"
    elif os.path.exists("data/multimodal_dataset_clean.csv"):
        import warnings
        warnings.warn(
            "AE dataset not found — loading full gene set. "
            "Run pretrain_gene_autoencoder.py then train_lung_model.py.",
            UserWarning, stacklevel=2,
        )
        csv_path = "data/multimodal_dataset_clean.csv"
    else:
        raise FileNotFoundError(
            "Neither multimodal_dataset_ae.csv nor "
            "multimodal_dataset_clean.csv found. "
            "Run train_lung_model.py first."
        )

    df = pd.read_csv(csv_path, low_memory=False)

    gene_cols = [col for col in df.columns if col.startswith("ENSG")]
    gene_dim  = len(gene_cols)

    if gene_dim == 0:
        raise ValueError("No ENSG* gene columns found in the dataset CSV.")

    # ------------------------------------------------------------------
    # Fix: load per-gene normalisation stats saved by train_lung_model.py.
    # The original predict.py applied a per-PATIENT z-score (mean/std
    # across genes for one sample), while datasets.py used a per-GENE
    # z-score (mean/std across patients for one gene). These are entirely
    # different operations — the model was trained with one and inferred
    # with the other, so risk scores at inference were nonsense.
    # ------------------------------------------------------------------
    # AE pipeline: load AE normalisation stats (2000-gene space).
    # These are the stats applied BEFORE the frozen encoder runs.
    # gene_mean.npy / gene_std.npy from train_lung_model are for the
    # same 2000-gene space, computed on the 31-patient training split.
    gene_mean = np.load("trained_models/gene_mean.npy").astype(np.float32)
    gene_std  = np.load("trained_models/gene_std.npy").astype(np.float32)

    if gene_mean.shape[0] != gene_dim:
        raise ValueError(
            f"gene_mean has {gene_mean.shape[0]} entries but dataset has "
            f"{gene_dim} gene columns. Re-run train_lung_model.py."
        )

    # --- Clinical features ---
    clin_df = clin_cols = clin_mean = clin_std = None
    clin_csv = "data/clinical_features.csv"
    if os.path.exists(clin_csv):
        clin_df   = pd.read_csv(clin_csv)
        clin_cols = [c for c in clin_df.columns if c != "patient_id"]
        clin_mean = np.load("trained_models/clinical_mean.npy").astype(np.float32)
        clin_std  = np.load("trained_models/clinical_std.npy").astype(np.float32)
    else:
        import warnings
        warnings.warn(
            "clinical_features.csv not found — running without clinical data. "
            "Run build_clinical_features.py to enable the clinical branch.",
            UserWarning, stacklevel=2,
        )

    # clinical_dim derived from loaded columns — no double file read needed
    _clin_dim = len(clin_cols) if clin_cols else 0

    model = MultimodalSurvivalModel(
        gene_dim, clinical_dim=_clin_dim,
    ).to(device)
    model.load_state_dict(
        torch.load("trained_models/best_lung_model.pth", weights_only=False,
                   map_location=device)
    )
    model.eval()

    return model, df, gene_cols, gene_mean, gene_std, clin_df, clin_cols, clin_mean, clin_std


# ======================
# PREDICTION FUNCTION
# ======================
def predict_survival(ct_path, patient_id, model, df, gene_cols, gene_mean, gene_std,
                     clin_df=None, clin_cols=None, clin_mean=None, clin_std=None):
    """
    Parameters
    ----------
    ct_path    : path to a preprocessed (128,128,128) float32 .npy CT volume.
                 Must have been produced by prepare_ct_volumes.py (HU-clipped,
                 rescaled to [0, 1]). No further normalisation is applied here.
    patient_id : TCGA patient ID used to look up gene expression in df.
    model      : loaded MultimodalSurvivalModel in eval mode.
    df         : deduplicated patient DataFrame with ENSG* gene columns.
    gene_cols  : list of ENSG* column names (must match training order).
    gene_mean  : per-gene mean computed on the training split (shape: n_genes,).
    gene_std   : per-gene std  computed on the training split (shape: n_genes,).
    """

    # ---- CT ------------------------------------------------------------
    if not isinstance(ct_path, str):
        raise ValueError("ct_path must be a file path string.")
    if not os.path.exists(ct_path):
        raise FileNotFoundError(f"CT file not found: {ct_path}")

    ct = np.load(ct_path).astype(np.float32)

    if ct.size == 0:
        raise ValueError("CT file is empty.")

    # Fix: removed the second min-max normalisation.
    # prepare_ct_volumes.py already clips HU values and rescales to [0, 1].
    # The original predict.py applied (ct - min) / (max - min) again, making
    # inference inconsistent with training (datasets.py loaded CT as-is after
    # the corrected version; even the old datasets.py used z-score, not min-max).
    if ct.shape != (128, 128, 128):
        raise ValueError(
            f"CT shape must be (128, 128, 128), got {ct.shape}. "
            "Re-run prepare_ct_volumes.py on this scan."
        )

    # Basic sanity: preprocessed volumes should be in [0, 1]
    if ct.min() < -0.1 or ct.max() > 1.1:
        raise ValueError(
            f"CT value range [{ct.min():.3f}, {ct.max():.3f}] is outside [0, 1]. "
            "This scan may not have been preprocessed by prepare_ct_volumes.py."
        )

    ct_tensor = torch.tensor(ct).unsqueeze(0).to(device)   # (1, 128, 128, 128)

    # ---- Genes ---------------------------------------------------------
    rows = df[df["patient_id"] == patient_id]
    if len(rows) == 0:
        raise ValueError(f"Patient '{patient_id}' not found in the dataset.")

    row   = rows.iloc[0]
    genes = row[gene_cols].values.astype(np.float32)

    if np.isnan(genes).any():
        raise ValueError(
            f"Gene data for patient '{patient_id}' contains NaN values."
        )

    # Fix: column-wise z-score using training stats (consistent with datasets.py).
    # Original used per-patient z-score (genes - genes.mean()) / genes.std(),
    # which is a completely different normalisation from what was used at training.
    genes = (genes - gene_mean) / (gene_std + 1e-8)

    genes_tensor = torch.tensor(genes).unsqueeze(0).to(device)  # (1, gene_dim)

    # ---- Clinical features ---------------------------------------------
    clinical_tensor = None
    if clin_df is not None and clin_cols and clin_mean is not None:
        clin_row = clin_df[clin_df["patient_id"] == patient_id]
        if len(clin_row) == 0:
            import warnings
            warnings.warn(
                f"Patient {patient_id} not found in clinical_features.csv. "
                "Using zero clinical vector.",
                UserWarning, stacklevel=2,
            )
            clinical = np.zeros(len(clin_cols), dtype=np.float32)
        else:
            clinical = clin_row.iloc[0][clin_cols].values.astype(np.float32)
        clinical = (clinical - clin_mean) / (clin_std + 1e-8)
        clinical_tensor = torch.tensor(clinical).unsqueeze(0).to(device)

    # ---- Inference -----------------------------------------------------
    with torch.no_grad():
        risk = model(ct_tensor, genes_tensor, clinical_tensor)

    return float(risk.cpu().item())