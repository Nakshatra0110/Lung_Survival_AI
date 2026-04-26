"""
pretrain_gene_autoencoder.py
----------------------------
Trains a bottleneck autoencoder on gene expression data from ALL available
TCGA-LUAD patients (up to 528 primary-tumour samples), then saves the
encoder half for use inside MultimodalSurvivalModel.

Why a separate script?
  The autoencoder only reconstructs gene expression — it never sees survival
  labels or CT scans. That means it can train on the full gene expression
  cohort, not just the 31 patients who also have CT scans and clinical data.
  This sidesteps the small-N problem for the gene compression step entirely.

Run order:
  1. build_multimodal_dataset.py   (creates multimodal_dataset.csv)
  2. pretrain_gene_autoencoder.py  (creates gene_encoder.pth + ae_gene_cols.npy
                                    + ae_gene_mean.npy + ae_gene_std.npy)
  3. train_lung_model.py           (trains survival model using frozen encoder)

Architecture:
  Input  : top-2000 genes by variance (selected from all AE training patients)
  Encoder: 2000 -> 256 -> ReLU -> 32   (latent vector)
  Decoder: 32   -> 256 -> ReLU -> 2000 (reconstruction)
  Loss   : MSE on normalised gene expression
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# ======================
# PATHS + HYPERPARAMETERS
# ======================
GENE_TSV     = "data/gene_expression.tsv"
CLEAN_CSV    = "data/multimodal_dataset_clean.csv"   # for aligning gene cols
OUT_DIR      = "trained_models"
os.makedirs(OUT_DIR, exist_ok=True)

N_INPUT_GENES = 2000     # top-variance genes fed to AE
LATENT_DIM    = 32       # compressed gene representation
HIDDEN_DIM    = 256      # intermediate layer width
BATCH_SIZE    = 32
MAX_EPOCHS    = 100
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
DROPOUT       = 0.2
VAL_FRAC      = 0.15     # fraction of AE patients held out for val
PATIENCE      = 10       # early stopping patience

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ======================
# STEP 1 — LOAD ALL GENE EXPRESSION DATA
# Use the full gene expression TSV, not the 31-patient multimodal CSV.
# Filter to primary tumour samples (-01A, -01B, -01C) only so we don't
# mix tumour and normal tissue expression in the same training set.
# ======================
print("\nLoading gene expression TSV...")
gene_df = pd.read_csv(GENE_TSV, sep="\t", index_col=0)
print(f"  Raw shape (genes x samples): {gene_df.shape}")

# Keep only primary tumour samples (sample type code starts with '01')
tumor_cols = [c for c in gene_df.columns if c.split("-")[3][:2] == "01"]
gene_df    = gene_df[tumor_cols]
print(f"  Primary tumour samples: {len(tumor_cols)}")

# Truncate sample IDs to TCGA-XX-XXXX (3-part)
gene_df.columns = ["-".join(c.split("-")[:3]) for c in gene_df.columns]

# Some patients have multiple primary tumour aliquots (e.g. -01A and -01B).
# Average them so each patient contributes exactly one row.
gene_df = gene_df.T                                  # samples x genes
gene_df = gene_df.groupby(level=0).mean()            # average duplicate patients
print(f"  After dedup (patients x genes): {gene_df.shape}")

# Strip Ensembl version numbers (ENSG00000000003.15 -> ENSG00000000003)
gene_df.columns = gene_df.columns.str.split(".").str[0]

# ======================
# STEP 2 — SELECT TOP-N GENES BY VARIANCE
# Variance is computed across all AE patients (not just the 31 survival
# patients), giving a much more reliable estimate of which genes vary
# meaningfully across the LUAD population.
# ======================
print(f"\nSelecting top {N_INPUT_GENES} genes by variance...")
gene_vals = gene_df.values.astype(np.float32)
variances = gene_vals.var(axis=0)                    # ddof=0, per gene

zero_var = (variances == 0).sum()
print(f"  Zero-variance genes removed: {zero_var}")

top_indices  = np.argsort(variances)[::-1][:N_INPUT_GENES]
top_gene_ids = gene_df.columns[top_indices].tolist()
gene_matrix  = gene_vals[:, top_indices]             # (n_patients, N_INPUT_GENES)

print(f"  Selected gene variance range: "
      f"{variances[top_indices[-1]]:.4f} – {variances[top_indices[0]]:.4f}")

# Save the selected gene IDs — train_lung_model.py uses this list to
# build multimodal_dataset_ae.csv with the same columns in the same order.
np.save(os.path.join(OUT_DIR, "ae_gene_cols.npy"), np.array(top_gene_ids))
print(f"  Saved ae_gene_cols.npy  ({len(top_gene_ids)} gene IDs)")

# ======================
# STEP 3 — NORMALISE (fit on train split only)
# z-score per gene across patients (column-wise), same convention as the
# survival pipeline. Stats are computed on ALL AE patients here because
# the AE's job is reconstruction — it is not predicting survival outcomes,
# so there is no label leakage from using the full cohort for normalisation.
# The survival model will use its own separate normalisation stats computed
# on its own training split (gene_mean.npy / gene_std.npy).
# ======================
ae_gene_mean = gene_matrix.mean(axis=0)              # (N_INPUT_GENES,)
ae_gene_std  = gene_matrix.std(axis=0)               # ddof=0

gene_matrix_norm = (gene_matrix - ae_gene_mean) / (ae_gene_std + 1e-8)

# ae_gene_mean/std are the AE's internal normalisation stats (applied
# across all 528 AE training patients). These are SEPARATE from
# gene_mean.npy / gene_std.npy which train_lung_model.py computes on
# the survival training split only. ae_gene_mean/std are saved for
# reference and reproducibility but are not loaded at inference time.
np.save(os.path.join(OUT_DIR, "ae_gene_mean.npy"), ae_gene_mean)
np.save(os.path.join(OUT_DIR, "ae_gene_std.npy"),  ae_gene_std)
print(f"  Saved ae_gene_mean.npy and ae_gene_std.npy (AE internal stats)")

# ======================
# STEP 4 — TRAIN / VAL SPLIT FOR AE
# ======================
X = torch.tensor(gene_matrix_norm, dtype=torch.float32)
dataset   = TensorDataset(X)
n_val     = max(1, int(len(dataset) * VAL_FRAC))
n_train   = len(dataset) - n_val

torch.manual_seed(42)
train_ds, val_ds = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=len(val_ds), shuffle=False)

print(f"\nAE split — train: {n_train}, val: {n_val}")

# ======================
# STEP 5 — DEFINE AUTOENCODER
# ======================
class GeneAutoencoder(nn.Module):
    """
    Bottleneck autoencoder for gene expression compression.

    Encoder: N_INPUT_GENES -> HIDDEN_DIM -> LATENT_DIM
    Decoder: LATENT_DIM    -> HIDDEN_DIM -> N_INPUT_GENES

    Only the encoder half is used by the survival model.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            # No activation on latent layer — keeps the space unconstrained
            # so downstream linear layers can use the full range
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            # No activation — reconstruction is of z-scored values which
            # can be negative; sigmoid/tanh would clip them
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon  = self.decoder(latent)
        return recon, latent

    def encode(self, x):
        """Encode only — used at inference time."""
        return self.encoder(x)


ae    = GeneAutoencoder(N_INPUT_GENES, HIDDEN_DIM, LATENT_DIM, DROPOUT).to(device)
opt   = optim.Adam(ae.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
loss_fn = nn.MSELoss()

total_params = sum(p.numel() for p in ae.parameters() if p.requires_grad)
print(f"\nAutoencoder parameters: {total_params:,}")
print(f"Parameters per patient: {total_params // n_train:,}")

# ======================
# STEP 6 — TRAINING LOOP
# ======================
print(f"\nTraining autoencoder for up to {MAX_EPOCHS} epochs...")

best_val_loss = float("inf")
patience_ctr  = 0

for epoch in range(MAX_EPOCHS):
    # ---- Train ----
    ae.train()
    train_loss = 0.0
    for (xb,) in train_loader:
        xb = xb.to(device)
        opt.zero_grad()
        recon, _ = ae(xb)
        loss = loss_fn(recon, xb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)
        opt.step()
        train_loss += loss.item()

    # ---- Validate ----
    ae.eval()
    with torch.no_grad():
        (xv,) = next(iter(val_loader))
        xv     = xv.to(device)
        recon_v, _ = ae(xv)
        val_loss   = loss_fn(recon_v, xv).item()

    sched.step(val_loss)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:03d}/{MAX_EPOCHS} | "
              f"Train MSE: {train_loss/len(train_loader):.6f} | "
              f"Val MSE: {val_loss:.6f} | "
              f"LR: {opt.param_groups[0]['lr']:.2e}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_ctr  = 0
        torch.save(ae.encoder.state_dict(),
                   os.path.join(OUT_DIR, "gene_encoder.pth"))
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

print(f"\nBest val MSE: {best_val_loss:.6f}")
print(f"Encoder saved to: {os.path.join(OUT_DIR, 'gene_encoder.pth')}")

# ======================
# STEP 7 — SANITY CHECK: reconstruction quality
# Run the full val set through and report per-gene reconstruction error.
# A well-trained AE should have val MSE close to train MSE (no overfitting).
# ======================
ae.eval()
ae.encoder.load_state_dict(
    torch.load(os.path.join(OUT_DIR, "gene_encoder.pth"), weights_only=True)
)

with torch.no_grad():
    (xv,) = next(iter(val_loader))
    xv = xv.to(device)
    recon_v, latent_v = ae(xv)

recon_np  = recon_v.cpu().numpy()
orig_np   = xv.cpu().numpy()
per_gene_mse = ((recon_np - orig_np) ** 2).mean(axis=0)

print(f"\nReconstruction quality (val set):")
print(f"  Mean per-gene MSE : {per_gene_mse.mean():.6f}")
print(f"  Worst gene MSE    : {per_gene_mse.max():.6f}")
print(f"  Best  gene MSE    : {per_gene_mse.min():.6f}")
print(f"  Latent dim stats  : "
      f"mean={latent_v.cpu().numpy().mean():.4f}, "
      f"std={latent_v.cpu().numpy().std():.4f}")
print("\nAutoencoder pretraining complete.")
print("Next step: run train_lung_model.py")