import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from predict import load_model, predict_survival

# ======================
# LOAD MODEL + DATA
# ======================
model, df, gene_cols, gene_mean, gene_std, \
    clin_df, clin_cols, clin_mean, clin_std = load_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ct_dir = "data/ct_processed"

# Fix: removed dead matplotlib import (was imported but never used)

# load_model() reads multimodal_dataset_clean.csv, which is already
# deduplicated by train_lung_model.py. No further deduplication needed.
unique_patients = df["patient_id"].unique()

print(f"Total unique patients: {len(unique_patients)}")

risks       = []
patient_ids = []
skipped     = []
failed      = []

# ======================
# LOOP THROUGH PATIENTS
# Fix: added tqdm progress bar so long runs are visible.
# Fix: track skipped/failed separately to report coverage at the end.
# ======================
for patient_id in tqdm(unique_patients, desc="Scoring patients"):

    ct_path = os.path.join(ct_dir, f"{patient_id}.npy")

    if not os.path.exists(ct_path):
        skipped.append(patient_id)
        continue

    try:
        risk = predict_survival(ct_path, patient_id, model, df, gene_cols, gene_mean, gene_std,
                                    clin_df, clin_cols, clin_mean, clin_std)
        risks.append(risk)
        patient_ids.append(patient_id)

    except Exception as e:
        failed.append((patient_id, str(e)))

# ======================
# COVERAGE REPORT
# Fix: warn loudly if a significant fraction of patients could not be scored,
# because a biased risk_distribution.npy will break app.py's calibration.
# ======================
total       = len(unique_patients)
n_scored    = len(risks)
n_skipped   = len(skipped)
n_failed    = len(failed)
coverage    = n_scored / total if total > 0 else 0.0

print(f"\n📋 Coverage Report:")
print(f"   Scored  : {n_scored} / {total}  ({coverage:.1%})")
print(f"   Skipped (CT missing): {n_skipped}")
print(f"   Failed  (errors)    : {n_failed}")

if n_skipped:
    print(f"\n⚠️  Patients with missing CT:")
    for pid in skipped:
        print(f"   {pid}")

if n_failed:
    print(f"\n❌ Patients with errors:")
    for pid, err in failed:
        print(f"   {pid}: {err}")

# Fix: raise if coverage is too low to produce a meaningful risk distribution.
# A distribution built from <50% of patients will give unreliable thresholds
# in app.py's calibration step.
if coverage < 0.5:
    raise RuntimeError(
        f"Only {coverage:.1%} of patients scored successfully. "
        "risk_distribution.npy would be unreliable. Fix data paths first."
    )

if n_scored == 0:
    raise ValueError("No valid risks computed. Check your data paths.")

# ======================
# CONVERT TO ARRAY
# ======================
risks       = np.array(risks, dtype=np.float32)
patient_ids = np.array(patient_ids)

# ======================
# PRINT STATS
# ======================
print("\n📊 Risk Statistics:")
print(f"   N        : {len(risks)}")
print(f"   Min      : {risks.min():.4f}")
print(f"   Max      : {risks.max():.4f}")
print(f"   Mean     : {risks.mean():.4f}")
print(f"   Std      : {risks.std():.4f}")
print(f"   P33      : {np.percentile(risks, 33):.4f}")
print(f"   P66      : {np.percentile(risks, 66):.4f}")

# ======================
# SAVE FILES
# Fix: also save patient_ids alongside risks so downstream code can
# cross-reference scores with clinical data (e.g. Kaplan-Meier plots).
# Fix: save to data/ subdirectory, not the project root, to keep things tidy.
# app.py must match this path.
# ======================
os.makedirs("data", exist_ok=True)

risk_path = "data/risk_distribution.npy"
ids_path  = "data/risk_patient_ids.npy"

np.save(risk_path, risks)
np.save(ids_path,  patient_ids)

print(f"\n✅ Risk distribution saved  : {os.path.abspath(risk_path)}")
print(f"✅ Patient ID index saved   : {os.path.abspath(ids_path)}")