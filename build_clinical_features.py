"""
build_clinical_features.py
--------------------------
Extracts and encodes clinical features from clinical.tsv for use as a
third input modality alongside CT and gene expression.

Features selected:
  - age_at_diagnosis  : patient age in years (stored as days in TCGA, divided by 365.25)
  - gender            : binary 0=female, 1=male
  - stage_ordinal     : AJCC pathologic stage mapped to 1–7 integer scale
  - pack_years_smoked : cumulative smoking burden (strongest lifestyle predictor)

Why these four?
  - All have established prognostic value in LUAD survival literature
  - They are largely non-collinear (age, sex, tumour burden, smoking are independent axes)
  - T/N/M sub-stages are excluded because they are components of stage (collinear)
  - years_smoked and cigarettes_per_day are excluded (>65% null in the 31-patient cohort)

Nulls are imputed with the median (numeric) or mode (categorical) computed
on the FULL clinical cohort (583 patients), not just the 31 survival patients.
Imputation statistics are saved so predict.py can apply exactly the same
fill values at inference time without re-reading clinical.tsv.

Run order:
  This script runs BEFORE train_lung_model.py.
  Full run order:
    1. build_multimodal_dataset.py
    2. pretrain_gene_autoencoder.py
    3. build_clinical_features.py      <- this script
    4. train_lung_model.py
    5. analyze_model.py
    6. app.py
"""

import os
import numpy as np
import pandas as pd

# ======================
# PATHS
# ======================
CLINICAL_TSV  = "data/clinical.tsv"
MULTIMODAL_CSV = "data/multimodal_dataset.csv"   # to know which patients to keep
OUT_CSV       = "data/clinical_features.csv"
OUT_DIR       = "trained_models"
os.makedirs(OUT_DIR, exist_ok=True)

# ======================
# STAGE ORDINAL MAP
# Coarser groupings handle the "Stage I" / "Stage IA" / "Stage IB"
# inconsistency in TCGA staging annotations.
# ======================
STAGE_MAP = {
    "Stage I":    1,
    "Stage IA":   1,
    "Stage IB":   2,
    "Stage II":   3,
    "Stage IIA":  3,
    "Stage IIB":  4,
    "Stage III":  5,
    "Stage IIIA": 5,
    "Stage IIIB": 6,
    "Stage IV":   7,
}

# ======================
# LOAD + DEDUPLICATE CLINICAL DATA
# ======================
print("Loading clinical data...")
clin = pd.read_csv(CLINICAL_TSV, sep="\t", low_memory=False)
print(f"  Raw rows: {len(clin)}")
clin = clin.drop_duplicates(subset="submitter_id").reset_index(drop=True)
print(f"  After dedup: {len(clin)} unique patients")

# ======================
# ENCODE FEATURES
# ======================

# Age: stored as days in TCGA -> convert to years
clin["age_years"] = pd.to_numeric(
    clin["age_at_diagnosis.diagnoses"], errors="coerce"
) / 365.25

# Gender: binary
clin["gender_enc"] = clin["gender.demographic"].map({"male": 1, "female": 0})

# Stage: ordinal 1-7
clin["stage_ordinal"] = clin["ajcc_pathologic_stage.diagnoses"].map(STAGE_MAP)

# Pack years: direct numeric
clin["pack_years"] = pd.to_numeric(
    clin["pack_years_smoked.exposures"], errors="coerce"
)

CLINICAL_COLS = ["age_years", "gender_enc", "stage_ordinal", "pack_years"]

# ======================
# COMPUTE IMPUTATION STATISTICS ON FULL COHORT
# Using the full 583-patient cohort for imputation fills gives better
# estimates than using only the 31 survival patients.
# ======================
impute_values = {}
for col in CLINICAL_COLS:
    if clin[col].dtype == object or col == "gender_enc":
        fill = clin[col].mode()[0]
    else:
        fill = clin[col].median()
    impute_values[col] = float(fill)
    print(f"  Impute {col}: {fill:.4f}  "
          f"(nulls={clin[col].isnull().sum()}/{len(clin)})")

# Save imputation values so predict.py can apply the same fill
np.save(os.path.join(OUT_DIR, "clinical_impute.npy"),
        np.array([impute_values[c] for c in CLINICAL_COLS], dtype=np.float32))
print(f"\nImputation values saved to {OUT_DIR}/clinical_impute.npy")

# Apply imputation
for col in CLINICAL_COLS:
    clin[col] = clin[col].fillna(impute_values[col])

# ======================
# FILTER TO SURVIVAL PATIENTS AND SAVE
# We save clinical features for ALL patients in multimodal_dataset.csv so
# train_lung_model.py can merge on patient_id without manual alignment.
# ======================
multi = pd.read_csv(MULTIMODAL_CSV, low_memory=False)
multi = multi.drop_duplicates(subset="patient_id").reset_index(drop=True)
survival_ids = set(multi["patient_id"])

out = clin[clin["submitter_id"].isin(survival_ids)][
    ["submitter_id"] + CLINICAL_COLS
].copy()
out = out.rename(columns={"submitter_id": "patient_id"})
out = out.reset_index(drop=True)

print(f"\nClinical features for {len(out)} survival patients:")
for col in CLINICAL_COLS:
    s = out[col]
    print(f"  {col}: min={s.min():.2f}, max={s.max():.2f}, "
          f"mean={s.mean():.2f}, nulls={s.isnull().sum()}")

out.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}  ({out.shape[0]} patients x {len(CLINICAL_COLS)} features)")
print("\nNext step: run train_lung_model.py")