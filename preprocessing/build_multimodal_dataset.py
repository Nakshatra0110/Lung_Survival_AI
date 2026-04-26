import pandas as pd
import numpy as np
import os

ct_dir = "data/ct_processed"
gene_file = "data/gene_expression.tsv"
clinical_file = "data/clinical.tsv"

ct_patients = [f.replace(".npy", "") for f in os.listdir(ct_dir)]
print("CT patients:", len(ct_patients))

gene_df = pd.read_csv(gene_file, sep="\t", index_col=0)
gene_df = gene_df.loc[:, gene_df.columns.str.startswith("TCGA")]

# Fix (Loophole 2): Keep only primary tumour samples (-01*) BEFORE
# truncating the patient ID. Without this, normal tissue samples
# (-11A) and tumour samples (-01A) both truncate to the same patient
# ID. drop_duplicates may then keep the normal sample, meaning the
# model trains on healthy-tissue gene expression instead of tumour.
tumour_mask = gene_df.columns.str.split("-").str[3].str.startswith("01")
n_before_filter = gene_df.shape[1]
gene_df = gene_df.loc[:, tumour_mask]
print(f"Tumour samples selected: {gene_df.shape[1]} / {n_before_filter} total")

gene_df.columns = ["-".join(c.split("-")[:3]) for c in gene_df.columns]
gene_df = gene_df.T

clinical_df = pd.read_csv(clinical_file, sep="\t")

# survival columns (fixed)
death_col = "days_to_death.demographic"
follow_col = "days_to_last_follow_up.diagnoses"
vital_col = "vital_status.demographic"

clinical_df["OS_months"] = clinical_df[death_col].fillna(
    clinical_df[follow_col]
) / 30.4

clinical_df["event"] = clinical_df[vital_col].map(
    {"Dead": 1, "Alive": 0}
)

clinical_df["patient_id"] = clinical_df["submitter_id"]

merged = gene_df.merge(
    clinical_df,
    left_index=True,
    right_on="patient_id",
    how="inner"
)

merged = merged[merged["patient_id"].isin(ct_patients)]

# Fix (Loophole 6): Drop patients where OS_months or event is NaN.
# Occurs when clinical.tsv has neither days_to_death nor
# days_to_last_follow_up filled in. A NaN label silently corrupts
# the Cox loss and poisons all model weights during training.
before_nan = len(merged)
merged = merged.dropna(subset=["OS_months", "event"])
if before_nan != len(merged):
    print(f"Dropped {before_nan - len(merged)} patients with NaN survival labels.")

print(f"Final dataset: {len(merged)} patients with CT + genomics + valid labels")

merged.to_csv("data/multimodal_dataset.csv", index=False)