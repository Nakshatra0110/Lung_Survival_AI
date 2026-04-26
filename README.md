# 🫁 LungSurvival AI — Multimodal Lung Cancer Survival Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange?style=for-the-badge)

**A multimodal deep learning framework integrating CT imaging, gene expression, and clinical data for lung cancer survival prediction.**

[Overview](#overview) • [Architecture](#architecture) • [Dataset](#dataset-setup) • [Installation](#installation) • [Pipeline](#running-the-pipeline) • [Web App](#web-application) • [Results](#results) • [Project Structure](#project-structure)

</div>

---

> ⚠️ **Research Disclaimer:** This is a research prototype built on TCGA-LUAD public data. It is **not** a validated clinical diagnostic tool and must **not** be used for treatment or diagnostic decisions.

---

## Overview

LungSurvival AI predicts lung cancer survival by fusing three complementary data modalities through a gated deep learning architecture:

| Modality | Encoder | Output | Status |
|---|---|---|---|
| **CT Scan** (128³ volume) | MedicalNet ResNet10 (layer2 + layer4) | 128-dim | Frozen |
| **Gene Expression** (2,000 genes) | Pretrained Autoencoder → 32-dim | 128-dim | Frozen |
| **Clinical Data** (4 features) | Small MLP | 128-dim | Trainable |

All three branches are combined via a **learned gated fusion** layer and trained with **Cox partial likelihood loss**. The web application provides real-time survival prediction, Kaplan-Meier curves, modality contribution analysis, and PDF report export.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LungSurvival AI                              │
│                                                                     │
│  CT Scan (128³)          Gene Expression        Clinical (4 feat)   │
│       │                   (2,000 values)               │            │
│       ▼                        │                       ▼            │
│  MedicalNet ResNet10           ▼                  MLP (4→32→128)   │
│  ┌─────────────┐        Frozen AE Encoder         Fully Trainable   │
│  │ layer2 ──► pool│     (2000→256→32)                   │            │
│  │    128-dim  │         Trainable Proj                 │            │
│  │ layer4 ──► pool│     (32→128)                        │            │
│  │    512-dim  │              │                         │            │
│  │ concat 640  │              │                         │            │
│  │ proj → 128  │              │                         │            │
│  └─────────────┘              │                         │            │
│  ALL FROZEN (6.27M)           │                         │            │
│       │                       │                         │            │
│       ▼                       ▼                         ▼            │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │            Gated Fusion (sigmoid gates per branch)           │   │
│  │   CT_gate × 128  +  Gene_gate × 128  +  Clin_gate × 128    │   │
│  │                    concat → 384-dim                          │   │
│  │              MLP (384 → 128 → 1)                            │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│                      Risk Score (float)                              │
│                Higher = worse prognosis                              │
└─────────────────────────────────────────────────────────────────────┘
```

### Pipeline Flow

```
Raw Data
   │
   ├── gene_expression.tsv ──► build_multimodal_dataset.py ──► multimodal_dataset.csv
   ├── clinical.tsv        ──►        (filter -01A tumour    ──► multimodal_dataset_clean.csv
   └── DICOM CT files      ──►   samples, drop NaN labels)
          │
          ▼
   convert_ct_to_numpy.py ──► ct_volumes/*.npy
          │
          ▼
   prepare_ct_volumes.py ──► ct_processed/*.npy
   (resample 1mm, HU clip, normalise [0,1], resize 128³)
          │
          ├──────────────────────────────────────────────────────────────┐
          ▼                                                              ▼
   pretrain_gene_autoencoder.py                          build_clinical_features.py
   (528 TCGA patients, 2000 genes)                       (age, sex, stage, pack-years)
   ──► trained_models/gene_encoder.pth                   ──► data/clinical_features.csv
   ──► trained_models/ae_gene_cols.npy
          │                                                              │
          └──────────────────────┬───────────────────────────────────────┘
                                 ▼
                       train_lung_model.py
                       (21 train / 10 val, Cox loss,
                        C-index early stopping)
                       ──► trained_models/best_lung_model.pth
                                 │
                                 ▼
                       analyze_model.py
                       (score all 31 patients)
                       ──► data/risk_distribution.npy
                                 │
                                 ▼
                       streamlit run app.py
                       (Risk prediction, KM curves,
                        gate weights, PDF export)
```

---

## Dataset Setup

> ⚠️ **The dataset is NOT included in this repository** due to file size constraints. You must download all three data types from the TCGA GDC Portal manually.

### Step 1 — Download Gene Expression Data

1. Go to **[GDC Data Portal](https://portal.gdc.cancer.gov/)**
2. Click **Repository** → Filter by:
   - **Project:** `TCGA-LUAD`
   - **Data Category:** `Transcriptome Profiling`
   - **Data Type:** `Gene Expression Quantification`
   - **Workflow Type:** `STAR - Counts` or `HTSeq - FPKM`
3. Add all files to cart → Download as **TSV**
4. Save as `data/gene_expression.tsv`

### Step 2 — Download Clinical Data

1. On the same GDC Portal, go to **Projects → TCGA-LUAD**
2. Click **Clinical** tab → Download **TSV**
3. Save as `data/clinical.tsv`

### Step 3 — Download CT Scans (DICOM)

1. Go to **[TCIA Collections — TCGA-LUAD](https://wiki.cancerimagingarchive.net/display/Public/TCGA-LUAD)**
2. Download the **NBIA Data Retriever** tool
3. Download CT scans for TCGA-LUAD patients
4. Place DICOM folders under `data/raw_ct/TCGA-XX-XXXX/` (one folder per patient)

### Step 4 — Download MedicalNet Pretrained Weights

```powershell
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='TencentMedicalNet/MedicalNet-Resnet10', filename='resnet_10_23dataset.pth', local_dir='trained_models')"
```

### Step 5 — Download ResNet definition file

```powershell
curl -L -o resnet.py https://raw.githubusercontent.com/Tencent/MedicalNet/master/models/resnet.py
```

### Expected Data Directory Structure

```
lung_survival_transformer/
├── data/
│   ├── gene_expression.tsv          ← downloaded from GDC
│   ├── clinical.tsv                 ← downloaded from GDC
│   └── raw_ct/
│       ├── TCGA-38-7271/            ← DICOM folders per patient
│       ├── TCGA-44-2659/
│       └── ...
├── resnet.py                        ← downloaded from MedicalNet
└── trained_models/
    └── resnet_10_23dataset.pth      ← downloaded from HuggingFace (~57.5 MB)
```

---

## Installation

### Requirements

- Python 3.10+
- CUDA-compatible GPU recommended (CPU also works)
- Windows / Linux / macOS

### Install Dependencies

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install streamlit numpy pandas scikit-learn SimpleITK pydicom scikit-image tqdm lifelines matplotlib reportlab huggingface_hub lungmask
```

Or install all at once from requirements:

```powershell
pip install -r requirements.txt
```

### requirements.txt

```
torch>=2.0.0
torchvision>=0.15.0
streamlit>=1.30.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
SimpleITK>=2.3.0
pydicom>=2.4.0
scikit-image>=0.21.0
tqdm>=4.65.0
lifelines>=0.27.0
matplotlib>=3.7.0
reportlab>=4.0.0
huggingface_hub>=0.20.0
lungmask>=0.2.0
```

---

## Running the Pipeline

Run each script **in order**, one at a time. Wait for each to finish before starting the next.

### Full Pipeline (first time setup)

```powershell
# Step 1 — Merge all three datasets into one CSV
python build_multimodal_dataset.py

# Step 2 — Convert DICOM CT scans to NumPy arrays
python convert_ct_to_numpy.py

# Step 3 — Preprocess CT volumes (resample, clip, normalise, resize)
python prepare_ct_volumes.py

# Step 4 — Pretrain gene autoencoder on 528 TCGA patients
python pretrain_gene_autoencoder.py

# Step 5 — Build and encode clinical features
python build_clinical_features.py

# Step 6 — Train the survival model
python train_lung_model.py

# Step 7 — Score all patients and save risk distribution
python analyze_model.py

# Step 8 — Launch the web application
streamlit run app.py
```

### Retraining Only (after code changes, data unchanged)

```powershell
# Delete old incompatible weights first
Remove-Item "trained_models\best_lung_model.pth" -Force
Remove-Item "trained_models\lung_model.pth" -Force

python train_lung_model.py
python analyze_model.py
streamlit run app.py
```

> ⚠️ **Always run `analyze_model.py` after `train_lung_model.py`.** The risk distribution must match the current model or thresholds will be wrong.

---

## Web Application

```powershell
streamlit run app.py
```

The app opens at `http://localhost:8501` and provides:

| Feature | Description |
|---|---|
| **CT Upload** | Upload a `.npy` CT scan — filename must be the TCGA patient ID |
| **CT Slice Viewer** | Axial, coronal, sagittal mid-slices of the uploaded volume |
| **Clinical Profile** | Auto-loaded age, sex, AJCC stage, pack-years with hover tooltips |
| **Risk Prediction** | Low / Moderate / High risk badge with survival index gauge |
| **Gate Weight Chart** | CT vs Gene vs Clinical modality contribution percentages |
| **Risk Distribution** | Cohort histogram with patient marker |
| **Kaplan-Meier Curves** | Estimated survival curves per risk group |
| **PDF Export** | One-page clinical report with all charts and interpretation |

---

## Results

| Metric | Value | Notes |
|---|---|---|
| Training patients | 21 | 70% of 31 total |
| Validation patients | 10 | 30% of 31 total |
| Best C-index | ~0.68 | Limited by N=21 training patients |
| Trainable parameters | ~107K | Out of 6.92M total |
| Frozen parameters | ~6.81M | MedicalNet + Gene AE |
| CT gate weight | ~33% | Roughly equal fusion |
| Gene gate weight | ~34% | Slightly dominant |
| Clinical gate weight | ~33% | Active contribution |

> **Important context:** A C-index of 0.68 is above random (0.5). This is **expected** at N=21 training patients and is a data limitation, not an architectural failure. Published literature achieves C-index 0.65–0.71 with 200–500 matched patients using the same type of architecture.

## Technology Stack

| Component | Technology |
|---|---|
| Deep Learning | PyTorch 2.0+ |
| CT Encoder | MedicalNet ResNet10 (pretrained) |
| Gene Compression | Custom Autoencoder |
| Survival Loss | Cox Partial Likelihood |
| CT Preprocessing | SimpleITK, pydicom, scikit-image |
| Survival Analysis | lifelines |
| Web UI | Streamlit |
| PDF Generation | ReportLab |
| Visualisation | matplotlib |

---

## Key Design Decisions

**Why freeze the gene encoder?**
Only 21 training patients. An unfrozen encoder would memorise those 21 profiles in a few epochs. Freezing preserves knowledge learned from 528 patients during autoencoder pretraining.

**Why multi-scale MedicalNet?**
Single-scale (layer2 only) captures local texture. Adding layer4 captures global context — tumour size relative to anatomy, pleural contact, lymph node patterns. Both scales together give the fusion layer complementary information.

**Why gated fusion?**
With only 21 training patients, no single modality is guaranteed to be informative. Gates let the model learn to trust each modality differently per patient. If CT features are noisy, the CT gate reduces its contribution automatically.

**Why Cox loss instead of classification?**
Survival data is censored — some patients were still alive at last follow-up. Cox loss correctly handles censored observations using only the relative ordering of risk scores, not absolute survival times.

---

## Limitations

- N=21 training patients — insufficient for deep CT feature generalisation
- MedicalNet pretrained on segmentation, not survival prediction
- No external validation cohort (NLST, LIDC-IDRI, or GEO)
- Gene normalisation computed on 21 patients — may not represent population
- No k-fold cross-validation — single split results have high variance

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{lungsurvivalai2026,
  author       = {Nakshatra Gupta},
  title        = {LungSurvival AI: Multimodal Lung Cancer Survival Prediction},
  year         = {2026},
  institution  = {Bennett University, School of CSE\&T},
  note         = {Research Prototype. GitHub repository.}
}
```

---

## References

1. Chen, S., Ma, K., & Zheng, Y. (2019). Med3D: Transfer Learning for 3D Medical Image Analysis. *arXiv:1904.00625*
2. Cox, D. R. (1972). Regression models and life-tables. *Journal of the Royal Statistical Society: Series B*, 34(2), 187–202
3. The Cancer Genome Atlas Research Network (2014). Comprehensive molecular profiling of lung adenocarcinoma. *Nature*, 511, 543–550
4. GDC Data Portal: https://portal.gdc.cancer.gov
5. TCIA TCGA-LUAD CT Collection: https://wiki.cancerimagingarchive.net/display/Public/TCGA-LUAD
6. MedicalNet (Tencent Research): https://github.com/Tencent/MedicalNet

---
Still working to improve it
<div align="center">

**Bennett University | School of CSE&T | 2025–2026**

Made with PyTorch · Streamlit · MedicalNet · TCGA-LUAD

</div>
