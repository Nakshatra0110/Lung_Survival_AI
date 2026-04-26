"""
convert_ct_to_numpy.py
----------------------
Converts raw TCGA DICOM CT series to oriented, spacing-aware .npy volumes.

Fixes vs original:
  1. Deterministic series selection — all series UIDs enumerated; the one
     with the most slices is used. Original used os.walk order which is
     filesystem-dependent and non-reproducible across machines.
  2. Orientation standardisation — every volume reoriented to LPS standard
     before any further processing. Ensures axis 0 is always the same
     anatomical direction regardless of how the scanner stored it.
  3. Voxel spacing saved — physical spacing (mm per voxel) saved alongside
     the volume as <patient>.spacing.npy. prepare_ct_volumes.py reads this
     to resample to isotropic voxels, preserving physical scale.
  4. Error isolation — per-patient failures logged and skipped, not crashed.
"""

import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# ======================
# PATHS
# ======================
INPUT_DIR  = "data/TCGA-LUAD/manifest-1Rd7jPNd5199284876140322680/TCGA-LUAD"
OUTPUT_DIR = "data/ct_volumes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LPS: x=Left, y=Posterior, z=Superior.
# Standard for most medical imaging tools and libraries.
TARGET_ORIENTATION = "LPS"


# ======================
# HELPERS
# ======================
def select_best_series(patient_dir):
    """
    Return (dicom_filenames, series_uid) for the series with the most slices,
    or (None, None) if no DICOM series is found anywhere under patient_dir.

    The largest series is almost always the primary diagnostic acquisition.
    Smaller series are typically scouts, localisers, or thin-slab reformats.
    """
    reader     = sitk.ImageSeriesReader()
    all_series = []   # list of (n_slices, filenames, uid)

    for root, _, _ in os.walk(patient_dir):
        series_ids = reader.GetGDCMSeriesIDs(root)
        for uid in series_ids:
            filenames = reader.GetGDCMSeriesFileNames(root, uid)
            if len(filenames) > 0:
                all_series.append((len(filenames), list(filenames), uid))

    if not all_series:
        return None, None

    # Deterministic: sort descending by slice count, take first
    all_series.sort(key=lambda x: x[0], reverse=True)
    n_slices, filenames, uid = all_series[0]
    return filenames, uid


def load_and_orient(dicom_filenames):
    """
    Load a DICOM series and reorient to TARGET_ORIENTATION.
    DICOMOrient performs a lossless axis permutation — no interpolation,
    no resampling, no change to voxel values.
    """
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_filenames)
    image = reader.Execute()
    image = sitk.DICOMOrient(image, TARGET_ORIENTATION)
    return image


# ======================
# MAIN LOOP
# ======================
patients  = sorted(os.listdir(INPUT_DIR))   # sorted = reproducible order
n_ok      = 0
n_skipped = 0
failed    = []

print(f"Processing {len(patients)} patients")
print(f"Output: {OUTPUT_DIR}")
print(f"Orientation: {TARGET_ORIENTATION}")
print()

for patient in tqdm(patients, desc="Converting DICOM"):
    patient_path = os.path.join(INPUT_DIR, patient)
    if not os.path.isdir(patient_path):
        continue

    out_vol     = os.path.join(OUTPUT_DIR, f"{patient}.npy")
    out_spacing = os.path.join(OUTPUT_DIR, f"{patient}.spacing.npy")

    # Resume support: skip already-completed patients
    if os.path.exists(out_vol) and os.path.exists(out_spacing):
        n_ok += 1
        continue

    try:
        filenames, series_uid = select_best_series(patient_path)
        if filenames is None:
            tqdm.write(f"  Skip {patient}: no DICOM series found")
            n_skipped += 1
            continue

        image = load_and_orient(filenames)

        # sitk stores (x,y,z); GetArrayFromImage transposes to (z,y,x)
        # which is the standard NumPy medical convention (slices, rows, cols)
        volume  = sitk.GetArrayFromImage(image).astype(np.int16)

        # GetSpacing() returns (x_mm, y_mm, z_mm); reverse to match (z,y,x) array
        spacing = np.array(image.GetSpacing()[::-1], dtype=np.float32)

        if volume.ndim != 3:
            tqdm.write(f"  Skip {patient}: unexpected shape {volume.shape}")
            n_skipped += 1
            continue

        np.save(out_vol,     volume)
        np.save(out_spacing, spacing)
        n_ok += 1

    except Exception as e:
        tqdm.write(f"  Error {patient}: {e}")
        failed.append((patient, str(e)))

# ======================
# SUMMARY
# ======================
print(f"\nConverted : {n_ok}")
print(f"Skipped   : {n_skipped}")
print(f"Errors    : {len(failed)}")
if failed:
    print("Failed patients:")
    for pid, err in failed:
        print(f"  {pid}: {err}")
print("\nNext step: run prepare_ct_volumes.py")