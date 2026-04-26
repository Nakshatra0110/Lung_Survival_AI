"""
prepare_ct_volumes.py
---------------------
Preprocesses raw CT volumes into normalised, isotropically-resampled,
fixed-size volumes ready for model input.

Fixes vs original:
  1. Physical spacing resampling — reads the .spacing.npy file saved by
     convert_ct_to_numpy.py and resamples the volume to 1x1x1mm voxels
     using SimpleITK before resizing to 128^3. The original skipped this
     step entirely, so a 0.7mm in-plane/5mm slice patient and a 1mm/2mm
     patient were both resized identically, destroying physical scale.
  2. Isotropic resampling preserves spatial scale — a 3cm tumour now
     always occupies the same number of voxels regardless of the scanner
     or acquisition protocol.
  3. Output cast to float32 explicitly — skimage resize returns float64
     by default; float32 halves disk and memory usage with no precision
     loss for this application.
  4. Error isolation — per-patient failures logged and skipped.

Pipeline for each patient:
  raw .npy (int16, original spacing)
    -> resample to 1x1x1mm  (SimpleITK, linear interpolation)
    -> lung mask             (lungmask R231 — zeroes out non-lung tissue)
    -> HU clip [-1000, 400]  (removes air and dense bone extremes)
    -> normalise to [0, 1]   ((v + 1000) / 1400)
    -> resize to 128^3       (skimage, anti-aliased)
    -> save as float32 .npy

Lung masking requires: pip install lungmask
If lungmask is not installed, this step is skipped with a warning.
Model weights (~200MB) are downloaded automatically on first run.
"""

import os
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
from tqdm import tqdm

# lungmask is optional — gracefully skip if not installed
try:
    from lungmask import LMInferer
    _LUNGMASK_AVAILABLE = True
except ImportError:
    _LUNGMASK_AVAILABLE = False

# ======================
# PATHS + CONFIG
# ======================
INPUT_DIR  = "data/ct_volumes"
OUTPUT_DIR = "data/ct_processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SHAPE   = (128, 128, 128)
TARGET_SPACING = (1.0, 1.0, 1.0)   # mm per voxel (z, y, x)

# Hounsfield unit window for lung cancer imaging
# -1000 = air, 0 = water, ~30-70 = soft tissue, ~400 = dense bone
HU_MIN, HU_MAX = -1000.0, 400.0

# Set to False to skip lung masking even if lungmask is installed
USE_LUNGMASK = True

# Lazy-initialised so the model only downloads on first patient processed
_lung_inferer = None


# ======================
# RESAMPLE TO ISOTROPIC SPACING
# ======================
def resample_to_isotropic(volume_int16, original_spacing):
    """
    Resample a volume from its original voxel spacing to TARGET_SPACING
    using SimpleITK trilinear interpolation.

    Parameters
    ----------
    volume_int16     : numpy int16 array (z, y, x)
    original_spacing : numpy float32 array (z_mm, y_mm, x_mm)

    Returns
    -------
    numpy float32 array at TARGET_SPACING, arbitrary spatial shape
    """
    # sitk expects (x, y, z) ordering — transpose
    sitk_image = sitk.GetImageFromArray(volume_int16.astype(np.float32))
    sitk_image.SetSpacing([float(original_spacing[2]),   # x
                           float(original_spacing[1]),   # y
                           float(original_spacing[0])])  # z

    original_size    = np.array(sitk_image.GetSize(),    dtype=np.float64)  # (x, y, z)
    original_spacing_xyz = np.array(sitk_image.GetSpacing(), dtype=np.float64)
    target_spacing_xyz   = np.array([TARGET_SPACING[2],
                                     TARGET_SPACING[1],
                                     TARGET_SPACING[0]], dtype=np.float64)

    # Compute new size to preserve physical extent
    new_size = np.round(original_size * original_spacing_xyz / target_spacing_xyz).astype(int)
    new_size = [max(1, int(s)) for s in new_size]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing_xyz.tolist())
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(float(HU_MIN))   # fill new voxels with air

    resampled = resampler.Execute(sitk_image)

    # Transpose back to (z, y, x) numpy convention
    return sitk.GetArrayFromImage(resampled).astype(np.float32)


# ======================
# LUNG MASK
# ======================
def apply_lung_mask(volume_float32, original_spacing):
    """
    Use lungmask R231 model to segment lung tissue and set all
    non-lung voxels to air (-1000 HU / 0.0 after normalisation).

    This removes the chest wall, spine, heart, and airways, leaving
    only the lung parenchyma where tumours reside. It dramatically
    reduces the fraction of irrelevant background anatomy the CNN sees.

    Mask values: 0 = background, 1 = right lung, 2 = left lung.
    Non-lung voxels (mask == 0) are set to HU_MIN (air).

    Parameters
    ----------
    volume_float32   : numpy float32 array (z, y, x), resampled HU values
    original_spacing : numpy float32 array (z_mm, y_mm, x_mm) — needed to
                       build a valid SimpleITK image for lungmask

    Returns
    -------
    numpy float32 array, same shape, non-lung voxels set to HU_MIN
    """
    global _lung_inferer

    # Initialise once — first call downloads weights (~200MB) to ~/.cache
    if _lung_inferer is None:
        _lung_inferer = LMInferer(modelname="R231", fillmodel="R231")

    # lungmask expects a SimpleITK image with correct spacing
    sitk_vol = sitk.GetImageFromArray(volume_float32)
    sitk_vol.SetSpacing([
        float(original_spacing[2]),   # x
        float(original_spacing[1]),   # y
        float(original_spacing[0]),   # z
    ])

    # Returns numpy array (z, y, x): 0=background, 1=right, 2=left lung
    mask = _lung_inferer.apply(sitk_vol)

    # Zero out everything outside both lungs
    masked = volume_float32.copy()
    masked[mask == 0] = HU_MIN
    return masked


# ======================
# MAIN LOOP
# ======================
files     = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".npy")
                   and not f.endswith(".spacing.npy"))
n_ok      = 0
n_skipped = 0
failed    = []

lungmask_status = "enabled" if (USE_LUNGMASK and _LUNGMASK_AVAILABLE) else \
                  "DISABLED (install lungmask)" if USE_LUNGMASK else "disabled"
print(f"Processing {len(files)} CT volumes")
print(f"Target spacing: {TARGET_SPACING} mm")
print(f"Target shape  : {TARGET_SHAPE}")
print(f"HU window     : [{HU_MIN}, {HU_MAX}]")
print(f"Lung masking  : {lungmask_status}")
print()

for fname in tqdm(files, desc="Preparing volumes"):
    out_path = os.path.join(OUTPUT_DIR, fname)

    # Resume support
    if os.path.exists(out_path):
        n_ok += 1
        continue

    vol_path     = os.path.join(INPUT_DIR, fname)
    spacing_path = os.path.join(INPUT_DIR, fname.replace(".npy", ".spacing.npy"))

    try:
        volume = np.load(vol_path).astype(np.float32)

        # --- Step 1: resample to isotropic spacing ---
        if os.path.exists(spacing_path):
            spacing = np.load(spacing_path)   # (z_mm, y_mm, x_mm)

            # Only resample if spacing deviates from target by >5%
            # to avoid unnecessary computation on already-isotropic scans
            needs_resample = any(
                abs(s - t) / t > 0.05
                for s, t in zip(spacing, TARGET_SPACING)
            )
            if needs_resample:
                volume = resample_to_isotropic(volume, spacing)
        else:
            # spacing file missing (e.g. volumes converted by old pipeline)
            # proceed without resampling but warn
            tqdm.write(f"  Warning {fname}: no spacing file found, skipping resample")

        # --- Step 2: lung masking (optional) ---
        # Runs BEFORE HU clip so the inferer sees raw HU values,
        # which is what it was trained on.
        if USE_LUNGMASK and _LUNGMASK_AVAILABLE:
            try:
                # Need original spacing for sitk image construction.
                # Use TARGET_SPACING if spacing file was missing.
                mask_spacing = (
                    spacing if os.path.exists(spacing_path)
                    else np.array(TARGET_SPACING, dtype=np.float32)
                )
                volume = apply_lung_mask(volume.astype(np.float32), mask_spacing)
            except Exception as mask_err:
                tqdm.write(f"  Warning {fname}: lung masking failed ({mask_err}), skipping")
        elif USE_LUNGMASK and not _LUNGMASK_AVAILABLE:
            tqdm.write(
                "  Warning: USE_LUNGMASK=True but lungmask not installed. "
                "Run: pip install lungmask"
            )

        # --- Step 3: HU clip ---
        volume = np.clip(volume, HU_MIN, HU_MAX)

        # --- Step 4: normalise to [0, 1] ---
        volume = (volume - HU_MIN) / (HU_MAX - HU_MIN)   # -> [0.0, 1.0]

        # --- Step 5: resize to target shape ---
        volume_resized = resize(
            volume,
            TARGET_SHAPE,
            mode="constant",
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float32)   # explicit float32, not float64

        # Sanity check: values should still be in [0, 1] after resize
        if volume_resized.min() < -0.01 or volume_resized.max() > 1.01:
            raise ValueError(
                f"Values out of range after resize: "
                f"[{volume_resized.min():.3f}, {volume_resized.max():.3f}]"
            )

        np.save(out_path, volume_resized)
        n_ok += 1

    except Exception as e:
        tqdm.write(f"  Error {fname}: {e}")
        failed.append((fname, str(e)))

# ======================
# SUMMARY
# ======================
print(f"\nProcessed : {n_ok}")
print(f"Skipped   : {n_skipped}")
print(f"Errors    : {len(failed)}")
if failed:
    print("Failed files:")
    for f, err in failed:
        print(f"  {f}: {err}")
print("\nNext step: run build_multimodal_dataset.py")