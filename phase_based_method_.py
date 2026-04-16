# %%

import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from affine import Affine
from skimage.registration import phase_cross_correlation
from scipy import ndimage

# Inputs
chm_path = r""
ndvi_path = r""
output_path = r""

# Temp folder
temp_dir = os.path.join(os.path.dirname(output_path), "phase_debug")
os.makedirs(temp_dir, exist_ok=True)

# Funtions
def save_raster(path, arr, ref_profile, transform):
    profile = ref_profile.copy()
    profile.update(dtype="float32", count=1)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype("float32"), 1)
        dst.transform = transform

def normalize(img):
    img = img.astype("float32")
    img = img - np.nanmean(img)
    std = np.nanstd(img)
    return img if std == 0 else img / std

def preprocess(img):
    img = np.nan_to_num(img, nan=0)

    # Sobel edge detection
    edges = ndimage.sobel(img)

    return normalize(edges)

def apply_window(img):
    win = np.outer(np.hanning(img.shape[0]), np.hanning(img.shape[1]))
    return img * win

# Load NDVI
with rasterio.open(ndvi_path) as ndvi_src:
    ndvi = ndvi_src.read(1)
    ndvi_transform = ndvi_src.transform
    ndvi_crs = ndvi_src.crs
    ndvi_profile = ndvi_src.profile

# Load CHM
with rasterio.open(chm_path) as chm_src:
    chm = chm_src.read(1)
    chm_transform = chm_src.transform
    chm_crs = chm_src.crs

# Resample CHM
chm_reproj = np.empty_like(ndvi, dtype="float32")

reproject(
    source=chm,
    destination=chm_reproj,
    src_transform=chm_transform,
    src_crs=chm_crs,
    dst_transform=ndvi_transform,
    dst_crs=ndvi_crs,
    resampling=Resampling.bilinear,
    dst_nodata=np.nan  
)

save_raster(os.path.join(temp_dir, "01_chm_reprojected.tif"),
            chm_reproj, ndvi_profile, ndvi_transform)

# Clip
valid_mask = (
    np.isfinite(ndvi) &
    np.isfinite(chm_reproj)
)

ndvi_masked = np.where(valid_mask, ndvi, 0)
chm_masked = np.where(valid_mask, chm_reproj, 0)

save_raster(os.path.join(temp_dir, "02_ndvi_masked.tif"),
            ndvi_masked, ndvi_profile, ndvi_transform)

save_raster(os.path.join(temp_dir, "03_chm_masked.tif"),
            chm_masked, ndvi_profile, ndvi_transform)

# Pre-process
ndvi_proc = preprocess(ndvi_masked)
chm_proc = preprocess(chm_masked)

# Edge blur
ndvi_proc = apply_window(ndvi_proc)
chm_proc = apply_window(chm_proc)

save_raster(os.path.join(temp_dir, "04_ndvi_processed.tif"),
            ndvi_proc, ndvi_profile, ndvi_transform)

save_raster(os.path.join(temp_dir, "05_chm_processed.tif"),
            chm_proc, ndvi_profile, ndvi_transform)

# Phase Correlation
shift, error, _ = phase_cross_correlation(
    ndvi_proc,
    chm_proc,
    upsample_factor=20
)

print("Pixel shift (dy, dx):", shift)

# Try flipped version
shift_alt = (-shift[0], -shift[1])

print("Alt shift (dy, dx):", shift_alt)

# Convert pix to M
px = ndvi_transform.a
py = -ndvi_transform.e

print("Meters shift (original):", shift[1]*px, shift[0]*py)
print("Meters shift (alt):", shift_alt[1]*px, shift_alt[0]*py)

final_shift = shift_alt  

# Apply shift to orginal dataset
new_transform = ndvi_transform * Affine.translation(-final_shift[1], -final_shift[0])

# Save shifted CHM
profile = ndvi_profile.copy()
profile.update(dtype="float32", count=1)

with rasterio.open(output_path, "w", **profile) as dst:
    dst.write(chm_reproj.astype("float32"), 1)
    dst.transform = new_transform

print("DONE")
print("Aligned CHM:", output_path)
print("Debug folder:", temp_dir)