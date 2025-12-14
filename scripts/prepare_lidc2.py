# scripts/prepare_lidc.py
# Prepare 2D slices + masks from already-available paired 3D volumes and 3D masks.
#
# IMPORTANT:
# LIDC-IDRI raw distribution is DICOM + XML; parsing XML to masks is non-trivial.
# This script assumes you already have (or have generated) for each patient:
#   - CT volume as NIfTI (.nii or .nii.gz) in HU
#   - Consensus 3D mask as NIfTI (.nii or .nii.gz) aligned with CT
#
# What this script does:
# 1) Resample CT and mask to 1mm isotropic spacing
# 2) Clip HU to [-1000, 400] and scale to [0,1]
# 3) 3D -> 2D: keep ALL axial slices where mask has >=1 foreground voxel
#    - multi-slice nodules: all slices kept (by definition)
#    - multi-nodule slices: mask is already unioned in the 3D mask
# 4) Lung-field crop (512x512):
#    - build coarse lung mask from HU threshold + morphological cleanup
#    - crop centered on lung-field bounding box center (NOT nodule centroid)
#    - pad if needed
# 5) Save 2D images/masks as PNG (or NPY) + write a JSONL manifest
#
# Output structure:
#   out_root/
#     images/*.png
#     masks/*.png
#     manifest.jsonl
#
# Dependencies:
#   pip install nibabel scipy pillow
# Optional (better resampling):
#   pip install SimpleITK

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_opening, binary_fill_holes, label

import nibabel as nib


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_png(arr01: np.ndarray, path: Path) -> None:
    arr_u8 = (np.clip(arr01, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(arr_u8).save(str(path))


def save_png_mask(mask01: np.ndarray, path: Path) -> None:
    mask_u8 = (np.clip(mask01, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(mask_u8).save(str(path))


def clip_and_scale_hu(vol_hu: np.ndarray, lo: float = -1000.0, hi: float = 400.0) -> np.ndarray:
    vol = np.clip(vol_hu, lo, hi).astype(np.float32)
    vol = (vol - lo) / (hi - lo)
    return np.clip(vol, 0.0, 1.0).astype(np.float32)


def get_spacing_from_affine(affine: np.ndarray) -> Tuple[float, float, float]:
    # voxel sizes from affine (absolute diagonal of rotation-scaling)
    sx = float(np.linalg.norm(affine[:3, 0]))
    sy = float(np.linalg.norm(affine[:3, 1]))
    sz = float(np.linalg.norm(affine[:3, 2]))
    return sx, sy, sz


def resample_to_1mm(ct: nib.Nifti1Image, msk: nib.Nifti1Image) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample CT (linear) and mask (nearest) to 1mm isotropic spacing.
    Prefers SimpleITK if installed; otherwise uses a conservative fallback (no resample).
    """
    try:
        import SimpleITK as sitk

        ct_s = sitk.ReadImage(ct.get_filename()) if ct.get_filename() else sitk.GetImageFromArray(ct.get_fdata().astype(np.float32))
        m_s = sitk.ReadImage(msk.get_filename()) if msk.get_filename() else sitk.GetImageFromArray(msk.get_fdata().astype(np.uint8))

        # If filename isn't available, copy spacing/direction/origin from nibabel header approximations
        # (Best practice: provide files on disk; then SITK reads full metadata.)
        if not ct.get_filename():
            sx, sy, sz = get_spacing_from_affine(ct.affine)
            ct_s.SetSpacing((sx, sy, sz))
        if not msk.get_filename():
            sx, sy, sz = get_spacing_from_affine(msk.affine)
            m_s.SetSpacing((sx, sy, sz))

        target_spacing = (1.0, 1.0, 1.0)

        def _resample(img, is_label: bool):
            orig_spacing = img.GetSpacing()
            orig_size = img.GetSize()
            new_size = [
                int(round(orig_size[i] * (orig_spacing[i] / target_spacing[i])))
                for i in range(3)
            ]

            res = sitk.ResampleImageFilter()
            res.SetOutputSpacing(target_spacing)
            res.SetSize(new_size)
            res.SetOutputDirection(img.GetDirection())
            res.SetOutputOrigin(img.GetOrigin())
            res.SetTransform(sitk.Transform())
            res.SetDefaultPixelValue(0)
            res.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
            return res.Execute(img)

        ct_r = _resample(ct_s, is_label=False)
        m_r = _resample(m_s, is_label=True)

        ct_np = sitk.GetArrayFromImage(ct_r).astype(np.float32)  # z,y,x
        m_np = sitk.GetArrayFromImage(m_r).astype(np.uint8)
        return ct_np, m_np

    except Exception:
        # Fallback: no resampling (still works if your inputs are already 1mm isotropic)
        ct_np = ct.get_fdata().astype(np.float32)
        m_np = msk.get_fdata().astype(np.uint8)
        # nibabel is typically x,y,z; align to z,y,x like SITK for consistent slicing
        if ct_np.ndim == 3:
            ct_np = np.transpose(ct_np, (2, 1, 0))
        if m_np.ndim == 3:
            m_np = np.transpose(m_np, (2, 1, 0))
        return ct_np, m_np


def largest_cc(binary: np.ndarray) -> np.ndarray:
    lbl, n = label(binary)
    if n == 0:
        return binary
    counts = np.bincount(lbl.ravel())
    counts[0] = 0
    k = int(np.argmax(counts))
    return (lbl == k)


def coarse_lung_mask_from_hu(slice_hu: np.ndarray) -> np.ndarray:
    """
    Coarse lung field from HU:
    - threshold (air-ish): HU < -320
    - cleanup + keep largest components
    """
    lung = slice_hu < -320.0
    lung = binary_opening(lung, structure=np.ones((3, 3), dtype=bool))
    lung = binary_closing(lung, structure=np.ones((5, 5), dtype=bool))
    lung = binary_fill_holes(lung)
    lung = largest_cc(lung)
    return lung.astype(bool)


def bbox_from_mask(m: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(m)
    if len(xs) == 0 or len(ys) == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return y0, y1, x0, x1


def center_crop_pad(img: np.ndarray, msk: np.ndarray, cy: int, cx: int, size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop size x size centered at (cy,cx). Pads with zeros if outside bounds.
    """
    H, W = img.shape
    half = size // 2
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half

    out_img = np.zeros((size, size), dtype=img.dtype)
    out_msk = np.zeros((size, size), dtype=msk.dtype)

    sy0 = max(0, y0)
    sy1 = min(H, y1)
    sx0 = max(0, x0)
    sx1 = min(W, x1)

    dy0 = sy0 - y0
    dx0 = sx0 - x0

    out_img[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = img[sy0:sy1, sx0:sx1]
    out_msk[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = msk[sy0:sy1, sx0:sx1]

    return out_img, out_msk


def iter_cases(root: Path, ct_glob: str, mask_glob: str) -> List[Tuple[str, Path, Path]]:
    """
    Finds matching CT and mask files by patient id derived from filename stem.
    Example:
      ct_glob="*ct.nii.gz", mask_glob="*mask.nii.gz"
    Patient id is taken as the leading part before first underscore.
    Adjust logic if needed.
    """
    ct_files = sorted(root.glob(ct_glob))
    mask_files = sorted(root.glob(mask_glob))

    def pid_of(p: Path) -> str:
        return p.name.split("_")[0]

    ct_map: Dict[str, Path] = {pid_of(p): p for p in ct_files}
    m_map: Dict[str, Path] = {pid_of(p): p for p in mask_files}

    pids = sorted(set(ct_map.keys()) & set(m_map.keys()))
    out = []
    for pid in pids:
        out.append((pid, ct_map[pid], m_map[pid]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, type=str, help="Root directory containing CT and mask nifti files")
    ap.add_argument("--out_root", required=True, type=str, help="Output directory for 2D slices and manifest")
    ap.add_argument("--ct_glob", default="*ct.nii.gz", type=str, help="Glob for CT files under in_root")
    ap.add_argument("--mask_glob", default="*mask.nii.gz", type=str, help="Glob for mask files under in_root")
    ap.add_argument("--min_nodule_mm", default=3.0, type=float, help="Assumes your 3D mask already encodes ≥3mm; kept for documentation")
    ap.add_argument("--crop_size", default=512, type=int, help="Crop size (square)")
    ap.add_argument("--hu_lo", default=-1000.0, type=float)
    ap.add_argument("--hu_hi", default=400.0, type=float)
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    img_dir = ensure_dir(out_root / "images")
    msk_dir = ensure_dir(out_root / "masks")
    manifest_path = out_root / "manifest.jsonl"

    cases = iter_cases(in_root, args.ct_glob, args.mask_glob)
    if not cases:
        raise FileNotFoundError(f"No cases found. Check --ct_glob/--mask_glob under {in_root}")

    print(f"Found {len(cases)} cases")
    print(f"Writing to {out_root}")

    # overwrite manifest
    if manifest_path.exists():
        manifest_path.unlink()

    n_slices_total = 0

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for pid, ct_path, mask_path in cases:
            # Load NIfTI
            ct_nii = nib.load(str(ct_path))
            m_nii = nib.load(str(mask_path))

            # Resample to 1mm
            ct_z_y_x, m_z_y_x = resample_to_1mm(ct_nii, m_nii)

            # Mask -> binary
            m_z_y_x = (m_z_y_x > 0).astype(np.uint8)

            # 3D -> 2D selection: keep all axial slices with any foreground voxels
            z_indices = np.where(m_z_y_x.reshape(m_z_y_x.shape[0], -1).sum(axis=1) > 0)[0]
            if len(z_indices) == 0:
                continue

            for z in z_indices:
                slice_hu = ct_z_y_x[z].astype(np.float32)
                slice_m = m_z_y_x[z].astype(np.uint8)

                # HU clip + scale
                slice01 = clip_and_scale_hu(slice_hu, lo=args.hu_lo, hi=args.hu_hi)

                # lung-field crop center (NOT nodule centroid)
                lung = coarse_lung_mask_from_hu(slice_hu)
                bb = bbox_from_mask(lung)
                if bb is None:
                    cy, cx = slice01.shape[0] // 2, slice01.shape[1] // 2
                else:
                    y0, y1, x0, x1 = bb
                    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2

                crop_img, crop_m = center_crop_pad(slice01, slice_m.astype(np.float32), cy, cx, size=int(args.crop_size))
                crop_m = (crop_m > 0.5).astype(np.float32)

                # Save slice
                sid = f"{pid}_z{int(z):04d}"
                img_out = img_dir / f"{sid}.png"
                msk_out = msk_dir / f"{sid}.png"
                save_png(crop_img, img_out)
                save_png_mask(crop_m, msk_out)

                # area for stratification (pixels)
                area = float((crop_m > 0.5).sum())

                rec = {
                    "id": sid,
                    "patient_id": pid,
                    "z_index": int(z),
                    "image": str(img_out.as_posix()),
                    "mask": str(msk_out.as_posix()),
                    "area": area,
                }
                mf.write(json.dumps(rec) + "\n")
                n_slices_total += 1

    print(f"Done. Wrote {n_slices_total} nodule-bearing slices.")
    print(f"Manifest: {manifest_path}")
    print("Next: run scripts/create_folds.py on this manifest.")


if __name__ == "__main__":
    main()
