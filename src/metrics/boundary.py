# src/metrics/boundary.py
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def _surface_distances_2d(a: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    a = a.astype(bool)
    b = b.astype(bool)

    if a.sum() == 0 and b.sum() == 0:
        return np.array([], dtype=np.float32)
    if a.sum() == 0 or b.sum() == 0:
        return None

    struct = np.ones((3, 3), dtype=bool)
    a_er = binary_erosion(a, structure=struct, border_value=0)
    b_er = binary_erosion(b, structure=struct, border_value=0)

    a_s = np.logical_xor(a, a_er)
    b_s = np.logical_xor(b, b_er)

    if a_s.sum() == 0 and b_s.sum() == 0:
        return np.array([], dtype=np.float32)
    if a_s.sum() == 0 or b_s.sum() == 0:
        return None

    dt_b = distance_transform_edt(np.logical_not(b_s))
    dt_a = distance_transform_edt(np.logical_not(a_s))

    d_ab = dt_b[a_s]
    d_ba = dt_a[b_s]
    return np.concatenate([d_ab, d_ba]).astype(np.float32)


def hd_hd95_assd_2d(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float]:
    d = _surface_distances_2d(pred, gt)
    if d is None:
        return float("nan"), float("nan"), float("nan")
    if d.size == 0:
        return 0.0, 0.0, 0.0
    hd = float(np.max(d))
    hd95 = float(np.percentile(d, 95))
    assd = float(np.mean(d))
    return hd, hd95, assd
