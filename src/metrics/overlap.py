# src/metrics/overlap.py
from __future__ import annotations

import numpy as np


def _bin(x: np.ndarray, thr: float = 0.5) -> np.ndarray:
    if x.dtype.kind == "b":
        return x
    return (x >= thr)


def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    p = _bin(pred)
    g = _bin(gt)
    tp = np.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    return float((2.0 * tp + eps) / (denom + eps))


def iou_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    p = _bin(pred)
    g = _bin(gt)
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, np.logical_not(g)).sum()
    fn = np.logical_and(np.logical_not(p), g).sum()
    return float((tp + eps) / (tp + fp + fn + eps))


def sensitivity_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    p = _bin(pred)
    g = _bin(gt)
    tp = np.logical_and(p, g).sum()
    fn = np.logical_and(np.logical_not(p), g).sum()
    return float((tp + eps) / (tp + fn + eps))


def ppv_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    p = _bin(pred)
    g = _bin(gt)
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, np.logical_not(g)).sum()
    return float((tp + eps) / (tp + fp + eps))
