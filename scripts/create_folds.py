# scripts/create_folds.py
# Create 5-fold, patient-level splits (train/val/test) from a prepared 2D manifest.
#
# Input:
#   A manifest JSONL produced by scripts/prepare_lidc.py
#   Each line is a dict with at least:
#     {
#       "patient_id": "LIDC-IDRI-XXXX",
#       "image": "path/to/slice.png",
#       "mask":  "path/to/mask.png",
#       "id":    "unique_slice_id",
#       "area":  <optional nodule area in pixels for stratification>
#     }
#
# Output:
#   data/splits/fold_1.json ... fold_5.json
#   Each has keys: train/val/test and each item contains: image, mask, id, patient_id
#
# Notes:
# - Patient-level grouping prevents leakage.
# - Stratification uses per-patient nodule burden (sum of mask area) binned into quantiles.

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import random

import numpy as np


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def group_by_patient(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    g: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        pid = str(it["patient_id"])
        g.setdefault(pid, []).append(it)
    return g


def patient_burden_bins(patient_items: Dict[str, List[Dict[str, Any]]], n_bins: int = 5) -> Dict[str, int]:
    """
    Build a stratification label per patient based on total mask area across slices.
    If 'area' absent, uses number of slices as a proxy.
    """
    pids = sorted(patient_items.keys())
    burdens = []
    for pid in pids:
        its = patient_items[pid]
        if "area" in its[0]:
            b = float(sum(float(x.get("area", 0.0)) for x in its))
        else:
            b = float(len(its))
        burdens.append(b)

    burdens_np = np.array(burdens, dtype=np.float32)
    # Quantile bins -> labels 0..n_bins-1
    # Avoid duplicate edges by using unique quantiles
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(burdens_np, qs)
    edges = np.unique(edges)
    if len(edges) <= 2:
        # no variability; all same label
        return {pid: 0 for pid in pids}

    labels = {}
    for pid, b in zip(pids, burdens_np):
        # place b into bins [edges[i], edges[i+1])
        lab = int(np.searchsorted(edges[1:-1], b, side="right"))
        labels[pid] = lab
    return labels


def stratified_kfold_patients(pids: List[str], labels: Dict[str, int], k: int, seed: int) -> List[List[str]]:
    """
    Pure-numpy stratified k-fold split on patient IDs.
    Returns folds: list of k lists of patient IDs.
    """
    rng = random.Random(seed)
    # group pids by label
    by_lab: Dict[int, List[str]] = {}
    for pid in pids:
        by_lab.setdefault(int(labels.get(pid, 0)), []).append(pid)
    for lab in by_lab:
        rng.shuffle(by_lab[lab])

    folds: List[List[str]] = [[] for _ in range(k)]
    # round-robin allocate per label
    for lab, lab_pids in by_lab.items():
        for i, pid in enumerate(lab_pids):
            folds[i % k].append(pid)

    # shuffle each fold for nicer distribution
    for f in folds:
        rng.shuffle(f)
    return folds


def split_train_val(pids: List[str], labels: Dict[str, int], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    """
    Stratified train/val split of patient IDs.
    """
    rng = random.Random(seed)
    by_lab: Dict[int, List[str]] = {}
    for pid in pids:
        by_lab.setdefault(int(labels.get(pid, 0)), []).append(pid)
    for lab in by_lab:
        rng.shuffle(by_lab[lab])

    train, val = [], []
    for lab, lab_pids in by_lab.items():
        n = len(lab_pids)
        n_val = max(1, int(round(n * val_ratio))) if n >= 5 else max(1, int(round(n * val_ratio)))
        val.extend(lab_pids[:n_val])
        train.extend(lab_pids[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def pack_items(patient_items: Dict[str, List[Dict[str, Any]]], pids: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for pid in pids:
        for it in patient_items[pid]:
            out.append(
                {
                    "image": it["image"],
                    "mask": it["mask"],
                    "id": it.get("id", ""),
                    "patient_id": it["patient_id"],
                }
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=str, help="Path to JSONL manifest from prepare_lidc.py")
    ap.add_argument("--outdir", required=True, type=str, help="Directory to write fold_*.json files")
    ap.add_argument("--k", default=5, type=int, help="Number of folds")
    ap.add_argument("--seed", default=1234, type=int, help="Split seed")
    ap.add_argument("--val_ratio", default=0.10, type=float, help="Validation ratio from non-test patients")
    args = ap.parse_args()

    items = read_jsonl(args.manifest)
    patient_items = group_by_patient(items)
    pids = sorted(patient_items.keys())

    labels = patient_burden_bins(patient_items, n_bins=5)
    folds = stratified_kfold_patients(pids, labels, k=args.k, seed=args.seed)

    outdir = ensure_dir(args.outdir)

    for fold_idx in range(args.k):
        test_pids = folds[fold_idx]
        trainval_pids = [pid for i, f in enumerate(folds) if i != fold_idx for pid in f]

        train_pids, val_pids = split_train_val(trainval_pids, labels, val_ratio=args.val_ratio, seed=args.seed + fold_idx + 1)

        fold_json = {
            "fold": fold_idx + 1,
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "train_patients": train_pids,
            "val_patients": val_pids,
            "test_patients": test_pids,
            "train": pack_items(patient_items, train_pids),
            "val": pack_items(patient_items, val_pids),
            "test": pack_items(patient_items, test_pids),
        }

        out_path = outdir / f"fold_{fold_idx + 1}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fold_json, f, indent=2)

        print(f"[fold {fold_idx+1}] train_patients={len(train_pids)} val_patients={len(val_pids)} test_patients={len(test_pids)} -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
