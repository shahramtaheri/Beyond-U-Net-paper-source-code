# src/utils/checkpoint.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def save_checkpoint(path: str | Path, state: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))


def load_checkpoint(
    model: nn.Module,
    ckpt_path: str | Path,
    device: torch.device,
    strict: bool = True,
) -> Dict[str, Any]:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=strict)
        return ckpt
    if isinstance(ckpt, dict):
        model.load_state_dict(ckpt, strict=strict)
        return {"model_state": ckpt}
    raise ValueError("Unsupported checkpoint format.")
