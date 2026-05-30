"""Shared helpers for ROGII."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # optional — only if torch is installed
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def parse_submission_id(sub_id: str) -> tuple[str, int]:
    """'000d7d20_1442' -> ('000d7d20', 1442). Row index is into horizontal_well.csv."""
    well_id, row = sub_id.rsplit("_", 1)
    return well_id, int(row)
