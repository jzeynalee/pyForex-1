from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch


PathLike = Union[str, Path]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def schema_tagged_path(path: PathLike, schema_version: str) -> Path:
    p = Path(path)
    stem = p.stem

    if stem.endswith('_best'):
        stem = stem[:-5] + f"_{schema_version}_best"
    else:
        stem = f"{stem}_{schema_version}"

    return p.with_name(stem + p.suffix)


def copy_schema_tagged(legacy_path: PathLike, schema_version: str) -> Optional[Path]:
    legacy = Path(legacy_path)
    schema_path = schema_tagged_path(legacy, schema_version)

    try:
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy, schema_path)
        return schema_path
    except Exception:
        return None


def move_with_schema_copy(
    source_path: PathLike,
    legacy_dest_path: PathLike,
    schema_version: str,
) -> Tuple[Path, Optional[Path]]:
    source = Path(source_path)
    legacy_dest = Path(legacy_dest_path)
    schema_dest = schema_tagged_path(legacy_dest, schema_version)

    schema_written = None
    try:
        schema_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, schema_dest)
        schema_written = schema_dest
    except Exception:
        schema_written = None

    legacy_dest.parent.mkdir(parents=True, exist_ok=True)
    source.replace(legacy_dest)
    return legacy_dest, schema_written
