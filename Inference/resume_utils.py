"""
Inference/resume_utils.py
Created on July 14, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import json
import os
from typing import Callable, List, Optional

import numpy as np
import pandas as pd


class MissingInput(Exception):
    pass


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def unit_cache_path(partial_dir: str, group: str, unit: str) -> str:
    return os.path.join(partial_dir, f"{_safe(group)}__{_safe(unit)}.jsonl")


def unit_done(partial_dir: str, group: str, unit: str) -> bool:
    return os.path.exists(unit_cache_path(partial_dir, group, unit))


def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def write_unit_rows(partial_dir: str, group: str, unit: str, rows: List[dict]) -> None:
    os.makedirs(partial_dir, exist_ok=True)
    path = unit_cache_path(partial_dir, group, unit)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=_json_safe) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_group_rows(partial_dir: str, group: str) -> List[dict]:
    rows: List[dict] = []
    if not os.path.isdir(partial_dir):
        return rows
    prefix = f"{_safe(group)}__"
    for name in sorted(os.listdir(partial_dir)):
        if not name.startswith(prefix) or not name.endswith(".jsonl"):
            continue
        with open(os.path.join(partial_dir, name)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def run_units_resumable(
    partial_dir: str,
    group: str,
    units: List[str],
    compute_unit: Callable[[str], List[dict]],
    progress_desc: Optional[str] = None,
    status_dir: Optional[str] = None,
) -> pd.DataFrame:
    from tqdm import tqdm
    from Inference.rows import append_status

    todo = [u for u in units if not unit_done(partial_dir, group, u)]
    if status_dir:
        append_status(status_dir, group,
                      f"{len(todo)} of {len(units)} units to compute "
                      f"({len(units) - len(todo)} already cached)")
    print(f"[{group}] {len(todo)} of {len(units)} units to compute "
          f"({len(units) - len(todo)} already cached).", flush=True)

    for unit in tqdm(units, desc=progress_desc or f"[{group}]", unit="unit"):
        if unit_done(partial_dir, group, unit):
            continue
        print(f"[{group}] START unit {unit}", flush=True)
        if status_dir:
            append_status(status_dir, group, f"START unit {unit}")
        try:
            rows = compute_unit(unit)
        except (MissingInput, FileNotFoundError) as e:
            print(f"[{group}] SKIP  unit {unit}: input missing ({e}); will retry.",
                  flush=True)
            if status_dir:
                append_status(status_dir, group,
                              f"SKIP unit {unit}: input missing; will retry")
            continue
        if not rows:
            print(f"[{group}] EMPTY unit {unit}: produced 0 rows; NOT caching it as "
                  f"done, so it retries once its inputs exist.", flush=True)
            if status_dir:
                append_status(status_dir, group,
                              f"EMPTY unit {unit}: 0 rows, not cached as done")
            continue
        write_unit_rows(partial_dir, group, unit, rows)
        print(f"[{group}] DONE  unit {unit} -> {len(rows)} rows", flush=True)
        if status_dir:
            append_status(status_dir, group, f"DONE unit {unit}: {len(rows)} rows")

    return pd.DataFrame(load_group_rows(partial_dir, group))


def clear_partial_dir(partial_dir: str) -> None:
    import shutil
    shutil.rmtree(partial_dir, ignore_errors=True)
