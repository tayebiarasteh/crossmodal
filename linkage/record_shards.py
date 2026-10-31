"""
linkage/record_shards.py
Created on October 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""
import json
import os
import shutil
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_loader.build_utils import read_csv_defensively, write_manifest
from linkage.retrieval import _digest


_BOOKKEEPING = ["kept", "drop_reason"]


def shard_dir(path: str, params: dict) -> str:
    sha = _digest([json.dumps(params, sort_keys=True)])[:12]
    return path.replace(".csv", f"_shards_{sha}")


def _shard_path(sd: str, k: int, keys: List[str]) -> str:
    return os.path.join(sd, f"shard{k:05d}__{_digest(keys)[:12]}.csv")


def records_are_current(path: str, params: dict) -> bool:
    meta = path.replace(".csv", ".buildparams.json")
    if not (os.path.exists(path) and os.path.exists(meta)):
        return False
    with open(meta) as f:
        stored = json.load(f)
    moved = [k for k, v in params.items() if k in stored and stored[k] != v]
    if moved:
        return False
    return True


def write_records(path: str, df: pd.DataFrame, params: dict, counts: dict) -> str:
    if df.empty:
        raise ValueError(f"refusing to write an empty records file to {path}")
    write_manifest(df, path)
    for name, payload in ((".buildparams.json", params), (".counts.json", counts)):
        p = path.replace(".csv", name)
        with open(p + ".tmp", "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(p + ".tmp", p)
    return path


def run_record_shards(path: str, params: dict, keys: List[str], chunk: int,
                      build_chunk: Callable[[int, int], List[Dict]], desc: str,
                      on_status: Optional[Callable[[str], None]] = None) -> pd.DataFrame:
    sd = shard_dir(path, params)
    os.makedirs(sd, exist_ok=True)
    with open(os.path.join(sd, "params.json"), "w") as f:
        json.dump(params, f, indent=2, sort_keys=True)
    n = len(keys)
    n_shards = int(np.ceil(n / chunk)) if n else 0
    frames = []
    for k in tqdm(range(n_shards), desc=f"{desc}:shards", unit="shard"):
        lo, hi = k * chunk, min((k + 1) * chunk, n)
        spath = _shard_path(sd, k, [str(x) for x in keys[lo:hi]])
        if os.path.exists(spath):
            frames.append(read_csv_defensively(
                spath, float_precision="round_trip",
                dtype={"case_id": str, "patient_id": str, "study_id": str,
                       "top1_study_id": str}))
            continue
        rows = build_chunk(lo, hi)
        if not rows:
            raise RuntimeError(f"{desc}: chunk {k} evaluated {hi - lo} queries and returned no row")
        frame = pd.DataFrame(rows)
        missing = [c for c in _BOOKKEEPING if c not in frame.columns]
        if missing:
            raise RuntimeError(f"{desc}: chunk {k} rows are missing {missing}")
        write_manifest(frame, spath)
        frames.append(frame)
        if on_status:
            on_status(f"{desc}: chunk {k + 1}/{n_shards}, {int(frame['kept'].sum())} of "
                      f"{len(frame)} queries kept")
    if not frames:
        raise RuntimeError(f"{desc}: no chunk produced anything")
    return pd.concat(frames, ignore_index=True)


def split_kept(evaluated: pd.DataFrame) -> tuple:
    kept = evaluated[evaluated["kept"].astype(int) == 1].drop(columns=_BOOKKEEPING)
    reasons = (evaluated.loc[evaluated["kept"].astype(int) == 0, "drop_reason"]
               .astype(str).value_counts().to_dict())
    counts = {"n_query_total": int(len(evaluated)), "n_kept": int(len(kept)),
              "n_dropped": int(len(evaluated) - len(kept)),
              "n_singleton_pool": int(reasons.get("singleton_pool", 0)),
              "n_below_min_pool": int(reasons.get("below_min_pool", 0)),
              "n_no_corpus_row": int(reasons.get("no_corpus_row", 0))}
    return kept.reset_index(drop=True), counts


def drop_shards(path: str, params: dict) -> None:
    shutil.rmtree(shard_dir(path, params), ignore_errors=True)
