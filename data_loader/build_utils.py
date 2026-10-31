"""
data_loader/build_utils.py
Created on July 14, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""
import hashlib
import os
import re
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

CORE_COLUMNS: List[str] = [
    "case_id", "dataset", "modality", "stratum", "split",
    "image_key", "image_subdir", "image_relpath",
]


def normalize_split(split) -> np.ndarray:
    mapping = {
        "training": "train", "trainval": "train",
        "valid": "val", "validation": "val", "validate": "val",
        "testing": "test",
    }
    return pd.Series(split).astype(str).str.lower().replace(mapping).values


def read_csv_defensively(path: str, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin-1", **kwargs)


def read_excel_defensively(path: str, **kwargs) -> pd.DataFrame:
    return pd.read_excel(path, **kwargs)


def find_column(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    norm = {str(c).strip().lower().replace(" ", "").replace("[", "").replace("]", ""): c
            for c in df.columns}
    for c in candidates:
        k = str(c).strip().lower().replace(" ", "").replace("[", "").replace("]", "")
        if k in norm:
            return norm[k]
    return None


def new_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def stable_hash_bucket(keys: Iterable, n_buckets: int, salt: str = "") -> np.ndarray:
    out = np.empty(len(list(keys)) if not hasattr(keys, "__len__") else len(keys),
                   dtype=np.int64)
    for i, k in enumerate(keys):
        h = hashlib.md5(f"{salt}::{k}".encode("utf-8")).hexdigest()
        out[i] = int(h[:16], 16) % n_buckets
    return out


def binarize_presence(
    series: pd.Series,
    positive_codes: Iterable,
    negative_codes: Iterable,
    exclude_codes: Iterable = (),
) -> pd.Series:
    num = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=series.index, dtype=float)
    for pc in positive_codes:
        out[out.isna() & (num == pc)] = 1.0
    for nc in negative_codes:
        out[out.isna() & (num == nc)] = 0.0
    _ = list(exclude_codes)
    return out


def cap_per_group(
    df: pd.DataFrame, group_cols, cap: Optional[int], seed: int = 42,
) -> pd.DataFrame:
    if not cap:
        return df
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    rng = np.random.RandomState(seed)
    parts = []
    for _, g in df.groupby(group_cols, dropna=False):
        parts.append(g.sample(n=cap, random_state=rng) if len(g) > cap else g)
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]


def stratified_sample(
    df: pd.DataFrame, strata_col: str, fractions: Sequence[float] = (0.6, 0.1, 0.3),
    seed: int = 42, group_col: Optional[str] = None,
) -> pd.Series:
    rng = np.random.RandomState(seed)
    split = pd.Series(index=df.index, dtype=object)
    for _, idx in df.groupby(strata_col, dropna=False).groups.items():
        sub = df.loc[idx]
        if group_col and group_col in df.columns:
            groups = sub[group_col].astype(str).values
            uniq = np.unique(groups)
            rng.shuffle(uniq)
            n = len(uniq)
            n_train = int(round(fractions[0] * n))
            n_val = int(round(fractions[1] * n))
            assign = {}
            for g in uniq[:n_train]:
                assign[g] = "train"
            for g in uniq[n_train:n_train + n_val]:
                assign[g] = "val"
            for g in uniq[n_train + n_val:]:
                assign[g] = "test"
            split.loc[idx] = [assign[g] for g in groups]
        else:
            arr = np.array(idx)
            rng.shuffle(arr)
            n = len(arr)
            n_train = int(round(fractions[0] * n))
            n_val = int(round(fractions[1] * n))
            split.loc[arr[:n_train]] = "train"
            split.loc[arr[n_train:n_train + n_val]] = "val"
            split.loc[arr[n_train + n_val:]] = "test"
    return split


def balanced_presence_sample(
    df: pd.DataFrame, label_col: str, n_per_class: Optional[int], seed: int = 42,
) -> pd.DataFrame:
    if not n_per_class:
        return df
    return cap_per_group(df, [label_col], n_per_class, seed=seed)


def make_case_ids(dataset: str, keys: Iterable) -> List[str]:
    def _norm(k) -> str:
        return str(k).strip().replace("/", "_").replace("\\", "_")
    return [f"{dataset}__{_norm(k)}" for k in keys]


def assert_unique_case_ids(df: pd.DataFrame) -> None:
    dupes = df["case_id"][df["case_id"].duplicated()]
    if len(dupes):
        raise ValueError(
            f"Duplicate case_id values found ({len(dupes)} total), for example: "
            f"{dupes.unique()[:10].tolist()}"
        )


def finalize_manifest(
    df: pd.DataFrame, label_cols: List[str], meta_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    meta_cols = meta_cols or []
    ordered = [c for c in CORE_COLUMNS if c in df.columns]
    ordered += [c for c in meta_cols if c in df.columns and c not in ordered]
    ordered += [c for c in label_cols if c in df.columns and c not in ordered]
    remaining = [c for c in df.columns if c not in ordered]
    return df[ordered + remaining].reset_index(drop=True)


def write_manifest(df: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path


def box_raw_dir(cfg: dict) -> str:
    return os.path.join(cfg["outputs_root"], "box_raw")


_WS_RUN = re.compile(r"[ \t]{2,}")
_BLANKLINE = re.compile(r"(?m)^[\s.,;:]*$")
_NL_RUN = re.compile(r"\n{3,}")


def normalize_report_text(text) -> str:
    out = "" if text is None else str(text)
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = _WS_RUN.sub(" ", out)
    out = _BLANKLINE.sub("", out)
    return _NL_RUN.sub("\n\n", out).strip()
