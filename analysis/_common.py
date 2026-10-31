"""
analysis/_common.py
Created on August 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""
import os
from typing import Callable, List, Optional

import pandas as pd

from Inference.report_utils import add_fdr
from Inference.resume_utils import MissingInput, run_units_resumable
from Inference.rows import write_rows


def analysis_dir(cfg: dict, name: str) -> str:
    d = os.path.join(cfg["paths"]["analysis_dir"], name)
    os.makedirs(d, exist_ok=True)
    return d


def partial_dir(cfg: dict, name: str) -> str:
    d = os.path.join(analysis_dir(cfg, name), "_partial")
    os.makedirs(d, exist_ok=True)
    return d


def enabled_cohorts(cfg: dict) -> List[str]:
    return [c for c, cc in cfg["cohorts"].items() if cc.get("enabled", True)]


def panel_models(cfg: dict) -> List[str]:
    return list(cfg["models"].keys())


def conditioning_axes(cfg: dict, analysis: str) -> List[str]:
    axes = cfg["conditioning"].get("analysis_axes", {})
    if analysis not in axes:
        raise KeyError(f"conditioning.analysis_axes has no entry for {analysis!r}; "
                       f"it declares {sorted(axes)}.")
    known = set(cfg["conditioning"]["sets"]) | {"hard_negative"}
    bad = [k for k in axes[analysis] if k not in known]
    if bad:
        raise KeyError(f"conditioning.analysis_axes[{analysis!r}] names {bad}, which "
                       f"conditioning.sets does not define.")
    return list(axes[analysis])


def ladder_models(cfg: dict) -> List[str]:
    names = list(cfg["conditioning"]["ladder_models"])
    bad = [m for m in names if m not in cfg["models"]]
    if bad:
        raise KeyError(f"conditioning.ladder_models names {bad}, which are not in the panel.")
    return names


def unit_key(**parts) -> str:
    return "__".join(f"{k}-{v}" for k, v in parts.items() if v not in (None, ""))


def run_analysis(cfg: dict, name: str, units: List[str],
                 compute_unit: Callable[[str], List[dict]],
                 fdr_family: Optional[List[str]] = None,
                 force: bool = False) -> pd.DataFrame:
    from Inference.resume_utils import clear_partial_dir
    pdir = partial_dir(cfg, name)
    if force:
        clear_partial_dir(pdir)
        os.makedirs(pdir, exist_ok=True)

    df = run_units_resumable(pdir, name, units, compute_unit,
                             progress_desc=f"[{name}]",
                             status_dir=cfg["paths"]["status_dir"])
    if df.empty:
        write_rows([], os.path.join(analysis_dir(cfg, name), f"{name}_rows.csv"))
        return df

    method = str(cfg["stats"].get("fdr_method", "fdr_bh"))
    if method != "fdr_bh":
        raise KeyError(f"stats.fdr_method={method!r} is configured and not implemented; "
                       "a configured name that nothing implements must fail loudly.")
    df = add_fdr(df, family_cols=fdr_family or ["fdr_family"])
    alpha = float(cfg["stats"].get("alpha", 0.05))
    if "p_fdr" in df.columns:
        df["significant_fdr05"] = df["p_fdr"] < alpha
    out = os.path.join(analysis_dir(cfg, name), f"{name}_rows.csv")
    write_rows(df.to_dict("records"), out)
    return df


def invalidate_partials(cfg: dict, name: str, contains: str = "") -> int:
    pdir = partial_dir(cfg, name)
    hit = [f for f in sorted(os.listdir(pdir))
           if f.endswith(".jsonl") and (contains in f if contains else True)]
    for f in hit:
        os.remove(os.path.join(pdir, f))
    return len(hit)


def load_analysis(cfg: dict, name: str) -> pd.DataFrame:
    path = os.path.join(analysis_dir(cfg, name), f"{name}_rows.csv")
    if not os.path.exists(path):
        raise MissingInput(f"{name} has not run; its rows are not on disk")
    return pd.read_csv(path, low_memory=False, float_precision="round_trip")
