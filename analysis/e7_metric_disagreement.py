"""
analysis/e7_metric_disagreement.py
Created on August 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from analysis._common import (conditioning_axes, enabled_cohorts, load_analysis,
                              run_analysis, unit_key)
from Inference.resume_utils import MissingInput
from config.serde import read_config
from Inference import rows as R


def run_e7(cfg_path: str, force: bool = False) -> pd.DataFrame:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    budgets = list(cfg["instrument"]["budgets_k"])
    ops = [""] + list(conditioning_axes(cfg, "e7_ranking"))
    units = [unit_key(cohort=c, op=o, budget=k)
             for c in enabled_cohorts(cfg) for o in ops for k in budgets]

    def compute(unit: str) -> List[Dict]:
        p = dict(x.split("-", 1) for x in unit.split("__"))
        cohort, k, op = p["cohort"], int(p["budget"]), p.get("op", "")
        frames = [load_analysis(cfg, "e1_linkage" if not op else "e2_conditioning")]
        try:
            frames.append(load_analysis(cfg, "e6_model_defense"))
        except MissingInput:
            pass
        e1 = pd.concat(frames, ignore_index=True)
        e1 = e1[(e1["cohort"] == cohort) & (e1["kind"] == "metric")]
        cond = (e1["conditioning"].astype(str) if "conditioning" in e1.columns
                else pd.Series([""] * len(e1), index=e1.index))
        e1 = e1[cond.isin([op]) if op else cond.isin(["", "nan"])]
        if len(e1) == 0:
            return []
        e1 = e1.assign(release=e1["model"].astype(str) + "|" + e1["variant"].astype(str))
        stat = str(cfg["instrument"].get("default_statistic", "margin"))

        r1 = (e1[e1["metric"] == "recall_at_1"].drop_duplicates("release")
              .set_index("release")["mean_pct"])
        spf = e1[(e1["metric"] == "selective_precision")
                 & (pd.to_numeric(e1["budget_k"], errors="coerce") == float(k))]
        if "statistic" in spf.columns:
            spf = spf[spf["statistic"].astype(str).isin([stat, ""])]
        sp = spf.drop_duplicates("release").set_index("release")["mean_pct"]
        common = sorted(set(r1.index) & set(sp.index))
        if len(common) < 3:
            return []
        a, b = r1.loc[common].to_numpy(float), sp.loc[common].to_numpy(float)
        tau, p_tau = stats.kendalltau(a, b)

        ident = {"cohort": cohort, "budget_k": k, "statistic": stat, "conditioning": op,
                 "experiment": "E7", "analysis": "metric_disagreement",
                 "fdr_family": f"E7_{cohort}_{op or 'full'}"}
        out = [R.raw_stat(float(tau), name="kendall_tau_recall_vs_selective",
                          n=len(common), p_raw=float(p_tau), **ident)]

        inversions = []
        for i in range(len(common)):
            for j in range(i + 1, len(common)):
                if np.sign(a[i] - a[j]) != np.sign(b[i] - b[j]) and a[i] != a[j] and b[i] != b[j]:
                    inversions.append(f"{common[i]}|{common[j]}")
        out.append(R.raw_stat(float(len(inversions)), name="ranking_inversions",
                              n=len(common), extra=";".join(inversions[:20]), **ident))
        return out

    return run_analysis(cfg, "e7_metric_disagreement", units, compute, force=force)
