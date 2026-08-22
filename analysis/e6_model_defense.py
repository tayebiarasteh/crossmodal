"""
analysis/e6_model_defense.py
Created on August 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from analysis._common import conditioning_axes, enabled_cohorts, run_analysis, unit_key
from config.serde import read_config
from Inference import rows as R
from Inference.stats_utils import paired_bootstrap_statistic
from linkage.conditioning import load_conditioned
from linkage.instrument import run_instrument, selective_precision
from linkage.retrieval import load_records


_VARIANTS = ["off_shelf", "dp", "dp_inbatch", "plain", "random_disruption"]

_DELTAS = ["dp", "dp_inbatch", "plain", "random_disruption"]


def _paired_delta(a: pd.DataFrame, b: pd.DataFrame, cfg: dict, k: int,
                  ident: Dict) -> List[Dict]:
    common = sorted(set(a["case_id"].astype(str)) & set(b["case_id"].astype(str)))
    if len(common) < 100:
        raise ValueError(
            f"the two releases share only {len(common)} query images, and they score the same "
            f"cohort against the same corpus, so they should share all of them. Check that both "
            f"variants were re-embedded from the same manifest.")
    aa = a.set_index(a["case_id"].astype(str)).loc[common]
    bb = b.set_index(b["case_id"].astype(str)).loc[common]

    def stat(ma, ca, mb, cb):
        return selective_precision(mb, cb, k) - selective_precision(ma, ca, k)

    res = paired_bootstrap_statistic(
        [aa["margin"].to_numpy(float), aa["correct"].to_numpy(float),
         bb["margin"].to_numpy(float), bb["correct"].to_numpy(float)],
        stat, cluster_ids=aa["patient_id"].astype(str).to_numpy(),
        n_boot=int(cfg["stats"]["n_boot"]), return_boot=True)
    from Inference.report_utils import _pack, keep_p
    packed = _pack("v", res["point"], res["std"], res["ci_lower"], res["ci_upper"],
                   res["n"], is_percent=True)
    boot, point = res.get("boot", np.array([])), res["point"]
    if boot.size and np.isfinite(point):
        p = float(np.mean(np.abs(boot - point) >= abs(point)))
        packed["p_raw"] = keep_p(max(p, 1.0 / boot.size))
    else:
        packed["p_raw"] = float("nan")
    return [R._finish_metric(packed, "v", "selective_precision", "percentage_points",
                             R._ident(**{**ident, "budget_k": k}))]


def run_e6(cfg_path: str, force: bool = False) -> pd.DataFrame:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    budgets = [int(k) for k in cfg["instrument"]["budgets_k"]]
    conds = conditioning_axes(cfg, "e6_defense")
    units = [unit_key(cohort=c, variant=v) for c in enabled_cohorts(cfg) for v in _VARIANTS]
    units += [unit_key(cohort=c, delta=f"{v}_vs_off_shelf")
              for c in enabled_cohorts(cfg) for v in _DELTAS]
    units += [unit_key(cohort=c, variant=v, cond=k)
              for c in enabled_cohorts(cfg) for v in _VARIANTS for k in conds]
    units += [unit_key(cohort=c, delta=f"{v}_vs_off_shelf", cond=k)
              for c in enabled_cohorts(cfg) for v in _DELTAS for k in conds]

    def compute(unit: str) -> List[Dict]:
        p = dict(x.split("-", 1) for x in unit.split("__"))
        cohort, cond = p["cohort"], p.get("cond")
        backbone = cfg["heads"]["backbone"]

        def records(variant: str):
            if cond:
                return load_conditioned(cfg, backbone, cohort, cond, variant=variant)
            return load_records(cfg, backbone, cohort, "T0", variant, "test")

        if "delta" in p:
            variant = p["delta"].split("_vs_")[0]
            ident = {"cohort": cohort, "variant": variant,
                     "comparison": f"{variant}_minus_off_shelf",
                     "experiment": "E6", "analysis": "defense_delta",
                     "fdr_family": f"E6_delta_{cohort}" + (f"_{cond}" if cond else "")}
            if cond:
                ident["conditioning"] = cond
            base, other = records("off_shelf"), records(variant)
            rows = []
            for k in budgets:
                rows += _paired_delta(base, other, cfg, k, ident)
            return rows

        variant = p["variant"]
        rec = records(variant)
        rows = run_instrument(rec, cfg, extra_ident={"conditioning": cond} if cond else None)
        fam = f"E6_{cohort}_{cond}" if cond else f"E6_{cohort}"
        for r in rows:
            r["experiment"] = "E6"
            r["analysis"] = "defense_level"
            r["fdr_family"] = f"{fam}_{r.get('metric') or r.get('measure')}"
        return rows

    return run_analysis(cfg, "e6_model_defense", units, compute, force=force)
