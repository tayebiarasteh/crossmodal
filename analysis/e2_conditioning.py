"""
analysis/e2_conditioning.py
Created on August 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

from typing import Dict, List

import pandas as pd

from analysis._common import (conditioning_axes, enabled_cohorts, panel_models,
                              run_analysis, unit_key)
from config.serde import read_config
from Inference import rows as R
from linkage.conditioning import load_conditioned, load_conditioned_counts
from linkage.instrument import agreement_rows, run_instrument


def run_e2(cfg_path: str, force: bool = False) -> pd.DataFrame:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    models, cohorts = panel_models(cfg), enabled_cohorts(cfg)
    conds = list(cfg["conditioning"]["sets"].keys()) + ["hard_negative"]
    units = [unit_key(model=m, cohort=c, cond=k)
             for m in models for c in cohorts for k in conds]

    for level in conditioning_axes(cfg, "e2_agreement"):
        units += [unit_key(model_a=a, model_b=b, cohort=c, cond=level)
                  for c in cohorts
                  for i, a in enumerate(models) for b in models[i + 1:]]

    def compute(unit: str) -> List[Dict]:
        p = dict(x.split("-", 1) for x in unit.split("__"))
        if "model_a" in p:
            ra = load_conditioned(cfg, p["model_a"], p["cohort"], p["cond"])
            rb = load_conditioned(cfg, p["model_b"], p["cohort"], p["cond"])
            rows = agreement_rows(ra, rb, cfg, p["model_a"], p["model_b"])
            for r in rows:
                r["experiment"] = "E2"
                r["analysis"] = "cross_model_agreement"
                r["conditioning"] = p["cond"]
                r["fdr_family"] = f"E2_agreement_{p['cohort']}_{p['cond']}"
            return rows

        rec = load_conditioned(cfg, p["model"], p["cohort"], p["cond"])
        rows = run_instrument(rec, cfg, extra_ident={"conditioning": p["cond"]})
        counts = load_conditioned_counts(cfg, p["model"], p["cohort"], p["cond"])
        if counts:
            ident = {"model": rec["model"].iloc[0], "cohort": p["cohort"], "transform": "T0",
                     "variant": "off_shelf", "split": "test", "conditioning": p["cond"]}
            rows.append(R.raw_stat(int(counts["n_dropped"]),
                                   name="queries_dropped_below_min_pool",
                                   n=int(counts["n_query_total"]), **ident))
            rows.append(R.raw_stat(int(counts["n_singleton_pool"]),
                                   name="singleton_pool_queries",
                                   n=int(counts["n_query_total"]), **ident))
        for r in rows:
            r["experiment"] = "E2"
            r["analysis"] = "conditioned_linkage"
            r["fdr_family"] = f"E2_{p['cohort']}_{p['cond']}_{r.get('metric') or r.get('measure')}"
        return rows

    return run_analysis(cfg, "e2_conditioning", units, compute, force=force)
