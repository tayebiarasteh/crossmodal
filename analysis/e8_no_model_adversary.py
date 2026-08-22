"""
analysis/e8_no_model_adversary.py
Created on August 14, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

from typing import Dict, List

import pandas as pd

from analysis._common import enabled_cohorts, run_analysis, unit_key
from config.serde import read_config
from Inference import rows as R
from linkage.baselines import load_baseline, load_baseline_counts
from linkage.instrument import run_instrument


def run_e8(cfg_path: str, force: bool = False) -> pd.DataFrame:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    cohorts = enabled_cohorts(cfg)
    conds = ["none"] + [k for k in cfg["conditioning"]["sets"] if k != "none"]

    units = [unit_key(baseline="metadata_only", cohort=c, cond="none") for c in cohorts]
    units += [unit_key(baseline="txrv_labels", cohort=c, cond=k)
              for c in cohorts for k in conds]

    def compute(unit: str) -> List[Dict]:
        p = dict(x.split("-", 1) for x in unit.split("__"))
        baseline, cohort, cond = p["baseline"], p["cohort"], p["cond"]
        rec = load_baseline(cfg, baseline, cohort, "test", cond)
        rows = run_instrument(rec, cfg, extra_ident={"conditioning": cond,
                                                     "baseline": baseline})
        counts = load_baseline_counts(cfg, baseline, cohort, "test", cond)
        if counts:
            ident = {"model": f"baseline_{baseline}", "cohort": cohort, "transform": "T0",
                     "variant": "baseline", "split": "test", "conditioning": cond,
                     "baseline": baseline}
            rows.append(R.raw_stat(int(counts["n_dropped"]),
                                   name="queries_dropped_below_min_pool",
                                   n=int(counts["n_query_total"]), **ident))
            rows.append(R.raw_stat(int(counts["n_singleton_pool"]),
                                   name="singleton_pool_queries",
                                   n=int(counts["n_query_total"]), **ident))
        for r in rows:
            r["experiment"] = "E8"
            r["analysis"] = "no_model_adversary"
            r["fdr_family"] = f"E8_{cohort}_{cond}_{r.get('metric') or r.get('measure')}"
        return rows

    return run_analysis(cfg, "e8_no_model_adversary", units, compute, force=force)
