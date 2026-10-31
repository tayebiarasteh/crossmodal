"""
analysis/e1_linkage.py
Created on August 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""
from itertools import combinations
from typing import Dict, List

import pandas as pd

from analysis._common import enabled_cohorts, panel_models, run_analysis, unit_key
from config.serde import read_config
from linkage.instrument import agreement_rows, run_instrument
from linkage.retrieval import load_records


_TRANSFORM = "T0"
_VARIANT = "off_shelf"
_SPLIT = "test"


def run_e1(cfg_path: str, force: bool = False) -> pd.DataFrame:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    models, cohorts = panel_models(cfg), enabled_cohorts(cfg)

    units = [unit_key(model=m, cohort=c) for m in models for c in cohorts]
    units += [unit_key(model_a=a, model_b=b, cohort=c)
              for c in cohorts for a, b in combinations(models, 2)]

    def compute(unit: str) -> List[Dict]:
        parts = dict(p.split("-", 1) for p in unit.split("__"))
        cohort = parts["cohort"]

        if "model_a" in parts:
            a, b = parts["model_a"], parts["model_b"]
            ra = load_records(cfg, a, cohort, _TRANSFORM, _VARIANT, _SPLIT)
            rb = load_records(cfg, b, cohort, _TRANSFORM, _VARIANT, _SPLIT)
            rows = agreement_rows(ra, rb, cfg, a, b)
            for r in rows:
                r["experiment"] = "E1"
                r["analysis"] = "cross_model_agreement"
                r["fdr_family"] = f"E1_agreement_{cohort}"
            return rows

        model = parts["model"]
        rec = load_records(cfg, model, cohort, _TRANSFORM, _VARIANT, _SPLIT)
        rows = run_instrument(rec, cfg)
        for r in rows:
            r["experiment"] = "E1"
            r["analysis"] = "selective_linkage"
            r["fdr_family"] = f"E1_{cohort}_{r.get('metric') or r.get('measure')}"
        return rows

    return run_analysis(cfg, "e1_linkage", units, compute,
                        fdr_family=["fdr_family"], force=force)
