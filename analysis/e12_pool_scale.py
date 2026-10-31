"""
analysis/e12_pool_scale.py
Created on October 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""
from typing import Dict, List

import pandas as pd

from analysis._common import enabled_cohorts, run_analysis, unit_key
from config.serde import read_config
from Inference import rows as R
from Inference.resume_utils import MissingInput
from linkage.conditioning import load_conditioned
from linkage.instrument import paired_selective_delta, run_instrument
from linkage.union_corpus import load_union_corpus, load_union_corpus_counts


def run_e12(cfg_path: str, force: bool = False) -> pd.DataFrame:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    budgets = [int(k) for k in cfg["instrument"]["budgets_k"]]
    pcfg = cfg["pool_scale"]
    cond = str(pcfg["conditioning"])
    enabled = set(enabled_cohorts(cfg))
    units = [unit_key(model=m, cohort=c, scope=s)
             for s, spec in pcfg["scopes"].items()
             for m in spec["models"]
             for c in spec["query_cohorts"] if c in enabled]

    def compute(unit: str) -> List[Dict]:
        p = dict(x.split("-", 1) for x in unit.split("__"))
        model, cohort, scope = p["model"], p["cohort"], p["scope"]
        union = load_union_corpus(cfg, model, cohort, scope, cond)
        rows = run_instrument(union, cfg, extra_ident={"conditioning": cond,
                                                       "corpus_scope": scope})
        counts = load_union_corpus_counts(cfg, model, cohort, scope, cond)
        ident = {"model": union["model"].iloc[0], "cohort": cohort, "transform": "T0",
                 "variant": "off_shelf", "split": "test", "conditioning": cond,
                 "corpus_scope": scope}
        if counts:
            rows.append(R.raw_stat(int(counts["n_singleton_pool"]),
                                   name="singleton_pool_queries",
                                   n=int(counts["n_query_total"]), **ident))

        small = load_conditioned(cfg, model, cohort, cond)
        common = sorted(set(union["case_id"].astype(str)) & set(small["case_id"].astype(str)))
        if len(common) < 100:
            raise MissingInput(f"{model}/{cohort}/{scope}: only {len(common)} queries are in "
                               f"both the pooled and the test-split records")
        u = union.set_index(union["case_id"].astype(str)).loc[common]
        s = small.set_index(small["case_id"].astype(str)).loc[common]
        rows += run_instrument(s.reset_index(drop=True), cfg,
                               extra_ident={"conditioning": cond, "corpus_scope": "test_split"})
        rows.append(R.raw_stat(len(common), name="n_shared_queries", n=len(common), **ident))
        for k in budgets:
            if k > len(common):
                continue
            rows.append(paired_selective_delta(
                s["margin"].to_numpy(float), s["correct"].to_numpy(float),
                u["margin"].to_numpy(float), u["correct"].to_numpy(float),
                u["patient_id"].astype(str).to_numpy(), k, cfg,
                {**ident, "comparison": f"{scope}_minus_test_split",
                 "analysis": "pool_scale_delta"}))
        for r in rows:
            r["experiment"] = "E12"
            if not r.get("analysis"):
                r["analysis"] = "pool_scale_level"
            r["fdr_family"] = (f"E12_{cohort}_{scope}_{r.get('corpus_scope') or 'na'}_"
                               f"{r.get('metric') or r.get('measure')}")
        return rows

    return run_analysis(cfg, "e12_pool_scale", units, compute, force=force)
