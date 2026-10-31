"""
analysis/e9_openset.py
Created on October 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""
from typing import Dict, List

import pandas as pd

from analysis._common import (conditioning_axes, enabled_cohorts, ladder_models,
                              run_analysis, unit_key)
from config.serde import read_config
from Inference import rows as R
from Inference.resume_utils import MissingInput
from linkage.instrument import paired_selective_delta, run_instrument, run_openset_instrument
from linkage.openset import load_openset, load_openset_counts, load_openset_study_level


def _single_image_side(per_image: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    keys = set(groups["case_id"].astype(str))
    pi = per_image.copy()
    pi["_key"] = pi["study_id"].astype(str) + "|" + pi["view"].astype(str)
    pi = pi[pi["_key"].isin(keys)]
    if pi.empty:
        raise MissingInput("no per-image record belongs to a study-level group; the two "
                           "passes were built from different manifests")
    best = pi.sort_values("margin", ascending=False).drop_duplicates("_key")
    return best.set_index("_key")


def run_e9(cfg_path: str, force: bool = False) -> pd.DataFrame:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    budgets = [int(k) for k in cfg["instrument"]["budgets_k"]]
    conds = conditioning_axes(cfg, "e9_openset")
    units = [unit_key(model=m, cohort=c, cond=k)
             for m in ladder_models(cfg) for c in enabled_cohorts(cfg) for k in conds]
    units += [unit_key(model=m, cohort=c, cond=k, level="studylevel")
              for m in ladder_models(cfg) for c in enabled_cohorts(cfg) for k in conds]

    def compute(unit: str) -> List[Dict]:
        p = dict(x.split("-", 1) for x in unit.split("__"))
        model, cohort, cond = p["model"], p["cohort"], p["cond"]

        if p.get("level") == "studylevel":
            groups = load_openset_study_level(cfg, model, cohort, cond)
            per_image = load_openset(cfg, model, cohort, cond)
            rows = run_instrument(groups, cfg, extra_ident={"conditioning": cond,
                                                            "case_set": "study_level"})
            single = _single_image_side(per_image, groups)
            g = groups.set_index(groups["case_id"].astype(str))
            common = sorted(set(g.index) & set(single.index))
            if len(common) < 10:
                raise MissingInput(f"{cohort}/{cond}: only {len(common)} groups have both a "
                                   f"pooled and a single-image record")
            g, single = g.loc[common], single.loc[common]
            rows += run_instrument(
                single.reset_index(drop=True), cfg,
                extra_ident={"conditioning": cond, "case_set": "study_level_single_image"})
            ident = {"model": g["model"].iloc[0], "cohort": cohort, "transform": "T0",
                     "variant": "off_shelf", "split": "test", "conditioning": cond,
                     "case_set": "study_level", "comparison": "pooled_minus_single_image",
                     "analysis": "study_level_delta"}
            for k in budgets:
                if k > len(common):
                    continue
                rows.append(paired_selective_delta(
                    single["margin"].to_numpy(float), single["correct"].to_numpy(float),
                    g["margin"].to_numpy(float), g["correct"].to_numpy(float),
                    g["patient_id"].astype(str).to_numpy(), k, cfg, ident))
            rows.append(R.raw_stat(len(common), name="n_shared_queries", n=len(common),
                                   extra=f"mean_images_per_group="
                                         f"{float(g['n_images'].mean()):.2f}", **ident))
            for r in rows:
                r["experiment"] = "E9"
                if not r.get("analysis"):
                    r["analysis"] = "study_level_attack"
                r["fdr_family"] = (f"E9_studylevel_{cohort}_{cond}_"
                                   f"{r.get('metric') or r.get('measure')}")
            return rows

        rec = load_openset(cfg, model, cohort, cond)
        rows = run_instrument(rec, cfg, extra_ident={"conditioning": cond,
                                                     "case_set": "openset_kept"})
        rows += run_openset_instrument(rec, cfg, extra_ident={"conditioning": cond})
        counts = load_openset_counts(cfg, model, cohort, cond)
        if counts:
            ident = {"model": rec["model"].iloc[0], "cohort": cohort, "transform": "T0",
                     "variant": "off_shelf", "split": "test", "conditioning": cond}
            rows.append(R.raw_stat(int(counts["n_dropped"]),
                                   name="queries_dropped_below_min_pool",
                                   n=int(counts["n_query_total"]), **ident))
            rows.append(R.raw_stat(int(counts["n_singleton_pool"]),
                                   name="singleton_pool_queries",
                                   n=int(counts["n_query_total"]), **ident))
        for r in rows:
            r["experiment"] = "E9"
            r["analysis"] = "openset_linkage"
            r["fdr_family"] = (f"E9_open_{cohort}_{cond}_{r.get('absent_frac') or 'closed'}_"
                               f"{r.get('metric') or r.get('measure')}")
        return rows

    return run_analysis(cfg, "e9_openset", units, compute, force=force)
