"""
analysis/e10_matched_adversary.py
Created on October 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""
import re
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis._common import (conditioning_axes, enabled_cohorts, ladder_models,
                              run_analysis, unit_key)
from config.serde import read_config
from data_loader.build_paired_pool import load_pool, load_reports
from Inference import rows as R
from Inference.resume_utils import MissingInput
from linkage.baselines import load_baseline
from linkage.conditioning import load_conditioned, load_conditioned_counts
from linkage.instrument import paired_selective_delta, selective_row
from linkage.retrieval import load_records


_SEX_IN_TEXT = re.compile(r"\b(?:male|female|man|woman|gentleman|lady|boy|girl)\b", re.I)
_AGE_IN_TEXT = re.compile(r"\b\d{1,3}\s*[- ]?\s*(?:year|yr|y/o|yo)\b|\baged?\s*\d{1,3}\b", re.I)
_VIEW_WORDS = re.compile(r"\b(?:anteroposterior|posteroanterior|frontal|portable|lateral|"
                         r"upright|supine)\b", re.I)
_VIEW_ABBREV = re.compile(r"\b(?:AP|PA)\b")

_POLICIES = ["metadata_as_submitted", "metadata_smallest_pool", "model_only", "combined"]


def _aligned(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    common = None
    for df in frames.values():
        ids = set(df["case_id"].astype(str))
        common = ids if common is None else (common & ids)
    common = sorted(common or [])
    if len(common) < 100:
        raise MissingInput(f"only {len(common)} queries are held by every adversary; the "
                           f"matched comparison needs the conditioned, the unconditioned and "
                           f"the metadata records of the same manifest")
    return {k: df.set_index(df["case_id"].astype(str)).loc[common] for k, df in frames.items()}


def run_e10(cfg_path: str, force: bool = False) -> pd.DataFrame:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    budgets = [int(k) for k in cfg["instrument"]["budgets_k"]]
    conjunctions = conditioning_axes(cfg, "e10_matched")
    units = [unit_key(model=m, cohort=c, conj=k)
             for m in ladder_models(cfg) for c in enabled_cohorts(cfg) for k in conjunctions]
    units += [unit_key(textattr=c) for c in enabled_cohorts(cfg)]

    def compute(unit: str) -> List[Dict]:
        p = dict(x.split("-", 1) for x in unit.split("__"))

        if "textattr" in p:
            cohort = p["textattr"]
            reports = load_reports(cfg, cohort, "T0").drop_duplicates("study_id")
            pool = (load_pool(cfg, cohort, split="test", usecols=["study_id", "patient_id"])
                    .drop_duplicates("study_id"))
            keep = reports[reports["study_id"].astype(str)
                           .isin(set(pool["study_id"].astype(str)))]
            if len(keep) < 100:
                raise MissingInput(f"{cohort}: {len(keep)} test studies have a report")
            pid = (pool.set_index(pool["study_id"].astype(str))
                   .reindex(keep["study_id"].astype(str))["patient_id"].astype(str).to_numpy())
            text = keep["report_text"].astype(str)
            ident = {"cohort": cohort, "experiment": "E10", "analysis": "attribute_in_report",
                     "fdr_family": f"E10_textattr_{cohort}"}
            states = {"sex": text.str.contains(_SEX_IN_TEXT),
                      "age": text.str.contains(_AGE_IN_TEXT),
                      "view": text.str.contains(_VIEW_WORDS) | text.str.contains(_VIEW_ABBREV)}
            rows = []
            for attr in ("sex", "age", "view"):
                v = states[attr].to_numpy(dtype=float)
                rows.append(R.metric(v, name="report_states_attribute", cluster_ids=pid,
                                     **{**ident, "covariate": attr}))
            return rows

        model, cohort, conj = p["model"], p["cohort"], p["conj"]
        attributes = cfg["conditioning"]["sets"][conj]
        primary = cfg["baselines"]["metadata_only"]["attributes"]
        metadata_condition = "none" if set(attributes) == set(primary) else conj
        frames = _aligned({
            "meta": load_baseline(cfg, "metadata_only", cohort, "test", metadata_condition),
            "model": load_records(cfg, model, cohort, "T0", "off_shelf", "test"),
            "cond": load_conditioned(cfg, model, cohort, conj),
        })
        meta, mod, cond = frames["meta"], frames["model"], frames["cond"]
        clusters = cond["patient_id"].astype(str).to_numpy()
        m = cond["n_corpus"].to_numpy(dtype=float)
        meta_margin = meta["margin"].to_numpy(dtype=float)
        score = {"metadata_as_submitted": meta_margin,
                 "metadata_smallest_pool": -m + np.nan_to_num(meta_margin),
                 "model_only": mod["margin"].to_numpy(dtype=float),
                 "combined": cond["margin"].to_numpy(dtype=float)}
        correct = {"metadata_as_submitted": meta["correct"].to_numpy(dtype=float),
                   "metadata_smallest_pool": meta["correct"].to_numpy(dtype=float),
                   "model_only": mod["correct"].to_numpy(dtype=float),
                   "combined": cond["correct"].to_numpy(dtype=float)}

        base = {"model": cond["model"].iloc[0], "cohort": cohort, "transform": "T0",
                "variant": "off_shelf", "split": "test", "conditioning": conj,
                "experiment": "E10", "analysis": "matched_adversary"}
        rows: List[Dict] = [R.raw_stat(len(cond), name="n_shared_queries", n=len(cond),
                                       extra=f"median_matching_set={int(np.median(m))}", **base)]
        for pol in tqdm(_POLICIES, desc=f"e10:{model}/{cohort}/{conj}", unit="policy"):
            for k in budgets:
                if k > len(cond):
                    continue
                rows.append(selective_row(score[pol], correct[pol], clusters, k, cfg,
                                          "selective_precision",
                                          {**base, "policy": pol, "case_set": "conjunction_kept"}))
                rows.append(selective_row(score["combined"], correct[pol], clusters, k, cfg,
                                          "selective_precision",
                                          {**base, "policy": pol, "case_set": "model_selected"}))
        for k in budgets:
            if k > len(cond):
                continue
            idx = np.argsort(-score["combined"], kind="stable")[:k]
            rows.append(R.raw_stat(float(np.mean(1.0 / m[idx])), name="chance_precision", n=k,
                                   **{**base, "policy": "uniform_guess",
                                      "case_set": "model_selected", "budget_k": k}))
            for pol in ("metadata_smallest_pool", "model_only"):
                rows.append(paired_selective_delta(
                    score[pol], correct[pol], score["combined"], correct["combined"],
                    clusters, k, cfg,
                    {**base, "policy": "combined", "comparison": f"combined_minus_{pol}",
                     "analysis": "matched_adversary_delta"}))
        counts = load_conditioned_counts(cfg, model, cohort, conj)
        if counts:
            rows.append(R.raw_stat(int(counts["n_singleton_pool"]),
                                   name="singleton_pool_queries",
                                   n=int(counts["n_query_total"]), **base))
        for r in rows:
            r["experiment"] = "E10"
            if not r.get("analysis"):
                r["analysis"] = "matched_adversary"
            r["fdr_family"] = (f"E10_{cohort}_{conj}_{r.get('case_set') or 'na'}_"
                               f"{r.get('metric') or r.get('measure')}")
        return rows

    return run_analysis(cfg, "e10_matched_adversary", units, compute, force=force)
