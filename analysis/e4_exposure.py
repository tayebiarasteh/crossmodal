"""
analysis/e4_exposure.py
Created on August 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import re
from typing import Dict, List

import numpy as np
import pandas as pd

from analysis._common import (conditioning_axes, enabled_cohorts, ladder_models,
                              panel_models, run_analysis, unit_key)
from config.serde import read_config
from data_loader.build_paired_pool import load_pool, load_reports
from Inference import rows as R
from linkage.conditioning import load_conditioned
from linkage.retrieval import load_records


_MEASURE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:cm|mm)\b", re.I)
_COMPARISON = re.compile(r"\b(compar|prior|previous|interval change|since)\b", re.I)
_DEVICE = re.compile(r"\b(tube|catheter|line|pacemaker|stent|drain|wire|valve)\b", re.I)
_DEIDENT = re.compile(r"_{3,}")
_LATERAL = re.compile(r"\b(left|right|bilateral)\b", re.I)


def _features(text: str) -> Dict[str, float]:
    from data_loader.report_transforms import split_sections
    t = str(text or "")
    words = t.split()
    sections = split_sections(t)
    return {
        "report_tokens": float(len(words)),
        "has_measurement": float(bool(_MEASURE.search(t))),
        "has_comparison": float(bool(_COMPARISON.search(t))),
        "has_device": float(bool(_DEVICE.search(t))),
        "has_laterality": float(bool(_LATERAL.search(t))),
        "n_deident_placeholders": float(len(_DEIDENT.findall(t))),
        "n_sections": float(len([k for k, v in sections.items() if v])),
    }


def run_e4(cfg_path: str, force: bool = False) -> pd.DataFrame:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    findings = list(cfg["labels"]["findings"])
    units = [unit_key(model=m, cohort=c)
             for m in panel_models(cfg) for c in enabled_cohorts(cfg)]
    units += [unit_key(model=m, cohort=c, cond=k)
              for m in ladder_models(cfg) for c in enabled_cohorts(cfg)
              for k in conditioning_axes(cfg, "e4_exposure")]

    def compute(unit: str) -> List[Dict]:
        p = dict(x.split("-", 1) for x in unit.split("__"))
        model, cohort, cond = p["model"], p["cohort"], p.get("cond")
        rec = (load_conditioned(cfg, model, cohort, cond) if cond
               else load_records(cfg, model, cohort, "T0", "off_shelf", "test"))
        reports = load_reports(cfg, cohort, "T0").drop_duplicates("study_id").set_index("study_id")
        pool = load_pool(cfg, cohort, split="test",
                         usecols=["study_id"] + findings).drop_duplicates("study_id")
        npos = pool.set_index(pool["study_id"].astype(str))[findings].apply(
            lambda r: float(np.nansum(r.to_numpy(dtype=float) == 1.0)), axis=1)

        feats = pd.DataFrame([_features(reports["report_text"].get(s, ""))
                              for s in rec["study_id"].astype(str)])
        feats["n_positive_findings"] = [float(npos.get(s, np.nan))
                                        for s in rec["study_id"].astype(str)]
        rec = rec.reset_index(drop=True)
        recovered = rec["correct"].to_numpy() == 1
        n_rec = int(recovered.sum())
        base = {"model": model, "cohort": cohort, "experiment": "E4",
                "analysis": "exposure", "fdr_family": f"E4_{cohort}"}
        if cond:
            base["conditioning"] = cond
            base["fdr_family"] = f"E4_{cohort}_{cond}"

        rows = [R.raw_stat(float(n_rec), name="n_recovered", n=len(rec), **base)]
        if n_rec < 10:
            return rows

        clusters = rec["patient_id"].astype(str).to_numpy()
        for col in feats.columns:
            v = feats[col].to_numpy(dtype=float)
            ok = np.isfinite(v)
            if ok.sum() < 10:
                continue
            is_share = col.startswith("has_")
            rows.append(R.metric(v[ok & recovered], name="exposure_feature",
                                 cluster_ids=clusters[ok & recovered], is_percent=is_share,
                                 **{**base, "covariate": col, "case_set": "recovered"}))
            rows.append(R.metric(v[ok], name="exposure_feature",
                                 cluster_ids=clusters[ok], is_percent=is_share,
                                 **{**base, "covariate": col, "case_set": "cohort"}))
            rest = ok & ~recovered
            if rest.sum() >= 10 and (ok & recovered).sum() >= 10:
                rows.append(R.metric_diff_unpaired(
                    v[ok & recovered], v[rest], name="exposure_feature",
                    comparison="recovered_minus_not_recovered", is_percent=is_share,
                    **{**base, "covariate": col, "case_set": "recovered_vs_rest"}))
        return rows

    return run_analysis(cfg, "e4_exposure", units, compute, force=force)
