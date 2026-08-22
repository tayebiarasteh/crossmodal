"""
analysis/e3_report_map.py
Created on August 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

from typing import Dict, List

import numpy as np
import pandas as pd

import os

from analysis._common import (analysis_dir, conditioning_axes, enabled_cohorts,
                              panel_models, run_analysis, unit_key)
from Inference.rows import write_rows
from analysis.e4_exposure import _features
from config.serde import read_config
from data_loader.build_paired_pool import load_pool, load_reports
from Inference import rows as R
from linkage.conditioning import load_conditioned
from linkage.retrieval import load_records


_COVARIATES = ["report_tokens", "n_positive_findings", "has_measurement",
               "has_comparison", "has_device", "has_laterality",
               "n_deident_placeholders", "n_sections", "has_rare_finding"]


def _standardize(v: np.ndarray) -> np.ndarray:
    sd = np.nanstd(v)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(v)
    return (v - np.nanmean(v)) / sd


def run_e3(cfg_path: str, force: bool = False) -> pd.DataFrame:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    findings = list(cfg["labels"]["findings"])
    rare_prevalence = float(cfg["report_map"]["rare_prevalence"])
    units = [unit_key(model=m, cohort=c)
             for m in panel_models(cfg) for c in enabled_cohorts(cfg)]
    units += [unit_key(model=m, cohort=c, cond=k)
              for m in panel_models(cfg) for c in enabled_cohorts(cfg)
              for k in conditioning_axes(cfg, "e3_report_map")]

    def compute(unit: str) -> List[Dict]:
        p = dict(x.split("-", 1) for x in unit.split("__"))
        model, cohort, cond = p["model"], p["cohort"], p.get("cond", "")
        rec = (load_conditioned(cfg, model, cohort, cond) if cond
               else load_records(cfg, model, cohort, "T0", "off_shelf", "test"))
        rec = rec.reset_index(drop=True)
        reports = load_reports(cfg, cohort, "T0").drop_duplicates("study_id").set_index("study_id")
        pool = load_pool(cfg, cohort, split="test",
                         usecols=["study_id"] + findings).drop_duplicates("study_id")
        npos = pool.set_index(pool["study_id"].astype(str))[findings].apply(
            lambda r: float(np.nansum(r.to_numpy(dtype=float) == 1.0)), axis=1)

        sids = rec["study_id"].astype(str)
        feats = pd.DataFrame([_features(reports["report_text"].get(s, "")) for s in sids])
        feats["n_positive_findings"] = [float(npos.get(s, np.nan)) for s in sids]

        lab = pool.set_index(pool["study_id"].astype(str))[findings]
        prev = (lab == 1.0).sum(axis=0) / np.maximum(lab.notna().sum(axis=0), 1)
        rare = [f for f in findings if prev.get(f, 1.0) < rare_prevalence]
        if rare:
            has_rare = (lab[rare] == 1.0).any(axis=1).astype(float)
            feats["has_rare_finding"] = [float(has_rare.get(s, 0.0)) for s in sids]
        else:
            feats["has_rare_finding"] = 0.0

        margin = rec["margin"].to_numpy(dtype=float)
        correct = rec["correct"].to_numpy(dtype=float)
        groups = rec["patient_id"].astype(str).to_numpy()
        base = {"model": model, "cohort": cohort, "conditioning": cond, "experiment": "E3",
                "analysis": "report_map", "fdr_family": f"E3_{cohort}_{cond or 'full'}_{model}"}

        chance = float(np.nanmean(1.0 / rec["n_corpus"].to_numpy(dtype=float)))
        fold = float(np.nanmean(correct) / chance) if chance > 0 else float("nan")
        rows: List[Dict] = [
            R.raw_stat(float(correct.sum()), name="n_recovered", n=len(rec), **base),
            R.raw_stat(fold, name="fold_over_chance", n=len(rec),
                       **{**base, "at": "recall_at_1"}),
        ]
        for cov in _COVARIATES:
            if cov not in feats.columns:
                continue
            x = feats[cov].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(margin)
            if ok.sum() < 50 or np.nanstd(x[ok]) == 0:
                continue
            xs = _standardize(x[ok])
            for outcome, y in (("margin", margin), ("recovered", correct)):
                if not np.isfinite(np.nanstd(y[ok])) or np.nanstd(y[ok]) == 0:
                    continue
                rows.append(R.mixedlm_slope(xs, y[ok], groups[ok],
                                            name="report_covariate_slope",
                                            **{**base, "covariate": cov, "outcome": outcome}))
        return rows

    df = run_analysis(cfg, "e3_report_map", units, compute, force=force)
    if df.empty:
        return df

    if "fit_method" in df.columns:
        got = df["fit_method"].astype(str).value_counts().to_dict()
        print(f"[e3_report_map] slope fits: {got}", flush=True)
        n_naive = int((df["fit_method"].astype(str) == "ols_fallback").sum())
        if n_naive:
            raise ValueError(
                f"{n_naive} rows carry fit_method=ols_fallback, which is a naive "
                "least-squares p-value and must never reach a table. Nothing writes that "
                "tag any more, so its presence means a partial from before 2026-08-16 "
                "survived; run this stage with force=True.")

    if {"covariate", "outcome", "measure"} <= set(df.columns):
        min_n = int(cfg["report_map"]["min_recovered_for_agreement"])
        min_fold = float(cfg["report_map"]["min_fold_for_agreement"])
        rec_rows = df[df["measure"].astype(str) == "n_recovered"]
        fold_rows = df[df["measure"].astype(str) == "fold_over_chance"]
        slopes = df[df["measure"].astype(str) == "report_covariate_slope"]
        cond_col = (df["conditioning"].astype(str).replace("nan", "")
                    if "conditioning" in df.columns else "")
        rec_rows = rec_rows.assign(_op=cond_col.reindex(rec_rows.index).fillna(""))
        fold_rows = fold_rows.assign(_op=cond_col.reindex(fold_rows.index).fillna(""))
        slopes = slopes.assign(_op=cond_col.reindex(slopes.index).fillna(""))
        enough_n = {(c, o): set(g.loc[pd.to_numeric(g["value"], errors="coerce") >= min_n, "model"]
                               .astype(str))
                    for (c, o), g in rec_rows.groupby(["cohort", "_op"])}
        above_chance = {(c, o): set(g.loc[pd.to_numeric(g["value"], errors="coerce") >= min_fold,
                                          "model"].astype(str))
                        for (c, o), g in fold_rows.groupby(["cohort", "_op"])}
        linking = {k: sorted(v & above_chance.get(k, set())) for k, v in enough_n.items()}
        agree = []
        for (cohort, op, cov, outcome), g in slopes.groupby(["cohort", "_op", "covariate", "outcome"]):
            keep = linking.get((cohort, op), [])
            g = g[g["model"].astype(str).isin(keep)]
            v = pd.to_numeric(g["value"], errors="coerce").dropna()
            if len(v) < 3:
                continue
            n_pos = int((v > 0).sum())
            n_maj = max(n_pos, len(v) - n_pos)
            direction = "positive" if n_pos >= len(v) - n_pos else "negative"
            agree.append(R.raw_stat(
                float(n_maj) / len(v), name="sign_agreement_across_models", n=len(v),
                extra=f"{n_maj} of {len(v)} share a {direction} sign; models: {','.join(keep)}",
                cohort=cohort, conditioning=op, covariate=cov, outcome=outcome,
                experiment="E3", analysis="map_agreement",
                fdr_family=f"E3_agreement_{cohort}_{op or 'full'}"))
        counted = {"%s/%s" % (c, o or "full"): v for (c, o), v in linking.items()}
        print(f"[e3_report_map] models counted in the agreement, per cohort and operating "
              f"point (>= {min_n} recoveries AND >= {min_fold} fold over chance): "
              f"{counted}", flush=True)
        if not agree:
            print("[e3_report_map] fewer than three models qualify anywhere, so the "
                  "cross-model agreement is not reported.", flush=True)
        if agree:
            out = os.path.join(analysis_dir(cfg, "e3_report_map"), "e3_map_agreement_rows.csv")
            write_rows(agree, out)
            print(f"[e3_report_map] {out} ({len(agree)} rows)", flush=True)
    return df
