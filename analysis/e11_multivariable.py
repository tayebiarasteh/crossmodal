"""
analysis/e11_multivariable.py
Created on October 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""
from typing import Dict, List

import numpy as np
import pandas as pd

from analysis._common import (conditioning_axes, enabled_cohorts, ladder_models,
                              run_analysis, unit_key)
from analysis.e3_report_map import _COVARIATES, _standardize, build_covariates
from config.serde import read_config
from Inference import rows as R
from Inference.report_utils import report_cluster_robust_multivariable
from Inference.resume_utils import MissingInput
from linkage.conditioning import load_conditioned
from linkage.retrieval import load_records


_LOG_POOL = "log_pool_size"


def run_e11(cfg_path: str, force: bool = False) -> pd.DataFrame:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    findings = list(cfg["labels"]["findings"])
    rare_prevalence = float(cfg["report_map"]["rare_prevalence"])
    units = [unit_key(model=m, cohort=c)
             for m in ladder_models(cfg) for c in enabled_cohorts(cfg)]
    units += [unit_key(model=m, cohort=c, cond=k)
              for m in ladder_models(cfg) for c in enabled_cohorts(cfg)
              for k in conditioning_axes(cfg, "e11_multivariable")]

    def compute(unit: str) -> List[Dict]:
        p = dict(x.split("-", 1) for x in unit.split("__"))
        model, cohort, cond = p["model"], p["cohort"], p.get("cond", "")
        rec = (load_conditioned(cfg, model, cohort, cond) if cond
               else load_records(cfg, model, cohort, "T0", "off_shelf", "test"))
        rec = rec.reset_index(drop=True)
        feats = build_covariates(cfg, cohort, rec, findings, rare_prevalence)

        nc = rec["n_corpus"].to_numpy(dtype=float)
        names = [c for c in _COVARIATES if c in feats.columns]
        cols = [feats[c].to_numpy(dtype=float) for c in names]
        if np.isfinite(nc).all() and nc.max() > nc.min():
            names.append(_LOG_POOL)
            cols.append(np.log(np.maximum(nc, 1.0)))
        keep = [i for i, v in enumerate(cols) if np.isfinite(np.nanstd(v)) and np.nanstd(v) > 0]
        if len(keep) < 2:
            raise MissingInput(f"{model}/{cohort}/{cond or 'full'}: fewer than two covariates "
                               f"vary, so no adjusted model is identified")
        names = [names[i] for i in keep]
        X = np.column_stack([_standardize(cols[i]) for i in keep])
        groups = rec["patient_id"].astype(str).to_numpy()
        base = {"model": rec["model"].iloc[0], "cohort": cohort, "conditioning": cond,
                "experiment": "E11", "analysis": "report_map_adjusted",
                "fdr_family": f"E11_{cohort}_{cond or 'full'}_{model}"}

        rows: List[Dict] = []
        for outcome, y in (("margin", rec["margin"].to_numpy(dtype=float)),
                           ("recovered", rec["correct"].to_numpy(dtype=float))):
            if not np.isfinite(np.nanstd(y)) or np.nanstd(y) == 0:
                continue
            fit = report_cluster_robust_multivariable(X, y, groups)
            note = (f"adjusted_for={len(names)} covariates jointly; "
                    f"r2={fit['r_squared']:.4f}; n_groups={fit['n_groups']}; "
                    f"fit_method=ols_cluster_robust")
            for j, cov in enumerate(names):
                rows.append(R.raw_stat(float(fit["beta"][j]),
                                       name="report_covariate_slope_adjusted",
                                       n=fit["n"], p_raw=float(fit["p"][j]), extra=note,
                                       **{**base, "covariate": cov, "outcome": outcome}))
                rows.append(R.raw_stat(float(fit["vif"][j]), name="covariate_vif",
                                       n=fit["n"],
                                       **{**base, "covariate": cov, "outcome": outcome}))
        if not rows:
            raise MissingInput(f"{model}/{cohort}/{cond or 'full'}: neither outcome varies")
        return rows

    return run_analysis(cfg, "e11_multivariable", units, compute, force=force)
