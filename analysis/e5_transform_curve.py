"""
analysis/e5_transform_curve.py
Created on August 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""
import os
import re
from typing import Dict, List

import numpy as np
import pandas as pd

from analysis._common import (conditioning_axes, enabled_cohorts, ladder_models,
                              panel_models, run_analysis, unit_key)
from analysis.e4_exposure import _COMPARISON, _DEVICE, _LATERAL, _MEASURE
from config.serde import read_config
from data_loader.build_paired_pool import load_pool, load_reports
from data_loader.build_utils import read_csv_defensively, write_manifest
from Inference import rows as R
from Inference.resume_utils import MissingInput
from linkage.conditioning import load_conditioned, load_conditioned_counts
from linkage.instrument import run_instrument
from linkage.retrieval import load_records


_AXES = {"laterality": _LATERAL, "measurement": _MEASURE, "device": _DEVICE,
         "comparison": _COMPARISON,
         "severity": re.compile(r"\b(?:mild|moderate|severe|small|large|extensive|trace|"
                                r"minimal|marked|slight|significant)\w*\b", re.I),
         "hedging": re.compile(r"\b(?:may|might|could|possible|possibly|probable|probably|"
                               r"suspect\w*|equivocal|questionable|apparent\w*|likely|"
                               r"cannot exclude|not excluded|concerning for)\b", re.I)}


def _axis_retention(cfg: dict, cohort: str, transform: str, sample: List[str]) -> pd.DataFrame:
    ref = load_reports(cfg, cohort, "T0").drop_duplicates("study_id")
    ref = ref[ref["study_id"].astype(str).isin(set(sample))]
    new = load_reports(cfg, cohort, transform).drop_duplicates("study_id")
    new = new.set_index(new["study_id"].astype(str)).reindex(ref["study_id"].astype(str))
    if ref.empty or new["report_text"].isna().all():
        raise MissingInput(f"{cohort}/{transform}: no sampled study has both reports")
    pool = (load_pool(cfg, cohort, usecols=["study_id", "patient_id"])
            .drop_duplicates("study_id"))
    pid = (pool.set_index(pool["study_id"].astype(str))
           .reindex(ref["study_id"].astype(str))["patient_id"].astype(str))
    out = pd.DataFrame({"study_id": ref["study_id"].astype(str).values,
                        "patient_id": pid.values})
    before = ref["report_text"].astype(str)
    after = new["report_text"].fillna("").astype(str)
    for name, rx in _AXES.items():
        b = before.map(lambda s, _r=rx: bool(_r.search(s))).to_numpy()
        a = after.map(lambda s, _r=rx: bool(_r.search(s))).to_numpy()
        col = np.full(len(out), np.nan)
        col[b] = a[b].astype(float)
        out[f"axis__{name}"] = col
    return out


def _sample_path(cfg: dict, cohort: str) -> str:
    return os.path.join(cfg["paths"]["analysis_dir"], "e5_transform_curve",
                        f"content_sample_{cohort}.csv")


def _content_sample(cfg: dict, cohort: str, n: int, seed: int) -> List[str]:
    path = _sample_path(cfg, cohort)
    if os.path.exists(path):
        return read_csv_defensively(path, dtype={"study_id": str})["study_id"].tolist()
    pool = load_pool(cfg, cohort, split="test", usecols=["study_id"])
    studies = np.sort(pool["study_id"].astype(str).unique())
    rng = np.random.RandomState(seed)
    take = studies if len(studies) <= n else rng.choice(studies, size=n, replace=False)
    out = pd.DataFrame({"study_id": np.sort(take)})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_manifest(out, path)
    return out["study_id"].tolist()


def _content_retained(cfg: dict, cohort: str, transform: str, sample: List[str],
                      findings: List[str], status_dir: str) -> pd.DataFrame:
    from data_loader.llm_local import assert_serving_ready, extract_findings
    assert_serving_ready(cfg)
    base = load_reports(cfg, cohort, transform)
    base = base[base["study_id"].astype(str).isin(set(sample))].drop_duplicates("study_id")
    if base.empty:
        raise MissingInput(f"{cohort}/{transform}: no sampled study has a transformed report")

    lab = extract_findings(cfg, f"{cohort}_{transform}", base, findings, status_dir)
    sids = lab["study_id"].astype(str)

    ref0 = load_reports(cfg, cohort, "T0")
    ref0 = ref0[ref0["study_id"].astype(str).isin(set(sample))].drop_duplicates("study_id")
    if ref0.empty:
        raise MissingInput(f"{cohort}: no sampled study has an untransformed report, so the "
                           "content reference cannot be read")
    ref_lab = extract_findings(cfg, f"{cohort}_T0", ref0, findings, status_dir)
    ref = ref_lab.set_index(ref_lab["study_id"].astype(str)).reindex(sids)

    pool = (load_pool(cfg, cohort, usecols=["study_id", "patient_id"])
            .drop_duplicates("study_id"))
    pid = pool.set_index(pool["study_id"].astype(str)).reindex(sids)

    out = pd.DataFrame({"study_id": sids.values,
                        "patient_id": pid["patient_id"].astype(str).values})
    agree_cols = []
    for f in findings:
        a = pd.to_numeric(lab[f], errors="coerce").to_numpy(float)
        b = pd.to_numeric(ref[f], errors="coerce").to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        col = np.full(len(a), np.nan)
        col[both] = (a[both] == b[both]).astype(float)
        out[f"agree__{f}"] = col
        agree_cols.append(f"agree__{f}")

    A = out[agree_cols].to_numpy(float)
    out["macro_agreement"] = np.nanmean(np.where(np.isfinite(A), A, np.nan), axis=1)

    pos_ref = np.vstack([pd.to_numeric(ref[f], errors="coerce").to_numpy(float) == 1.0
                         for f in findings]).T
    pos_new = np.vstack([pd.to_numeric(lab[f], errors="coerce").to_numpy(float) == 1.0
                         for f in findings]).T
    n_ref = pos_ref.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["concept_recall"] = np.where(n_ref > 0, (pos_ref & pos_new).sum(axis=1) / n_ref, np.nan)
    return out


def run_e5(cfg_path: str, force: bool = False) -> pd.DataFrame:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    findings = list(cfg["labels"]["findings"])
    n_sample = int(cfg.get("content_check", {}).get("sample_per_cohort", 2000))
    seed = int(cfg.get("content_check", {}).get("seed", 11))
    status_dir = cfg["paths"]["status_dir"]
    transforms = list(cfg["report_transforms"]["order"])

    units = [unit_key(model=m, cohort=c, transform=t)
             for m in panel_models(cfg) for c in enabled_cohorts(cfg) for t in transforms]
    units += [unit_key(model=m, cohort=c, transform=t, cond=k)
              for m in ladder_models(cfg) for c in enabled_cohorts(cfg) for t in transforms
              for k in conditioning_axes(cfg, "e5_transform")]
    units += [unit_key(content=c, transform=t) for c in enabled_cohorts(cfg) for t in transforms]
    units += [unit_key(axes=c, transform=t) for c in enabled_cohorts(cfg) for t in transforms]

    def compute(unit: str) -> List[Dict]:
        p = dict(x.split("-", 1) for x in unit.split("__"))

        if "axes" in p:
            cohort, transform = p["axes"], p["transform"]
            sample = _content_sample(cfg, cohort, n_sample, seed)
            per_study = _axis_retention(cfg, cohort, transform, sample)
            clusters = per_study["patient_id"].astype(str).to_numpy()
            ident = {"cohort": cohort, "transform": transform, "experiment": "E5",
                     "analysis": "content_axis", "fdr_family": f"E5_axis_{cohort}"}
            rows = []
            for name in _AXES:
                v = per_study[f"axis__{name}"].to_numpy(float)
                ok = np.isfinite(v)
                if ok.sum() < 10:
                    continue
                rows.append(R.metric(v[ok], name="content_axis_retained",
                                     cluster_ids=clusters[ok],
                                     **{**ident, "covariate": name}))
            if not rows:
                raise MissingInput(f"{cohort}/{transform}: fewer than ten sampled studies state "
                                   f"any of the finer axes in their original report")
            return rows

        if "content" in p:
            cohort, transform = p["content"], p["transform"]
            sample = _content_sample(cfg, cohort, n_sample, seed)
            per_study = _content_retained(cfg, cohort, transform, sample, findings, status_dir)
            ident = {"cohort": cohort, "transform": transform, "experiment": "E5",
                     "analysis": "content_retained", "fdr_family": f"E5_content_{cohort}"}
            clusters = per_study["patient_id"].astype(str).to_numpy()

            rows: List[Dict] = []
            macro = per_study["macro_agreement"].to_numpy(float)
            ok = np.isfinite(macro)
            if ok.sum() < 10:
                raise MissingInput(
                    f"{cohort}/{transform}: only {int(ok.sum())} sampled studies produced a "
                    "comparable label vector on both sides; the labeler output is too sparse to "
                    "read a content number from.")
            rows.append(R.metric(macro[ok], name="content_retained",
                                 cluster_ids=clusters[ok], **ident))
            for f in findings:
                v = per_study[f"agree__{f}"].to_numpy(float)
                fok = np.isfinite(v)
                if fok.sum() < 10:
                    continue
                rows.append(R.metric(v[fok], name="content_retained", cluster_ids=clusters[fok],
                                     **{**ident, "finding": f}))
            cr = per_study["concept_recall"].to_numpy(float)
            cok = np.isfinite(cr)
            if cok.sum() >= 10:
                rows.append(R.metric(cr[cok], name="content_concept_recall",
                                     cluster_ids=clusters[cok], **ident))
            return rows

        model, cohort, transform = p["model"], p["cohort"], p["transform"]
        cond = p.get("cond")
        if cond:
            rec = load_conditioned(cfg, model, cohort, cond, transform=transform)
            rows = run_instrument(rec, cfg, extra_ident={"conditioning": cond})
            counts = load_conditioned_counts(cfg, model, cohort, cond, transform=transform)
            if counts:
                ident = {"model": rec["model"].iloc[0], "cohort": cohort, "transform": transform,
                         "variant": "off_shelf", "split": "test", "conditioning": cond}
                rows.append(R.raw_stat(int(counts["n_dropped"]),
                                       name="queries_dropped_below_min_pool",
                                       n=int(counts["n_query_total"]), **ident))
                rows.append(R.raw_stat(int(counts["n_singleton_pool"]),
                                       name="singleton_pool_queries",
                                       n=int(counts["n_query_total"]), **ident))
        else:
            rec = load_records(cfg, model, cohort, transform, "off_shelf", "test")
            rows = run_instrument(rec, cfg)
        fam = f"E5_{cohort}_{cond}" if cond else f"E5_{cohort}"
        for r in rows:
            r["experiment"] = "E5"
            r["analysis"] = "transform_privacy"
            r["fdr_family"] = f"{fam}_{r.get('metric') or r.get('measure')}"
        return rows

    return run_analysis(cfg, "e5_transform_curve", units, compute, force=force)
