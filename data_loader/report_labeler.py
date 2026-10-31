"""
data_loader/report_labeler.py
Created on August 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""
import hashlib
import json
import os
from typing import List

import numpy as np

from config.serde import read_config
from data_loader.build_paired_pool import load_pool, load_reports
from data_loader.build_utils import write_manifest
from data_loader.llm_local import extract_findings
from Inference.resume_utils import MissingInput


def _content_digest(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return hashlib.blake2b(f.read(), digest_size=16).hexdigest()


def label_reports(cfg_path: str, cohort: str, force: bool = False) -> str:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    paths = cfg["paths"]
    findings: List[str] = list(cfg["labels"]["findings"])
    ccfg = cfg["cohorts"].get(cohort)
    if ccfg is None:
        raise KeyError(f"cohort {cohort!r} is not in config CrossModal.cohorts")
    if ccfg.get("labels_source") != "llm_labeler":
        raise MissingInput(
            f"{cohort} carries CheXpert-labeler labels already, so there is nothing to label. "
            "Only a cohort whose config says labels_source: llm_labeler runs this stage.")

    from data_loader.llm_local import assert_serving_ready
    assert_serving_ready(cfg)

    from Inference.rows import append_status
    status_dir, stage = paths["status_dir"], f"labeler_{cohort}"

    pool_path = os.path.join(paths["manifests_dir"], f"pool_{cohort}.csv")
    meta_path = os.path.join(paths["manifests_dir"], f"pool_{cohort}.labels.buildparams.json")
    if not os.path.exists(pool_path):
        raise MissingInput(f"pool not built for {cohort}; run main_build_paired_pool(cohort='{cohort}')")

    want = {
        "cohort": cohort,
        "findings": findings,
        "llm": {"hf_id": cfg["llm"]["hf_id"],
                "served_name": cfg["llm"].get("served_name", cfg["llm"]["hf_id"]),
                "temperature": cfg["llm"]["temperature"],
                "max_new_tokens": cfg["llm"]["max_new_tokens"], "seed": cfg["llm"]["seed"]},
        "backend": cfg["serving"]["backend"],
        "pool_build": _content_digest(
            os.path.join(paths["manifests_dir"], f"pool_{cohort}.buildparams.json")),
        "reports_t0": _content_digest(
            os.path.join(paths["reports_dir"], f"reports_{cohort}_T0.csv")),
    }

    pool = load_pool(cfg, cohort)
    if not force and "labels_source" in pool.columns:
        done = pool["labels_source"].astype(str).eq("llm_labeler").all()
        stored = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                stored = json.load(f)
        moved = [k for k, v in want.items() if k in stored and stored[k] != v]
        if done and not moved:
            append_status(status_dir, stage, "labels already written; skipping")
            return pool_path
        if moved:
            pass

    base = load_reports(cfg, cohort, "T0")
    lab = extract_findings(cfg, cohort, base, findings, status_dir)

    merged = pool.drop(columns=[c for c in findings if c in pool.columns], errors="ignore")
    merged = merged.merge(lab, on="study_id", how="left")
    for f in findings:
        if f not in merged.columns:
            merged[f] = np.nan

    n_rows_with_any = int(merged[findings].notna().any(axis=1).sum())
    if n_rows_with_any == 0:
        raise ValueError(f"{cohort}: no row received a single label; refusing to write")
    merged["labels_source"] = np.where(merged[findings].notna().any(axis=1),
                                       "llm_labeler", "llm_labeler_failed")

    cols = [c for c in pool.columns if c in merged.columns]
    cols += [c for c in merged.columns if c not in cols]
    merged = merged[cols]

    write_manifest(merged, pool_path)
    tmp = meta_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(want, f, indent=2, sort_keys=True)
    os.replace(tmp, meta_path)

    covered = 100.0 * float(merged[findings].notna().to_numpy().mean())
    append_status(status_dir, stage,
                  f"wrote labels for {n_rows_with_any} of {len(merged)} rows, "
                  f"{covered:.1f}% of finding cells populated")
    return pool_path
