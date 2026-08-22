"""
linkage/conditioning.py
Created on August 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import json
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config.serde import read_config
from data_loader.build_paired_pool import load_pool
from data_loader.build_utils import write_manifest
from encoders.extract_embeddings import load_embeddings
from linkage.retrieval import _dedupe_corpus, add_tied_at_top, embedding_key
from Inference.resume_utils import MissingInput


def _age_band(age, width: int) -> str:
    a = pd.to_numeric(age, errors="coerce")
    if not np.isfinite(a):
        return "NA"
    return str(int(np.floor(float(a) / width) * width))


def _study_attributes(pool: pd.DataFrame, width: int) -> pd.DataFrame:
    g = pool.groupby("study_id", dropna=False)
    out = pd.DataFrame({
        "sex": g["sex"].first(),
        "age_band": g["age"].first().map(lambda a: _age_band(a, width)),
        "study_year": g["study_year"].first().astype(str),
        "site": g["site"].first().astype(str),
        "views": g["view"].apply(lambda s: frozenset(s.astype(str))),
    })
    return out.reset_index()


def _match_mask(q: pd.Series, studies: pd.DataFrame, attrs: List[str]) -> np.ndarray:
    mask = np.ones(len(studies), dtype=bool)
    for a in attrs:
        if a == "view":
            mask &= np.array([str(q["view"]) in vs for vs in studies["views"]])
        else:
            mask &= (studies[a].astype(str).to_numpy() == str(q[a]))
    return mask


def _hard_negative_mask(q_labels: np.ndarray, corpus_labels: np.ndarray,
                        true_row: int, n_keep: int) -> np.ndarray:
    obs = np.isfinite(q_labels) & np.isfinite(corpus_labels)
    diff = (np.nan_to_num(corpus_labels, nan=0.0) != np.nan_to_num(q_labels, nan=0.0)) & obs
    dist = diff.sum(axis=1).astype(float)
    dist[~obs.any(axis=1)] = np.inf
    dist[true_row] = -1.0
    order = np.argsort(dist, kind="stable")[:max(2, n_keep)]
    mask = np.zeros(len(corpus_labels), dtype=bool)
    mask[order] = True
    mask[true_row] = True
    return mask


def _records_path(cfg: dict, model: str, cohort: str, transform: str, variant: str,
                  split: str, conditioning: str) -> str:
    d = os.path.join(cfg["paths"]["linkage_dir"], model, cohort)
    return os.path.join(d, f"cond_{conditioning}_{variant}_{transform}_{split}.csv")


def build_conditioned_records(cfg_path: str, model: str, conditioning: str,
                              cohorts: Optional[List[str]] = None, transform: str = "T0",
                              variant: str = "off_shelf", split: str = "test",
                              force: bool = False, device: str = "cuda") -> List[str]:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    from Inference.rows import append_status
    status_dir, stage = cfg["paths"]["status_dir"], f"cond_{model}"

    ccfg = cfg["conditioning"]
    if conditioning not in ccfg["sets"] and conditioning != "hard_negative":
        raise KeyError(f"conditioning {conditioning!r} is not in config; "
                       f"available: {sorted(ccfg['sets'])} plus 'hard_negative'")
    attrs = [] if conditioning == "hard_negative" else list(ccfg["sets"][conditioning])
    width = int(ccfg.get("age_band_years", 5))
    min_pool = int(ccfg.get("min_pool_size", 50))
    findings = list(cfg["labels"]["findings"])
    cohorts = cohorts or [c for c, cc in cfg["cohorts"].items() if cc.get("enabled", True)]
    device = device if torch.cuda.is_available() else "cpu"

    written = []
    for cohort in cohorts:
        if conditioning == "site" and not cfg["cohorts"][cohort].get("site_col"):
            append_status(status_dir, stage, f"{cohort}: no site column, skipping the site conditioning")
            continue

        path = _records_path(cfg, embedding_key(model, variant), cohort, transform,
                             variant, split, conditioning)
        meta = path.replace(".csv", ".buildparams.json")
        want = {"model": embedding_key(model, variant), "cohort": cohort, "transform": transform,
                "variant": variant, "split": split, "conditioning": conditioning,
                "attributes": attrs, "age_band_years": width, "min_pool_size": min_pool,
                "hard_neg_pool": int(ccfg.get("hard_neg_pool", 10000))}
        if not force and os.path.exists(path) and os.path.exists(meta):
            with open(meta) as f:
                stored = json.load(f)
            moved = [k for k, v in want.items() if k in stored and stored[k] != v]
            if not moved:
                append_status(status_dir, stage, f"{cohort}/{conditioning}: cached, skipping")
                written.append(path)
                continue
            print(f"[{stage}] REBUILD {os.path.basename(path)}: {moved}", flush=True)

        pool = load_pool(cfg, cohort, usecols=["case_id", "split", "patient_id", "study_id",
                                               "view", "age", "sex", "study_year", "site"] + findings)
        pool_split = pool[pool["split"] == split].reset_index(drop=True)
        c2s = dict(zip(pool["case_id"].astype(str), pool["study_id"].astype(str)))

        emb_model = embedding_key(model, variant)
        img = load_embeddings(cfg, emb_model, cohort, split, "img")
        txt = load_embeddings(cfg, emb_model, cohort, split, "txt", transform)
        studies, C = _dedupe_corpus(txt["case_ids"], np.asarray(txt["embeddings"], np.float32), c2s)
        sid_to_row = {s: i for i, s in enumerate(studies)}

        sattr = _study_attributes(pool_split, width).set_index("study_id").reindex(studies).reset_index()
        slab = (pool_split.drop_duplicates("study_id").set_index("study_id")
                .reindex(studies)[findings].to_numpy(dtype=float))

        q_ids = img["case_ids"]
        Q = torch.from_numpy(np.asarray(img["embeddings"], np.float32))
        qmeta = pool.set_index(pool["case_id"].astype(str)).reindex(q_ids)
        qmeta["age_band"] = qmeta["age"].map(lambda a: _age_band(a, width))
        Cd = torch.from_numpy(C).to(device)
        n_keep = int(ccfg.get("hard_neg_pool", 10000))

        append_status(status_dir, stage,
                      f"{cohort}/{conditioning}: {len(q_ids)} queries, corpus {len(studies)} studies")

        chunk = max(1, min(int(cfg["instrument"].get("score_chunk_rows", 2048)),
                           int(float(cfg["instrument"].get("score_chunk_max_gb", 4.0))
                               * 1e9 / (4.0 * max(len(studies), 1)))))
        rows, dropped, singleton = [], 0, 0
        pid_col = qmeta["patient_id"].astype(str).to_numpy()
        for lo in tqdm(range(0, len(q_ids), chunk), desc=f"{model}:{cohort}/{conditioning}"):
            hi = min(lo + chunk, len(q_ids))
            S = (Q[lo:hi].to(device) @ Cd.T).cpu().numpy()
            for j in range(lo, hi):
                cid = q_ids[j]
                sid = c2s.get(cid)
                tr = sid_to_row.get(sid)
                if tr is None:
                    continue
                if conditioning == "hard_negative":
                    mask = _hard_negative_mask(slab[tr], slab, tr, n_keep)
                else:
                    mask = _match_mask(qmeta.iloc[j], sattr, attrs)
                    mask[tr] = True
                n_i = int(mask.sum())
                if n_i <= 1:
                    singleton += 1
                    dropped += 1
                    continue
                if n_i < min_pool:
                    dropped += 1
                    continue
                idx = np.flatnonzero(mask)
                s = S[j - lo][idx]
                local_true = int(np.searchsorted(idx, tr))
                ts = float(s[local_true])
                rank = int((s > ts).sum()) + 1
                order = np.argsort(-s, kind="stable")[:2]
                s1 = float(s[order[0]])
                s2 = float(s[order[1]]) if n_i > 1 else float("nan")
                rows.append({
                    "case_id": cid, "patient_id": pid_col[j], "study_id": sid,
                    "model": emb_model, "cohort": cohort, "transform": transform, "variant": variant,
                    "split": split, "conditioning": conditioning,
                    "rank": rank, "true_score": ts, "s1": s1, "s2": s2, "margin": s1 - s2,
                    "top1_study_id": studies[int(idx[order[0]])],
                    "n_corpus": n_i,
                })
            del S
        del Cd
        if device == "cuda":
            torch.cuda.empty_cache()

        if not rows:
            raise ValueError(
                f"{cohort}/{conditioning}: all {len(q_ids)} queries fell below the minimum "
                f"candidate-set size of {min_pool} over a corpus of {len(studies)} studies, so "
                f"this conditioning is not measurable here. Either lower "
                f"conditioning.min_pool_size, or drop this conditioning for this cohort and say "
                f"in the paper that the conjunction narrows it past the point of measurement.")
        out = pd.DataFrame(rows)
        out["correct"] = (out["rank"] == 1).astype(int)
        out["reciprocal_rank"] = 1.0 / out["rank"]
        add_tied_at_top(out)
        write_manifest(out, path)
        with open(meta + ".tmp", "w") as f:
            json.dump(want, f, indent=2, sort_keys=True)
        os.replace(meta + ".tmp", meta)
        counts = path.replace(".csv", ".counts.json")
        with open(counts + ".tmp", "w") as f:
            json.dump({"n_query_total": int(len(q_ids)), "n_kept": int(len(out)),
                       "n_dropped": int(dropped), "n_singleton_pool": int(singleton),
                       "min_pool_size": min_pool}, f, indent=2, sort_keys=True)
        os.replace(counts + ".tmp", counts)
        append_status(status_dir, stage,
                      f"{cohort}/{conditioning}: {len(out)} queries kept, {dropped} below the "
                      f"minimum candidate set ({singleton} of them uniquely identified), "
                      f"median candidates {int(out['n_corpus'].median())}, "
                      f"Recall@1 {100.0 * out['correct'].mean():.3f}%")
        written.append(path)
        print(f"[{stage}] {path}", flush=True)
    return written


def load_conditioned(cfg: dict, model: str, cohort: str, conditioning: str,
                     transform: str = "T0", variant: str = "off_shelf",
                     split: str = "test") -> pd.DataFrame:
    path = _records_path(cfg, embedding_key(model, variant), cohort, transform,
                         variant, split, conditioning)
    if not os.path.exists(path):
        raise MissingInput(f"no conditioned records for {model}/{cohort}/{conditioning}. "
                           f"Run main_build_conditioned_records(model='{model}', "
                           f"conditioning='{conditioning}').")
    df = pd.read_csv(path, dtype={"case_id": str, "patient_id": str, "study_id": str,
                                  "top1_study_id": str})
    if "tied_at_top" not in df.columns:
        add_tied_at_top(df)
    return df


def load_conditioned_counts(cfg: dict, model: str, cohort: str, conditioning: str,
                            transform: str = "T0", variant: str = "off_shelf",
                            split: str = "test") -> dict:
    path = _records_path(cfg, embedding_key(model, variant), cohort, transform,
                         variant, split, conditioning).replace(".csv", ".counts.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)
