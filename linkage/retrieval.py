"""
linkage/retrieval.py
Created on August 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config.serde import read_config
from data_loader.build_paired_pool import load_pool
from data_loader.build_utils import write_manifest
from encoders.extract_embeddings import load_embeddings
from Inference.resume_utils import MissingInput


_TIE_EPS = 1e-6


def add_tied_at_top(records: pd.DataFrame) -> pd.DataFrame:
    picked_true = records["top1_study_id"].astype(str) == records["study_id"].astype(str)
    tied = (records["true_score"].astype(float)
            >= records["s1"].astype(float) - _TIE_EPS) & ~picked_true
    records["tied_at_top"] = tied.astype(int)
    return records


def embedding_key(model: str, variant: str) -> str:
    return model if variant == "off_shelf" else f"{model}_{variant}"


def _records_path(cfg: dict, model: str, cohort: str, transform: str,
                  variant: str, split: str) -> str:
    d = os.path.join(cfg["paths"]["linkage_dir"], model, cohort)
    return os.path.join(d, f"records_{variant}_{transform}_{split}.csv")


def _digest(items: List[str]) -> str:
    h = hashlib.blake2b(digest_size=16)
    for x in items:
        h.update(str(x).encode())
        h.update(b"\x00")
    return h.hexdigest()


def _build_params(model: str, cohort: str, transform: str, variant: str, split: str,
                  query_ids: List[str], corpus_ids: List[str]) -> dict:
    return {"model": model, "cohort": cohort, "transform": transform, "variant": variant,
            "split": split, "n_corpus": len(corpus_ids), "n_query": len(query_ids),
            "corpus_unit": "study",
            "query_ids_digest": _digest(query_ids),
            "corpus_ids_digest": _digest(corpus_ids)}


def _dedupe_corpus(case_ids: List[str], emb: np.ndarray,
                   case_to_study: Dict[str, str]) -> Tuple[List[str], np.ndarray]:
    seen, keep_idx, studies = set(), [], []
    for i, cid in enumerate(case_ids):
        sid = case_to_study.get(cid)
        if sid is None or sid in seen:
            continue
        seen.add(sid)
        keep_idx.append(i)
        studies.append(sid)
    return studies, emb[np.asarray(keep_idx, dtype=np.int64)]


def build_linkage_records(cfg_path: str, model: str, cohorts: Optional[List[str]] = None,
                          transforms: Optional[List[str]] = None, variant: str = "off_shelf",
                          splits: Optional[List[str]] = None, force: bool = False,
                          device: str = "cuda") -> List[str]:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    from Inference.rows import append_status
    status_dir, stage = cfg["paths"]["status_dir"], f"linkage_{model}"

    cohorts = cohorts or [c for c, cc in cfg["cohorts"].items() if cc.get("enabled", True)]
    if transforms is None:
        transforms = (list(cfg["report_transforms"]["order"]) if variant == "off_shelf"
                      else ["T0"])
    splits = splits or list(cfg["extraction"].get("splits", ["test"]))
    emb_model = embedding_key(model, variant)
    chunk_cfg = int(cfg["instrument"].get("score_chunk_rows", 2048))
    chunk_budget_gb = float(cfg["instrument"].get("score_chunk_max_gb", 4.0))
    device = device if torch.cuda.is_available() else "cpu"

    written = []
    for cohort in cohorts:
        pool = load_pool(cfg, cohort, usecols=["case_id", "split", "patient_id", "study_id"])
        c2s = dict(zip(pool["case_id"].astype(str), pool["study_id"].astype(str)))
        c2p = dict(zip(pool["case_id"].astype(str), pool["patient_id"].astype(str)))

        for split in splits:
            img = load_embeddings(cfg, emb_model, cohort, split, "img")
            q_ids = img["case_ids"]
            Q = torch.from_numpy(np.asarray(img["embeddings"], dtype=np.float32))

            for transform in transforms:
                path = _records_path(cfg, emb_model, cohort, transform, variant, split)
                meta = path.replace(".csv", ".buildparams.json")

                txt = load_embeddings(cfg, emb_model, cohort, split, "txt", transform)
                studies, C = _dedupe_corpus(txt["case_ids"], np.asarray(txt["embeddings"], dtype=np.float32), c2s)
                n_corpus = len(studies)
                if n_corpus < 2:
                    raise MissingInput(f"{cohort}/{split}: the corpus has {n_corpus} distinct studies")
                want = _build_params(emb_model, cohort, transform, variant, split, q_ids, studies)

                if not force and os.path.exists(path) and os.path.exists(meta):
                    with open(meta) as f:
                        stored = json.load(f)
                    moved = [k for k, v in want.items() if k in stored and stored[k] != v]
                    if not moved:
                        append_status(status_dir, stage,
                                      f"{cohort}/{transform}/{split}/{variant}: cached, skipping")
                        written.append(path)
                        continue
                    print(f"[{stage}] REBUILD {os.path.basename(path)}: {moved}", flush=True)

                sid_to_row = {s: i for i, s in enumerate(studies)}
                true_row = np.array([sid_to_row.get(c2s.get(cid, ""), -1) for cid in q_ids], dtype=np.int64)
                missing = int((true_row < 0).sum())
                if missing:
                    raise MissingInput(
                        f"{cohort}/{split}: {missing} query images have no report row in the corpus. "
                        "Rebuild the text embeddings for this transform.")

                append_status(status_dir, stage,
                              f"{cohort}/{transform}/{split}/{variant}: {len(q_ids)} queries "
                              f"against {n_corpus} distinct studies")

                max_rows = max(1, int(chunk_budget_gb * 1e9 / (4.0 * max(n_corpus, 1))))
                chunk = min(chunk_cfg, max_rows)
                if chunk < chunk_cfg:
                    print(f"[{stage}] score chunk clamped from {chunk_cfg} to {chunk} rows "
                          f"to stay under {chunk_budget_gb:.1f} GB at n_corpus={n_corpus}",
                          flush=True)
                Cd = torch.from_numpy(np.ascontiguousarray(C)).to(device)
                n_q = len(q_ids)
                rank_a = np.empty(n_q, dtype=np.int64)
                ts_a = np.empty(n_q, dtype=np.float32)
                s1_a = np.empty(n_q, dtype=np.float32)
                s2_a = np.empty(n_q, dtype=np.float32)
                top1_a = np.empty(n_q, dtype=np.int64)
                for lo in tqdm(range(0, n_q, chunk),
                               desc=f"{model}:{cohort}/{transform}/{split}"):
                    hi = min(lo + chunk, n_q)
                    S = (Q[lo:hi].to(device) @ Cd.T)
                    tr = torch.from_numpy(true_row[lo:hi]).to(device)
                    true_score = S.gather(1, tr.view(-1, 1)).squeeze(1)
                    rank = (S > true_score.view(-1, 1)).sum(dim=1) + 1
                    top_v, top_i = torch.topk(S, k=2, dim=1)
                    rank_a[lo:hi] = rank.cpu().numpy()
                    ts_a[lo:hi] = true_score.cpu().numpy()
                    s1_a[lo:hi] = top_v[:, 0].cpu().numpy()
                    s2_a[lo:hi] = top_v[:, 1].cpu().numpy()
                    top1_a[lo:hi] = top_i[:, 0].cpu().numpy()
                    del S, top_v, top_i
                del Cd
                if device == "cuda":
                    torch.cuda.empty_cache()

                out = pd.DataFrame({
                    "case_id": q_ids,
                    "patient_id": [c2p.get(c, "") for c in q_ids],
                    "study_id": [c2s.get(c, "") for c in q_ids],
                    "model": emb_model, "cohort": cohort, "transform": transform,
                    "variant": variant, "split": split,
                    "rank": rank_a, "true_score": ts_a, "s1": s1_a, "s2": s2_a,
                    "margin": s1_a - s2_a,
                    "top1_study_id": [studies[i] for i in top1_a],
                    "n_corpus": n_corpus,
                })
                out["correct"] = (out["rank"] == 1).astype(int)
                out["reciprocal_rank"] = 1.0 / out["rank"]

                picked_true = (out["top1_study_id"] == out["study_id"])
                add_tied_at_top(out)
                tied_top = out["tied_at_top"].astype(bool)
                n_tied = int(tied_top.sum())
                bad = (out["correct"] == 1) & ~picked_true & ~tied_top
                bad |= (out["correct"] == 0) & picked_true
                if bad.any():
                    raise ValueError(
                        f"{cohort}/{transform}: rank and top1_study_id disagree on "
                        f"{int(bad.sum())} queries with no tie to explain it; one of the two is "
                        "wrong")
                if n_tied:
                    msg = (f"{cohort}/{transform}/{split}/{variant}: {n_tied} of {n_q} queries tie "
                           f"for the top score with a study that is not their own, which means two "
                           f"studies share identical report text. Counted as recovered, the "
                           f"conservative direction for a privacy claim.")
                    print(f"[{stage}] {msg}", flush=True)
                    append_status(status_dir, stage, msg)

                write_manifest(out, path)
                tmp = meta + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(want, f, indent=2, sort_keys=True)
                os.replace(tmp, meta)
                r1 = 100.0 * out["correct"].mean()
                append_status(status_dir, stage,
                              f"{cohort}/{transform}/{split}/{variant}: Recall@1 {r1:.3f}% "
                              f"against a chance of {100.0 / n_corpus:.4f}%")
                written.append(path)
                print(f"[{stage}] {path}", flush=True)

    return written


def load_records(cfg: dict, model: str, cohort: str, transform: str = "T0",
                 variant: str = "off_shelf", split: str = "test") -> pd.DataFrame:
    path = _records_path(cfg, embedding_key(model, variant), cohort, transform, variant, split)
    if not os.path.exists(path):
        raise MissingInput(
            f"no linkage records for {model}/{cohort}/{transform}/{variant}/{split}. "
            f"Run main_build_linkage_records(model='{model}', variant='{variant}').")
    return pd.read_csv(path, dtype={"case_id": str, "patient_id": str, "study_id": str,
                                    "top1_study_id": str})
