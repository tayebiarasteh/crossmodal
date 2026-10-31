"""
linkage/instrument.py
Created on August 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""
import hashlib
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from Inference import rows as R
from Inference.report_utils import _pack_bootstrap, keep_p
from Inference.stats_utils import BOOT_SEED, paired_bootstrap_statistic, _percentile_ci
from Inference.selection_bootstrap import selective_bootstrap


def decision_statistic(records: pd.DataFrame, name: str) -> np.ndarray:
    s1 = records["s1"].to_numpy(dtype=float)
    s2 = records["s2"].to_numpy(dtype=float)
    if name == "margin":
        return s1 - s2
    if name == "top1_score":
        return s1
    if name == "ratio":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(s2 > 1e-12, s1 / s2, np.nan)
    raise KeyError(f"unknown decision statistic {name!r}")


def selective_precision(margin: np.ndarray, correct: np.ndarray, k: int) -> float:
    n = len(margin)
    if n == 0 or k <= 0:
        return float("nan")
    k = min(int(k), n)
    idx = np.argsort(-np.asarray(margin, dtype=float), kind="stable")[:k]
    return float(np.asarray(correct, dtype=float)[idx].mean())


def tpr_at_fpr(margin: np.ndarray, correct: np.ndarray, fpr: float) -> float:
    margin = np.asarray(margin, dtype=float)
    correct = np.asarray(correct, dtype=float)
    pos, neg = margin[correct == 1], margin[correct == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thr = np.quantile(neg, 1.0 - float(fpr))
    return float((pos >= thr).mean())


def _packed(stat_fn, arrays: List[np.ndarray], cluster_ids, n_boot: int) -> Dict:
    res = paired_bootstrap_statistic(arrays, stat_fn, cluster_ids=cluster_ids,
                                     n_boot=n_boot, seed=BOOT_SEED)
    return _pack_bootstrap("v", res)


def _selection_result(score, outcome, clusters, k, cfg):
    budgets = list(cfg["instrument"]["budgets_k"])
    budgets += [int(round(float(f)*len(score))) for f in cfg["instrument"].get("budget_curve_fracs", [])]
    budgets.append(int(k))
    return selective_bootstrap(score, outcome, clusters, budgets,
                               n_boot=int(cfg["stats"]["n_boot"]), seed=BOOT_SEED)[int(k)]


def _metric_row(packed: Dict, name: str, ident: Dict) -> Dict:
    return R._finish_metric(packed, "v", name, "percent", R._ident(**ident))


def selective_row(score: np.ndarray, outcome: np.ndarray, clusters: np.ndarray, k: int,
                  cfg: dict, name: str, ident: Dict) -> Dict:
    packed = _pack_bootstrap("v", _selection_result(score, outcome, clusters, k, cfg))
    return _metric_row(packed, name, {**ident, "budget_k": int(k)})


def paired_selective_delta(score_a, outcome_a, score_b, outcome_b, clusters, k: int,
                           cfg: dict, ident: Dict, name: str = "selective_precision") -> Dict:
    a = _selection_result(score_a, outcome_a, clusters, k, cfg)
    b = _selection_result(score_b, outcome_b, clusters, k, cfg)
    boot = b["boot"] - a["boot"]
    std, lo, hi = _percentile_ci(boot)
    res = {"point": b["point"] - a["point"], "mean": float(boot.mean()),
           "std": std, "ci_lower": lo, "ci_upper": hi, "n": len(score_a),
           "n_boot_finite": int(np.isfinite(boot).sum()), "boot": boot}
    packed = _pack_bootstrap("v", res)
    boot, point = res.get("boot", np.array([])), res["point"]
    if boot.size and np.isfinite(point):
        p = float(np.mean(np.abs(boot - point) >= abs(point)))
        packed["p_raw"] = keep_p(max(p, 1.0 / boot.size))
    else:
        packed["p_raw"] = float("nan")
    return R._finish_metric(packed, "v", name, "percentage_points",
                            R._ident(**{**ident, "budget_k": int(k)}))


def run_instrument(records: pd.DataFrame, cfg: dict, extra_ident: Optional[Dict] = None) -> List[Dict]:
    icfg = cfg["instrument"]
    n_boot = int(cfg["stats"]["n_boot"])
    ident = {
        "model": records["model"].iloc[0], "cohort": records["cohort"].iloc[0],
        "transform": records["transform"].iloc[0], "variant": records["variant"].iloc[0],
        "split": records["split"].iloc[0],
    }
    ident.update(extra_ident or {})

    statistics = list(icfg.get("decision_statistics", ["margin"]))
    default_stat = str(icfg.get("default_statistic", "margin"))
    margin = decision_statistic(records, default_stat)
    correct = records["correct"].to_numpy(dtype=float)
    rank = records["rank"].to_numpy(dtype=float)
    unit = str(cfg["stats"].get("cluster_unit", "patient_id"))
    if unit not in records.columns:
        raise KeyError(f"stats.cluster_unit={unit!r} is not a column of the records; "
                       "the resampling unit must exist or the bootstrap silently "
                       "falls back to resampling rows.")
    clusters = records[unit].astype(str).to_numpy()
    out: List[Dict] = []

    nc = records["n_corpus"].to_numpy(dtype=float)
    varies = bool(nc.max() > nc.min())
    if varies:
        n_corpus = int(np.median(nc))
        chance = float(np.mean(1.0 / nc))
    else:
        n_corpus = int(nc[0])
        chance = 1.0 / n_corpus

    def chance_of_top(stat_values: np.ndarray, k: int) -> float:
        if not varies:
            return chance
        idx = np.argsort(-np.asarray(stat_values, dtype=float), kind="stable")[:min(int(k), len(nc))]
        return float(np.mean(1.0 / nc[idx]))

    out.append(R.raw_stat(n_corpus, name="n_corpus", n=len(records), **ident))
    out.append(R.raw_stat(chance, name="chance_precision", n=len(records), **ident))

    for k in icfg["budgets_k"]:
        if k > len(records):
            continue
        packed = _pack_bootstrap("v", _selection_result(margin, correct, clusters, k, cfg))
        out.append(_metric_row(packed, "selective_precision", {**ident, "budget_k": int(k)}))
        hits = int(round(selective_precision(margin, correct, k) * k))
        out.append(R.raw_stat(hits, name="selective_hits", n=int(k),
                              **{**ident, "budget_k": int(k)}))
        prec = selective_precision(margin, correct, k)
        ref = chance_of_top(margin, k)
        out.append(R.raw_stat(prec / ref if ref > 0 else np.nan,
                              name="fold_over_chance", n=int(k),
                              **{**ident, "budget_k": int(k), "at": "selective"}))

    for frac in icfg.get("budget_curve_fracs", []):
        k = int(round(float(frac) * len(records)))
        if k < 1:
            continue
        packed = _pack_bootstrap("v", _selection_result(margin, correct, clusters, k, cfg))
        out.append(_metric_row(packed, "selective_precision",
                               {**ident, "budget_k": k, "budget_frac": float(frac)}))

    for fpr in icfg.get("fpr_levels", []):
        packed = _packed(lambda m, c, _f=float(fpr): tpr_at_fpr(m, c, _f),
                         [margin, correct], clusters, n_boot)
        out.append(_metric_row(packed, "tpr_at_fpr", {**ident, "fpr": float(fpr)}))

    out.append(R.metric(correct, name="recall_at_1", cluster_ids=clusters,
                        n_boot=n_boot, **ident))
    out.append(R.metric((rank <= 5).astype(float), name="recall_at_5",
                        cluster_ids=clusters, n_boot=n_boot, **ident))
    out.append(R.metric((rank <= 10).astype(float), name="recall_at_10",
                        cluster_ids=clusters, n_boot=n_boot, **ident))
    out.append(R.metric(records["reciprocal_rank"].to_numpy(dtype=float),
                        name="mrr", cluster_ids=clusters, n_boot=n_boot, **ident))
    out.append(R.raw_stat(float(correct.mean()) / chance if chance > 0 else np.nan,
                          name="fold_over_chance", n=len(records),
                          **{**ident, "at": "recall_at_1"}))

    out.append(R.metric(correct, name="recovered_fraction", cluster_ids=clusters,
                        n_boot=n_boot, **ident))

    if "tied_at_top" in records.columns:
        tied = pd.to_numeric(records["tied_at_top"], errors="coerce").fillna(0).to_numpy() > 0
        out.append(R.raw_stat(int(tied.sum()), name="tied_at_top_count",
                              n=len(records), **ident))
        if tied.any() and (~tied).sum() >= 10:
            keep = ~tied
            out.append(R.metric(correct[keep], name="recall_at_1", cluster_ids=clusters[keep],
                                n_boot=n_boot, **{**ident, "case_set": "untied"}))
            out.append(R.metric(correct[keep], name="recovered_fraction",
                                cluster_ids=clusters[keep], n_boot=n_boot,
                                **{**ident, "case_set": "untied"}))

    for stat in statistics:
        if stat == default_stat:
            continue
        alt = decision_statistic(records, stat)
        n_undef = int((~np.isfinite(alt)).sum())
        if n_undef:
            out.append(R.raw_stat(n_undef / len(alt), name="statistic_undefined_fraction",
                                  n=len(alt), **{**ident, "statistic": stat}))
        if n_undef == len(alt):
            continue
        for k in icfg["budgets_k"]:
            if k > len(records):
                continue
            packed = _pack_bootstrap("v", _selection_result(alt, correct, clusters, k, cfg))
            out.append(_metric_row(packed, "selective_precision",
                                   {**ident, "budget_k": int(k), "statistic": stat}))
    for row in out:
        if row.get("metric") == "selective_precision" and not row.get("statistic"):
            row["statistic"] = default_stat
    return out


def _query_uniform(case_ids: List[str], seed: int) -> np.ndarray:
    out = np.empty(len(case_ids), dtype=float)
    for i, cid in enumerate(case_ids):
        h = hashlib.blake2b(f"{seed}|{cid}".encode(), digest_size=8).digest()
        out[i] = int.from_bytes(h, "big") / float(1 << 64)
    return out


def run_openset_instrument(records: pd.DataFrame, cfg: dict,
                           extra_ident: Optional[Dict] = None) -> List[Dict]:
    icfg = cfg["instrument"]
    ocfg = cfg.get("openset", {})
    n_boot = int(cfg["stats"]["n_boot"])
    ident = {
        "model": records["model"].iloc[0], "cohort": records["cohort"].iloc[0],
        "transform": records["transform"].iloc[0], "variant": records["variant"].iloc[0],
        "split": records["split"].iloc[0],
    }
    ident.update(extra_ident or {})
    unit = str(cfg["stats"].get("cluster_unit", "patient_id"))
    if unit not in records.columns:
        raise KeyError(f"stats.cluster_unit={unit!r} is not a column of the records")
    clusters = records[unit].astype(str).to_numpy()

    m_closed = records["margin"].to_numpy(dtype=float)
    m_open = records["margin_open"].to_numpy(dtype=float)
    correct = records["correct"].to_numpy(dtype=float)
    u = _query_uniform(records["case_id"].astype(str).tolist(), int(ocfg.get("seed", 0)))
    out: List[Dict] = []

    both = np.isfinite(m_closed) & np.isfinite(m_open)
    if int(both.sum()) >= 10:
        n = int(both.sum())
        out.append(R.auroc(np.concatenate([np.ones(n), np.zeros(n)]),
                           np.concatenate([m_closed[both], m_open[both]]),
                           name="openset_margin_auroc",
                           cluster_ids=np.concatenate([clusters[both], clusters[both]]),
                           n_boot=n_boot, **ident))
    out.append(R.raw_stat(int((~np.isfinite(m_open)).sum()), name="openset_undefined_count",
                          n=len(records), **ident))

    fracs = [float(f) for f in ocfg.get("absent_fracs", [0.25, 0.5, 1.0])]
    for frac in tqdm(fracs, desc=f"openset:{ident['cohort']}", unit="frac"):
        absent = u < frac
        obs = np.where(absent, m_open, m_closed)
        hit = np.where(absent, 0.0, correct)
        ok = np.isfinite(obs)
        fid = {**ident, "absent_frac": frac}
        out.append(R.raw_stat(float(absent[ok].mean()), name="openset_absent_share",
                              n=int(ok.sum()), **fid))
        for k in icfg["budgets_k"]:
            if k > int(ok.sum()):
                continue
            out.append(selective_row(obs[ok], hit[ok], clusters[ok], k, cfg,
                                     "selective_precision", fid))
            out.append(selective_row(obs[ok], absent[ok].astype(float), clusters[ok], k, cfg,
                                     "openset_false_match_rate", fid))
    return out


def recovered_case_ids(records: pd.DataFrame) -> set:
    return set(records.loc[records["correct"] == 1, "case_id"].astype(str))


def agreement_rows(rec_a: pd.DataFrame, rec_b: pd.DataFrame, cfg: dict,
                   name_a: str, name_b: str) -> List[Dict]:
    common = sorted(set(rec_a["case_id"].astype(str)) & set(rec_b["case_id"].astype(str)))
    if not common:
        return []
    ia = rec_a.set_index(rec_a["case_id"].astype(str)).loc[common]
    ib = rec_b.set_index(rec_b["case_id"].astype(str)).loc[common]
    a = ia["correct"].to_numpy(dtype=float)
    b = ib["correct"].to_numpy(dtype=float)
    pool = ia["n_corpus"].to_numpy(dtype=float)
    ident = {"cohort": rec_a["cohort"].iloc[0], "transform": rec_a["transform"].iloc[0],
             "variant": rec_a["variant"].iloc[0], "split": rec_a["split"].iloc[0],
             "model_a": name_a, "model_b": name_b}

    pair = f"{name_a}_vs_{name_b}"
    n_strata = int(cfg["instrument"].get("overlap_strata", 10))
    overlap = R.stratified_overlap(a == 1, b == 1, pool, name="recovered_jaccard",
                                   comparison=pair, n_strata=n_strata, **ident)
    parts = overlap.pop("_parts")
    n11 = float(((a == 1) & (b == 1)).sum())
    return [
        overlap,
        R.raw_stat(parts["odds_ratio_mh"], name="recovered_odds_ratio_mh",
                   n=len(common), p_raw=parts["p_raw"], **{**ident, "comparison": pair}),
        R.raw_stat(n11, name="recovered_both", n=len(common),
                   **{**ident, "comparison": pair}),
        R.raw_stat(parts["expected"], name="recovered_both_expected_stratified",
                   n=len(common), **{**ident, "comparison": pair}),
        R.raw_stat(parts["n_strata"], name="recovered_overlap_n_strata",
                   n=len(common), **{**ident, "comparison": pair}),
    ]
