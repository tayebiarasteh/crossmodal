"""
Inference/report_utils.py
Created on July 14, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from Inference.stats_utils import (
    N_BOOT, N_PERM, BOOT_SEED,
    bootstrap_proportion, paired_bootstrap_diff, bootstrap_two_sample_diff,
    paired_bootstrap_statistic, bootstrap_scalar_metric_2arr,
    permutation_test_2groups, permutation_test_kgroups, bh_fdr,
)


def fmt_pct(x: float) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float("nan")
    v = float(x)
    if np.isfinite(v) and abs(v) > 2.0:
        raise ValueError(
            f"[report] {v!r} was passed to the percent formatter, but a value on its "
            f"natural scale cannot exceed 2 in magnitude (levels sit in [0, 1], "
            f"differences and cosine-scale readouts in [-2, 2]). This is the "
            f"signature of a value that was ALREADY multiplied by 100 upstream: "
            f"scaling happens in exactly one place, and this is it. Emit the raw "
            f"proportion and let the formatter scale it once.")
    return round(v * 100.0, 1)


def fmt_one(x: float) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float("nan")
    return round(float(x), 1)


def keep_p(p: float) -> float:
    if p is None:
        return float("nan")
    return float(p)


def _pack(prefix: str, point, std, lo, hi, n, is_percent: bool) -> Dict:
    f = fmt_pct if is_percent else fmt_one
    return {
        f"{prefix}_mean":     f(point),
        f"{prefix}_std":      f(std),
        f"{prefix}_ci_low":   f(lo),
        f"{prefix}_ci_high":  f(hi),
        f"{prefix}_mean_raw": float(point) if point is not None and np.isfinite(point) else float("nan"),
        f"{prefix}_std_raw":  float(std)   if std   is not None and np.isfinite(std)   else float("nan"),
        f"{prefix}_ci_low_raw":  float(lo) if lo    is not None and np.isfinite(lo)    else float("nan"),
        f"{prefix}_ci_high_raw": float(hi) if hi    is not None and np.isfinite(hi)    else float("nan"),
        "n": int(n) if n is not None and not (isinstance(n, float) and np.isnan(n)) else 0,
    }


def _pack_stat(prefix: str, estimate, n) -> Dict:
    def _r(v):
        return round(float(v), 6) if v is not None and np.isfinite(v) else float("nan")
    return {
        f"{prefix}_value":     _r(estimate),
        f"{prefix}_value_raw": float(estimate) if estimate is not None and np.isfinite(estimate) else float("nan"),
        "n": int(n) if n is not None and not (isinstance(n, float) and np.isnan(n)) else 0,
    }


def _anova_f_stat(groups: List[np.ndarray]) -> float:
    groups = [np.asarray(g, dtype=float) for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return float("nan")
    k = len(groups)
    pooled  = np.concatenate(groups)
    n_total = len(pooled)
    grand   = pooled.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_within  = sum(((g - g.mean()) ** 2).sum() for g in groups)
    if ss_within == 0:
        return float("inf")
    return (ss_between / (k - 1)) / (ss_within / (n_total - k))


def report_metric(
    values: np.ndarray,
    prefix: str = "value",
    is_percent: bool = True,
    cluster_ids: Optional[np.ndarray] = None,
    n_boot: int = N_BOOT,
) -> Dict:
    res = bootstrap_proportion(np.asarray(values, dtype=float),
                               cluster_ids=cluster_ids, n_boot=n_boot)
    return _pack(prefix, res["point"], res["std"], res["ci_lower"],
                 res["ci_upper"], res["n"], is_percent)


def report_metric_statistic(
    values: np.ndarray,
    stat_fn,
    prefix: str = "value",
    is_percent: bool = True,
    cluster_ids: Optional[np.ndarray] = None,
    n_boot: int = N_BOOT,
) -> Dict:
    res = paired_bootstrap_statistic([np.asarray(values, dtype=float)], stat_fn,
                                     cluster_ids=cluster_ids, n_boot=n_boot)
    return _pack(prefix, res["point"], res["std"], res["ci_lower"],
                 res["ci_upper"], res["n"], is_percent)


def report_paired_diff(
    a: np.ndarray,
    b: np.ndarray,
    prefix: str = "diff",
    is_percent: bool = True,
    cluster_ids: Optional[np.ndarray] = None,
    n_boot: int = N_BOOT,
) -> Dict:
    res = paired_bootstrap_diff(np.asarray(a, float), np.asarray(b, float),
                                cluster_ids=cluster_ids, n_boot=n_boot)
    out = _pack(prefix, res["point"], res["std"], res["ci_lower"],
                res["ci_upper"], res["n"], is_percent)
    out["p_raw"] = keep_p(res["p_value"])
    out["p_fdr"] = float("nan")
    return out


def report_permutation_2(
    a: np.ndarray,
    b: np.ndarray,
    prefix: str = "diff",
    is_percent: bool = True,
    n_boot: int = N_BOOT,
    n_perm: int = N_PERM,
) -> Dict:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    boot = bootstrap_two_sample_diff(a, b, n_boot=n_boot)
    p    = permutation_test_2groups(a, b, n_perm=n_perm)
    out  = _pack(prefix, boot["point"], boot["std"], boot["ci_lower"],
                 boot["ci_upper"], boot["n_a"] + boot["n_b"], is_percent)
    f = fmt_pct if is_percent else fmt_one
    out[f"{prefix}_group_a_mean"] = f(np.nanmean(a)) if len(a) else float("nan")
    out[f"{prefix}_group_b_mean"] = f(np.nanmean(b)) if len(b) else float("nan")
    out["n_a"] = int(len(a))
    out["n_b"] = int(len(b))
    out["p_raw"] = keep_p(p)
    out["p_fdr"] = float("nan")
    return out


def report_auroc(
    y_true: np.ndarray, y_score: np.ndarray,
    prefix: str = "auroc", cluster_ids: Optional[np.ndarray] = None,
    n_boot: int = N_BOOT,
) -> Dict:
    from Inference.stats_utils import bootstrap_auroc
    res = bootstrap_auroc(np.asarray(y_true, float), np.asarray(y_score, float),
                          cluster_ids=cluster_ids, n_boot=n_boot)
    return _pack(prefix, res["point"], res["std"], res["ci_lower"],
                 res["ci_upper"], res["n"], is_percent=True)


def report_average_precision(
    y_true: np.ndarray, y_score: np.ndarray,
    prefix: str = "ap", cluster_ids: Optional[np.ndarray] = None,
    n_boot: int = N_BOOT,
) -> Dict:
    from Inference.stats_utils import bootstrap_average_precision
    res = bootstrap_average_precision(np.asarray(y_true, float), np.asarray(y_score, float),
                                      cluster_ids=cluster_ids, n_boot=n_boot)
    return _pack(prefix, res["point"], res["std"], res["ci_lower"],
                 res["ci_upper"], res["n"], is_percent=True)


def report_paired_auroc_diff(
    y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray,
    prefix: str = "auroc_diff", cluster_ids: Optional[np.ndarray] = None,
    n_boot: int = N_BOOT,
) -> Dict:
    from Inference.stats_utils import bootstrap_auroc_diff
    y_true  = np.asarray(y_true, float)
    score_a = np.asarray(score_a, float)
    score_b = np.asarray(score_b, float)
    mask = np.isfinite(y_true) & np.isfinite(score_a) & np.isfinite(score_b)
    y_true, score_a, score_b = y_true[mask], score_a[mask], score_b[mask]
    cids = np.asarray(cluster_ids)[mask] if cluster_ids is not None else None
    n = len(y_true)
    if n < 2 or len(np.unique(y_true)) < 2:
        out = _pack(prefix, np.nan, np.nan, np.nan, np.nan, n, is_percent=True)
        out["p_raw"] = float("nan"); out["p_fdr"] = float("nan")
        return out

    res = bootstrap_auroc_diff(y_true, score_a, score_b, cluster_ids=cids, n_boot=n_boot)
    out = _pack(prefix, res["point"], res["std"], res["ci_lower"],
                res["ci_upper"], res["n"], is_percent=True)
    out["p_raw"] = keep_p(res["p_value"])
    out["p_fdr"] = float("nan")
    return out


def report_spearman(x: np.ndarray, y: np.ndarray, prefix: str = "spearman") -> Dict:
    from scipy.stats import spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4:
        out = _pack_stat(prefix, np.nan, len(x))
        out["p_raw"] = float("nan"); out["p_fdr"] = float("nan")
        return out
    rho, p = spearmanr(x, y)
    out = _pack_stat(prefix, rho, len(x))
    out["p_raw"] = keep_p(p)
    out["p_fdr"] = float("nan")
    return out


def report_partial_spearman(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    prefix: str = "partial_spearman",
) -> Dict:
    from scipy.stats import spearmanr
    from sklearn.linear_model import LinearRegression
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    if len(x) < 5:
        out = _pack_stat(prefix, np.nan, len(x))
        out["p_raw"] = float("nan"); out["p_fdr"] = float("nan")
        return out

    def _resid(a, b):
        lr = LinearRegression().fit(b.reshape(-1, 1), a)
        return a - lr.predict(b.reshape(-1, 1))

    rx = _resid(x, z); ry = _resid(y, z)
    rho, p = spearmanr(rx, ry)
    out = _pack_stat(prefix, rho, len(x))
    out["p_raw"] = keep_p(p)
    out["p_fdr"] = float("nan")
    return out


def report_slope(x: np.ndarray, y: np.ndarray, prefix: str = "slope") -> Dict:
    from scipy.stats import linregress
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        out = _pack_stat(prefix, np.nan, len(x))
        out["p_raw"] = float("nan"); out["r_squared"] = float("nan")
        out["intercept_raw"] = float("nan"); out["p_fdr"] = float("nan")
        return out
    lr = linregress(x, y)
    out = _pack_stat(prefix, lr.slope, len(x))
    out["p_raw"] = keep_p(lr.pvalue)
    out["r_squared"] = round(float(lr.rvalue ** 2), 4)
    out["intercept_raw"] = float(lr.intercept)
    out["p_fdr"] = float("nan")
    return out


def _cluster_robust_slope(x: np.ndarray, y: np.ndarray, groups: np.ndarray):
    from scipy.stats import t as _t
    x = np.asarray(x, float); y = np.asarray(y, float)
    g = np.asarray(groups)
    n = len(x)
    xc = x - x.mean()
    sxx = float((xc * xc).sum())
    if sxx <= 0:
        return float("nan"), float("nan"), 0
    slope = float((xc * (y - y.mean())).sum() / sxx)
    resid = y - (float(y.mean() - slope * x.mean()) + slope * x)
    _, inv = np.unique(g, return_inverse=True)
    n_groups = int(inv.max()) + 1
    if n_groups < 3 or n < 3:
        return slope, float("nan"), n_groups
    per_group = np.bincount(inv, weights=xc * resid, minlength=n_groups)
    meat = float((per_group * per_group).sum())
    corr = (n_groups / (n_groups - 1.0)) * ((n - 1.0) / (n - 2.0))
    var = corr * meat / (sxx * sxx)
    if not np.isfinite(var) or var <= 0:
        return slope, float("nan"), n_groups
    stat = slope / np.sqrt(var)
    return slope, float(2.0 * _t.sf(abs(stat), df=n_groups - 1)), n_groups


def report_mixedlm_slope(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray,
    prefix: str = "mixedlm_slope",
) -> Dict:
    x = np.asarray(x, float); y = np.asarray(y, float)
    groups = np.asarray(groups)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y, groups = x[mask], y[mask], groups[mask]
    n = len(x)
    if n < 5 or len(np.unique(groups)) < 3:
        out = _pack_stat(prefix, np.nan, n)
        out["p_raw"] = float("nan"); out["p_fdr"] = float("nan")
        out["fit_method"] = "insufficient_groups"
        out[f"{prefix}_ols_crosscheck"] = float("nan")
        return out

    slope = np.nan
    p = np.nan
    method = "mixedlm_random_intercept"
    try:
        import statsmodels.formula.api as smf
        df = pd.DataFrame({"y": y, "x": x, "g": groups})
        fit = smf.mixedlm("y ~ x", df, groups=df["g"]).fit(reml=True)
        slope = float(fit.params["x"])
        p = float(fit.pvalues["x"])
    except Exception as e:
        slope, p, n_groups = _cluster_robust_slope(x, y, groups)
        method = "ols_cluster_robust"
        print(f"[report_mixedlm_slope] MixedLM did not fit ({type(e).__name__}: {e}); "
              f"used an OLS slope with a cluster-robust standard error over {n_groups} "
              f"groups instead. The row is tagged fit_method=ols_cluster_robust. Its "
              f"p-value accounts for correlation inside a group and is NOT a naive "
              f"least-squares p.")

    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    ols_slope = float(coef[0])

    out = _pack_stat(prefix, slope, n)
    out["p_raw"] = keep_p(p)
    out["p_fdr"] = float("nan")
    out["fit_method"] = method
    out["n_groups"] = int(len(np.unique(groups)))
    out[f"{prefix}_ols_crosscheck"] = ols_slope if np.isfinite(ols_slope) else float("nan")
    return out


def report_cramers_v(a: np.ndarray, b: np.ndarray, prefix: str = "cramers_v") -> Dict:
    from scipy.stats import chi2_contingency
    a = pd.Series(a).astype(str)
    b = pd.Series(b).astype(str)
    keep = (a != "nan") & (b != "nan")
    a, b = a[keep], b[keep]
    n = len(a)
    if n < 5 or a.nunique() < 2 or b.nunique() < 2:
        out = _pack_stat(prefix, np.nan, n)
        out["p_raw"] = float("nan"); out["p_fdr"] = float("nan")
        return out
    table = pd.crosstab(a, b).values
    chi2, p, _, _ = chi2_contingency(table)
    r, c = table.shape
    v = float(np.sqrt((chi2 / n) / max(min(r - 1, c - 1), 1)))
    out = _pack_stat(prefix, v, n)
    out["p_raw"] = keep_p(p)
    out["p_fdr"] = float("nan")
    return out


def report_kappa(
    rater_a: np.ndarray, rater_b: np.ndarray,
    prefix: str = "kappa", weights: Optional[str] = None,
    n_perm: int = N_PERM, seed: int = BOOT_SEED,
) -> Dict:
    from sklearn.metrics import cohen_kappa_score
    a = np.asarray(rater_a, dtype=float)
    b = np.asarray(rater_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask].astype(int), b[mask].astype(int)
    n = len(a)
    if n < 5 or (len(np.unique(a)) < 2 and len(np.unique(b)) < 2):
        out = _pack_stat(prefix, np.nan, n)
        out["p_raw"] = float("nan"); out["p_fdr"] = float("nan")
        return out

    obs = float(cohen_kappa_score(a, b, weights=weights))
    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_perm):
        if float(cohen_kappa_score(a, rng.permutation(b), weights=weights)) >= obs:
            count += 1
    out = _pack_stat(prefix, obs, n)
    out["p_raw"] = keep_p((count + 1) / (n_perm + 1))
    out["p_fdr"] = float("nan")
    out["kappa_weights"] = weights or "unweighted"
    return out


def report_permutation_k(
    groups: List[np.ndarray],
    prefix: str = "anova",
    n_perm: int = N_PERM,
) -> Dict:
    f_stat = _anova_f_stat(groups)
    p = permutation_test_kgroups(groups, n_perm=n_perm)
    k_valid = len([g for g in groups if len(g) >= 2])
    n_total = sum(len(g) for g in groups)
    out = _pack_stat(prefix, f_stat, n_total)
    out["k_groups"] = k_valid
    out["p_raw"] = keep_p(p)
    out["p_fdr"] = float("nan")
    return out


def report_hypergeom_overlap(
    mask_a: np.ndarray, mask_b: np.ndarray, prefix: str = "overlap",
) -> Dict:
    from scipy.stats import hypergeom
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    n = len(mask_a)
    k_a = int(mask_a.sum())
    k_b = int(mask_b.sum())
    k_obs = int((mask_a & mask_b).sum())
    union = k_a + k_b - k_obs
    jaccard_obs = k_obs / union if union > 0 else float("nan")

    if n == 0 or k_a == 0 or k_b == 0:
        out = _pack_stat(prefix, jaccard_obs, n)
        out[f"{prefix}_null_value"] = float("nan")
        out[f"{prefix}_excess"] = float("nan")
        out["p_raw"] = float("nan")
        out["p_fdr"] = float("nan")
        return out

    null_mean_intersection = k_a * k_b / n
    null_union = k_a + k_b - null_mean_intersection
    jaccard_null = null_mean_intersection / null_union if null_union > 0 else float("nan")
    p_upper = float(hypergeom.sf(k_obs - 1, n, k_a, k_b))

    out = _pack_stat(prefix, jaccard_obs, n)
    out[f"{prefix}_null_value"] = round(jaccard_null, 6) if np.isfinite(jaccard_null) else float("nan")
    out[f"{prefix}_excess"] = round(jaccard_obs - jaccard_null, 6) \
        if np.isfinite(jaccard_obs) and np.isfinite(jaccard_null) else float("nan")
    out["p_raw"] = keep_p(p_upper)
    out["p_fdr"] = float("nan")
    return out


def _quantile_strata(values: np.ndarray, n_strata: int) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if not np.isfinite(v).any():
        return np.zeros(len(v), dtype=int)
    edges = np.unique(np.nanquantile(v, np.linspace(0.0, 1.0, int(n_strata) + 1)))
    if len(edges) < 3:
        return np.zeros(len(v), dtype=int)
    lab = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2)
    lab[~np.isfinite(v)] = 0
    return lab.astype(int)


def report_stratified_overlap(
    mask_a: np.ndarray, mask_b: np.ndarray, strata_value: np.ndarray,
    prefix: str = "overlap", n_strata: int = 10, n_perm: int = N_PERM,
) -> Dict:
    from scipy.stats import hypergeom
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    n, k_a, k_b = len(mask_a), int(mask_a.sum()), int(mask_b.sum())
    k_obs = int((mask_a & mask_b).sum())
    union = k_a + k_b - k_obs
    jaccard_obs = k_obs / union if union > 0 else float("nan")

    out = _pack_stat(prefix, jaccard_obs, n)
    out[f"{prefix}_null_value"] = float("nan")
    out[f"{prefix}_excess"] = float("nan")
    out[f"{prefix}_expected"] = float("nan")
    out[f"{prefix}_odds_ratio_mh"] = float("nan")
    out[f"{prefix}_n_strata"] = 0
    out["p_raw"] = float("nan")
    out["p_fdr"] = float("nan")
    if n == 0 or k_a == 0 or k_b == 0:
        return out

    labels = _quantile_strata(strata_value, n_strata)
    rng = np.random.RandomState(BOOT_SEED)
    expected, mh_num, mh_den, used = 0.0, 0.0, 0.0, 0
    null_tot = np.zeros(int(n_perm), dtype=np.int64)
    for s in np.unique(labels):
        sel = labels == s
        n_s, a_s, b_s = int(sel.sum()), int(mask_a[sel].sum()), int(mask_b[sel].sum())
        if n_s == 0:
            continue
        used += 1
        n11 = float((mask_a[sel] & mask_b[sel]).sum())
        n10, n01 = float(a_s) - n11, float(b_s) - n11
        n00 = float(n_s) - n11 - n10 - n01
        expected += a_s * b_s / n_s
        mh_num += n11 * n00 / n_s
        mh_den += n10 * n01 / n_s
        if a_s > 0 and b_s > 0:
            null_tot += hypergeom.rvs(n_s, a_s, b_s, size=int(n_perm), random_state=rng)

    null_union = float(k_a + k_b) - null_tot.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        null_jacc = np.where(null_union > 0, null_tot / null_union, np.nan)
    jaccard_null = float(np.nanmean(null_jacc)) if np.isfinite(null_jacc).any() else float("nan")
    odds_mh = (mh_num + 0.5) / (mh_den + 0.5)

    out[f"{prefix}_null_value"] = round(jaccard_null, 6) if np.isfinite(jaccard_null) else float("nan")
    out[f"{prefix}_excess"] = round(jaccard_obs - jaccard_null, 6) \
        if np.isfinite(jaccard_obs) and np.isfinite(jaccard_null) else float("nan")
    out[f"{prefix}_expected"] = float(expected)
    out[f"{prefix}_odds_ratio_mh"] = float(odds_mh)
    out[f"{prefix}_n_strata"] = int(used)
    out["p_raw"] = keep_p((1.0 + float((null_tot >= k_obs).sum())) / (1.0 + int(n_perm)))
    return out


def ece_value(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    keep = np.isfinite(y) & np.isfinite(p)
    y, p = y[keep], p[keep]
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_binned = 0
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
        n_binned += int(m.sum())
        if m.sum() == 0:
            continue
        ece += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(ece / n_binned) if n_binned else float("nan")


def risk_coverage_auc_value(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    keep = np.isfinite(y) & np.isfinite(p)
    y, p = y[keep], p[keep]
    if len(y) < 5:
        return float("nan")
    order = np.argsort(-np.abs(p - 0.5))
    yo = y[order]; po = (p[order] > 0.5).astype(int)
    err = (yo != po).astype(float)
    step = max(1, len(y) // 20)
    covs, risks = [], []
    for k in range(step, len(y) + 1, step):
        covs.append(k / len(y)); risks.append(float(err[:k].mean()))
    return float(np.trapezoid(risks, covs)) if len(covs) >= 2 else float("nan")


def report_ece(y, p, prefix: str = "value", cluster_ids=None,
               n_boot: int = N_BOOT) -> Dict:
    res = bootstrap_scalar_metric_2arr(y, p, ece_value, cluster_ids=cluster_ids,
                                       n_boot=n_boot)
    return _pack(prefix, res["point"], res["std"], res["ci_lower"],
                 res["ci_upper"], res["n"], is_percent=True)


def report_risk_coverage_auc(y, p, prefix: str = "value", cluster_ids=None,
                             n_boot: int = N_BOOT) -> Dict:
    res = bootstrap_scalar_metric_2arr(y, p, risk_coverage_auc_value,
                                       cluster_ids=cluster_ids, n_boot=n_boot)
    return _pack(prefix, res["point"], res["std"], res["ci_lower"],
                 res["ci_upper"], res["n"], is_percent=True)


def report_raw_stat(value: float, prefix: str, n=None, p_raw=None) -> Dict:
    out = _pack_stat(prefix, value, n if n is not None else 0)
    out["p_raw"] = keep_p(p_raw) if p_raw is not None else float("nan")
    out["p_fdr"] = float("nan")
    return out


def add_fdr(
    df: pd.DataFrame,
    family_cols: Optional[List[str]] = None,
    p_col: str = "p_raw",
    out_col: str = "p_fdr",
) -> pd.DataFrame:
    df = df.copy()
    if p_col not in df.columns:
        return df
    if family_cols:
        present = [c for c in family_cols if c in df.columns]
        if not present:
            df[out_col] = bh_fdr(df[p_col].values)
        else:
            for _, idx in df.groupby(present, dropna=False).groups.items():
                sub = df.loc[idx, p_col].values
                df.loc[idx, out_col] = bh_fdr(sub)
    else:
        df[out_col] = bh_fdr(df[p_col].values)
    df["significant_fdr05"] = df[out_col].apply(
        lambda q: bool(q < 0.05) if pd.notna(q) else False
    )
    return df
