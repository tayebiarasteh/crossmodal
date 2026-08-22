"""
Inference/stats_utils.py
Created on July 14, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

from typing import List, Optional

import numpy as np

N_BOOT    = 1000
N_PERM    = 1000
BOOT_SEED = 0


def assert_resampling_config(cfg: dict) -> None:
    st = cfg.get("stats", {}) or {}
    mismatch = {}
    for key, const, name in (("n_boot", N_BOOT, "N_BOOT"),
                             ("n_perm", N_PERM, "N_PERM"),
                             ("boot_seed", BOOT_SEED, "BOOT_SEED")):
        if key in st and int(st[key]) != int(const):
            mismatch[key] = (int(st[key]), f"{name}={const}")
    if mismatch:
        raise RuntimeError(
            f"[stats] config.stats disagrees with Inference/stats_utils.py: "
            f"{mismatch}. The locked contract is ONE resampling count project-wide, "
            f"and these two are the only places it may live. Change both, or the "
            f"reported methods will not describe the runs.")

_MAX_CHUNK_ELEMS = 64_000_000


def _checked_clusters(cluster_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(cluster_ids)
    if ids.dtype.kind in "fc":
        bad = ~np.isfinite(ids)
    else:
        flat = ids.astype(str)
        bad = np.isin(np.char.strip(np.char.lower(flat)), ("", "nan", "none", "<na>"))
    n_bad = int(bad.sum())
    if n_bad:
        raise ValueError(
            f"[stats] {n_bad} of {len(ids)} rows carry no cluster id. The resampling unit is the "
            "patient, and a missing id would collapse every affected row into one cluster, which "
            "silently changes both n and the interval. Fix the manifest instead of dropping the "
            "check.")
    return ids


def _cluster_sums_setup(values: np.ndarray, cluster_ids: np.ndarray):
    unique_c, inverse = np.unique(_checked_clusters(cluster_ids), return_inverse=True)
    K = len(unique_c)
    counts = np.bincount(inverse, minlength=K).astype(float)
    sums = np.bincount(inverse, weights=values, minlength=K)
    return K, inverse, counts, sums


def _cluster_bootstrap_means(
    value_arrays: List[np.ndarray], cluster_ids: np.ndarray, n_boot: int, seed: int,
) -> List[np.ndarray]:
    K, inverse, counts, _ = _cluster_sums_setup(value_arrays[0], cluster_ids)
    rng = np.random.RandomState(seed)
    draw = rng.randint(0, K, size=(n_boot, K))
    total_count = counts[draw].sum(axis=1)
    results = []
    for values in value_arrays:
        sums = np.bincount(inverse, weights=values, minlength=K)
        total_sum = sums[draw].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            results.append(total_sum / total_count)
    return results


def _cluster_pad_index(cluster_ids: np.ndarray):
    unique_c, inverse = np.unique(_checked_clusters(cluster_ids), return_inverse=True)
    K = len(unique_c)
    counts = np.bincount(inverse, minlength=K)
    max_size = int(counts.max()) if K > 0 else 0
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    cluster_start = np.zeros(K, dtype=np.int64)
    cluster_start[1:] = np.cumsum(counts)[:-1]
    pos_within = np.arange(len(inverse)) - cluster_start[sorted_inverse]
    padded = np.zeros((K, max_size), dtype=np.int64)
    valid = np.zeros((K, max_size), dtype=bool)
    padded[sorted_inverse, pos_within] = order
    valid[sorted_inverse, pos_within] = True
    return K, padded, valid


def _iter_cluster_draws(padded: np.ndarray, valid: np.ndarray, n_boot: int, seed: int):
    K, _ = padded.shape
    sizes = valid.sum(axis=1).astype(np.int64)
    flat = padded[valid]
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    n_rows = int(sizes.sum())

    rng = np.random.RandomState(seed)
    per_replicate = max(2 * n_rows, 1)
    chunk_boot = max(1, min(n_boot, _MAX_CHUNK_ELEMS // per_replicate))
    done = 0
    while done < n_boot:
        c = min(chunk_boot, n_boot - done)
        draw = rng.randint(0, K, size=(c, K))
        lens = sizes[draw]
        totals = lens.sum(axis=1)
        max_total = int(totals.max()) if c else 0
        row_idx = np.zeros((c, max_total), dtype=np.int64)
        row_valid = np.zeros((c, max_total), dtype=bool)
        for i in range(c):
            li = lens[i]
            t = int(totals[i])
            if t == 0:
                continue
            csum = np.cumsum(li)
            off = np.repeat(starts[draw[i]], li)
            within = np.arange(t) - np.repeat(csum - li, li)
            row_idx[i, :t] = flat[off + within]
            row_valid[i, :t] = True
        yield row_idx, row_valid
        done += c


def bootstrap_proportion(
    values: np.ndarray,
    cluster_ids: Optional[np.ndarray] = None,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n": 0}
    if cluster_ids is not None:
        boot_means, = _cluster_bootstrap_means([values], np.asarray(cluster_ids), n_boot, seed)
    else:
        rng = np.random.RandomState(seed)
        idx = rng.randint(0, n, size=(n_boot, n))
        boot_means = values[idx].mean(axis=1)
    return {
        "point":    float(values.mean()),
        "std":      float(boot_means.std(ddof=1)),
        "ci_lower": float(np.percentile(boot_means, 2.5)),
        "ci_upper": float(np.percentile(boot_means, 97.5)),
        "n":        int(n),
    }


def bootstrap_scalar_metric_2arr(
    y: np.ndarray,
    p: np.ndarray,
    metric_fn,
    cluster_ids: Optional[np.ndarray] = None,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = int(len(y))
    if n == 0:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n": 0}
    point = float(metric_fn(y, p))
    boot = np.full(n_boot, np.nan, dtype=float)
    if cluster_ids is not None:
        _, padded, valid = _cluster_pad_index(np.asarray(cluster_ids))
        b = 0
        for row_idx, row_valid in _iter_cluster_draws(padded, valid, n_boot, seed):
            for i in range(row_idx.shape[0]):
                idx = row_idx[i][row_valid[i]]
                if b < n_boot:
                    boot[b] = metric_fn(y[idx], p[idx]) if len(idx) else np.nan
                    b += 1
    else:
        rng = np.random.RandomState(seed)
        for b in range(n_boot):
            idx = rng.randint(0, n, size=n)
            boot[b] = metric_fn(y[idx], p[idx])
    boot = boot[np.isfinite(boot)]
    if len(boot) == 0:
        return {"point": point, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n": n}
    return {"point": point, "std": float(np.std(boot, ddof=1)),
            "ci_lower": float(np.percentile(boot, 2.5)),
            "ci_upper": float(np.percentile(boot, 97.5)), "n": n}


def paired_bootstrap_diff(
    values_a: np.ndarray,
    values_b: np.ndarray,
    cluster_ids: Optional[np.ndarray] = None,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    n = len(values_a)
    assert len(values_b) == n, "Both arrays must have equal length."
    if n == 0:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "p_value": np.nan, "n": 0}
    point = float(values_a.mean() - values_b.mean())
    if cluster_ids is not None:
        mean_a, mean_b = _cluster_bootstrap_means(
            [values_a, values_b], np.asarray(cluster_ids), n_boot, seed
        )
        boot_diffs = mean_a - mean_b
    else:
        rng = np.random.RandomState(seed)
        idx = rng.randint(0, n, size=(n_boot, n))
        boot_diffs = values_a[idx].mean(axis=1) - values_b[idx].mean(axis=1)
    ci_lower   = float(np.percentile(boot_diffs, 2.5))
    ci_upper   = float(np.percentile(boot_diffs, 97.5))
    std        = float(boot_diffs.std(ddof=1))
    centered   = boot_diffs - point
    p_value    = float(np.mean(np.abs(centered) >= abs(point)))
    p_value    = max(p_value, 1.0 / n_boot)
    return {
        "point":    point,
        "std":      std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value":  p_value,
        "n":        int(n),
    }


def bootstrap_two_sample_diff(
    values_a: np.ndarray,
    values_b: np.ndarray,
    cluster_ids_a: Optional[np.ndarray] = None,
    cluster_ids_b: Optional[np.ndarray] = None,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    n_a, n_b = len(values_a), len(values_b)
    if n_a == 0 or n_b == 0:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n_a": n_a, "n_b": n_b}
    point = float(values_a.mean() - values_b.mean())
    if cluster_ids_a is not None:
        boot_a, = _cluster_bootstrap_means([values_a], np.asarray(cluster_ids_a), n_boot, seed)
    else:
        rng_a = np.random.RandomState(seed)
        idx_a = rng_a.randint(0, n_a, size=(n_boot, n_a))
        boot_a = values_a[idx_a].mean(axis=1)
    if cluster_ids_b is not None:
        boot_b, = _cluster_bootstrap_means([values_b], np.asarray(cluster_ids_b), n_boot, seed + 1)
    else:
        rng_b = np.random.RandomState(seed + 1)
        idx_b = rng_b.randint(0, n_b, size=(n_boot, n_b))
        boot_b = values_b[idx_b].mean(axis=1)
    boot = boot_a - boot_b
    return {
        "point":    point,
        "std":      float(boot.std(ddof=1)),
        "ci_lower": float(np.percentile(boot, 2.5)),
        "ci_upper": float(np.percentile(boot, 97.5)),
        "n_a":      int(n_a),
        "n_b":      int(n_b),
    }


def permutation_test_2groups(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_perm: int = N_PERM,
    seed: int = BOOT_SEED,
) -> float:
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    if len(values_a) < 2 or len(values_b) < 2:
        return float("nan")
    obs    = abs(float(values_a.mean() - values_b.mean()))
    if not np.isfinite(obs):
        return float("nan")
    pooled = np.concatenate([values_a, values_b])
    n_a    = len(values_a)
    rng    = np.random.RandomState(seed)
    count  = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        if abs(perm[:n_a].mean() - perm[n_a:].mean()) >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def permutation_test_kgroups(
    groups: List[np.ndarray],
    n_perm: int = N_PERM,
    seed: int = BOOT_SEED,
) -> float:
    groups = [np.asarray(g, dtype=float) for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return float("nan")
    sizes   = [len(g) for g in groups]
    pooled  = np.concatenate(groups)
    n_total = len(pooled)
    k       = len(groups)

    def _f(gs: List[np.ndarray]) -> float:
        grand      = np.concatenate(gs).mean()
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in gs)
        ss_within  = sum(((g - g.mean()) ** 2).sum() for g in gs)
        if ss_within == 0:
            return float("inf")
        return (ss_between / (k - 1)) / (ss_within / (n_total - k))

    obs   = _f(groups)
    if not np.isfinite(obs):
        return float("nan")
    rng   = np.random.RandomState(seed)
    count = 0
    for _ in range(n_perm):
        perm  = rng.permutation(pooled)
        start = 0
        pgs   = []
        for s in sizes:
            pgs.append(perm[start:start + s])
            start += s
        if _f(pgs) >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    if n == 0:
        return np.array([])
    nan_mask  = np.isnan(p_values)
    valid_idx = np.where(~nan_mask)[0]
    if len(valid_idx) == 0:
        return p_values.copy()
    valid_p  = p_values[valid_idx]
    m        = len(valid_p)
    order    = np.argsort(valid_p)
    p_sorted = valid_p[order]
    adj = p_sorted * m / (np.arange(m, dtype=float) + 1)
    for i in range(m - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    adj = np.minimum(adj, 1.0)
    inv_order = np.empty(m, dtype=int)
    inv_order[order] = np.arange(m)
    result = p_values.copy()
    result[valid_idx] = adj[inv_order]
    return result


def _percentile_ci(boot: np.ndarray) -> tuple:
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return (np.nan, np.nan, np.nan)
    std = float(np.std(boot, ddof=1)) if boot.size > 1 else np.nan
    return (std, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def paired_bootstrap_statistic(
    arrays: List[np.ndarray],
    stat_fn,
    cluster_ids: Optional[np.ndarray] = None,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    return_boot: bool = False,
) -> dict:
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    n = len(arrays[0])
    assert all(len(a) == n for a in arrays), "all arrays must be equal length"
    if n == 0:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n": 0}
    point = float(stat_fn(*arrays))
    rng   = np.random.RandomState(seed)
    boot  = np.empty(n_boot, dtype=float)

    if cluster_ids is not None:
        unique_c, inverse = np.unique(_checked_clusters(cluster_ids), return_inverse=True)
        K = len(unique_c)
        order = np.argsort(inverse, kind="stable")
        counts = np.bincount(inverse, minlength=K)
        cluster_rows = np.split(order, np.cumsum(counts)[:-1])
        for b in range(n_boot):
            draw = rng.randint(0, K, size=K)
            resample_idx = np.concatenate([cluster_rows[k] for k in draw])
            resampled = [a[resample_idx] for a in arrays]
            try:
                boot[b] = stat_fn(*resampled)
            except Exception:
                boot[b] = np.nan
    else:
        idx = rng.randint(0, n, size=(n_boot, n))
        for b in range(n_boot):
            resampled = [a[idx[b]] for a in arrays]
            try:
                boot[b] = stat_fn(*resampled)
            except Exception:
                boot[b] = np.nan

    std, lo, hi = _percentile_ci(boot)
    out = {"point": point, "std": std, "ci_lower": lo, "ci_upper": hi, "n": int(n)}
    if return_boot:
        out["boot"] = boot[np.isfinite(boot)]
    return out


def _rank_rows(mat: np.ndarray) -> np.ndarray:
    order = np.argsort(mat, axis=1, kind="quicksort")
    ranks = np.empty_like(order, dtype=np.float64)
    row_idx = np.arange(mat.shape[0])[:, None]
    ranks[row_idx, order] = np.arange(1, mat.shape[1] + 1)[None, :]
    return ranks


def bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    cluster_ids: Optional[np.ndarray] = None,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    from scipy.stats import spearmanr
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 4:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n": int(n)}
    point = float(spearmanr(x, y).correlation)

    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)[mask]
        K, padded, valid = _cluster_pad_index(cluster_ids)
        boot = np.empty(n_boot, dtype=float)
        pos = 0
        for row_idx, row_valid in _iter_cluster_draws(padded, valid, n_boot, seed):
            c = row_idx.shape[0]
            x_g, y_g = x[row_idx], y[row_idx]
            rx, ry = _rank_rows(x_g), _rank_rows(y_g)
            n_valid = row_valid.sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                rx_mean = (rx * row_valid).sum(axis=1, keepdims=True) / n_valid
                ry_mean = (ry * row_valid).sum(axis=1, keepdims=True) / n_valid
            rx_c = (rx - rx_mean) * row_valid
            ry_c = (ry - ry_mean) * row_valid
            num = (rx_c * ry_c).sum(axis=1)
            den = np.sqrt((rx_c ** 2).sum(axis=1) * (ry_c ** 2).sum(axis=1))
            with np.errstate(invalid="ignore", divide="ignore"):
                boot[pos:pos + c] = num / den
            pos += c
    else:
        rng = np.random.RandomState(seed)
        idx = rng.randint(0, n, size=(n_boot, n))
        rx, ry = _rank_rows(x[idx]), _rank_rows(y[idx])
        rx_c = rx - rx.mean(axis=1, keepdims=True)
        ry_c = ry - ry.mean(axis=1, keepdims=True)
        num = (rx_c * ry_c).sum(axis=1)
        den = np.sqrt((rx_c ** 2).sum(axis=1) * (ry_c ** 2).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = num / den

    std, lo, hi = _percentile_ci(boot)
    return {"point": point, "std": std, "ci_lower": lo, "ci_upper": hi, "n": int(n)}


def bootstrap_slope(
    x: np.ndarray,
    y: np.ndarray,
    cluster_ids: Optional[np.ndarray] = None,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n": int(len(x))}

    def _slope(a, b):
        A = np.vstack([a, np.ones_like(a)]).T
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        return coef[0]
    cids = np.asarray(cluster_ids)[mask] if cluster_ids is not None else None
    return paired_bootstrap_statistic([x, y], _slope, cluster_ids=cids,
                                      n_boot=n_boot, seed=seed)


def _auroc_batch(
    yt: np.ndarray, ys: np.ndarray, valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    n = yt.shape[1]
    if valid_mask is not None:
        ys = np.where(valid_mask, ys, -np.inf)
        yt = np.where(valid_mask, yt, 0.0)
        n_valid = valid_mask.sum(axis=1)
        n_invalid = n - n_valid
    else:
        n_valid = n
        n_invalid = 0

    ranks = _rank_rows(ys)
    n_pos = yt.sum(axis=1)
    n_neg = n_valid - n_pos
    sum_pos_ranks = (ranks * yt).sum(axis=1)
    if valid_mask is not None:
        sum_pos_ranks = sum_pos_ranks - n_pos * n_invalid
    denom = n_pos * n_neg
    with np.errstate(invalid="ignore", divide="ignore"):
        auroc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / denom
    auroc[(n_pos == 0) | (n_neg == 0)] = np.nan
    return auroc


def _bfill_last_valid(arr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    n_boot, n = arr.shape
    idx = np.where(valid_mask, np.arange(n)[None, :], n)
    idx_rev = idx[:, ::-1]
    filled_idx = np.minimum.accumulate(idx_rev, axis=1)[:, ::-1]
    row_idx = np.arange(n_boot)[:, None]
    return arr[row_idx, filled_idx]


def _average_precision_batch(
    yt: np.ndarray, ys: np.ndarray, valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    n_boot, n = yt.shape
    if valid_mask is not None:
        ys = np.where(valid_mask, ys, -np.inf)
        yt = np.where(valid_mask, yt, 0.0)

    order = np.argsort(-ys, axis=1, kind="stable")
    row_idx = np.arange(n_boot)[:, None]
    ys_sorted = ys[row_idx, order]
    yt_sorted = yt[row_idx, order]

    is_group_end = np.ones((n_boot, n), dtype=bool)
    is_group_end[:, :-1] = ys_sorted[:, :-1] != ys_sorted[:, 1:]

    tp_cum = np.cumsum(yt_sorted, axis=1)
    rank = np.arange(1, n + 1)[None, :]
    precision_at_k = tp_cum / rank
    group_end_precision = _bfill_last_valid(
        np.where(is_group_end, precision_at_k, 0.0), is_group_end
    )

    n_pos = yt.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ap = (group_end_precision * yt_sorted).sum(axis=1) / n_pos
    ap[n_pos == 0] = np.nan
    return ap


def bootstrap_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    cluster_ids: Optional[np.ndarray] = None,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    from sklearn.metrics import roc_auc_score
    y_true  = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    n = len(y_true)
    if n < 2 or len(np.unique(y_true)) < 2:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n": int(n)}
    point = float(roc_auc_score(y_true, y_score))

    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)[mask]
        K, padded, valid = _cluster_pad_index(cluster_ids)
        boot = np.empty(n_boot, dtype=float)
        pos = 0
        for row_idx, row_valid in _iter_cluster_draws(padded, valid, n_boot, seed):
            c = row_idx.shape[0]
            boot[pos:pos + c] = _auroc_batch(y_true[row_idx], y_score[row_idx],
                                             valid_mask=row_valid)
            pos += c
    else:
        rng  = np.random.RandomState(seed)
        idx  = rng.randint(0, n, size=(n_boot, n))
        boot = _auroc_batch(y_true[idx], y_score[idx])

    std, lo, hi = _percentile_ci(boot)
    return {"point": point, "std": std, "ci_lower": lo, "ci_upper": hi, "n": int(n)}


def bootstrap_average_precision(
    y_true: np.ndarray,
    y_score: np.ndarray,
    cluster_ids: Optional[np.ndarray] = None,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    from sklearn.metrics import average_precision_score
    y_true  = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    n = len(y_true)
    if n < 2 or y_true.sum() == 0:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n": int(n)}
    point = float(average_precision_score(y_true, y_score))

    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)[mask]
        K, padded, valid = _cluster_pad_index(cluster_ids)
        boot = np.empty(n_boot, dtype=float)
        pos = 0
        for row_idx, row_valid in _iter_cluster_draws(padded, valid, n_boot, seed):
            c = row_idx.shape[0]
            boot[pos:pos + c] = _average_precision_batch(y_true[row_idx], y_score[row_idx],
                                                         valid_mask=row_valid)
            pos += c
    else:
        rng  = np.random.RandomState(seed)
        idx  = rng.randint(0, n, size=(n_boot, n))
        boot = _average_precision_batch(y_true[idx], y_score[idx])

    std, lo, hi = _percentile_ci(boot)
    return {"point": point, "std": std, "ci_lower": lo, "ci_upper": hi, "n": int(n)}


def bootstrap_auroc_diff(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    cluster_ids: Optional[np.ndarray] = None,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    from sklearn.metrics import roc_auc_score
    y_true  = np.asarray(y_true, dtype=float)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(score_a) & np.isfinite(score_b)
    y_true, score_a, score_b = y_true[mask], score_a[mask], score_b[mask]
    n = len(y_true)
    if n < 2 or len(np.unique(y_true)) < 2:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "p_value": np.nan, "n": int(n)}
    point = float(roc_auc_score(y_true, score_a) - roc_auc_score(y_true, score_b))

    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)[mask]
        K, padded, valid = _cluster_pad_index(cluster_ids)
        boot = np.empty(n_boot, dtype=float)
        pos = 0
        for row_idx, row_valid in _iter_cluster_draws(padded, valid, n_boot, seed):
            c = row_idx.shape[0]
            yt_b = y_true[row_idx]
            boot[pos:pos + c] = (
                _auroc_batch(yt_b, score_a[row_idx], valid_mask=row_valid)
                - _auroc_batch(yt_b, score_b[row_idx], valid_mask=row_valid)
            )
            pos += c
    else:
        rng  = np.random.RandomState(seed)
        idx  = rng.randint(0, n, size=(n_boot, n))
        yt_b = y_true[idx]
        boot = _auroc_batch(yt_b, score_a[idx]) - _auroc_batch(yt_b, score_b[idx])

    finite = boot[np.isfinite(boot)]
    if finite.size == 0:
        return {"point": point, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "p_value": np.nan, "n": int(n)}
    std = float(finite.std(ddof=1)) if finite.size > 1 else np.nan
    ci_lower = float(np.percentile(finite, 2.5))
    ci_upper = float(np.percentile(finite, 97.5))
    centered = finite - point
    p_value = float(np.mean(np.abs(centered) >= abs(point)))
    p_value = max(p_value, 1.0 / max(finite.size, 1))
    return {"point": point, "std": std, "ci_lower": ci_lower,
            "ci_upper": ci_upper, "p_value": p_value, "n": int(n)}
