"""
Inference/selection_bootstrap.py
Created on October 30, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""
from collections import OrderedDict
import hashlib

import numpy as np

from Inference.stats_utils import BOOT_SEED, N_BOOT, _checked_clusters, _percentile_ci

_CACHE = OrderedDict()
_CACHE_SIZE = 12


def selective_bootstrap(score, outcome, clusters, budgets, n_boot=N_BOOT, seed=BOOT_SEED):
    score, outcome = np.asarray(score, float), np.asarray(outcome, float)
    budgets = np.asarray(sorted({int(k) for k in budgets if int(k) > 0}), int)
    if len(score) != len(outcome) or len(score) != len(clusters):
        raise ValueError("scores, outcomes, and patients must be aligned")
    if len(score) == 0 or not len(budgets):
        raise ValueError("selection requires observations and positive budgets")
    unique, inverse = np.unique(_checked_clusters(clusters), return_inverse=True)
    digest = hashlib.sha256()
    for a in (score, outcome, inverse, budgets, np.array([n_boot, seed])):
        digest.update(np.ascontiguousarray(a).tobytes())
    key = digest.hexdigest()
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]

    order = np.argsort(-score, kind="stable")
    sorted_score, y = score[order], outcome[order]
    sorted_cluster = inverse[order]
    ties = (sorted_score[1:] == sorted_score[:-1]) | (np.isnan(sorted_score[1:]) & np.isnan(sorted_score[:-1]))
    edges = np.r_[0, np.flatnonzero(~ties) + 1, len(score)]
    groups = np.repeat(np.arange(len(edges)-1), np.diff(edges))
    mixed = np.array([len(np.unique(y[a:b])) > 1 for a,b in zip(edges[:-1], edges[1:])])
    cluster_order = np.argsort(inverse, kind="stable")
    sizes = np.bincount(inverse, minlength=len(unique))
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    rng = np.random.RandomState(seed)
    boot = np.empty((int(n_boot), len(budgets)), float)
    fallbacks = 0
    for b in range(int(n_boot)):
        draw = rng.randint(0, len(unique), size=len(unique))
        weights = np.bincount(draw, minlength=len(unique))[sorted_cluster]
        cumulative = np.cumsum(weights)
        take = np.minimum(budgets, cumulative[-1])
        last = np.searchsorted(cumulative, take, side="left")
        group = groups[last]
        end = edges[group+1]-1
        tied_cut = mixed[group] & (cumulative[end] > take)
        if tied_cut.any():
            lens = sizes[draw]
            within = np.arange(int(lens.sum())) - np.repeat(np.cumsum(lens)-lens, lens)
            idx = cluster_order[np.repeat(starts[draw], lens) + within]
            selected = np.argsort(-score[idx], kind="stable")
            sums = np.cumsum(outcome[idx[selected]])
            boot[b] = sums[take-1] / take
            fallbacks += 1
        else:
            sums = np.cumsum(weights*y)
            boot[b] = (sums[last]-(cumulative[last]-take)*y[last]) / take
    sums = np.cumsum(y)
    results = {}
    for j,k in enumerate(budgets):
        vals = boot[:,j].copy()
        std,lo,hi = _percentile_ci(vals)
        vals.setflags(write=False)
        results[int(k)] = {"point": float(sums[min(k,len(y))-1]/min(k,len(y))),
                           "mean": float(vals.mean()), "std": std,
                           "ci_lower": lo, "ci_upper": hi, "n": len(y),
                           "n_boot_finite": int(np.isfinite(vals).sum()),
                           "boot": vals, "tie_fallbacks": fallbacks}
    _CACHE[key] = results
    while len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return results
