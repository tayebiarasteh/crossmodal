"""
analysis/e6_model_defense.py
Created on August 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""
from typing import Dict, List

import pandas as pd

from analysis._common import conditioning_axes, enabled_cohorts, run_analysis, unit_key
from config.serde import read_config
from linkage.conditioning import load_conditioned
from linkage.instrument import paired_selective_delta, run_instrument
from linkage.retrieval import load_records


def _variants(cfg: dict) -> List[str]:
    return ["off_shelf"] + list(cfg["heads"]["variants"].keys())


def _contrasts(cfg: dict) -> List[tuple]:
    known = _variants(cfg)
    pairs = [(v, "off_shelf") for v in known if v != "off_shelf"]
    for c in cfg["heads"].get("contrasts") or []:
        pair = (str(c["variant"]), str(c["baseline"]))
        for side in pair:
            if side not in known:
                raise ValueError(f"heads.contrasts names '{side}', which is not a release "
                                 f"variant. Known: {', '.join(known)}")
        if pair[0] == pair[1]:
            raise ValueError(f"heads.contrasts compares '{pair[0]}' with itself")
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def _paired_delta(a: pd.DataFrame, b: pd.DataFrame, cfg: dict, k: int,
                  ident: Dict) -> List[Dict]:
    common = sorted(set(a["case_id"].astype(str)) & set(b["case_id"].astype(str)))
    if len(common) < 100:
        raise ValueError(
            f"the two releases share only {len(common)} query images, and they score the same "
            f"cohort against the same corpus, so they should share all of them. Check that both "
            f"variants were re-embedded from the same manifest.")
    aa = a.set_index(a["case_id"].astype(str)).loc[common]
    bb = b.set_index(b["case_id"].astype(str)).loc[common]
    return [paired_selective_delta(
        aa["margin"].to_numpy(float), aa["correct"].to_numpy(float),
        bb["margin"].to_numpy(float), bb["correct"].to_numpy(float),
        aa["patient_id"].astype(str).to_numpy(), k, cfg, ident)]


def run_e6(cfg_path: str, force: bool = False) -> pd.DataFrame:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    budgets = [int(k) for k in cfg["instrument"]["budgets_k"]]
    conds = conditioning_axes(cfg, "e6_defense")
    variants = _variants(cfg)
    deltas = _contrasts(cfg)
    units = [unit_key(cohort=c, variant=v) for c in enabled_cohorts(cfg) for v in variants]
    units += [unit_key(cohort=c, delta=f"{v}_vs_{b}")
              for c in enabled_cohorts(cfg) for v, b in deltas]
    units += [unit_key(cohort=c, variant=v, cond=k)
              for c in enabled_cohorts(cfg) for v in variants for k in conds]
    units += [unit_key(cohort=c, delta=f"{v}_vs_{b}", cond=k)
              for c in enabled_cohorts(cfg) for v, b in deltas for k in conds]

    def compute(unit: str) -> List[Dict]:
        p = dict(x.split("-", 1) for x in unit.split("__"))
        cohort, cond = p["cohort"], p.get("cond")
        backbone = cfg["heads"]["backbone"]

        def records(variant: str):
            if cond:
                return load_conditioned(cfg, backbone, cohort, cond, variant=variant)
            return load_records(cfg, backbone, cohort, "T0", variant, "test")

        if "delta" in p:
            variant, baseline = p["delta"].split("_vs_", 1)
            fam = "delta" if baseline == "off_shelf" else "mechanism"
            ident = {"cohort": cohort, "variant": variant,
                     "comparison": f"{variant}_minus_{baseline}",
                     "experiment": "E6", "analysis": "defense_delta",
                     "fdr_family": f"E6_{fam}_{cohort}" + (f"_{cond}" if cond else "")}
            if cond:
                ident["conditioning"] = cond
            base, other = records(baseline), records(variant)
            rows = []
            for k in budgets:
                rows += _paired_delta(base, other, cfg, k, ident)
            return rows

        variant = p["variant"]
        rec = records(variant)
        rows = run_instrument(rec, cfg, extra_ident={"conditioning": cond} if cond else None)
        fam = f"E6_{cohort}_{cond}" if cond else f"E6_{cohort}"
        for r in rows:
            r["experiment"] = "E6"
            r["analysis"] = "defense_level"
            r["fdr_family"] = f"{fam}_{r.get('metric') or r.get('measure')}"
        return rows

    return run_analysis(cfg, "e6_model_defense", units, compute, force=force)
