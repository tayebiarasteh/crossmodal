"""
aggregate/build_final_tables.py
Created on August 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import glob
import os
from typing import List

import pandas as pd

from config.serde import read_config
from data_loader.build_utils import write_manifest
from Inference.resume_utils import MissingInput


_EXPECTED = {
    "e1_linkage": "E1 selective linkage risk and cross-model agreement",
    "e2_conditioning": "E2 the instrument under auxiliary conditioning",
    "e3_report_map": "E3 which report covariates predict recoverability",
    "e4_exposure": "E4 what a successful linkage exposes",
    "e5_transform_curve": "E5 the privacy-against-content curve",
    "e6_model_defense": "E6 the model-side defense at the operating point, and the bypass",
    "e7_metric_disagreement": "E7 do the two metrics rank releases differently",
    "e8_no_model_adversary": "E8 what releasing a model adds over an adversary who has none",
}

_SIDE_FILES = {
    "e3_report_map": ["e3_map_agreement_rows.csv"],
}

_SUPERSEDED = ["table1_selective_linkage.csv"]


def _read_rows(cfg: dict, name: str) -> List[pd.DataFrame]:
    out = []
    base = os.path.join(cfg["paths"]["analysis_dir"], name)
    for fname in [f"{name}_rows.csv"] + _SIDE_FILES.get(name, []):
        path = os.path.join(base, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["source_analysis"] = name
        out.append(df)
    return out


def _utility_dirs(cfg: dict) -> List[str]:
    return sorted(os.path.basename(d)
                  for d in glob.glob(os.path.join(cfg["paths"]["analysis_dir"], "e6_*"))
                  if os.path.basename(d) not in _EXPECTED)


def _completeness(cfg: dict) -> pd.DataFrame:
    rows = []
    named = list(_EXPECTED.items()) + [(n, "E6 utility") for n in _utility_dirs(cfg)]
    for name, desc in named:
        path = os.path.join(cfg["paths"]["analysis_dir"], name, f"{name}_rows.csv")
        if not os.path.exists(path):
            state, n = "NEVER RAN", 0
        else:
            n = len(pd.read_csv(path))
            state = "ran, produced nothing" if n == 0 else "ok"
        rows.append({"analysis": name, "description": desc, "state": state, "n_rows": n})
    return pd.DataFrame(rows)


def _main_table(df: pd.DataFrame, cfg: dict, source: str = "e1_linkage",
                conditioning: str = "") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    m = df[(df["kind"] == "metric") & (df["source_analysis"] == source)]
    if "conditioning" in m.columns:
        col = m["conditioning"].astype(str)
        m = m[col == conditioning] if conditioning else m[col.isin(["", "nan"])]
    if m.empty:
        return pd.DataFrame()
    budgets = [int(k) for k in cfg["instrument"]["budgets_k"]]
    out = []
    for (model, cohort), g in m.groupby(["model", "cohort"]):
        row = {"model": model, "cohort": cohort, "conditioning": conditioning or "full_corpus"}
        r1 = g[g["metric"] == "recall_at_1"]
        if len(r1):
            row["recall_at_1_pct"] = r1["mean_pct"].iloc[0]
            row["recall_at_1_ci"] = f"[{r1['ci_low_pct'].iloc[0]}, {r1['ci_high_pct'].iloc[0]}]"
        for k in budgets:
            sp = g[(g["metric"] == "selective_precision")
                   & (pd.to_numeric(g["budget_k"], errors="coerce") == float(k))]
            if len(sp):
                row[f"selective_precision_at_{k}_pct"] = sp["mean_pct"].iloc[0]
                row[f"selective_precision_at_{k}_ci"] = (
                    f"[{sp['ci_low_pct'].iloc[0]}, {sp['ci_high_pct'].iloc[0]}]")
        st = df[(df["source_analysis"] == source) & (df["kind"] == "statistic")
                & (df["model"] == model) & (df["cohort"] == cohort)]
        if "conditioning" in st.columns:
            sc = st["conditioning"].astype(str)
            st = st[sc == conditioning] if conditioning else st[sc.isin(["", "nan"])]
        nc = st[st["measure"] == "n_corpus"]
        ch = st[st["measure"] == "chance_precision"]
        if len(nc):
            row["n_corpus_median"] = int(nc["value"].iloc[0])
        if len(ch):
            row["chance_pct"] = 100.0 * float(ch["value"].iloc[0])
        out.append(row)
    return pd.DataFrame(out).sort_values(["cohort", "model"]).reset_index(drop=True)


def build_final_tables(cfg_path: str) -> str:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    from Inference.rows import append_status
    res = cfg["paths"]["results_dir"]
    os.makedirs(res, exist_ok=True)
    status_dir, stage = cfg["paths"]["status_dir"], "final_tables"

    comp = _completeness(cfg)
    write_manifest(comp, os.path.join(res, "completeness.csv"))
    print(f"\n[{stage}] completeness", flush=True)
    for _, r in comp.iterrows():
        mark = "  " if r["state"] == "ok" else ">>"
        print(f"{mark} {r['analysis']:26s} {r['state']:22s} {r['n_rows']:6d} rows", flush=True)
    never = comp[comp["state"] == "NEVER RAN"]["analysis"].tolist()
    empty = comp[comp["state"] == "ran, produced nothing"]["analysis"].tolist()
    if never:
        append_status(status_dir, stage, f"NEVER RAN: {never}")
    if empty:
        append_status(status_dir, stage, f"ran and produced nothing: {empty}")

    frames = []
    for n in list(_EXPECTED) + _utility_dirs(cfg):
        frames += _read_rows(cfg, n)
    if not frames:
        raise MissingInput("no analysis has produced rows; nothing to consolidate")
    allrows = pd.concat(frames, ignore_index=True)
    write_manifest(allrows, os.path.join(res, "all_rows.csv"))

    from Inference.rows import NAME_ALIASES, NAME_MAP
    known = set(NAME_MAP) | set(NAME_ALIASES)
    unmapped = sorted({str(x) for x in pd.concat([allrows.get("metric", pd.Series(dtype=str)),
                                                  allrows.get("measure", pd.Series(dtype=str))])
                       .dropna().unique()} - known - {""})
    if unmapped:
        print(f"[{stage}] UNMAPPED names, add them to Inference/rows.NAME_MAP: {unmapped}", flush=True)
        append_status(status_dir, stage, f"unmapped names: {unmapped}")

    scope = pd.DataFrame([{"model": k, "hf_id": v["hf_id"], "type": v["type"],
                           "specialization": v.get("specialization", ""),
                           "axis_role": v.get("axis_role", ""),
                           "image_size": v.get("image_size"),
                           "embed_dim": v.get("embed_dim"),
                           "gated": bool(v.get("gated", False))}
                          for k, v in cfg["models"].items()])
    write_manifest(scope, os.path.join(res, "table_scope_models.csv"))

    for fname, src, cond in (
            ("table1_selective_linkage_full_corpus.csv", "e1_linkage", ""),
            ("table1_selective_linkage_conditioned.csv", "e2_conditioning",
             str(cfg["conditioning"]["analysis_axes"]["e7_ranking"][0]))):
        main = _main_table(allrows, cfg, source=src, conditioning=cond)
        if main.empty:
            print(f"[{stage}] {fname}: no rows at this operating point", flush=True)
            continue
        write_manifest(main, os.path.join(res, fname))
        print(f"[{stage}] {fname} ({len(main)} rows)", flush=True)

    for fname in _SUPERSEDED:
        stale = os.path.join(res, fname)
        if os.path.exists(stale):
            os.remove(stale)
            print(f"[{stage}] removed superseded {fname}; it was built at one operating "
                  f"point and is replaced by one file per operating point", flush=True)
            append_status(status_dir, stage, f"removed superseded output {fname}")

    for kind, fname in (("metric", "all_metrics.csv"), ("statistic", "all_statistics.csv")):
        sub = allrows[allrows["kind"] == kind]
        if len(sub):
            write_manifest(sub, os.path.join(res, fname))

    append_status(status_dir, stage,
                  f"consolidated {len(allrows)} rows from {len(frames)} analyses; "
                  f"never ran: {len(never)}, empty: {len(empty)}")
    print(f"[{stage}] {res}", flush=True)
    return res
