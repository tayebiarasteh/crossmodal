"""
data_loader/build_paired_pool.py
Created on August 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.serde import read_config
from data_loader.build_utils import (assert_unique_case_ids, binarize_presence,
                                     make_case_ids, normalize_report_text,
                                     normalize_split, read_csv_defensively,
                                     report_pool, write_manifest)
from Inference.resume_utils import MissingInput


_SHARD_ROWS = 20000

_clean_text = normalize_report_text


def _study_year(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="Int64")
    s = series.astype(str).str.extract(r"(\d{4})", expand=False)
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _assign_patient_split(patients: np.ndarray, test_frac: float, valid_frac: float,
                          seed: int) -> Dict[str, str]:
    uniq = np.sort(np.asarray(patients).astype(str))
    uniq = np.unique(uniq)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(uniq)
    n = len(perm)
    n_test = int(round(test_frac * n))
    n_val = int(round(valid_frac * (n - n_test)))
    out = {}
    for p in perm[:n_test]:
        out[p] = "test"
    for p in perm[n_test:n_test + n_val]:
        out[p] = "val"
    for p in perm[n_test + n_val:]:
        out[p] = "train"
    return out


def _assert_no_leak(df: pd.DataFrame) -> None:
    if df["patient_id"].isna().any():
        raise ValueError("rows with no patient_id: it is the split key and the bootstrap cluster unit")
    spanning = df.groupby("patient_id")["split"].nunique()
    bad = spanning[spanning > 1]
    if len(bad):
        raise ValueError(f"{len(bad)} patient(s) span more than one split; the test set leaks. "
                         f"examples: {bad.index[:5].tolist()}")
    missing = {"train", "val", "test"} - set(df["split"].unique())
    if missing:
        raise ValueError(f"split(s) {sorted(missing)} are empty; a zero-size partition is absent "
                         "from value_counts, not present as 0")


def _fingerprint(path: str) -> dict:
    if not os.path.exists(path):
        return {"name": os.path.basename(path), "size": None, "mtime": None}
    st = os.stat(path)
    return {"name": os.path.basename(path), "size": int(st.st_size), "mtime": int(st.st_mtime)}


def _build_params(cfg: dict, cohort: str, ccfg: dict) -> dict:
    labels = cfg["labels"]
    split = cfg["split"]
    srcs = [ccfg["master_csv"]]
    if ccfg.get("report_mode") == "join_csv":
        srcs.append(ccfg["report_csv"])
    extra_cols = {"full": list(ccfg.get("report_full_cols") or []),
                  "sections": list(ccfg.get("report_section_cols") or []),
                  "impression": list(ccfg.get("report_impression_cols") or []),
                  "sections_from": ccfg.get("report_sections_from", "parser")}
    return {
        "cohort": cohort,
        "sources": [_fingerprint(p) for p in srcs],
        "label_policy": [labels["positive_code"], labels["negative_codes"], labels["exclude_codes"]],
        "keep_views": list(ccfg.get("keep_views") or []),
        "keep_modalities": list(ccfg.get("keep_modalities") or []),
        "split_rule": [split["test_frac"], split["valid_frac"], split["seed"],
                       bool(split["rebuild_from_patients"])],
        "min_report_chars": split["min_report_chars"],
        "findings": list(labels["findings"]),
        "report_cols": extra_cols,
        "study_key": [ccfg.get("study_col"), ccfg.get("study_from_path")],
    }


def _reports_from_files(master: pd.DataFrame, ccfg: dict, shard_dir: str,
                        status_dir: str, cohort: str) -> pd.DataFrame:
    from Inference.rows import append_status
    root = ccfg["report_root"]
    col = ccfg["report_col"]
    studies = (master[["study_id", col]]
               .dropna(subset=[col])
               .drop_duplicates(subset=["study_id"])
               .reset_index(drop=True))
    os.makedirs(shard_dir, exist_ok=True)
    n_shards = int(np.ceil(len(studies) / _SHARD_ROWS))
    frames = []
    for k in range(n_shards):
        out = os.path.join(shard_dir, f"reports_shard{k:04d}.csv")
        if os.path.exists(out):
            frames.append(read_csv_defensively(out, dtype={"study_id": str}))
            continue
        sub = studies.iloc[k * _SHARD_ROWS:(k + 1) * _SHARD_ROWS]
        rows = []
        for sid, rel in tqdm(zip(sub["study_id"].astype(str), sub[col].astype(str)),
                             total=len(sub), desc=f"{cohort}:reports shard {k + 1}/{n_shards}"):
            path = os.path.join(root, rel)
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    txt = f.read()
            except OSError:
                continue
            rows.append({"study_id": sid, "report_text": _clean_text(txt)})
        if not rows:
            append_status(status_dir, f"pool_{cohort}",
                          f"report shard {k} produced 0 rows; not cached as done")
            continue
        frame = pd.DataFrame(rows)
        tmp = out + ".tmp"
        frame.to_csv(tmp, index=False)
        os.replace(tmp, out)
        frames.append(frame)
        append_status(status_dir, f"pool_{cohort}", f"report shard {k + 1}/{n_shards}: {len(frame)} studies")
    if not frames:
        raise MissingInput(f"{cohort}: no report files resolved under {root}")
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["study_id"])


def _join_cols(df: pd.DataFrame, cols) -> pd.Series:
    parts = [c for c in (cols or []) if c in df.columns]
    if not parts:
        return pd.Series([""] * len(df), index=df.index)
    return df[parts].fillna("").astype(str).agg("\n".join, axis=1).map(_clean_text)


def _reports_from_join(ccfg: dict) -> pd.DataFrame:
    path = ccfg["report_csv"]
    if not os.path.exists(path):
        raise MissingInput(f"report csv not found: {path}")
    join = ccfg["report_join_col"]
    want = ([join] + list(ccfg.get("report_full_cols") or [])
            + list(ccfg.get("report_section_cols") or [])
            + list(ccfg.get("report_impression_cols") or []))
    head = read_csv_defensively(path, nrows=1)
    use = [c for c in dict.fromkeys(want) if c in head.columns]
    if join not in use:
        raise ValueError(f"{path} has no join column {join!r}")
    plus = read_csv_defensively(path, usecols=use)
    full = _join_cols(plus, ccfg.get("report_full_cols"))
    sect = _join_cols(plus, ccfg.get("report_section_cols"))
    impr = _join_cols(plus, ccfg.get("report_impression_cols"))
    full = full.where(full.str.len() > 0, sect)
    out = pd.DataFrame({join: plus[join].astype(str), "report_text": full,
                        "sec_findings_impression": sect, "sec_impression": impr})
    return out.drop_duplicates(subset=[join])


def _reports_inline(master: pd.DataFrame, ccfg: dict) -> pd.DataFrame:
    full = _join_cols(master, ccfg.get("report_full_cols"))
    sect = _join_cols(master, ccfg.get("report_section_cols"))
    impr = _join_cols(master, ccfg.get("report_impression_cols"))
    if not full.str.len().gt(0).any():
        raise ValueError(f"none of {ccfg.get('report_full_cols')} present in the master list")
    return pd.DataFrame({"report_text": full.values,
                         "sec_findings_impression": sect.values,
                         "sec_impression": impr.values}, index=master.index)


def build_paired_pool(cfg_path: str, cohort: str, force: bool = False) -> str:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    paths = cfg["paths"]
    ccfg = cfg["cohorts"].get(cohort)
    if ccfg is None:
        raise KeyError(f"cohort {cohort!r} is not in config CrossModal.cohorts")
    if not ccfg.get("enabled", True):
        raise MissingInput(f"cohort {cohort!r} is disabled in config")

    from Inference.rows import append_status
    status_dir, stage = paths["status_dir"], f"pool_{cohort}"

    pool_path = os.path.join(paths["manifests_dir"], f"pool_{cohort}.csv")
    text_path = os.path.join(paths["reports_dir"], f"reports_{cohort}_T0.csv")
    meta_path = os.path.join(paths["manifests_dir"], f"pool_{cohort}.buildparams.json")
    want = _build_params(cfg, cohort, ccfg)

    if not force and os.path.exists(pool_path) and os.path.exists(text_path):
        stored = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                stored = json.load(f)
        moved = [k for k, v in want.items() if k in stored and stored[k] != v]
        if not moved:
            n = len(pd.read_csv(pool_path, usecols=["case_id"]))
            if n > 0:
                append_status(status_dir, stage, f"already built: {n} rows; skipping")
                print(f"[{stage}] already built ({n} rows). Use force=True to rebuild.", flush=True)
                return pool_path
        else:
            print(f"[{stage}] REBUILD: build parameters moved: {moved}", flush=True)

    master_csv = ccfg["master_csv"]
    if not os.path.exists(master_csv):
        raise MissingInput(f"master list not found: {master_csv}")
    append_status(status_dir, stage, f"reading {os.path.basename(master_csv)}")

    dtypes = {ccfg["patient_col"]: str}
    if ccfg.get("study_col"):
        dtypes[ccfg["study_col"]] = str
    wanted = [ccfg.get(k) for k in ("image_key_col", "patient_col", "study_col", "split_col",
                                    "view_col", "age_col", "sex_col", "date_col", "site_col",
                                    "modality_col", "report_col")]
    wanted += list(ccfg.get("report_full_cols") or [])
    wanted += list(ccfg.get("report_section_cols") or [])
    wanted += list(ccfg.get("report_impression_cols") or [])
    wanted += list(cfg["labels"]["findings"])
    wanted = [c for c in dict.fromkeys(w for w in wanted if w)]
    head = read_csv_defensively(master_csv, nrows=1)
    use = [c for c in wanted if c in head.columns]
    dropped = [c for c in wanted if c not in head.columns]
    if dropped:
        print(f"[{stage}] columns named in config but absent from the master list: {dropped}", flush=True)
    master = read_csv_defensively(master_csv, dtype=dtypes, usecols=use)
    n_raw = len(master)

    vcol = ccfg.get("view_col")
    keep_views = ccfg.get("keep_views") or []
    if vcol and vcol in master.columns and keep_views:
        master = master[master[vcol].astype(str).str.strip().isin(keep_views)]
    mcol = ccfg.get("modality_col")
    keep_mod = ccfg.get("keep_modalities") or []
    if mcol and mcol in master.columns and keep_mod:
        master = master[master[mcol].astype(str).str.strip().isin(keep_mod)]
    master = master.reset_index(drop=True)
    append_status(status_dir, stage, f"{n_raw} raw rows -> {len(master)} after view and modality filters")

    if ccfg.get("study_col") and ccfg["study_col"] in master.columns:
        master["study_id"] = master[ccfg["study_col"]].astype(str)
    elif ccfg.get("study_from_path"):
        pattern = str(ccfg["study_from_path"])
        derived = master[ccfg["image_key_col"]].astype(str).str.extract(pattern, expand=False)
        n_bad = int(derived.isna().sum())
        if n_bad:
            raise ValueError(
                f"{cohort}: study_from_path {pattern!r} matched no study on {n_bad} of "
                f"{len(master)} image paths, for example "
                f"{master.loc[derived.isna(), ccfg['image_key_col']].head(3).tolist()}. "
                "Falling back to the image path would put one report in the corpus once per "
                "image and the rank of the true report would not be defined.")
        master["study_id"] = derived.values
        print(f"[{stage}] study_id derived from the image path: {len(master)} images over "
              f"{derived.nunique()} studies", flush=True)
    else:
        master["study_id"] = master[ccfg["image_key_col"]].astype(str)

    mode = ccfg["report_mode"]
    if mode == "file":
        shard_dir = os.path.join(paths["reports_dir"], f"_shards_{cohort}")
        rep = _reports_from_files(master, ccfg, shard_dir, status_dir, cohort)
        master = master.merge(rep, on="study_id", how="left")
    elif mode == "join_csv":
        rep = _reports_from_join(ccfg)
        master = master.merge(rep, on=ccfg["report_join_col"], how="left")
    elif mode == "inline":
        rep = _reports_inline(master, ccfg)
        for c in rep.columns:
            master[c] = rep[c].values
    else:
        raise ValueError(f"unknown report_mode {mode!r}")

    for c in ("report_text", "sec_findings_impression", "sec_impression"):
        master[c] = master[c].map(_clean_text) if c in master.columns else ""
    min_chars = int(cfg["split"]["min_report_chars"])
    n_before = len(master)
    master = master[master["report_text"].str.len() >= min_chars].reset_index(drop=True)
    append_status(status_dir, stage,
                  f"{n_before} rows -> {len(master)} with a report of at least {min_chars} characters")
    if master.empty:
        raise MissingInput(f"{cohort}: every row lost its report; check report_mode and the roots")

    image_key = master[ccfg["image_key_col"]].astype(str).str.lstrip("./")
    out = pd.DataFrame({
        "case_id": make_case_ids(cohort, image_key),
        "dataset": cohort,
        "modality": "cxr",
        "image_key": image_key.values,
        "image_subdir": np.nan,
        "image_relpath": image_key.values,
        "patient_id": master[ccfg["patient_col"]].astype(str).values,
        "study_id": master["study_id"].values,
    })

    if vcol and vcol in master.columns:
        view = master[vcol].astype(str).str.strip()
    else:
        view = pd.Series(["NA"] * len(master), index=master.index)
    out["view"] = view.values
    out["stratum"] = (cohort + "__" + view).values

    acol = ccfg.get("age_col")
    out["age"] = pd.to_numeric(master[acol], errors="coerce").values if acol and acol in master.columns else np.nan
    scol = ccfg.get("sex_col")
    if scol and scol in master.columns:
        sex = master[scol].astype(str).str.strip().str.upper().str[0]
        out["sex"] = sex.where(sex.isin(["F", "M"]), "U").values
    else:
        out["sex"] = "U"
    dcol = ccfg.get("date_col")
    out["study_year"] = _study_year(master[dcol]).values if dcol and dcol in master.columns else pd.NA
    sitecol = ccfg.get("site_col")
    out["site"] = master[sitecol].astype(str).values if sitecol and sitecol in master.columns else "NA"

    labels = cfg["labels"]
    findings = list(labels["findings"])
    if ccfg.get("labels_source") == "llm_labeler":
        for t in findings:
            out[t] = np.nan
        out["labels_source"] = "llm_labeler_pending"
    else:
        for t in findings:
            if t in master.columns:
                out[t] = binarize_presence(master[t], [labels["positive_code"]],
                                           labels["negative_codes"], labels["exclude_codes"]).values
            else:
                out[t] = np.nan
        out["labels_source"] = "chexpert_labeler"

    scfg = cfg["split"]
    if scfg["rebuild_from_patients"]:
        assign = _assign_patient_split(out["patient_id"].values, float(scfg["test_frac"]),
                                       float(scfg["valid_frac"]), int(scfg["seed"]))
        out["split"] = [assign[p] for p in out["patient_id"].astype(str)]
    else:
        out["split"] = normalize_split(master[ccfg["split_col"]])

    assert_unique_case_ids(out)
    _assert_no_leak(out)

    text = out[["case_id", "study_id"]].copy()
    text["report_text"] = master["report_text"].values
    text["sec_findings_impression"] = master["sec_findings_impression"].values
    text["sec_impression"] = master["sec_impression"].values
    text["sections_from"] = ccfg.get("report_sections_from", "parser")
    text["transform"] = "T0"

    cols = ["case_id", "dataset", "modality", "stratum", "split", "image_key", "image_subdir",
            "image_relpath", "patient_id", "study_id", "view", "age", "sex", "study_year",
            "site", "labels_source"] + findings
    out = out[[c for c in cols if c in out.columns]].reset_index(drop=True)

    write_manifest(out, pool_path)
    write_manifest(text, text_path)
    tmp = meta_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(want, f, indent=2, sort_keys=True)
    os.replace(tmp, meta_path)

    side = os.path.join(paths["manifests_dir"], f"pool_{cohort}.split.json")
    counts = {s: {"images": int((out["split"] == s).sum()),
                  "patients": int(out.loc[out["split"] == s, "patient_id"].nunique()),
                  "studies": int(out.loc[out["split"] == s, "study_id"].nunique())}
              for s in ("train", "val", "test")}
    with open(side + ".tmp", "w") as f:
        json.dump({"seed": int(scfg["seed"]), "test_frac": float(scfg["test_frac"]),
                   "valid_frac": float(scfg["valid_frac"]), "counts": counts}, f,
                  indent=2, sort_keys=True)
    os.replace(side + ".tmp", side)

    report_pool(out, findings, stage)
    n_pat = out["patient_id"].nunique()
    append_status(status_dir, stage,
                  f"wrote {len(out)} rows, {n_pat} patients, {out['study_id'].nunique()} studies")
    print(f"[{stage}] {pool_path}\n[{stage}] {text_path}", flush=True)
    return pool_path


def load_pool(cfg: dict, cohort: str, split: Optional[str] = None,
              usecols: Optional[List[str]] = None) -> pd.DataFrame:
    path = os.path.join(cfg["paths"]["manifests_dir"], f"pool_{cohort}.csv")
    if not os.path.exists(path):
        raise MissingInput(f"pool not built for {cohort}; run main_build_paired_pool(cohort='{cohort}')")
    want = list(usecols) if usecols else None
    if want is not None and split and "split" not in want:
        want = want + ["split"]
    df = read_csv_defensively(path, usecols=want, dtype={"patient_id": str, "study_id": str})
    if split:
        df = df[df["split"] == split].reset_index(drop=True)
        if usecols is not None and "split" not in usecols:
            df = df.drop(columns=["split"])
    return df


def load_reports(cfg: dict, cohort: str, transform: str = "T0") -> pd.DataFrame:
    path = os.path.join(cfg["paths"]["reports_dir"], f"reports_{cohort}_{transform}.csv")
    if not os.path.exists(path):
        raise MissingInput(f"reports not built for {cohort}/{transform}; "
                           f"run main_build_report_transforms(cohort='{cohort}')")
    return read_csv_defensively(path, dtype={"case_id": str, "study_id": str})
