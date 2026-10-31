"""
data_loader/report_transforms.py
Created on August 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""
import json
import os
import re
from typing import Dict, List, Optional

import pandas as pd

from config.serde import read_config
from data_loader.build_paired_pool import load_pool, load_reports
from data_loader.build_utils import normalize_report_text, write_manifest
from Inference.resume_utils import MissingInput


_SECTION_NAMES = [
    "final report", "final addendum", "examination", "exam", "indication",
    "reason for examination", "reason for exam", "history", "clinical history",
    "comparison", "comparisons", "technique", "procedure", "findings",
    "provisional findings impression", "impression", "conclusion",
    "recommendation", "recommendations", "notification", "wet read",
    "addendum", "summary",
]
_ADMIN_KEYS = {"final report", "final addendum", "examination", "exam", "indication",
               "reason for examination", "reason for exam", "history", "clinical history",
               "comparison", "comparisons", "technique", "procedure", "notification",
               "wet read", "addendum"}
_SECTION_RE = re.compile(
    r"^[ \t]*(" + "|".join(re.escape(n) for n in _SECTION_NAMES) + r")[ \t]*:?[ \t]*$"
    r"|^[ \t]*(" + "|".join(re.escape(n) for n in _SECTION_NAMES) + r")[ \t]*:",
    re.IGNORECASE | re.MULTILINE)

_FINDINGS_KEYS = {"findings"}
_IMPRESSION_KEYS = {"impression", "conclusion", "summary"}

_MEASUREMENT_PATTERNS = [
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:x|by)\s*\d+(?:\.\d+)?\s*(?:x|by)?\s*(?:\d+(?:\.\d+)?)?\s*"
               r"(?:cm|mm|centimeter|centimetre|millimeter|millimetre)s?\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:cm|mm|centimeter|centimetre|millimeter|millimetre)s?\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:french|fr|gauge|ga)\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:percent|%)", re.I),
    re.compile(r"\bcardiothoracic ratio[^.\n]{0,20}\d+(?:\.\d+)?", re.I),
]
_DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}"
               r"(?:st|nd|rd|th)?,?\s*\d{0,4}\b", re.I),
    re.compile(r"\b\d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.?)?\b", re.I),
    re.compile(r"\b\d{1,3}\s*(?:year|yr)s?[- ]old\b", re.I),
    re.compile(r"\b\d{1,3}\s*y/?o\b", re.I),
]
_DEVICE_TERMS = [
    "endotracheal tube", "endotracheal", "et tube", "ett", "tracheostomy tube", "tracheostomy",
    "nasogastric tube", "ng tube", "enteric tube", "feeding tube", "dobhoff", "orogastric",
    "central venous catheter", "central line", "cvc", "picc line", "picc", "port-a-cath",
    "portacath", "swan-ganz", "swan ganz", "pulmonary artery catheter", "introducer sheath",
    "chest tube", "pigtail catheter", "pigtail", "thoracostomy tube", "pleural drain", "drain",
    "pacemaker", "pacer", "aicd", "icd", "defibrillator", "pacing lead", "epicardial lead",
    "sternotomy wire", "sternal wire", "surgical clip", "surgical clips", "mediastinal drain",
    "intra-aortic balloon pump", "iabp", "lvad", "ecmo cannula", "ecmo", "cannula",
    "esophageal stent", "tracheal stent", "stent", "prosthetic valve", "valve replacement",
]
_DEVICE_PATTERNS = [re.compile(r"\b" + re.escape(t).replace(r"\ ", r"\s+") + r"s?\b", re.I)
                    for t in _DEVICE_TERMS]

_STRIP_GROUPS = {
    "measurement": _MEASUREMENT_PATTERNS,
    "date": _DATE_PATTERNS,
    "device": _DEVICE_PATTERNS,
}

normalize = normalize_report_text


def split_sections(text: str) -> Dict[str, str]:
    text = "" if text is None else str(text)
    hits = []
    for m in _SECTION_RE.finditer(text):
        name = (m.group(1) or m.group(2) or "").strip().lower()
        if name:
            hits.append((m.start(), m.end(), name))
    if not hits:
        return {"preamble": text.strip()}
    out = {}
    if hits[0][0] > 0:
        pre = text[:hits[0][0]].strip()
        if pre:
            out["preamble"] = pre
    for i, (_, end, name) in enumerate(hits):
        stop = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        body = text[end:stop].strip()
        if name in out:
            out[name] = (out[name] + "\n" + body).strip()
        else:
            out[name] = body
    return out


def _sections_text(text: str, keys: set) -> str:
    sec = split_sections(text)
    parts = [sec[k] for k in sec if k in keys and sec[k]]
    if parts:
        return normalize("\n".join(parts))
    body = [v for k, v in sec.items() if k not in _ADMIN_KEYS and v]
    if body:
        return normalize("\n".join(body))
    return normalize("\n".join(v for v in sec.values() if v))


def _strip_groups(text: str, groups: List[str]) -> str:
    out = "" if text is None else str(text)
    for g in groups:
        for pat in _STRIP_GROUPS.get(g, []):
            out = pat.sub(" ", out)
    return normalize(out)


def _label_template(row: pd.Series, findings: List[str]) -> str:
    pos = [f.replace("_", " ") for f in findings
           if f != "no_finding" and pd.notna(row.get(f)) and float(row.get(f)) == 1.0]
    neg = [f.replace("_", " ") for f in findings
           if f != "no_finding" and pd.notna(row.get(f)) and float(row.get(f)) == 0.0]
    lines = ["FINDINGS:"]
    if pos:
        lines.append("There is evidence of " + ", ".join(sorted(pos)) + ".")
    else:
        lines.append("No acute cardiopulmonary abnormality is identified.")
    if neg:
        lines.append("There is no evidence of " + ", ".join(sorted(neg)) + ".")
    lines.append("IMPRESSION:")
    lines.append(("Findings consistent with " + ", ".join(sorted(pos)) + ".") if pos
                 else "No acute abnormality.")
    return "\n".join(lines)


def _fingerprint(path: str) -> dict:
    if not os.path.exists(path):
        return {"name": os.path.basename(path), "size": None, "mtime": None}
    st = os.stat(path)
    return {"name": os.path.basename(path), "size": int(st.st_size), "mtime": int(st.st_mtime)}


def _params_for(cfg: dict, cohort: str, tkey: str, tcfg: dict) -> dict:
    p = {"cohort": cohort, "transform": tkey, "spec": dict(tcfg),
         "source": _fingerprint(os.path.join(cfg["paths"]["reports_dir"],
                                             f"reports_{cohort}_T0.csv"))}
    if tcfg["kind"] == "llm_rewrite":
        llm = cfg["llm"]
        p["llm"] = {"hf_id": llm["hf_id"],
                    "served_name": llm.get("served_name", llm["hf_id"]),
                    "load_4bit": llm["load_4bit"],
                    "max_new_tokens": llm["max_new_tokens"], "temperature": llm["temperature"]}
    if tcfg["kind"] == "label_template":
        p["findings"] = list(cfg["labels"]["findings"])
    return p


def _apply_rule_based(tkey: str, tcfg: dict, base: pd.DataFrame, pool: pd.DataFrame,
                      findings: List[str]) -> pd.Series:
    kind = tcfg["kind"]
    if kind == "identity":
        return base["report_text"]

    if kind == "sections":
        keep = set(tcfg["keep"])
        col = "sec_findings_impression" if keep == {"findings", "impression"} else "sec_impression"
        have_cols = (col in base.columns
                     and base.get("sections_from", pd.Series(["parser"] * len(base))).iloc[0] == "columns")
        if have_cols:
            direct = base[col].fillna("").astype(str)
            if direct.str.len().gt(0).mean() > 0.5:
                keys = _FINDINGS_KEYS | _IMPRESSION_KEYS if "findings" in keep else _IMPRESSION_KEYS
                parsed = base["report_text"].map(lambda t: _sections_text(t, keys))
                return direct.where(direct.str.len() > 0, parsed)
        keys = (_FINDINGS_KEYS | _IMPRESSION_KEYS) if "findings" in keep else _IMPRESSION_KEYS
        return base["report_text"].map(lambda t: _sections_text(t, keys))

    if kind == "regex_strip":
        groups = list(tcfg["groups"])
        return base["report_text"].map(lambda t: _strip_groups(t, groups))

    if kind == "label_template":
        lab = pool.set_index("case_id")[findings]
        joined = base.set_index("case_id").join(lab, how="left")
        return pd.Series([_label_template(r, findings) for _, r in joined.iterrows()],
                         index=joined.index).reindex(base["case_id"]).values

    raise ValueError(f"unknown transform kind {kind!r}")


def build_report_transforms(cfg_path: str, cohort: str, transforms: Optional[List[str]] = None,
                            force: bool = False) -> List[str]:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    paths = cfg["paths"]
    findings = list(cfg["labels"]["findings"])
    tcfgs = cfg["report_transforms"]
    order = list(transforms or tcfgs["order"])

    if any(tcfgs[t]["kind"] == "llm_rewrite" for t in order):
        from data_loader.llm_local import assert_serving_ready
        assert_serving_ready(cfg)

    from Inference.rows import append_status
    status_dir, stage = paths["status_dir"], f"transforms_{cohort}"

    base = load_reports(cfg, cohort, "T0")
    pool = load_pool(cfg, cohort, usecols=["case_id", "split", "labels_source"] + findings)
    if pool["labels_source"].astype(str).str.contains("pending").any() and "T7" in order:
        raise MissingInput(
            f"{cohort}: labels are still pending, so T7 cannot be built. "
            f"Run main_label_reports_llm(cohort='{cohort}') first.")

    written = []
    for tkey in order:
        tcfg = tcfgs[tkey]
        out_path = os.path.join(paths["reports_dir"], f"reports_{cohort}_{tkey}.csv")
        meta_path = out_path.replace(".csv", ".buildparams.json")
        want = _params_for(cfg, cohort, tkey, tcfg)

        if not force and os.path.exists(out_path):
            stored = {}
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    stored = json.load(f)
            moved = [k for k, v in want.items() if k in stored and stored[k] != v]
            if not moved:
                n = len(pd.read_csv(out_path, usecols=["case_id"]))
                if n > 0:
                    append_status(status_dir, stage, f"{tkey} already built ({n} rows); skipping")
                    written.append(out_path)
                    continue
            else:
                pass

        append_status(status_dir, stage, f"building {tkey} ({tcfg['name']})")
        if tcfg["kind"] == "llm_rewrite":
            from data_loader.llm_local import rewrite_reports_canonical
            text = rewrite_reports_canonical(cfg, cohort, base, status_dir=status_dir)
        else:
            text = _apply_rule_based(tkey, tcfg, base, pool, findings)

        out = pd.DataFrame({"case_id": base["case_id"].values,
                            "study_id": base["study_id"].values,
                            "report_text": pd.Series(text).fillna("").astype(str).values,
                            "transform": tkey})

        n_empty = int((out["report_text"].str.len() == 0).sum())
        kept = float(out["report_text"].str.len().sum()) / max(float(base["report_text"].str.len().sum()), 1.0)
        if n_empty == len(out):
            raise ValueError(f"{cohort}/{tkey}: every report came out empty; the transform is wrong, "
                             "and an empty text column would read as a perfect defense")
        append_status(status_dir, stage,
                      f"{tkey}: {len(out)} rows, {n_empty} empty, {100.0 * kept:.1f}% of T0 characters retained")

        write_manifest(out, out_path)
        tmp = meta_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(want, f, indent=2, sort_keys=True)
        os.replace(tmp, meta_path)
        written.append(out_path)

    return written
