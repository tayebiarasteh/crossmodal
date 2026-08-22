"""
Inference/wiring_audit.py
Created on August 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import ast
import glob
import json
import os
import re
from typing import Dict, List

from config.serde import read_config


_SELF_REFERENCE = {"global_config_path"}


def _source_text(repo_root: str) -> str:
    out = []
    for path in sorted(glob.glob(os.path.join(repo_root, "**", "*.py"), recursive=True)):
        if os.sep + ".git" + os.sep in path:
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            out.append(f.read())
    return "\n".join(out)


def _read_anywhere(name: str, src: str) -> bool:
    return (f'"{name}"' in src) or (f"'{name}'" in src)


def _cached_config(hf_id: str) -> Dict:
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    hits = sorted(glob.glob(os.path.join(
        home, "hub", "models--" + hf_id.replace("/", "--"), "snapshots", "*", "config.json")))
    for path in reversed(hits):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            continue
    return {}


def _declared_width(conf: Dict) -> Dict[str, int]:
    mt = str(conf.get("model_type", ""))
    vis, txt = conf.get("vision_config", {}), conf.get("text_config", {})
    if mt.startswith("clip"):
        d = conf.get("projection_dim") or vis.get("projection_dim") or txt.get("projection_dim")
        return {"image": int(d), "text": int(d)} if d else {}
    if mt.startswith("siglip"):
        out = {}
        if vis.get("hidden_size"):
            out["image"] = int(vis["hidden_size"])
        if txt.get("projection_size") or txt.get("hidden_size"):
            out["text"] = int(txt.get("projection_size") or txt["hidden_size"])
        return out
    return {}


def audit_wiring(cfg_path: str, pending_blocks: List[str] = None,
                 pending_paths: List[str] = None, strict: bool = False) -> Dict:
    params = read_config(cfg_path)
    cfg = params["CrossModal"]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = _source_text(repo_root)
    sheet_path = os.path.join(repo_root, "main_crossmodal.py")
    sheet = open(sheet_path, encoding="utf-8").read() if os.path.exists(sheet_path) else ""
    pending = set(pending_blocks or [])
    pending_p = set(pending_paths or [])

    findings = {"orphan_config_keys": [], "pending_blocks": [], "unreachable_models": [],
                "unimplemented_kinds": [], "orphan_paths": [], "pending_paths": [],
                "missing_stage_modules": [], "dead_modules": [], "stray_main_blocks": [],
                "declared_width_mismatch": [], "width_unverifiable": []}

    for key in cfg:
        if key in _SELF_REFERENCE or _read_anywhere(key, src):
            continue
        (findings["pending_blocks"] if key in pending else findings["orphan_config_keys"]).append(key)

    for key in cfg.get("paths", {}):
        if _read_anywhere(key, src):
            continue
        (findings["pending_paths"] if key in pending_p else findings["orphan_paths"]).append(key)

    for model in cfg.get("models", {}):
        if f'model="{model}"' not in sheet:
            findings["unreachable_models"].append(model)

    for model, spec in cfg.get("models", {}).items():
        declared = spec.get("embed_dim")
        if not declared:
            continue
        widths = _declared_width(_cached_config(str(spec.get("hf_id", ""))))
        if not widths:
            findings["width_unverifiable"].append(f"{model} ({spec.get('type')})")
            continue
        for tower, real in widths.items():
            if real != int(declared):
                findings["declared_width_mismatch"].append(
                    f"{model}: config declares embed_dim {declared} and the checkpoint's own "
                    f"config.json gives {real} for the {tower} tower")

    import ast
    if sheet:
        for node in ast.walk(ast.parse(sheet)):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            mod = node.module
            rel = os.path.join(repo_root, *mod.split(".")) + ".py"
            pkg = os.path.join(repo_root, *mod.split("."), "__init__.py")
            if not os.path.exists(rel) and not os.path.exists(pkg):
                findings["missing_stage_modules"].append(mod)

    reachable, stack = set(), ["main_crossmodal"]
    local = {}
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", "logs", ".idea")]
        for fn in filenames:
            if fn.endswith(".py"):
                rel = os.path.relpath(os.path.join(dirpath, fn), repo_root)
                local[rel[:-3].replace(os.sep, ".")] = rel
    while stack:
        m = stack.pop()
        if m in reachable or m not in local:
            continue
        reachable.add(m)
        try:
            tree = ast.parse(open(os.path.join(repo_root, local[m])).read())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in local:
                stack.append(node.module)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name in local:
                        stack.append(a.name)
    findings["dead_modules"] = sorted(v for k, v in local.items()
                                      if k not in reachable and not v.endswith("__init__.py"))

    for mod, rel in sorted(local.items()):
        if rel == "main_crossmodal.py":
            continue
        if re.search(r'^if __name__ == ["\']__main__["\']', open(os.path.join(repo_root, rel)).read(), re.M):
            findings["stray_main_blocks"].append(rel)

    rt = cfg.get("report_transforms", {})
    for tkey in rt.get("order", []):
        if tkey not in rt:
            findings["unimplemented_kinds"].append(f"transform {tkey} is in `order` and has no block")
            continue
        kind = rt[tkey].get("kind")
        if kind and f'"{kind}"' not in src:
            findings["unimplemented_kinds"].append(f"transform kind {kind!r} has no implementation")

    print("[wiring_audit] " + ("PASS" if not any(
        findings[k] for k in ("orphan_config_keys", "unreachable_models",
                              "unimplemented_kinds", "orphan_paths",
                              "missing_stage_modules", "dead_modules",
                              "stray_main_blocks", "declared_width_mismatch")) else "FINDINGS"))
    for name, items in findings.items():
        if items:
            print(f"  {name}: {items}", flush=True)

    hard = (findings["orphan_config_keys"] + findings["unreachable_models"]
            + findings["unimplemented_kinds"] + findings["orphan_paths"]
            + findings["missing_stage_modules"] + findings["dead_modules"]
            + findings["stray_main_blocks"] + findings["declared_width_mismatch"])
    if strict and hard:
        raise ValueError(f"wiring audit failed: {hard}")
    return findings


def audit_environment(cfg_path: str) -> Dict:
    import numpy as np
    import pandas as pd

    read_config(cfg_path)
    out: Dict[str, object] = {}
    for mod in ("scipy", "statsmodels"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception as e:
            out[mod] = f"unavailable ({type(e).__name__})"

    rng = np.random.default_rng(0)
    n, ng = 4000, 900
    g = rng.integers(0, ng, n)
    x = rng.normal(size=n)
    y = 0.05 * x + rng.normal(0, 1, ng)[g] + rng.normal(0, 1, n)
    try:
        import statsmodels.formula.api as smf
        smf.mixedlm("y ~ x", pd.DataFrame({"y": y, "x": x, "g": g}), groups=g).fit(reml=True)
        out["slope_fit"] = "mixedlm_random_intercept"
        note = ""
    except Exception as e:
        out["slope_fit"] = "ols_cluster_robust"
        out["mixedlm_error"] = f"{type(e).__name__}: {e}"
        note = ("  MixedLM is unavailable, so E3's slopes get a cluster-robust "
                "standard error instead. Both account for repeated images of one "
                "patient and the run continues either way.")

    print(f"[audit_environment] scipy {out['scipy']}, statsmodels {out['statsmodels']}; "
          f"E3 slope fit will be {out['slope_fit']}", flush=True)
    if note:
        print(note, flush=True)
    return out
