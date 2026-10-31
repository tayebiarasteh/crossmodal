"""
data_loader/llm_local.py
Created on August 8, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""
import hashlib
import json
import os
import re
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_loader.build_utils import normalize_report_text, read_csv_defensively
from generation.model_wrappers import build_generation_prompt, split_reasoning
from Inference.resume_utils import MissingInput


_ALLOWED_BACKENDS = {"api"}


def assert_serving_ready(cfg: dict) -> str:
    serving = cfg.get("serving") or {}
    backend = str(serving.get("backend", "api")).lower()
    if backend not in _ALLOWED_BACKENDS:
        raise ValueError(
            f"serving.backend is {backend!r}; this project allows only {sorted(_ALLOWED_BACKENDS)}. "
            "Sending report text to a third-party API would breach the PhysioNet agreement.")
    from generation.model_wrappers import resolve_secret
    url_env = serving.get("api_base_url_env", "CROSSMODALITY_API_BASE_URL")
    key_env = serving.get("api_key_env", "CROSSMODALITY_API_KEY")
    url = resolve_secret(serving.get("api_base_url"), url_env)
    key = resolve_secret(serving.get("api_key"), key_env)
    missing = [n for n, v in ((f"the base URL (serving.api_base_url or ${url_env})", url),
                              (f"the API key (serving.api_key or ${key_env})", key)) if not v]
    if missing:
        raise MissingInput(
            "the local LLM server is not configured, so this stage cannot generate anything: "
            + " and ".join(missing) + " is not set.\n"
            "  Serving the model is a manual step:\n"
            "    1. load " + str(cfg.get("llm", {}).get("served_name", "the model")) +
            " on your own serving proxy; the pipeline never loads or evicts a model there.\n"
            f"    2. export {url_env} and {key_env} in the job script, or paste the two values "
            "into the serving block of config.yaml.\n"
            "")
    return backend


def _is_placeholder_url(v) -> bool:
    return (not v) or "YOUR_" in str(v)


def build_llm(cfg: dict):
    serving = cfg.get("serving") or {}
    assert_serving_ready(cfg)
    spec = dict(cfg["llm"])
    gen_cfg = dict(cfg["llm"])
    token = cfg.get("hf_token")
    from generation.model_wrappers import LocalServerModel
    return LocalServerModel("crossmodal_llm", spec, gen_cfg, token, serving)


class ServerUnusable(RuntimeError):
    pass


def _generate_isolating_failures(model, chat: List[str], cfg: dict, tag: str,
                                 breaker: dict, note) -> tuple:
    def _call(block: List[str], t: str):
        return model.generate(
            block, n=1,
            temperature=float(cfg["llm"].get("temperature", 0.0)),
            top_p=float(cfg["llm"].get("top_p", 1.0)),
            max_tokens=int(cfg["llm"].get("max_new_tokens", 512)),
            seed=int(cfg["llm"].get("seed", 0)), tag=t)

    def _run(block: List[str], base: int, t: str):
        if not block:
            return [], {}
        try:
            out = _call(block, t)
            breaker["consecutive"] = 0
            breaker["ok"] += len(block)
            return out, {}
        except ServerUnusable:
            raise
        except Exception as e:
            if len(block) == 1:
                breaker["consecutive"] += 1
                breaker["failed"] += 1
                if breaker["consecutive"] >= breaker["max"]:
                    raise ServerUnusable(
                        f"{breaker['consecutive']} items failed in a row with no success between "
                        f"them, so the server is down or is rejecting "
                        f"{cfg['llm'].get('served_name')}. Stopping instead of failing the "
                        f"remaining items one by one for hours. Everything already generated is "
                        f"on disk and the next run resumes from it. Last error: "
                        f"{type(e).__name__}: {e}") from e
                note(f"item {base} failed permanently ({type(e).__name__}: {str(e)[:90]})")
                return [[""]], {base: f"{type(e).__name__}: {str(e)[:200]}"}
            mid = len(block) // 2
            note(f"a block of {len(block)} failed ({type(e).__name__}); halving to isolate it")
            lo_out, lo_bad = _run(block[:mid], base, t + ":a")
            hi_out, hi_bad = _run(block[mid:], base + mid, t + ":b")
            return lo_out + hi_out, {**lo_bad, **hi_bad}

    return _run(chat, 0, tag)


def generation_identity(cfg: dict, prompt_template: str) -> dict:
    llm = cfg["llm"]
    return {"served_name": llm.get("served_name", llm["hf_id"]), "hf_id": llm["hf_id"],
            "backend": str((cfg.get("serving") or {}).get("backend", "api")),
            "temperature": llm.get("temperature"), "top_p": llm.get("top_p"),
            "max_new_tokens": llm.get("max_new_tokens"), "seed": llm.get("seed"),
            "prompt_sha": hashlib.blake2b(prompt_template.encode(), digest_size=8).hexdigest()}


def assert_thinking_configured(cfg: dict, model, probe_prompt: str) -> None:
    from Inference.rows import append_status
    llm = cfg["llm"]
    want = str(llm.get("thinking", "off"))
    group = llm.get("recipe_group")
    if want != "off":
        return

    on = build_generation_prompt(model.tokenizer, probe_prompt, group, "on")
    off = build_generation_prompt(model.tokenizer, probe_prompt, group, "off")
    if on == off:
        raise MissingInput(
            f"llm.thinking is 'off' and it had no effect: the prompt renders identically with "
            f"thinking on and off under recipe_group={group!r}.\n"
            f"  Either the family is not routed (set llm.recipe_group to qwen3 or gpt_oss to "
            f"match {llm.get('served_name')}), or this tokenizer's chat template does not accept "
            f"the toggle.\n"
            f"  Leaving it is not an option: every one of these generations would carry a "
            f"reasoning trace in front of the answer, and both parsers in this file read the "
            f"answer.")

    out = model.generate([off], n=1, temperature=0.0, top_p=1.0,
                         max_tokens=int(llm.get("max_new_tokens", 512)), seed=0, tag="thinkprobe")
    raw = str((out[0][0] if isinstance(out[0], (list, tuple)) else out[0]) or "")
    cot, body = split_reasoning(raw)
    if cot.strip():
        raise MissingInput(
            f"llm.thinking is 'off', the template toggle applied, and the served model STILL "
            f"returned a reasoning trace of {len(cot)} characters on the probe.\n"
            f"  The proxy is probably serving {llm.get('served_name')} with thinking forced on, "
            f"or the served build is a reasoning-only checkpoint.\n"
            f"  Trace begins: {cot[:160]!r}")
    msg = (f"thinking OFF verified: the toggle changes the prompt, and one real completion came "
           f"back with no trace ({len(body)} characters of answer)")
    append_status(cfg["paths"]["status_dir"], "thinking_probe", f"{llm.get('served_name')}: {msg}")


def generate_sharded(cfg: dict, keys: List[str], prompts: List[str], out_path: str,
                     status_dir: str, stage: str, identity: dict, tag: str = "gen",
                     postprocess: Optional[Callable[[str], str]] = None) -> pd.DataFrame:
    from Inference.rows import append_status
    if len(keys) != len(prompts):
        raise ValueError(f"{len(keys)} keys against {len(prompts)} prompts")

    shard_rows = int(cfg["llm"].get("shard_rows", 2000))
    ident_sha = hashlib.blake2b(json.dumps(identity, sort_keys=True).encode(),
                                digest_size=6).hexdigest()
    shard_dir = out_path.replace(".csv", f"_shards_{ident_sha}")
    os.makedirs(shard_dir, exist_ok=True)
    with open(os.path.join(shard_dir, "identity.json"), "w") as f:
        json.dump(identity, f, indent=2, sort_keys=True)
    append_status(status_dir, stage,
                  f"generating under {identity['served_name']} (identity {ident_sha})")
    n_shards = int(np.ceil(len(keys) / shard_rows)) if keys else 0

    model = None
    frames = []
    failures: List[dict] = []
    breaker = {"consecutive": 0, "max": int((cfg.get("serving") or {}).get(
        "max_consecutive_failures", 30)), "ok": 0, "failed": 0}
    thinking = str(cfg["llm"].get("thinking", "off"))
    for k in tqdm(range(n_shards), desc=f"{stage}:generate", unit="shard"):
        spath = os.path.join(shard_dir, f"shard{k:05d}.csv")
        lo, hi = k * shard_rows, min((k + 1) * shard_rows, len(keys))
        if os.path.exists(spath):
            cached = read_csv_defensively(spath, dtype={"case_id": str})
            if cached["case_id"].astype(str).tolist() == [str(x) for x in keys[lo:hi]]:
                frames.append(cached)
                continue
            append_status(status_dir, stage,
                          f"shard {k + 1}/{n_shards}: the cached keys are not this slice's "
                          f"({len(cached)} cached against {hi - lo} wanted); regenerating it")
        if model is None:
            model = build_llm(cfg)
            append_status(status_dir, stage, f"backend up: {cfg['serving'].get('backend')} "
                                             f"{cfg['llm']['hf_id']}")
            assert_thinking_configured(cfg, model, prompts[lo])
        append_status(status_dir, stage, f"shard {k + 1}/{n_shards}: generating {hi - lo} items")
        chat = [build_generation_prompt(model.tokenizer, p,
                                        recipe_group=cfg["llm"].get("recipe_group"),
                                        reasoning_setting=thinking)
                for p in prompts[lo:hi]]
        outs, dead = _generate_isolating_failures(
            model, chat, cfg, tag=f"{tag}:{k}", breaker=breaker,
            note=lambda m: append_status(status_dir, stage, f"shard {k + 1}/{n_shards}: {m}"))
        if dead:
            for j in sorted(dead):
                failures.append({"shard": k, "case_id": keys[lo + j], "reason": dead[j]})
            append_status(status_dir, stage,
                          f"shard {k + 1}/{n_shards}: {len(dead)} of {hi - lo} items failed "
                          f"permanently and were written empty; see failures.json")
        texts, raws, n_trunc = [], [], 0
        for o in outs:
            raw = str((o[0] if isinstance(o, (list, tuple)) and o else (o or "")) or "")
            raws.append(raw)
            body = split_reasoning(raw)[1]
            if not body.strip() and raw.strip():
                n_trunc += 1
            texts.append(postprocess(body) if postprocess else body.strip())
        if n_trunc:
            append_status(status_dir, stage,
                          f"shard {k + 1}/{n_shards}: {n_trunc} of {len(texts)} generations ended "
                          f"inside an unclosed reasoning trace, so llm.max_new_tokens "
                          f"({cfg['llm'].get('max_new_tokens')}) is too small for this model, or "
                          f"thinking should be off for this task")
        frame = pd.DataFrame({"case_id": keys[lo:hi], "generated": texts, "raw": raws})
        n_empty = int((frame["generated"].astype(str).str.len() == 0).sum())
        if n_empty == len(frame):
            append_status(status_dir, stage,
                          f"shard {k + 1}/{n_shards} produced 0 usable completions; NOT cached")
            continue
        tmp = spath + ".tmp"
        frame.to_csv(tmp, index=False)
        os.replace(tmp, spath)
        frames.append(frame)
        append_status(status_dir, stage,
                      f"shard {k + 1}/{n_shards} done: {len(frame)} rows, {n_empty} empty")

    if failures:
        fpath = os.path.join(shard_dir, "failures.json")
        with open(fpath + ".tmp", "w") as f:
            json.dump(failures, f, indent=2)
        os.replace(fpath + ".tmp", fpath)
        msg = (f"{len(failures)} of {len(keys)} items failed permanently "
               f"({100.0 * len(failures) / max(len(keys), 1):.2f}%) and are listed in {fpath}")
        append_status(status_dir, stage, msg)
    if not frames:
        raise MissingInput(f"{stage}: no shard produced any completion; check the serving backend")
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["case_id"])
    return out


_REWRITE_SYSTEM = (
    "You are a radiology editor. Rewrite the chest radiograph report you are given into one "
    "fixed, canonical style. Keep every clinical finding, its laterality and its severity. "
    "Remove wording that identifies the individual report: measurements, dates, times, "
    "comparisons to prior studies, technique and positioning notes, hedging phrases, and any "
    "turn of phrase specific to the dictating radiologist. Write exactly two sections, "
    "FINDINGS: and IMPRESSION:, in plain declarative sentences. Output the report only."
)


def _rewrite_prompt(text: str) -> str:
    return (f"{_REWRITE_SYSTEM}\n\nREPORT:\n{text}\n\nREWRITTEN REPORT:\n")


_FENCE = re.compile(r"^```[a-z]*\s*|\s*```$", re.I | re.M)
_FIRST_SECTION = re.compile(r"(?im)^[ \t]*(findings)[ \t]*:")
_LEADING_PREAMBLE = re.compile(r"\A[^\n]{0,160}?:[ \t]*\n+")


def _clean_rewrite(raw: str) -> str:
    out = _FENCE.sub("", str(raw or "")).strip()
    m = _FIRST_SECTION.search(out)
    if m:
        out = out[m.start():]
    else:
        out = _LEADING_PREAMBLE.sub("", out, count=1)
    return normalize_report_text(out.strip())


def rewrite_reports_canonical(cfg: dict, cohort: str, base: pd.DataFrame,
                              status_dir: str) -> pd.Series:
    from Inference.rows import append_status
    stage = f"t6_{cohort}"
    per_study = base.drop_duplicates(subset=["study_id"])[["study_id", "report_text"]]
    append_status(status_dir, stage,
                  f"{len(base)} images over {len(per_study)} studies; generating per study")

    out_path = os.path.join(cfg["paths"]["reports_dir"], f"_t6_{cohort}.csv")
    gen = generate_sharded(
        cfg,
        keys=per_study["study_id"].astype(str).tolist(),
        prompts=[_rewrite_prompt(t) for t in per_study["report_text"].astype(str)],
        out_path=out_path, status_dir=status_dir, stage=stage, tag="t6",
        identity=generation_identity(cfg, _REWRITE_SYSTEM),
        postprocess=_clean_rewrite,
    )
    gen = gen.rename(columns={"case_id": "study_id", "generated": "rewritten"})
    merged = base[["case_id", "study_id", "report_text"]].merge(gen, on="study_id", how="left")
    rewritten = merged["rewritten"].fillna("").astype(str)

    n_fallback = int((rewritten.str.len() == 0).sum())
    if n_fallback:
        append_status(status_dir, stage,
                      f"{n_fallback} of {len(merged)} rows kept their original text after a failed rewrite")
    return rewritten.where(rewritten.str.len() > 0, merged["report_text"].astype(str)).values


_LABEL_SYSTEM = (
    "You label chest radiograph reports. For each of the 14 findings below, answer 1 if the "
    "report states the finding is present, 0 if it states the finding is absent or is uncertain "
    "about it, and 3 if the report does not mention it at all. Answer with a JSON object only, "
    "with one key per finding and no other text."
)


def _label_prompt(text: str, findings: List[str]) -> str:
    keys = ", ".join(f'"{f}"' for f in findings)
    return (f"{_LABEL_SYSTEM}\n\nFINDINGS: {keys}\n\nREPORT:\n{text}\n\nJSON:\n")


def _parse_label_row(raw: str, findings: List[str]):
    out = {f: np.nan for f in findings}
    if not raw:
        return out, 0
    m = re.search(r"\{.*\}", str(raw), re.S)
    if not m:
        return out, 0
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return out, 0
    if not isinstance(obj, dict):
        return out, 0
    lower = {str(k).strip().lower().replace(" ", "_"): v for k, v in obj.items()}
    n_answered = 0
    for f in findings:
        v = lower.get(f)
        if v is None:
            continue
        try:
            code = int(float(v))
        except (TypeError, ValueError):
            continue
        if code not in (0, 1, 2, 3):
            continue
        n_answered += 1
        if code == 1:
            out[f] = 1.0
        elif code in (0, 2):
            out[f] = 0.0
    return out, n_answered


def parse_label_json(raw: str, findings: List[str]) -> Dict[str, float]:
    return _parse_label_row(raw, findings)[0]


def extract_findings(cfg: dict, cohort: str, base: pd.DataFrame, findings: List[str],
                     status_dir: str) -> pd.DataFrame:
    from Inference.rows import append_status
    stage = f"labeler_{cohort}"
    per_study = base.drop_duplicates(subset=["study_id"])[["study_id", "report_text"]]
    append_status(status_dir, stage, f"labeling {len(per_study)} studies")

    out_path = os.path.join(cfg["paths"]["reports_dir"], f"_labels_{cohort}.csv")
    gen = generate_sharded(
        cfg,
        keys=per_study["study_id"].astype(str).tolist(),
        prompts=[_label_prompt(t, findings) for t in per_study["report_text"].astype(str)],
        out_path=out_path, status_dir=status_dir, stage=stage, tag="labeler",
        identity=generation_identity(cfg, _LABEL_SYSTEM + "|".join(findings)),
    )
    if gen.empty:
        raise MissingInput(f"{cohort}: the labeler generated nothing to parse")
    parsed = [_parse_label_row(r, findings) for r in gen["generated"].astype(str)]
    lab = pd.DataFrame([p[0] for p in parsed], columns=findings)
    lab.insert(0, "study_id", gen["case_id"].astype(str).values)

    n_answered = np.array([p[1] for p in parsed], dtype=float)
    unanswered = float((n_answered == 0).mean())
    covered = float(lab[findings].notna().to_numpy().mean())
    msg = (f"parsed {len(lab)} studies, {100.0 * (1.0 - unanswered):.1f}% answered, "
           f"{n_answered.mean():.1f} of {len(findings)} findings answered per study, "
           f"{100.0 * covered:.1f}% of finding cells populated")
    append_status(status_dir, stage, msg)

    max_unanswered = float(cfg["llm"].get("max_unanswered_frac", 0.10))
    if unanswered > max_unanswered:
        raise ValueError(
            f"{cohort}: {100.0 * unanswered:.1f}% of studies came back with no parseable answer, "
            f"above the {100.0 * max_unanswered:.0f}% allowed by llm.max_unanswered_frac. "
            f"The model is not returning the JSON object the prompt asks for, so check the served "
            f"model, llm.max_new_tokens ({cfg['llm'].get('max_new_tokens')}) and whether a "
            f"reasoning trace is coming back. The raw completions are kept beside the shards, so "
            f"a parser fix is a re-parse and not a regeneration.")
    return lab
