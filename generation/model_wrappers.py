"""
generation/model_wrappers.py
Created on July 20, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""
import math
import os
import re
import threading
from typing import Dict, List, Optional, Tuple


import warnings
warnings.filterwarnings("ignore")


_THINK_OPEN  = "<think>"
_THINK_CLOSE = "</think>"


def _split_harmony(raw: str) -> Tuple[str, str]:
    text = raw
    end_markers = ["<|return|>", "<|end|>", "<|endoftext|>"]
    if "final<|message|>" in text:
        after = text.split("final<|message|>", 1)[1]
        for m in end_markers:
            after = after.split(m, 1)[0]
        cot = ""
        if "analysis<|message|>" in text:
            cot = text.split("analysis<|message|>", 1)[1].split("<|end|>", 1)[0]
        return cot.strip(), after.strip()
    if "assistantfinal" in text:
        cot, after = text.split("assistantfinal", 1)
        cot = cot.replace("analysis", "", 1)
        for m in end_markers:
            after = after.split(m, 1)[0]
        return cot.strip(), after.strip()
    return text.strip(), ""


def split_reasoning(raw: str, recipe_group: Optional[str] = None) -> Tuple[str, str]:
    if raw is None:
        return "", ""
    if _THINK_CLOSE in raw:
        head, tail = raw.split(_THINK_CLOSE, 1)
        cot = head.split(_THINK_OPEN, 1)[-1].strip()
        return cot, tail.strip()
    if recipe_group == "gpt_oss" or "<|channel|>" in raw or "assistantfinal" in raw:
        return _split_harmony(raw)
    if raw.lstrip().startswith(_THINK_OPEN):
        return raw.split(_THINK_OPEN, 1)[-1].strip(), ""
    return _split_on_answer_cue(raw)


_ANSWER_MARKER = re.compile(
    r"(?:\*\*|\#\#)?\s*(?:final\s+answer|the\s+answer\s+is|answer)\s*(?:\*\*)?\s*[:\-]|"
    r"\\boxed\{",
    re.IGNORECASE,
)


def _split_on_answer_cue(raw: str) -> Tuple[str, str]:
    text = raw.strip()
    if not text:
        return "", ""
    matches = list(_ANSWER_MARKER.finditer(text))
    if not matches:
        return "", text
    last = matches[-1]
    start = last.start() + (len(last.group()) - len(last.group().lstrip()))
    line_start = text.rfind("\n", 0, start) + 1
    cot = text[:line_start].strip()
    answer = text[line_start:].strip()
    if not cot:
        return "", text
    return cot, answer


def _safe_apply_chat_template(tokenizer, messages, extra_kwargs: dict) -> str:
    base = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, **base, **extra_kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **base)


def _reasoning_kwargs(recipe_group: Optional[str], reasoning_setting: Optional[str]) -> dict:
    if recipe_group == "qwen3":
        return {"enable_thinking": (reasoning_setting or "on") != "off"}
    if recipe_group == "gpt_oss":
        return {"reasoning_effort": reasoning_setting or "medium"}
    return {}


def build_generation_prompt(
    tokenizer,
    user_prompt: str,
    recipe_group: Optional[str] = None,
    reasoning_setting: Optional[str] = None,
) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    return _safe_apply_chat_template(
        tokenizer, messages, _reasoning_kwargs(recipe_group, reasoning_setting)
    )


def build_forced_trace_prompt(
    tokenizer,
    user_prompt: str,
    trace: str,
    answer_cue: str,
    recipe_group: Optional[str] = None,
    reasoning_setting: Optional[str] = None,
) -> str:
    base = build_generation_prompt(tokenizer, user_prompt, recipe_group, reasoning_setting)
    thinking_on = (reasoning_setting or "on") != "off"

    if recipe_group == "gpt_oss":
        return (base + "<|channel|>analysis<|message|>" + trace +
                "<|end|><|start|>assistant<|channel|>final<|message|>" +
                answer_cue.strip() + " ")
    if recipe_group in ("r1_distill", "qwen3", None) and thinking_on:
        opener = "" if base.rstrip().endswith(_THINK_OPEN) else _THINK_OPEN + "\n"
        return base + opener + trace + "\n" + _THINK_CLOSE + "\n\n" + answer_cue.strip() + " "
    return base + trace + "\n\n" + answer_cue.strip() + " "


def perplexity_from_logprobs(token_logprobs: List[float]) -> float:
    lps = [lp for lp in token_logprobs if lp is not None and math.isfinite(lp)]
    if not lps:
        return float("nan")
    return float(math.exp(-sum(lps) / len(lps)))


def load_tokenizer(hf_id: str, token: Optional[str] = None):
    from transformers import AutoTokenizer
    attempts = [
        ("local cache", dict(token=token, trust_remote_code=True, local_files_only=True)),
        ("hub", dict(token=token, trust_remote_code=True)),
    ]
    if token:
        attempts.append(("hub without token", dict(token=None, trust_remote_code=True)))
    errors = []
    for label, kwargs in attempts:
        try:
            return AutoTokenizer.from_pretrained(hf_id, **kwargs)
        except Exception as e:
            errors.append(f"{label}: {type(e).__name__}: {e}")
            if "expired" in str(e).lower() or "401" in str(e):
                pass
    detail = "\n  ".join(errors)
    raise RuntimeError(
        f"[model_wrappers] could not load the tokenizer for '{hf_id}'. Attempts:\n  {detail}\n"
        f"If the token is expired, refresh the configured hf_token (or unset it for "
        f"public repos); if the hub is unreachable, pre-cache the tokenizer and set "
        f"HF_HUB_OFFLINE=1."
    )


_PLACEHOLDER_MARKERS = ("YOUR_", "_HERE", "VERIFY_AT_RUNTIME")


def _is_placeholder(value: Optional[str]) -> bool:
    if not value:
        return True
    return any(m in value for m in _PLACEHOLDER_MARKERS)


def resolve_secret(cfg_value: Optional[str], env_var: str) -> Optional[str]:
    if not _is_placeholder(cfg_value):
        return cfg_value
    return os.environ.get(env_var)


def _is_rate_limit(e: Exception) -> bool:
    try:
        import openai
        if isinstance(e, openai.RateLimitError):
            return True
        if isinstance(e, openai.APIStatusError) and getattr(e, "status_code", 0) == 429:
            return True
    except Exception:
        pass
    return " 429" in str(e).lower() or "rate limit" in str(e).lower()


def _retry_after_s(e: Exception) -> Optional[float]:
    try:
        resp = getattr(e, "response", None)
        if resp is not None:
            ra = resp.headers.get("retry-after")
            if ra:
                return float(ra)
    except Exception:
        pass
    return None


_NOTICE_STATE: Dict[str, Tuple[float, int]] = {}
_NOTICE_LOCK = threading.Lock()


def _notice(key: str, message: str, min_interval_s: float = 60.0) -> None:
    import time
    now = time.time()
    with _NOTICE_LOCK:
        last, suppressed = _NOTICE_STATE.get(key, (0.0, 0))
        if last and (now - last) < min_interval_s:
            _NOTICE_STATE[key] = (last, suppressed + 1)
            return
        _NOTICE_STATE[key] = (now, 0)


def _is_transient_server_error(e: Exception) -> bool:
    try:
        import openai
        if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError,
                          openai.InternalServerError)):
            return True
        if isinstance(e, openai.APIStatusError) and getattr(e, "status_code", 0) in (502, 503, 504, 429):
            return True
    except Exception:
        pass
    try:
        import httpx
        if isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
                          httpx.WriteTimeout, httpx.PoolTimeout, httpx.RemoteProtocolError)):
            return True
    except Exception:
        pass
    if isinstance(e, (ConnectionError, ConnectionRefusedError, TimeoutError)):
        return True
    msg = str(e).lower()
    return any(s in msg for s in (
        "connection refused", "connection error", "connection reset",
        "connect", "timed out", "timeout", "temporarily unavailable",
        "reset by peer", "bad gateway", "service unavailable",
        "gateway timeout", " 502", " 503", " 504", " 429",
    ))


class LocalServerModel:

    def __init__(self, model_key: str, spec: dict, gen_cfg: dict,
                 hf_token: Optional[str], serving_cfg: dict):
        self.model_key = model_key
        self.spec = spec
        self.recipe_group = spec.get("recipe_group")
        self.hf_id = spec["hf_id"]
        self.served_name = spec.get("served_name", self.hf_id)
        self._gen_cfg = gen_cfg
        self._hf_token = hf_token
        self._serving = serving_cfg or {}
        self._client = None
        self._tokenizer = None

    def _ensure_client(self):
        if self._client is not None:
            return
        import httpx
        from openai import OpenAI
        base_url = resolve_secret(self._serving.get("api_base_url"),
                                  self._serving.get("api_base_url_env", "CROSSMODALITY_API_BASE_URL"))
        api_key  = resolve_secret(self._serving.get("api_key"),
                                  self._serving.get("api_key_env", "CROSSMODALITY_API_KEY"))
        if not base_url:
            raise RuntimeError(
                "[LocalServerModel] no serving base URL. Set serving.api_base_url "
                "or export the env var named in serving.api_base_url_env "
                f"({self._serving.get('api_base_url_env', 'CROSSMODALITY_API_BASE_URL')})."
            )
        verify = bool(self._serving.get("verify_ssl", False))
        floor = float(self._serving.get("timeout_s", 120))
        conc = max(1, int(self._serving.get("concurrency", 1)))
        limits = httpx.Limits(max_connections=conc + 8,
                              max_keepalive_connections=conc + 8)
        self._client = OpenAI(
            base_url=base_url, api_key=(api_key or "EMPTY"),
            http_client=httpx.Client(verify=verify, timeout=floor, limits=limits),
            max_retries=0,
        )

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(self.hf_id, self._hf_token)
        return self._tokenizer

    def build_prompt(self, user_prompt: str, reasoning_setting: Optional[str] = None) -> str:
        return build_generation_prompt(
            self.tokenizer, user_prompt, self.recipe_group, reasoning_setting
        )

    def build_forced_prompt(self, user_prompt: str, trace: str, answer_cue: str,
                            reasoning_setting: Optional[str] = None) -> str:
        return build_forced_trace_prompt(
            self.tokenizer, user_prompt, trace, answer_cue,
            self.recipe_group, reasoning_setting
        )

    def _call_timeout_s(self, kwargs: dict) -> float:
        s = self._serving
        floor = float(s.get("timeout_s", 120))
        per_tok = float(s.get("timeout_per_token_s", 0.12))
        cap = float(s.get("timeout_request_max_s", 3600.0))
        mt = int(kwargs.get("max_tokens", 0) or 0)
        return min(cap, floor + mt * per_tok)

    def _completion(self, **kwargs):
        import time, random
        self._ensure_client()
        s = self._serving
        max_retries  = max(1, int(s.get("max_retries", 5)))
        base         = float(s.get("retry_backoff_s", 2.0))
        cap          = float(s.get("retry_backoff_max_s", 30.0))
        connect_budget = float(s.get("connect_retry_max_s", 900.0))
        call_timeout = self._call_timeout_s(kwargs)
        t0 = time.time()
        last, attempt, waited, announced, rl_attempt = None, 0, 0.0, False, 0
        while True:
            try:
                try:
                    return self._client.completions.create(timeout=call_timeout, **kwargs)
                except TypeError:
                    return self._client.completions.create(**kwargs)
            except Exception as e:
                last = e
                if _is_rate_limit(e):
                    rl_attempt += 1
                    if rl_attempt > int(s.get("rate_limit_retries", 8)):
                        break
                    wait = _retry_after_s(e) or min(30.0, base * (2 ** min(rl_attempt, 5)))
                    wait *= 0.7 + 0.6 * random.random()
                    if not announced:
                        _notice(f"429:{self.served_name}",
                                f"[LocalServerModel] '{self.served_name}' throttled (429). "
                                f"Backing off; lower serving.concurrency if this persists.")
                        announced = True
                    time.sleep(wait)
                    continue
                if _is_transient_server_error(e):
                    if (time.time() - t0) >= connect_budget:
                        break
                    sleep = min(cap, base * (2 ** min(attempt, 8)))
                    sleep *= 0.7 + 0.6 * random.random()
                    if not announced:
                        _notice(f"outage:{self.served_name}",
                                f"[LocalServerModel] '{self.served_name}' unreachable "
                                f"({type(e).__name__}). Waiting for the server to come "
                                f"back (up to {connect_budget:.0f}s); the run will resume "
                                f"automatically.")
                        announced = True
                    time.sleep(sleep)
                    waited += sleep
                    attempt += 1
                    continue
                attempt += 1
                if attempt >= max_retries:
                    break
                time.sleep(min(cap, base * (2 ** attempt)))
        raise RuntimeError(
            f"[LocalServerModel] /v1/completions failed for '{self.served_name}' "
            f"(waited {waited:.0f}s for connectivity): {last}"
        ) from last

    def unload(self):
        return

    def generate(
        self,
        prompts: List[str],
        n: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        tag: str = "gen",
    ) -> List[List[str]]:
        from generation.gen_utils import progress
        from concurrent.futures import ThreadPoolExecutor, as_completed
        nn = int(n)
        mt = int(max_tokens or self._gen_cfg.get("max_new_tokens", 4096))
        bs = max(1, int(self._serving.get("request_batch_size",
                                          self._serving.get("batch_size", 8))))
        conc = max(1, int(self._serving.get("concurrency", 1)))
        self._ensure_client()
        starts = list(range(0, len(prompts), bs))

        def _do(start):
            batch = prompts[start:start + bs]
            resp = self._completion(
                model=self.served_name, prompt=batch, n=nn,
                temperature=float(temperature), top_p=float(top_p),
                max_tokens=mt, seed=seed,
            )
            per = [["" for _ in range(nn)] for _ in batch]
            for ch in resp.choices:
                p_idx, s_idx = divmod(int(getattr(ch, "index", 0)), nn)
                if 0 <= p_idx < len(batch) and 0 <= s_idx < nn:
                    per[p_idx][s_idx] = ch.text or ""
            return start, per

        show = bool(self._serving.get("progress_bars", True)) and len(starts) > 1
        by_start = {}
        with ThreadPoolExecutor(max_workers=conc) as ex:
            futs = {ex.submit(_do, s): s for s in starts}
            for f in progress(as_completed(futs), desc=f"{self.model_key} {tag}",
                              total=len(futs), enable=show, leave=False):
                s = futs[f]
                try:
                    start, per = f.result()
                    by_start[start] = per
                except Exception:
                    n_p = len(prompts[s:s + bs])
                    by_start[s] = [["" for _ in range(nn)] for _ in range(n_p)]
        results: List[List[str]] = []
        for s in starts:
            results.extend(by_start[s])
        return results

    def generate_requests(self, specs: List[dict], tag: str = "gen") -> List[List[str]]:
        from generation.gen_utils import progress
        from concurrent.futures import ThreadPoolExecutor, as_completed
        conc = max(1, int(self._serving.get("concurrency", 1)))
        default_mt = int(self._gen_cfg.get("max_new_tokens", 4096))
        self._ensure_client()

        def _do(i):
            s = specs[i]
            nn = int(s.get("n", 1))
            resp = self._completion(
                model=self.served_name, prompt=[s["prompt"]], n=nn,
                temperature=float(s.get("temperature", 0.0)),
                top_p=float(s.get("top_p", 1.0)),
                max_tokens=int(s.get("max_tokens") or default_mt),
                seed=s.get("seed"),
            )
            out = ["" for _ in range(nn)]
            for ch in resp.choices:
                _, s_idx = divmod(int(getattr(ch, "index", 0)), nn)
                if 0 <= s_idx < nn:
                    out[s_idx] = ch.text or ""
            return i, out

        results: List[List[str]] = [[] for _ in specs]
        show = bool(self._serving.get("progress_bars", True)) and len(specs) > 1
        with ThreadPoolExecutor(max_workers=conc) as ex:
            futs = {ex.submit(_do, i): i for i in range(len(specs))}
            for f in progress(as_completed(futs), desc=f"{self.model_key} {tag}",
                              total=len(futs), enable=show, leave=False):
                i = futs[f]
                try:
                    _, out = f.result()
                    results[i] = out
                except Exception:
                    results[i] = ["" for _ in range(int(specs[i].get("n", 1)))]
        return results

    def perplexity(self, texts: List[str]) -> List[float]:
        from generation.gen_utils import progress
        from concurrent.futures import ThreadPoolExecutor, as_completed
        bs = max(1, int(self._serving.get("request_batch_size",
                                          self._serving.get("batch_size", 8))))
        conc = max(1, int(self._serving.get("concurrency", 1)))
        self._ensure_client()
        ppls = [float("nan")] * len(texts)
        starts = list(range(0, len(texts), bs))

        def _do(start):
            batch = texts[start:start + bs]
            resp = self._completion(
                model=self.served_name, prompt=batch, max_tokens=0,
                echo=True, logprobs=0, temperature=0.0,
            )
            out = {}
            for ch in resp.choices:
                idx = int(getattr(ch, "index", 0))
                lp = ch.logprobs
                token_lps = [x for x in (getattr(lp, "token_logprobs", None) or [])
                             if x is not None]
                if 0 <= idx < len(batch):
                    out[start + idx] = perplexity_from_logprobs(token_lps)
            return out

        show = bool(self._serving.get("progress_bars", True)) and len(starts) > 1
        with ThreadPoolExecutor(max_workers=conc) as ex:
            futs = {ex.submit(_do, s): s for s in starts}
            for f in progress(as_completed(futs), desc=f"{self.model_key} ppl",
                              total=len(futs), enable=show, leave=False):
                try:
                    for i, v in f.result().items():
                        ppls[i] = v
                except Exception:
                    pass
        return ppls
