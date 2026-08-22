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

from config.serde import read_config

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


def count_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


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
                print(f"[model_wrappers] tokenizer for '{hf_id}' via {label} failed on "
                      f"an auth error; the HF token may be expired. Retrying.", flush=True)
    detail = "\n  ".join(errors)
    raise RuntimeError(
        f"[model_wrappers] could not load the tokenizer for '{hf_id}'. Attempts:\n  {detail}\n"
        f"If the token is expired, refresh CotLoadbearing.hf_token (or unset it for "
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


class VLLMModel:

    def __init__(self, model_key: str, spec: dict, gen_cfg: dict, hf_token: Optional[str]):
        self.model_key = model_key
        self.spec = spec
        self.recipe_group = spec.get("recipe_group")
        self.hf_id = spec["hf_id"]
        self._gen_cfg = gen_cfg
        self._hf_token = hf_token
        self._llm = None
        self._tokenizer = None
        self.revision = None

    def unload(self):
        import gc
        try:
            if getattr(self, "_llm", None) is not None:
                try:
                    from vllm.distributed.parallel_state import (
                        destroy_distributed_environment, destroy_model_parallel)
                    destroy_model_parallel()
                    destroy_distributed_environment()
                except Exception:
                    pass
                del self._llm
            self._llm = None
            self._tokenizer = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass
        except Exception as e:
            print(f"[model_wrappers] unload warning for '{self.model_key}': {e}")

    def _ensure_loaded(self):
        if self._llm is not None:
            return
        try:
            from vllm import LLM
            from transformers import AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                f"[model_wrappers] vLLM/transformers required to run '{self.model_key}'. "
                f"Install them (requirements.txt) and re-run."
            ) from e
        vc = self._gen_cfg.get("vllm", {})
        tp = int(self.spec.get("tensor_parallel", 1))
        if self._hf_token:
            os.environ.setdefault("HF_TOKEN", self._hf_token)
        try:
            self._llm = LLM(
                model=self.hf_id,
                dtype=vc.get("dtype", "bfloat16"),
                tensor_parallel_size=tp,
                gpu_memory_utilization=float(vc.get("gpu_memory_utilization", 0.90)),
                max_model_len=int(vc.get("max_model_len", 16384)),
                enforce_eager=bool(vc.get("enforce_eager", False)),
                trust_remote_code=True,
                seed=int(vc.get("seed", 2026)),
            )
        except Exception as e:
            raise RuntimeError(
                f"[model_wrappers] failed to load '{self.model_key}' ({self.hf_id}) in "
                f"vLLM: {e}. Check the hf_id, gated access, tensor_parallel, and that "
                f"the installed vLLM/transformers support this architecture."
            ) from e
        self._tokenizer = load_tokenizer(self.hf_id, self._hf_token)

    @property
    def tokenizer(self):
        self._ensure_loaded()
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

    def generate(
        self,
        prompts: List[str],
        n: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[str]]:
        self._ensure_loaded()
        from vllm import SamplingParams
        sp = SamplingParams(
            n=int(n),
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_tokens or self._gen_cfg.get("max_new_tokens", 4096)),
            seed=seed,
        )
        outs = self._llm.generate(prompts, sp)
        return [[o.text for o in req.outputs] for req in outs]

    def generate_requests(self, specs: List[dict], tag: str = "gen") -> List[List[str]]:
        groups: Dict[tuple, List[int]] = {}
        for i, s in enumerate(specs):
            key = (int(s.get("n", 1)), float(s.get("temperature", 0.0)),
                   int(s.get("max_tokens") or self._gen_cfg.get("max_new_tokens", 4096)),
                   s.get("seed"))
            groups.setdefault(key, []).append(i)
        out: List[List[str]] = [[] for _ in specs]
        for (nn, temp, mt, seed), idxs in groups.items():
            res = self.generate([specs[i]["prompt"] for i in idxs], n=nn,
                                temperature=temp, max_tokens=mt, seed=seed)
            for j, i in enumerate(idxs):
                out[i] = res[j]
        return out

    def perplexity(self, texts: List[str]) -> List[float]:
        self._ensure_loaded()
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        outs = self._llm.generate(texts, sp)
        ppls = []
        for req in outs:
            plps = req.prompt_logprobs or []
            token_lps = []
            for entry in plps:
                if not entry:
                    continue
                lp = next(iter(entry.values()))
                token_lps.append(getattr(lp, "logprob", None))
            ppls.append(perplexity_from_logprobs(token_lps))
        return ppls


class HFLocalModel:

    def __init__(self, model_key: str, spec: dict, gen_cfg: dict, hf_token: Optional[str]):
        self.model_key = model_key
        self.spec = spec
        self.recipe_group = spec.get("recipe_group")
        self.hf_id = spec["hf_id"]
        self._gen_cfg = gen_cfg
        self._hf_token = hf_token
        self._model = None
        self._tokenizer = None
        self.revision = None
        self._safe_bs: Dict[tuple, int] = {}
        self._last_shape: Optional[tuple] = None

    def unload(self):
        import gc
        try:
            if getattr(self, "_model", None) is not None:
                del self._model
            self._model = None
            self._tokenizer = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass
        except Exception as e:
            print(f"[model_wrappers] unload warning for '{self.model_key}': {e}")

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError as e:
            raise RuntimeError(
                f"[model_wrappers] torch and transformers are required to run "
                f"'{self.model_key}' locally. Install them (requirements.txt) and re-run."
            ) from e

        tok = load_tokenizer(self.hf_id, self._hf_token)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        self._tokenizer = tok

        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16 else torch.float32
        kwargs = dict(trust_remote_code=True, device_map="auto",
                      **hf_dtype_kwargs(dtype))
        attempts = [("local files", dict(local_files_only=True, **kwargs)),
                    ("hub", dict(token=self._hf_token, **kwargs))]
        if self._hf_token:
            attempts.append(("hub without token", dict(token=None, **kwargs)))
        errors = []
        for label, kw in attempts:
            try:
                self._model = AutoModelForCausalLM.from_pretrained(self.hf_id, **kw)
                break
            except Exception as e:
                errors.append(f"{label}: {type(e).__name__}: {e}")
        if self._model is None:
            detail = "\n  ".join(errors)
            raise RuntimeError(
                f"[model_wrappers] could not load '{self.model_key}' ({self.hf_id}) "
                f"with transformers. Attempts:\n  {detail}"
            )
        self._model.eval()
        print(f"[model_wrappers] loaded '{self.model_key}' from {self.hf_id} "
              f"via transformers (no vLLM).", flush=True)

    @property
    def tokenizer(self):
        self._ensure_loaded()
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

    def _generate_batch(self, batch: List[str], n: int, temperature: float,
                        top_p: float, max_tokens: int, seed: Optional[int]):
        import torch
        do_sample = float(temperature) > 0.0
        enc = self._tokenizer(batch, return_tensors="pt", padding=True,
                              add_special_tokens=False).to(self._model.device)
        if seed is not None:
            torch.manual_seed(int(seed))
        kw = dict(max_new_tokens=int(max_tokens), do_sample=do_sample,
                  num_return_sequences=int(n),
                  pad_token_id=self._tokenizer.pad_token_id)
        if do_sample:
            kw.update(temperature=float(temperature), top_p=float(top_p))
        with torch.no_grad():
            gen = self._model.generate(**enc, **kw)
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        return self._tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    def _batch_size_for(self, max_tokens: int, n: int) -> int:
        ceiling = max(1, int(self._gen_cfg.get("hf_batch_size", 8)))
        return max(1, min(ceiling, self._safe_bs.get((int(max_tokens), int(n)), ceiling)))

    def generate(
        self,
        prompts: List[str],
        n: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[str]]:
        self._ensure_loaded()
        mt = int(max_tokens or self._gen_cfg.get("max_new_tokens", 4096))
        shape = (mt, int(n))
        if self._last_shape is not None and self._last_shape != shape:
            release_gpu()
        self._last_shape = shape
        bs = self._batch_size_for(mt, n)
        out: List[List[str]] = []
        i = 0
        freed_at = None
        while i < len(prompts):
            batch = prompts[i:i + bs]
            try:
                texts = self._generate_batch(batch, n, temperature, top_p, mt, seed)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                release_gpu()
                if freed_at != (i, bs):
                    freed_at = (i, bs)
                    continue
                if bs == 1:
                    print(f"[model_wrappers] CUDA OOM in '{self.model_key}' at "
                          f"hf_batch_size=1 (max_tokens={mt}, n={n}); prompt {i} "
                          f"left EMPTY and will be retried on the next run. Lower "
                          f"generation.max_new_tokens for this stage or run it on "
                          f"a larger GPU.", flush=True)
                    out.append(["" for _ in range(int(n))])
                    i += 1
                    continue
                bs = max(1, bs // 2)
                self._safe_bs[shape] = bs
                print(f"[model_wrappers] CUDA OOM in '{self.model_key}' "
                      f"(max_tokens={mt}, n={n}); retrying at hf_batch_size={bs} "
                      f"and holding that for this shape.", flush=True)
                continue
            for j in range(len(batch)):
                out.append(texts[j * n:(j + 1) * n])
            i += len(batch)
        return out

    def generate_requests(self, specs: List[dict], tag: str = "gen") -> List[List[str]]:
        groups: Dict[tuple, List[int]] = {}
        for i, s in enumerate(specs):
            key = (int(s.get("n", 1)), float(s.get("temperature", 0.0)),
                   int(s.get("max_tokens") or self._gen_cfg.get("max_new_tokens", 4096)),
                   s.get("seed"))
            groups.setdefault(key, []).append(i)
        out: List[List[str]] = [[] for _ in specs]
        for (nn, temp, mt, seed), idxs in groups.items():
            res = self.generate([specs[i]["prompt"] for i in idxs], n=nn,
                                temperature=temp, max_tokens=mt, seed=seed)
            for j, i in enumerate(idxs):
                out[i] = res[j]
        return out

    def perplexity(self, texts: List[str]) -> List[float]:
        self._ensure_loaded()
        import torch
        max_len = int(self._gen_cfg.get("vllm", {}).get("max_model_len", 16384))
        ppls = []
        for text in texts:
            ids = self._tokenizer(text, add_special_tokens=False).input_ids[:max_len]
            if len(ids) < 2:
                ppls.append(float("nan"))
                continue
            t = torch.tensor([ids], device=self._model.device)
            with torch.no_grad():
                logits = self._model(t).logits.float()
            logprobs = torch.log_softmax(logits[0, :-1, :], dim=-1)
            chosen = logprobs.gather(1, t[0, 1:].unsqueeze(1)).squeeze(1)
            ppls.append(perplexity_from_logprobs(chosen.tolist()))
        return ppls


def build_local_model(model_key: str, spec: dict, gen_cfg: dict,
                      hf_token: Optional[str]):
    try:
        import vllm
        return VLLMModel(model_key, spec, gen_cfg, hf_token)
    except ImportError:
        print(f"[model_wrappers] vLLM not installed; running '{model_key}' through "
              f"transformers instead.", flush=True)
        return HFLocalModel(model_key, spec, gen_cfg, hf_token)


class ProviderUnavailable(RuntimeError):
    pass


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
    extra = ""
    if suppressed:
        extra = (f" [{suppressed} further occurrence(s) suppressed in the last "
                 f"{now - last:.0f}s]")
    print(message + extra, flush=True)


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
        print(f"[LocalServerModel] Ready: {self.model_key} -> '{self.served_name}' "
              f"via {base_url} (raw /v1/completions)")

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
                except Exception as e:
                    n_p = len(prompts[s:s + bs])
                    by_start[s] = [["" for _ in range(nn)] for _ in range(n_p)]
                    print(f"[LocalServerModel] request at offset {s} failed after "
                          f"retries ({type(e).__name__}); {n_p} item(s) left empty, "
                          f"will be retried on the next run.", flush=True)
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
                except Exception as e:
                    results[i] = ["" for _ in range(int(specs[i].get("n", 1)))]
                    print(f"[LocalServerModel] request {i} failed after retries "
                          f"({type(e).__name__}); left empty, retried on the next run.",
                          flush=True)
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
                except Exception as e:
                    print(f"[LocalServerModel] perplexity batch at offset {futs[f]} "
                          f"failed ({type(e).__name__}); left as nan.", flush=True)
        return ppls

class APIModel:

    def __init__(self, name: str, spec: dict, secret: Optional[str]):
        self.name = name
        self.provider = spec["provider"].lower()
        self.model = spec.get("model")
        self.spec = spec
        self._secret = secret
        self._client = None
        if _is_placeholder(self.model):
            print(f"[model_wrappers] WARNING: model id for '{name}' is a placeholder "
                  f"('{self.model}'); set the real id in config before running.")

    def _ensure_client(self):
        if self._client is not None:
            return
        base_url = self.spec.get("base_url")
        if self.provider != "openai_compatible":
            raise ValueError(
                f"[model_wrappers] unsupported provider '{self.provider}'. Only "
                f"'openai_compatible' (the local proxy) is supported; the study "
                f"calls no external provider."
            )
        if not base_url:
            raise ValueError(
                "[model_wrappers] provider 'openai_compatible' requires "
                "spec.base_url (a local proxy); this study makes no calls to "
                "OpenAI's own API."
            )
        from openai import OpenAI
        import httpx
        verify = bool(self.spec.get("verify_ssl", True))
        timeout = float(self.spec.get("timeout_s", 1800))
        self._client = OpenAI(
            api_key=(self._secret or "EMPTY"), base_url=base_url,
            http_client=httpx.Client(verify=verify, timeout=timeout),
        )

    def _retry(self, fn):
        import time, random
        s = self.spec
        max_retries    = max(1, int(s.get("max_retries", 6)))
        base           = float(s.get("retry_backoff_s", 2.0))
        cap            = float(s.get("retry_backoff_max_s", 30.0))
        connect_budget = float(s.get("connect_retry_max_s", 600.0))
        t0 = time.time()
        last, attempt, rl_attempt, announced = None, 0, 0, False
        while True:
            try:
                return fn()
            except Exception as e:
                last = e
                if _is_rate_limit(e):
                    rl_attempt += 1
                    if rl_attempt > int(s.get("rate_limit_retries", 8)):
                        break
                    wait = _retry_after_s(e) or min(30.0, base * (2 ** min(rl_attempt, 5)))
                    if not announced:
                        _notice(f"429:{self.name}",
                                f"[APIModel] '{self.name}' throttled (429). Backing off; "
                                f"lower the caller's concurrency if this persists.")
                        announced = True
                    time.sleep(wait * (0.7 + 0.6 * random.random()))
                    continue
                if _is_transient_server_error(e):
                    if (time.time() - t0) >= connect_budget:
                        break
                    if not announced:
                        _notice(f"outage:{self.name}",
                                f"[APIModel] '{self.name}' unreachable "
                                f"({type(e).__name__}). Waiting for the proxy to come "
                                f"back (up to {connect_budget:.0f}s); the run will "
                                f"resume automatically.")
                        announced = True
                    sleep = min(cap, base * (2 ** min(attempt, 8)))
                    time.sleep(sleep * (0.7 + 0.6 * random.random()))
                    attempt += 1
                    continue
                raise
        raise ProviderUnavailable(
            f"[APIModel] '{self.name}' failed after "
            f"{time.time() - t0:.0f}s of retries: {last}"
        ) from last

    def generate(
        self,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system: Optional[str] = None,
    ) -> Tuple[str, str]:
        self._ensure_client()
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": user_prompt}]
        kwargs = dict(model=self.model, messages=msgs,
                      temperature=temperature, max_tokens=max_tokens)
        thinking = self.spec.get("thinking")
        extra_body = {} if thinking is None else \
            {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}
        effort = self.spec.get("reasoning_effort")

        def _create():
            if extra_body:
                kwargs["extra_body"] = dict(extra_body)
            if effort:
                try:
                    return self._client.chat.completions.create(
                        reasoning_effort=effort, **kwargs)
                except TypeError:
                    eb = dict(kwargs.pop("extra_body", {}))
                    eb["reasoning_effort"] = effort
                    return self._client.chat.completions.create(
                        extra_body=eb, **kwargs)
            return self._client.chat.completions.create(**kwargs)

        resp = self._retry(_create)
        msg = resp.choices[0].message
        text = msg.content or ""
        cot = str(getattr(msg, "reasoning_content", "") or "")
        if _THINK_CLOSE in text:
            head, text = text.split(_THINK_CLOSE, 1)
            cot = cot or head.split(_THINK_OPEN, 1)[-1]
        elif not cot and text.lstrip().startswith(_THINK_OPEN):
            cot, text = text.lstrip()[len(_THINK_OPEN):], ""
        return cot.strip(), text.strip()


def _find_open_spec(cfg: dict, model_key: str) -> Optional[dict]:
    return cfg["model_panel"]["open"].get(model_key)


def hf_dtype_kwargs(dtype) -> dict:
    key = "torch_dtype"
    try:
        import inspect
        import transformers
        from transformers import AutoModelForCausalLM
        params = inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
        major = int(str(transformers.__version__).split(".")[0])
        if "dtype" in params or major >= 5:
            key = "dtype"
    except Exception:
        pass
    return {key: dtype}


def release_gpu():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def purge_model_cache(hf_id: str, cfg: dict) -> None:
    import gc
    import shutil

    rt = cfg.get("runtime", {}) or {}
    if not rt.get("purge_hf_cache_after_use", False):
        return
    if not hf_id:
        return
    if os.path.isdir(hf_id):
        print(f"[cache] '{hf_id}' is a local path; not purging.")
        return

    hub = os.path.expanduser(rt.get("hf_hub_cache") or "~/.cache/huggingface/hub")
    try:
        from huggingface_hub import repo_folder_name
        folder = repo_folder_name(repo_id=hf_id, repo_type="model")
    except Exception:
        folder = "models--" + hf_id.replace("/", "--")
    path = os.path.join(hub, folder)

    gc.collect()
    if not os.path.isdir(path):
        print(f"[cache] nothing to purge for '{hf_id}' (no folder at {path}).")
        return
    try:
        shutil.rmtree(path)
        if os.path.isdir(path):
            print(f"[cache] could NOT remove '{hf_id}' at {path} "
                  f"(folder still present; weights may be held open).")
        else:
            print(f"[cache] purged '{hf_id}' -> {path}")
    except Exception as e:
        print(f"[cache] could not purge '{hf_id}' at {path}: {e}")


def is_judge_key(model_key: str) -> bool:
    return model_key == "judge" or model_key.startswith("judge_")


def judge_spec(cfg: dict, judge_key: str = "judge") -> dict:
    spec = (cfg.get("model_panel", {}) or {}).get(judge_key)
    if not isinstance(spec, dict):
        declared = sorted(k for k in (cfg.get("model_panel", {}) or {}) if is_judge_key(k))
        raise KeyError(
            f"[model_wrappers] judge '{judge_key}' is not declared under "
            f"model_panel. Declared judges: {declared}. Check "
            f"model_panel.judge_routing.")
    return spec


def model_family(cfg: dict, model_key: str) -> str:
    spec = _find_open_spec(cfg, model_key) or {}
    return str(spec.get("family") or "")


def judge_key_for_family(cfg: dict, family: str, who: str = "") -> str:
    routing = (cfg.get("model_panel", {}) or {}).get("judge_routing", {}) or {}
    default_key = routing.get("default", "judge")
    fam = (family or "").strip()
    key = (routing.get("by_family", {}) or {}).get(fam, default_key) if fam else default_key
    spec = judge_spec(cfg, key)
    jfam = str(spec.get("family") or "")
    label = who or (fam or "an untagged model")
    if not fam or not jfam:
        _notice(f"judge_family:{label}",
                f"[model_wrappers] cannot verify judge independence for {label}: "
                f"model family='{fam or 'MISSING'}', judge '{key}' "
                f"family='{jfam or 'MISSING'}'. Tag both in config to enable the "
                f"check.")
        return key
    if jfam == fam:
        raise ValueError(
            f"[model_wrappers] judge '{key}' ({spec.get('model')}, family={jfam}) "
            f"is from the SAME family as {label}, so it would judge its own "
            f"lineage: the same weights would write the paraphrase and the "
            f"corruption and then run the recovery check on them. Add a "
            f"model_panel.judge_routing.by_family entry for '{fam}' pointing at a "
            f"judge outside it.")
    return key


def judge_key_for_model(cfg: dict, model_key: str) -> str:
    return judge_key_for_family(cfg, model_family(cfg, model_key), who=model_key)


def get_model_wrapper(model_key: str, global_config_path: str):
    cfg = read_config(global_config_path)["CotLoadbearing"]
    gen_cfg = cfg["generation"]

    open_spec = _find_open_spec(cfg, model_key)
    if open_spec is not None:
        backend = (cfg.get("serving", {}) or {}).get("backend", "vllm")
        if backend == "api":
            return LocalServerModel(model_key, open_spec, gen_cfg,
                                    cfg.get("hf_token"), cfg.get("serving", {}))
        return VLLMModel(model_key, open_spec, gen_cfg, cfg.get("hf_token"))

    if is_judge_key(model_key):
        spec = judge_spec(cfg, model_key)
        if not spec.get("base_url"):
            raise ValueError(
                f"[model_wrappers] judge '{model_key}' has no base_url. Every "
                f"model in this study is served by the local proxy; there is no "
                f"external-provider path.")
        serving = cfg.get("serving", {}) or {}
        secret = resolve_secret(
            spec.get("api_key", serving.get("api_key")),
            spec.get("api_key_env", serving.get("api_key_env", "CROSSMODALITY_API_KEY")))
        return APIModel(model_key, spec, secret)

    raise KeyError(
        f"[model_wrappers] unknown model key '{model_key}'. It is not in "
        f"model_panel.open and is not a judge key. (The closed panel was removed "
        f"on 2026-07-30; the study is open-weight only.)"
    )
