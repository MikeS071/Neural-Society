"""Planner back-ends.

AnthropicPlanner calls the Messages API with the standard library only (urllib).
ScriptedPlanner is a deterministic stand-in used to test the harness offline; it is
never used to produce results.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    thinking_chars: int = 0  # reasoning text the server returned separately (not shown to either arm)
    load_s: float = 0.0  # time the server spent loading the model inside these calls

    def add(self, other):
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.seconds += other.seconds
        self.thinking_chars += other.thinking_chars
        self.load_s += other.load_s


@dataclass
class PlannerReply:
    text: str
    usage: Usage = field(default_factory=Usage)


class AnthropicPlanner:
    def __init__(self, model="claude-sonnet-5-5", max_tokens=4000, api_key=None, base_url=None, temperature=0.0):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        # NS_ANTHROPIC_BASE_URL, not ANTHROPIC_BASE_URL: tool hosts often set the latter for themselves.
        self.base_url = (base_url or os.environ.get("NS_ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")

    def __call__(self, system, messages):
        body = json.dumps(
            {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "system": system,
                "messages": messages,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/v1/messages",
            data=body,
            method="POST",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        )
        delay = 2.0
        for attempt in range(6):
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    out = json.loads(r.read())
                text = "".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text")
                u = out.get("usage", {})
                return PlannerReply(
                    text,
                    Usage(
                        1,
                        u.get("input_tokens", 0)
                        + u.get("cache_read_input_tokens", 0)
                        + u.get("cache_creation_input_tokens", 0),
                        u.get("output_tokens", 0),
                        time.time() - t0,
                    ),
                )
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504, 529) and attempt < 5:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise RuntimeError(f"Anthropic API error {e.code}: {e.read()[:500]!r}") from None
            except (urllib.error.URLError, TimeoutError):
                if attempt < 5:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise


class OpenAICompatPlanner:
    """Any server speaking the OpenAI chat-completions API: Ollama, llama.cpp's llama-server,
    LM Studio, vLLM. Used for locally hosted (e.g. CPU-only) planners. No API key needed for
    local servers; NS_LOCAL_API_KEY is sent as a bearer token if set.

    `think`: None leaves the server's default. True/False is sent both as
    chat_template_kwargs.enable_thinking (llama-server, vLLM, SGLang) and as `think`
    (Ollama); servers ignore whichever they do not understand. Reasoning text is never
    kept in the conversation history, so both arms carry the same context."""

    def __init__(
        self,
        model,
        base_url="http://localhost:11434/v1",
        max_tokens=None,
        temperature=0.0,
        timeout=3600,
        think=None,
    ):
        self.model = model
        self.think = think
        if max_tokens is None:
            max_tokens = 12000 if think else 3000  # reasoning traces need room

        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout  # CPU inference on long prompts can take minutes
        self.api_key = os.environ.get("NS_LOCAL_API_KEY")

    def __call__(self, system, messages):
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "system", "content": system}, *messages],
        }
        if self.think is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": bool(self.think)}
            payload["think"] = bool(self.think)
        body = json.dumps(payload).encode()
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, method="POST", headers=headers)
        for attempt in range(3):
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read())
                text = strip_thinking(out["choices"][0]["message"].get("content") or "")
                u = out.get("usage") or {}
                return PlannerReply(
                    text, Usage(1, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), time.time() - t0)
                )
            except urllib.error.HTTPError as e:
                if e.code >= 500 and attempt < 2:
                    time.sleep(5)
                    continue
                raise RuntimeError(f"local model server error {e.code}: {e.read()[:500]!r}") from None


class OllamaPlanner:
    """Ollama's native /api/chat. Preferred over its OpenAI-compatible endpoint because it
    honours `think` (reasoning on/off) and per-request `num_ctx`, and reports load and
    generation timings. `base_url` is the server root, e.g. http://127.0.0.1:11434."""

    def __init__(
        self,
        model,
        base_url="http://127.0.0.1:11434",
        max_tokens=None,
        temperature=0.0,
        num_ctx=16384,
        timeout=3600,
        think=None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.think = think
        self.max_tokens = max_tokens or (12000 if think else 3000)
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout = timeout

    def _post(self, payload):
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode(),
            method="POST",
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def warmup(self):
        """Load the model into memory so the first timed call does not pay for it."""
        self._post(
            {
                "model": self.model,
                "messages": [],
                "stream": False,
                "keep_alive": "30m",
                "options": {"num_ctx": self.num_ctx},
            }
        )

    def __call__(self, system, messages):
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx, "num_predict": self.max_tokens},
        }
        if self.think is not None:
            payload["think"] = bool(self.think)
        t0 = time.time()
        try:
            out = self._post(payload)
        except urllib.error.HTTPError as e:
            body = e.read()[:500]
            if e.code == 400 and b"think" in body and "think" in payload:  # model has no thinking switch
                payload.pop("think")
                out = self._post(payload)
            else:
                raise RuntimeError(f"ollama error {e.code}: {body!r}") from None
        msg = out.get("message") or {}
        text = strip_thinking(msg.get("content") or "")
        return PlannerReply(
            text,
            Usage(
                1,
                out.get("prompt_eval_count", 0),
                out.get("eval_count", 0),
                time.time() - t0,
                thinking_chars=len(msg.get("thinking") or ""),
                load_s=out.get("load_duration", 0) / 1e9,
            ),
        )


class ScriptedPlanner:
    """Offline stand-in: replays a fixed list of replies (or a function of the conversation).
    Token counts are estimated as len(text) / 4 so harness accounting can be tested."""

    def __init__(self, replies):
        self.replies = replies
        self.i = 0

    def __call__(self, system, messages):
        r = (
            self.replies(system, messages)
            if callable(self.replies)
            else self.replies[min(self.i, len(self.replies) - 1)]
        )
        self.i += 1
        n_in = (len(system) + sum(len(m["content"]) for m in messages)) // 4
        return PlannerReply(r, Usage(1, n_in, len(r) // 4, 0.0))


def oracle_planner(task):
    """A ScriptedPlanner that reverts the task's injected bugs: proves a perfect planner can
    succeed through each arm's interface. For harness tests only."""

    def reply(system, messages):
        edits = [{"file": b.file, "old": b.new, "new": b.old} for b in task.bugs if b.new]
        if any(not b.new for b in task.bugs):  # a deleted block: rewrite the file wholesale
            from .tasks import PROGRAMS

            for b in task.bugs:
                if not b.new:
                    with open(os.path.join(PROGRAMS, task.program, b.file)) as f:
                        ref = f.read()
                    edits = [e for e in edits if e["file"] != b.file] + [{"file": b.file, "content": ref}]
        return "Reverting.\n```json\n" + json.dumps({"edits": edits}) + "\n```"

    return ScriptedPlanner(reply)


_THINK = re.compile(r"<think>.*?</think>\s*", re.S)


def strip_thinking(text):
    """Drop inline reasoning (<think>...</think>), including an unterminated trailing block."""
    text = _THINK.sub("", text)
    i = text.find("<think>")
    return text[:i] if i != -1 else text


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def parse_edits(text):
    """Extract {"edits": [...]} from a reply. Returns (edits, error_or_None)."""
    candidates = _JSON_BLOCK.findall(text)
    if not candidates:
        a, b = text.find("{"), text.rfind("}")
        if a != -1 and b > a:
            candidates = [text[a : b + 1]]
    for c in reversed(candidates):
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("edits"), list):
            edits = [e for e in obj["edits"] if isinstance(e, dict)]
            return edits, None
    return [], 'could not find a JSON object with an "edits" list in your reply'
