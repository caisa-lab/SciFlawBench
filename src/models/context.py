from __future__ import annotations

import contextlib
import json
import logging
import re
import urllib.request
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("sciflawbench.context")

# Default context window assumed when a model does not declare a larger one.
DEFAULT_MODEL_MAX_CONTEXT = 256_000
# Input tokens kept free on top of the trim budget to absorb estimation drift.
CONTEXT_TRIM_RESERVE = 512
# Placeholder that replaces a truncated observation.
CONTEXT_TRIM_NOTE = "[… observation truncated to fit the model context window]"
# Rough characters-per-token ratio when no real tokenizer is available.
CHARS_PER_TOKEN = 4
# Idempotency marker of the off-loaded memory index appended to the task message.
MEMORY_INDEX_MARKER = "<!-- sfb-memory-index -->"
# Timeout of the optional context-window probe against an OpenAI-compatible server.
WINDOW_PROBE_TIMEOUT_S = 5


def estimate_tokens(text: Any) -> int:
    """Estimate the token count of a string from its length (no tokenizer needed)."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def message_role(message: Any) -> str:
    """Message role as a plain string ('system', 'assistant', 'tool-response', ...)."""
    role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
    return str(getattr(role, "value", role) or "")


def message_text(message: Any) -> str:
    """Flatten a ChatMessage/dict into its text content."""
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text") or item.get("content") or "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return "" if content is None else str(content)


def set_message_text(message: Any, text: str) -> None:
    """Replace a ChatMessage/dict text content, preserving non-text blocks."""
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if isinstance(content, list):
        non_text = [part for part in content if isinstance(part, dict) and part.get("type") not in (None, "text")]
        new_content: Any = ([{"type": "text", "text": text}] if text else []) + non_text
    else:
        new_content = text
    if isinstance(message, dict):
        message["content"] = new_content
    else:
        message.content = new_content


def count_message_tokens(messages: list[Any], count_tokens: Callable[[str], int] = estimate_tokens) -> int:
    """Total estimated token count of a message list."""
    return sum(count_tokens(message_text(message)) for message in messages)


def compute_input_budget(
    context_window: int | None, max_new_tokens: int | None = None, reserve_tokens: int = CONTEXT_TRIM_RESERVE
) -> int:
    """
    Input-token budget: `context_window - max_new_tokens - reserve_tokens`.

    Returns 0 (no trimming) without a window. When the response allowance alone would swallow the window, half the
    window is kept for the input (never less than 1024 tokens) so the conversation stays trimmable.
    """
    if not context_window:
        return 0
    budget = int(context_window) - int(max_new_tokens or 0) - int(reserve_tokens or 0)
    if budget <= 0:
        budget = max(1024, int(context_window) // 2)
    return budget


def _probe_server_window(api_base: str, model_id: str, api_key: str | None) -> int | None:
    """`max_model_len` advertised by an OpenAI-compatible server (vLLM) for `model_id`, if any."""
    request = urllib.request.Request(api_base.rstrip("/") + "/models")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=WINDOW_PROBE_TIMEOUT_S) as response:  # noqa: S310 - operator-supplied URL
        payload = json.load(response)
    for entry in payload.get("data", []):
        value = entry.get("max_model_len") if isinstance(entry, dict) else None
        if isinstance(value, int) and value > 0 and entry.get("id") in (model_id, None):
            return value
    return None


def resolve_context_window(
    provider: str,
    model_id: str,
    api_base: str | None = None,
    api_key: str | None = None,
    explicit: int | None = None,
) -> int:
    """
    Resolve the model's context window (maximum input tokens), in decreasing order of authority:

    1. the explicit `model_max_context` from the config (operator override; always wins),
    2. the `max_model_len` an OpenAI-compatible server (vLLM) reports for the model: the limit it enforces,
    3. LiteLLM's model-info table (`max_input_tokens`),

    and finally `DEFAULT_MODEL_MAX_CONTEXT`. A wrong window silently disables trimming, so the fallback is logged.
    """
    if explicit:
        return int(explicit)
    if provider == "fake_model":
        return DEFAULT_MODEL_MAX_CONTEXT

    if provider == "openai_server" and api_base:
        with contextlib.suppress(Exception):
            value = _probe_server_window(api_base, model_id, api_key)
            if value:
                return value

    if provider in ("litellm", "openai_server", "hf_api"):
        with contextlib.suppress(Exception):
            import litellm

            value = litellm.get_model_info(model_id).get("max_input_tokens")
            if isinstance(value, int) and value > 0:
                return value

    logger.warning(
        "Could not infer the context window of %s/%s; assuming %d tokens (set model_max_context to override).",
        provider,
        model_id,
        DEFAULT_MODEL_MAX_CONTEXT,
    )
    return DEFAULT_MODEL_MAX_CONTEXT


def apply_memory_index(messages: list[Any], index_text: str) -> list[Any]:
    """Append the off-loaded-memory index to the task message (idempotent)."""
    if not index_text or len(messages) < 2:
        return messages
    target = messages[1]
    base = message_text(target)
    if MEMORY_INDEX_MARKER in base:
        base = base.split(MEMORY_INDEX_MARKER)[0].rstrip()
    set_message_text(target, f"{base}\n\n{index_text}" if base else index_text)
    return messages


# Python builtins that would otherwise be mistaken for tool calls when labelling an evicted observation.
_LABEL_STOPWORDS = frozenset(
    {
        *("print", "len", "str", "int", "float", "bool", "sum", "min", "max", "sorted", "list", "dict", "set"),
        *("tuple", "range", "open", "round", "abs", "eval", "zip", "input", "enumerate", "map", "filter"),
        *("any", "all", "isinstance", "type", "super", "getattr", "setattr", "format", "join", "split"),
        *("strip", "replace", "append", "keys", "values", "items", "lower", "upper", "count", "index"),
        *("startswith", "endswith"),
    }
)
_CALL_RE = re.compile(r"\b([a-z_][a-z0-9_]{2,})\s*\(")


def _guess_tool_names(text: str, limit: int = 2) -> list[str]:
    """Best-effort list of tool/function names called in a code block."""
    names: list[str] = []
    for match in _CALL_RE.finditer(text or ""):
        name = match.group(1)
        if name in _LABEL_STOPWORDS or name in names:
            continue
        names.append(name)
        if len(names) >= limit:
            break
    return names


def eviction_label(messages: list[Any], index: int) -> str:
    """Tool names behind an evicted observation (the preceding assistant code), for the memory index."""
    if not 0 < index < len(messages) or message_role(messages[index]) not in ("tool-response", "tool"):
        return ""
    return ", ".join(_guess_tool_names(message_text(messages[index - 1])))


def trim_messages_to_budget(
    messages: list[Any],
    budget_tokens: int,
    count_tokens: Callable[[str], int] = estimate_tokens,
    note: str = CONTEXT_TRIM_NOTE,
    offloader: Any = None,
) -> list[Any]:
    """
    Shrink a rendered conversation to `budget_tokens` without changing its structure.

    Observations are evicted oldest-first, then other non-essential messages, then assistant turns; only as a last
    resort are the oldest messages dropped entirely. The system prompt and the original task are always preserved, as
    are the most recent messages.

    When `offloader` (a `models.memory.MemoryOffloader`) is given, an evicted message is not lost: its full text is
    written to the task's memory file and the message is replaced by a short digest plus a pointer. A compact index of
    every block is appended to the task message, which is never trimmed, so pointers outlive the messages that carried
    them.

    Returns the (possibly shrunk) list of messages; the input list is returned unchanged when it already fits.
    Mutates the message objects in place; smolagents rebuilds the ChatMessage list from its memory on every step, so
    that is safe.
    """
    if not budget_tokens or budget_tokens <= 0 or not messages:
        return messages

    counts = [count_tokens(message_text(m)) for m in messages]
    total = sum(counts)
    if total <= budget_tokens:
        return messages

    n = len(messages)
    protect_head = min(2, n)  # system prompt + original task
    protect_tail = min(4, max(0, n - protect_head))
    trimmable = list(range(protect_head, n - protect_tail))

    roles = {i: message_role(messages[i]) for i in trimmable}
    observations = [i for i in trimmable if roles[i] in ("tool-response", "tool")]
    assistants = [i for i in trimmable if roles[i] == "assistant"]
    others = [i for i in trimmable if i not in observations and i not in assistants]
    order = observations + others + assistants

    can_offload = offloader is not None and getattr(offloader, "enabled", False)
    for index in order:
        if total <= budget_tokens:
            break
        replacement = note
        if can_offload:
            replacement = offloader.offload(
                message_text(messages[index]),
                fallback=note,
                role=roles[index],
                label=eviction_label(messages, index),
            )
        new_tokens = count_tokens(replacement)
        freed = counts[index] - new_tokens
        if freed <= 0:
            continue
        set_message_text(messages[index], replacement)
        counts[index] = new_tokens
        total -= freed

    if total > budget_tokens:
        drop: set[int] = set()
        for index in order:
            if total <= budget_tokens:
                break
            drop.add(index)
            total -= counts[index]
        if drop:
            messages = [m for i, m in enumerate(messages) if i not in drop]

    if offloader is not None and getattr(offloader, "has_blocks", False):
        messages = apply_memory_index(messages, offloader.index_text())

    return messages
