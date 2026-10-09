from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Default context window assumed when a model does not declare a larger one.
DEFAULT_MODEL_MAX_CONTEXT = 128_000
# Input tokens kept free on top of the trim budget to absorb estimation drift.
CONTEXT_TRIM_RESERVE = 512
# Placeholder that replaces a truncated observation.
CONTEXT_TRIM_NOTE = "[… observation truncated to fit the model context window]"
# Rough characters-per-token ratio when no real tokenizer is available.
CHARS_PER_TOKEN = 4


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


def trim_messages_to_budget(
    messages: list[Any],
    budget_tokens: int,
    count_tokens: Callable[[str], int] = estimate_tokens,
    note: str = CONTEXT_TRIM_NOTE,
) -> list[Any]:
    """Shrink a rendered conversation to `budget_tokens` without changing its structure.

    Observations are truncated oldest-first, then other non-essential messages, then
    assistant turns; only as a last resort are the oldest messages dropped entirely.  The
    system prompt and the original task are always preserved, as are the most recent messages.

    Returns the (possibly truncated) list of messages; the input list is returned unchanged
    when it already fits.  Mutates the message objects in place — smolagents rebuilds the
    ChatMessage list from its memory on every step, so that is safe.
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

    note_tokens = count_tokens(note)
    for index in order:
        if total <= budget_tokens:
            break
        freed = counts[index] - note_tokens
        if freed <= 0:
            continue
        set_message_text(messages[index], note)
        counts[index] = note_tokens
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

    return messages
