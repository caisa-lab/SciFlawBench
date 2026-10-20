from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, replace

from models.context import MEMORY_INDEX_MARKER
from tools.artifacts import ArtifactStore

logger = logging.getLogger("sciflawbench.memory")

DEFAULT_MEMORY_FILE_NAME = "memory.md"
# How much of an evicted text is shown inline as a digest.
DEFAULT_DIGEST_CHARS = 200
# Hard cap on the index that is injected into the task message.
DEFAULT_INDEX_MAX_CHARS = 4000
DEFAULT_LABEL_CHARS = 80
# Texts shorter than this are never off-loaded: the pointer would cost more context than the text it replaces.
DEFAULT_MIN_OFFLOAD_CHARS = 800

_WHITESPACE_RE = re.compile(r"\s+")


def _collapse(text: str) -> str:
    """Single-line, whitespace-collapsed rendering of `text`."""
    return _WHITESPACE_RE.sub(" ", str(text)).strip()


def _truncate(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


@dataclass
class MemoryBlock:
    """One evicted message, stored verbatim in the memory file."""

    id: int
    marker: str
    label: str
    digest: str
    chars: int
    text: str = ""


class MemoryOffloader:
    """
    Spills evicted context to a file and hands back a pointer to it.

    Args:
        store (ArtifactStore | None): the task's artifact store; the file goes into its *current scope*, so the
            artifact tools find it without extra wiring
        enabled (bool): master switch
        file_name (str): name of the memory file inside the scope
        digest_chars (int): characters of the evicted text shown in the pointer and the index (0 disables digests)
        index_max_chars (int): hard cap on the injected index
        label_chars (int): cap on the per-block label
        min_chars (int): do not off-load texts shorter than this
    """

    def __init__(
        self,
        store: ArtifactStore | None,
        *,
        enabled: bool = True,
        file_name: str = DEFAULT_MEMORY_FILE_NAME,
        digest_chars: int = DEFAULT_DIGEST_CHARS,
        index_max_chars: int = DEFAULT_INDEX_MAX_CHARS,
        label_chars: int = DEFAULT_LABEL_CHARS,
        min_chars: int = DEFAULT_MIN_OFFLOAD_CHARS,
    ):
        self.store = store
        self.file_name = file_name or DEFAULT_MEMORY_FILE_NAME
        self.digest_chars = max(0, int(digest_chars or 0))
        self.index_max_chars = max(0, int(index_max_chars or 0))
        self.label_chars = max(1, int(label_chars or DEFAULT_LABEL_CHARS))
        self.min_chars = max(0, int(min_chars or 0))
        self._enabled = bool(enabled)
        self._blocks: list[MemoryBlock] = []
        self._seen: dict[str, MemoryBlock] = {}
        self._chars_spilled = 0
        self._file_rel = ""

    @property
    def enabled(self) -> bool:
        """True when off-loading can actually write files."""
        return bool(self._enabled and self.store is not None and self.store.enabled)

    @property
    def has_blocks(self) -> bool:
        return bool(self._blocks)

    def reset(self) -> None:
        """Forget everything (call once per task attempt)."""
        self._blocks = []
        self._seen = {}
        self._chars_spilled = 0
        self._file_rel = ""

    def snapshot(self) -> dict:
        """Counters for logging and trace metadata."""
        return {"blocks": len(self._blocks), "chars": self._chars_spilled, "file": self._file_rel}

    def offload(self, text: str, *, fallback: str, role: str = "", label: str = "") -> str:
        """
        Spill `text` and return the pointer that replaces it.

        Returns `fallback` (the plain "truncated" note) whenever off-loading is unavailable, impossible or not
        worthwhile, so the caller can always use the return value verbatim.
        """
        if not self.enabled or not text:
            return fallback
        text = str(text)
        if len(text) < self.min_chars:
            return fallback

        digest = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
        block = self._seen.get(digest)
        if block is not None:
            # smolagents rebuilds the conversation every step, so the same eviction is re-derived each time
            return self._pointer(block)

        index = len(self._blocks) + 1
        block = MemoryBlock(
            id=index,
            marker=f"[block {index}]",
            label=_truncate(_collapse(label) or self._role_label(role), self.label_chars),
            digest=_truncate(_collapse(text), self.digest_chars),
            chars=len(text),
            text=text,
        )
        if len(self._pointer(block)) >= len(text):
            return fallback  # a pointer longer than the text frees nothing

        self._blocks.append(block)
        self._seen[digest] = block
        self._chars_spilled += len(text)
        if not self._write_file():
            self._blocks.pop()
            del self._seen[digest]
            self._chars_spilled -= len(text)
            return fallback

        logger.info("Memory off-loaded: %s (%s chars) -> %s", block.marker, f"{block.chars:,}", self._file_rel)
        # rebuilt now that the final file path is known, so a later dedup hit renders identical text
        return self._pointer(block)

    def _write_file(self) -> bool:
        """Rewrite the whole memory file (it is small and grows slowly)."""
        try:
            ref = self.store.save_named(self.file_name, self._render_file(), tool_name="memory")
        except Exception as exc:  # noqa: BLE001 - never break a run over memory
            logger.warning("Memory file could not be written: %s", exc)
            self._file_rel = ""
            return False
        self._file_rel = ref.rel_path
        return True

    def _render_file(self) -> str:
        sections = [
            "# Offloaded memory\n\n"
            "Maintained by the harness: context snippets (observations and reasoning) saved verbatim before being "
            "removed from the prompt to free space. Use `search_artifact` / `read_artifact` to read them back.\n"
        ]
        for block in self._blocks:
            title = f"\n### {block.marker}" + (f" — {block.label}" if block.label else "")
            sections.append(f"{title}\n<!-- chars: {block.chars} -->\n\n{block.text}\n")
        return "\n".join(sections)

    @staticmethod
    def _role_label(role: str) -> str:
        role = str(role or "").lower()
        if role in ("tool-response", "tool"):
            return "tool observation"
        if role == "assistant":
            return "reasoning/code"
        return role or "context"

    def _pointer(self, block: MemoryBlock) -> str:
        file_name = self._file_rel or self.file_name
        parts = [f"[offloaded memory · {block.marker}" + (f" · {block.label}" if block.label else "") + "]"]
        if block.digest:
            parts.append(f'"{block.digest}"')
        parts.append(
            f"Full content in `{file_name}` — retrieve it with "
            f'search_artifact(pattern="{block.marker}") or read_artifact(path="{file_name}", start_line=…).'
        )
        return " ".join(parts)

    @staticmethod
    def _entry_line(block: MemoryBlock) -> str:
        line = f"- {block.marker}"
        if block.label:
            line += f" {block.label}"
        if block.digest:
            line += f' — "{block.digest}"'
        return line

    def index_text(self) -> str:
        """
        Compact index of every off-loaded block (empty when there is none).

        The index is injected into the *task* message, which the trimmer never touches, so a pointer survives even
        after the message that carried it has been dropped.
        """
        if not self._blocks or self.index_max_chars <= 0:
            return ""
        file_name = self._file_rel or self.file_name
        head = [
            MEMORY_INDEX_MARKER,
            "## Offloaded memory (older context)",
            f"Older content was saved to `{file_name}` to free space. "
            f'Retrieve a block with search_artifact(pattern="[block N]") '
            f'(keyword search) or read_artifact(path="{file_name}", start_line=…).',
        ]

        blocks = list(self._blocks)
        text = "\n".join(head + [self._entry_line(b) for b in blocks])
        if len(text) <= self.index_max_chars:
            return text

        # 1) drop the digest from the oldest entries first (their pointers stay)
        for position in range(len(blocks)):
            if len(text) <= self.index_max_chars:
                break
            if not blocks[position].digest:
                continue
            blocks[position] = replace(blocks[position], digest="")
            text = "\n".join(head + [self._entry_line(b) for b in blocks])

        # 2) still too long: collapse the oldest entries into a single range line
        collapsed = 0
        while len(text) > self.index_max_chars and collapsed < len(blocks) - 1:
            collapsed += 1
            remaining = blocks[collapsed:]
            collapse_line = (
                f"- [blocks {blocks[0].id}-{remaining[0].id - 1}] {collapsed} offloaded block(s) — see `{file_name}`"
            )
            text = "\n".join(head + [collapse_line] + [self._entry_line(b) for b in remaining])
        return text
