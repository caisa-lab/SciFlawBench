from __future__ import annotations

import functools
import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("sciflawbench.artifacts")

# Cap applied to a tool output before it is spilled to disk (characters).
DEFAULT_TOOL_OUTPUT_MAX_CHARS = 20000
# How much of an oversized output is shown inline in the observation.
DEFAULT_PREVIEW_CHARS = 8000
# Hard upper bound on a single spilled file (bytes); larger outputs fall back
# to plain in-context truncation so a runaway page cannot fill the disk.
DEFAULT_ARTIFACT_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_MAX_MATCHES = 50
DEFAULT_CONTEXT_LINES = 2
DEFAULT_READ_MAX_CHARS = 20000
MAX_MATCH_LIMIT = 500

# Wikipedia pages spill only above this size (larger than the global default).
WIKIPEDIA_OUTPUT_CAP = 50000

_ARTIFACT_EXTENSIONS = (".md", ".txt")
_MARKDOWN_TOOLS = {"wikipedia_search", "web_search", "visit_webpage", "arxiv_search"}

# Name of the attribute set on a wrapped tool so capping stays idempotent.
_CAP_MARKER = "_sfb_output_cap"


class ArtifactError(Exception):
    """Raised for invalid artifact operations (bad path, bad pattern, ...)."""


@dataclass
class ArtifactRef:
    """A saved artifact."""

    path: Path
    rel_path: str
    chars: int
    lines: int
    created: bool


def _slug(text: str) -> str:
    """Filesystem-safe slug for names/scopes."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))[:120].strip("_") or "root"


class ArtifactStore:
    """Writes oversized tool outputs to disk and reads/searches them back.

    Args:
        root:               Directory that holds every artifact (created if
                            missing).  All paths passed to the retrieval tools
                            are sandboxed to this directory.
        enabled:            When False the store is a no-op (no files written).
        max_artifact_bytes: Skip spilling a single output larger than this.
        manifest_name:      File (under `root`) that records every artifact.
        preview_chars:      How much of an oversized output to inline.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        enabled: bool = True,
        max_artifact_bytes: int = DEFAULT_ARTIFACT_MAX_BYTES,
        manifest_name: str = "manifest.jsonl",
        preview_chars: int = DEFAULT_PREVIEW_CHARS,
    ):
        self.root = Path(root)
        self.enabled = enabled
        self.max_artifact_bytes = int(max_artifact_bytes)
        self.manifest_name = manifest_name
        self.preview_chars = int(preview_chars)
        self._scope = ""
        self._lock = threading.Lock()
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    # scope (one per trace/attempt)
    def set_scope(self, scope: str) -> None:
        """Set the sub-directory new artifacts are written to (per attempt)."""
        self._scope = _slug(scope) if scope else ""

    def clear_scope(self) -> None:
        self._scope = ""

    def _base_dir(self) -> Path:
        """Directory searched/listed by default: the scope dir, else the root."""
        if self._scope:
            directory = self.root / self._scope
            directory.mkdir(parents=True, exist_ok=True)
            return directory
        return self.root

    # writing
    def save(self, text: str, *, tool_name: str, extension: str = "md", args: dict | None = None) -> ArtifactRef:
        """Persist `text` under the current scope; identical content dedups."""
        if not self.enabled:
            raise ArtifactError("artifact store is disabled")
        data = text if isinstance(text, str) else str(text)
        if len(data.encode("utf-8", "replace")) > self.max_artifact_bytes:
            raise ArtifactError("artifact exceeds the maximum size")

        digest = hashlib.sha1(data.encode("utf-8", "replace")).hexdigest()[:10]
        target_dir = self._base_dir()
        filename = f"{_slug(tool_name)}_{digest}.{extension.lstrip('.')}"
        path = target_dir / filename
        rel_path = path.relative_to(self.root).as_posix()

        created = False
        with self._lock:
            if not path.exists():
                path.write_text(data, encoding="utf-8")
                created = True
                self._append_manifest(rel_path, tool_name, args, len(data), len(data.splitlines()))
        return ArtifactRef(path, rel_path, len(data), max(1, len(data.splitlines())), created)

    def _append_manifest(self, rel_path, tool_name, args, chars, lines) -> None:
        entry = {
            "time": datetime.now(UTC).isoformat(),
            "scope": self._scope,
            "path": rel_path,
            "tool": tool_name,
            "args": _jsonable_args(args),
            "chars": chars,
            "lines": lines,
        }
        try:
            with (self.root / self.manifest_name).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:  # never let bookkeeping break a run
            pass

    # path resolution (sandboxed)
    def resolve(self, path: str | Path) -> Path:
        """Resolve `path` to a file inside the artifact root, tolerant of form.

        Accepts an absolute path, a path relative to the root, a path that
        redundantly includes the root's own name, or a bare filename that is
        unique within the root.
        """
        raw = str(path).strip()
        if not raw:
            raise ArtifactError("empty path")

        candidates: list[Path] = []
        candidate = Path(raw)
        if candidate.is_absolute():
            candidates.append(candidate)
        else:
            candidates.append(self.root / raw)
            if raw.split("/", 1)[0] == self.root.name:
                candidates.append(self.root / raw.split("/", 1)[1])
            # Bare filename: look it up anywhere under the root.
            candidates.extend(sorted(self.root.rglob(candidate.name)))

        root = self.root.resolve()
        for item in candidates:
            try:
                resolved = item.resolve()
            except OSError:
                continue
            if resolved.is_file() and (resolved == root or root in resolved.parents):
                return resolved
        raise ArtifactError(f"could not find an artifact at '{raw}' under {self.root}")

    def _iter_files(self, base: Path | None = None) -> list[Path]:
        search_root = base if base is not None else self._base_dir()
        if not search_root.exists():
            return []
        return [
            item
            for item in sorted(search_root.rglob("*"))
            if item.is_file() and item.suffix.lower() in _ARTIFACT_EXTENSIONS
        ]

    # reading / searching
    def list_artifacts(self) -> list[dict]:
        """Metadata for every artifact under the current scope."""
        rows = []
        for item in self._iter_files():
            try:
                text = item.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rows.append(
                {
                    "path": item.relative_to(self.root).as_posix(),
                    "chars": len(text),
                    "lines": max(1, len(text.splitlines())),
                }
            )
        return rows

    def read(
        self,
        path,
        start_line: int = 1,
        end_line: int | None = None,
        max_chars: int = DEFAULT_READ_MAX_CHARS,
    ) -> dict:
        """Read a 1-based, inclusive line range from an artifact."""
        resolved = self.resolve(path)
        text = resolved.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        total = len(lines)

        start = max(1, int(start_line or 1))
        end = total if end_line is None else int(end_line)
        end = min(max(end, start), total) if total else 0
        chunk = "\n".join(lines[start - 1 : end]) if total else ""

        truncated = False
        if max_chars and len(chunk) > max_chars:
            chunk = chunk[:max_chars]
            truncated = True
        return {
            "rel_path": resolved.relative_to(self.root).as_posix(),
            "text": chunk,
            "start": start,
            "end": end,
            "total_lines": total,
            "truncated": truncated,
        }

    def search(
        self,
        pattern: str,
        *,
        path=None,
        regex: bool = False,
        ignore_case: bool = True,
        context_lines: int = DEFAULT_CONTEXT_LINES,
        max_matches: int = DEFAULT_MAX_MATCHES,
    ) -> dict:
        """grep/ctrl-F over one artifact, or over every artifact of the scope."""
        if not isinstance(pattern, str) or not pattern:
            raise ArtifactError("empty pattern")
        flags = re.IGNORECASE if ignore_case else 0
        try:
            compiled = re.compile(pattern if regex else re.escape(pattern), flags)
        except re.error as exc:
            raise ArtifactError(f"invalid regex: {exc}") from exc

        context = max(0, int(context_lines or 0))
        limit = max(1, min(int(max_matches or DEFAULT_MAX_MATCHES), MAX_MATCH_LIMIT))

        targets = [self.resolve(path)] if path is not None else self._iter_files()

        matches: list[dict] = []
        total = 0
        for file in targets:
            try:
                lines = file.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            rel = file.relative_to(self.root).as_posix()
            for index, line in enumerate(lines):
                if not compiled.search(line):
                    continue
                total += 1
                if len(matches) < limit:
                    matches.append(
                        {
                            "path": rel,
                            "line": index + 1,
                            "text": line,
                            "before": lines[max(0, index - context) : index],
                            "after": lines[index + 1 : index + 1 + context],
                        }
                    )
        return {
            "matches": matches,
            "total": total,
            "shown": len(matches),
            "truncated": total > len(matches),
        }


def _default_extension(tool_name: str) -> str:
    return "md" if tool_name in _MARKDOWN_TOOLS else "txt"


def _jsonable_args(args) -> dict:
    """Best-effort JSON-safe rendering of a tool's call arguments."""
    if not isinstance(args, dict):
        return {}
    out = {}
    for key, value in args.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)[:200]
    return out


def _plain_truncation(text: str, preview: str) -> str:
    """Fallback used when spilling is disabled or impossible."""
    total = f"{len(text):,}"
    shown = f"{len(preview):,}"
    lines = max(1, len(text.splitlines()))
    note = (
        f"\n\n⚠️ Output truncated: showing {shown} of {total} characters "
        f"({lines} lines total). The remainder is not available."
    )
    return preview + note


def _artifact_banner(text: str, ref: ArtifactRef, preview: str) -> str:
    """Preview + instructions telling the agent how to retrieve the full text."""
    total_c = f"{len(text):,}"
    shown_c = f"{len(preview):,}"
    total_l = f"{ref.lines:,}"
    rel = ref.rel_path
    note = (
        f"\n\n⚠️ Output truncated: showing {shown_c} of {total_c} characters "
        f"({total_l} lines total).\n"
        f"Full content saved to: {rel}\n"
        f"To search inside it (grep-like):\n"
        f'    search_artifact(path="{rel}", pattern="<term>")\n'
        f"To read a line range:\n"
        f'    read_artifact(path="{rel}", start_line=1, end_line=200)'
    )
    return preview + note


def cap_output(
    text,
    *,
    tool_name: str,
    store: ArtifactStore | None,
    cap_chars: int,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
    extension: str | None = None,
    args: dict | None = None,
):
    """Cap a tool output, spilling the full text to `store` when it is exceeded.

    Returns the text unchanged when it fits; otherwise a preview plus a banner
    pointing at the spilled file (or a plain-truncation note when spilling is
    unavailable).
    """
    if not isinstance(text, str) or not cap_chars or cap_chars <= 0:
        return text
    if len(text) <= cap_chars:
        return text

    preview_chars = max(0, min(int(preview_chars), int(cap_chars)))
    preview = text[:preview_chars]

    ref: ArtifactRef | None = None
    if store is not None and getattr(store, "enabled", False):
        try:
            ref = store.save(
                text,
                tool_name=tool_name,
                extension=extension or _default_extension(tool_name),
                args=args,
            )
        except Exception as exc:  # noqa: BLE001 - fall back to plain truncation
            logger.warning("Artifact spill failed for %s: %s", tool_name, exc)
            ref = None

    if ref is None:
        return _plain_truncation(text, preview)
    return _artifact_banner(text, ref, preview)


def _unwrap(tool):
    """Return the tool whose `forward` produces the observation.

    Harness tools are :class:`tools.base.WrappedTool` instances that delegate to
    an inner smolagents tool; capping the *inner* tool means the event watcher
    records the capped observation (keeping traces small) while the full text
    lives on disk.
    """
    return getattr(tool, "_wrapped", tool)


def install_output_cap(
    tools,
    store: ArtifactStore | None,
    *,
    default_cap: int = DEFAULT_TOOL_OUTPUT_MAX_CHARS,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
    overrides: dict | None = None,
):
    """Wrap every tool's `forward` so its output is capped and spilled.

    `overrides` maps a tool name to a custom cap.  Idempotent: a tool that was
    already wrapped is left untouched.  Tools with a non-positive cap are
    skipped (their output is passed through unchanged).
    """
    overrides = overrides or {}
    for tool in tools:
        target = _unwrap(tool)
        name = getattr(target, "name", "")
        if getattr(target, _CAP_MARKER, None) is not None:
            continue
        cap = overrides.get(name, default_cap)
        if not cap or cap <= 0:
            continue

        original = target.forward

        def _make_wrapper(original, name, cap):
            @functools.wraps(original)
            def _forward(*args, **kwargs):
                result = original(*args, **kwargs)
                call_args = {}
                if args:
                    call_args["args"] = args
                call_args.update(kwargs)
                return cap_output(
                    result,
                    tool_name=name,
                    store=store,
                    cap_chars=cap,
                    preview_chars=preview_chars,
                    args=call_args,
                )

            return _forward

        target.forward = _make_wrapper(original, name, cap)
        setattr(target, _CAP_MARKER, cap)
    return tools
