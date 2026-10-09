from __future__ import annotations

from smolagents import Tool

from tools.artifacts import (
    DEFAULT_CONTEXT_LINES,
    DEFAULT_MAX_MATCHES,
    DEFAULT_READ_MAX_CHARS,
    ArtifactStore,
)


def _format_context(number_width: int, line_no: int, text: str, marker: str = " ") -> str:
    return f"{marker} {line_no:>{number_width}}| {text}"


class SearchArtifactTool(Tool):
    name = "search_artifact"
    description = (
        "Searches the text of an artifact — a tool output that was too large and was saved "
        "to a file (the truncation banner gives its path). Use it like grep/ctrl-F to locate "
        "a fact: pass a `pattern` (plain text by default, or a regular expression with "
        "regex=true) and it returns the matching lines with their line numbers and a little "
        "surrounding context. Omit `path` to search every artifact saved for the current task."
    )
    inputs = {
        "pattern": {
            "type": "string",
            "description": "Text to find (literal, or a regex when regex=true).",
        },
        "path": {
            "type": "string",
            "description": (
                "Artifact path from the truncation banner; omit to search all artifacts of the current task."
            ),
            "nullable": True,
        },
        "regex": {
            "type": "boolean",
            "description": "Treat `pattern` as a regular expression (default false).",
            "nullable": True,
        },
        "ignore_case": {
            "type": "boolean",
            "description": "Case-insensitive matching (default true).",
            "nullable": True,
        },
        "context_lines": {
            "type": "integer",
            "description": "Lines of context shown around each match (default 2).",
            "nullable": True,
        },
        "max_matches": {
            "type": "integer",
            "description": "Maximum number of matches to return (default 50).",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, store: ArtifactStore | None = None):
        super().__init__()
        self.store = store

    def forward(
        self,
        pattern: str,
        path: str | None = None,
        regex: bool = False,
        ignore_case: bool = True,
        context_lines: int = DEFAULT_CONTEXT_LINES,
        max_matches: int = DEFAULT_MAX_MATCHES,
    ) -> str:
        if self.store is None:
            return "No artifacts are available."
        try:
            result = self.store.search(
                pattern,
                path=path,
                regex=bool(regex),
                ignore_case=bool(ignore_case),
                context_lines=context_lines,
                max_matches=max_matches,
            )
        except Exception as exc:  # noqa: BLE001
            return f"Artifact search failed ({type(exc).__name__}: {exc})."

        matches = result["matches"]
        if not matches:
            where = f" in {path}" if path else ""
            return (
                f"No occurrences of '{pattern}'{where}. Try another term, or use "
                "`read_artifact` to read the file by line range."
            )

        header = f"## {result['shown']} of {result['total']} occurrence(s) of '{pattern}'"
        blocks = []
        for match in matches:
            block_lines = [f"**{match['path']}**"]
            before = match["before"]
            start_line = match["line"] - len(before)
            width = len(str(match["line"] + len(match["after"])))
            for offset, text in enumerate(before):
                block_lines.append(_format_context(width, start_line + offset, text))
            block_lines.append(_format_context(width, match["line"], match["text"], marker=">"))
            for offset, text in enumerate(match["after"], start=1):
                block_lines.append(_format_context(width, match["line"] + offset, text))
            blocks.append("\n".join(block_lines))
        return header + "\n\n" + "\n\n".join(blocks)


class ReadArtifactTool(Tool):
    name = "read_artifact"
    description = (
        "Reads a range of lines from an artifact — a tool output that was too large and was "
        "saved to a file (the truncation banner gives its path). Use it to page through the "
        "content or to read a section you located with `search_artifact`. Lines are 1-based "
        "and the range is inclusive; the output is capped, so read in chunks for very long files."
    )
    inputs = {
        "path": {
            "type": "string",
            "description": "Artifact path from the truncation banner.",
        },
        "start_line": {
            "type": "integer",
            "description": "First line to read (1-based, default 1).",
            "nullable": True,
        },
        "end_line": {
            "type": "integer",
            "description": "Last line to read, inclusive (default: end of file).",
            "nullable": True,
        },
        "max_chars": {
            "type": "integer",
            "description": "Maximum characters returned (default 20000).",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, store: ArtifactStore | None = None):
        super().__init__()
        self.store = store

    def forward(
        self, path: str, start_line: int = 1, end_line: int | None = None, max_chars: int = DEFAULT_READ_MAX_CHARS
    ) -> str:
        if self.store is None:
            return "No artifacts are available."
        try:
            result = self.store.read(path, start_line=start_line, end_line=end_line, max_chars=max_chars)
        except Exception as exc:  # noqa: BLE001
            return f"Could not read the artifact ({type(exc).__name__}: {exc})."

        header = f"## {result['rel_path']} (lines {result['start']}-{result['end']} of {result['total_lines']})"
        body = result["text"] or "(empty range)"
        footer = ""
        if result["truncated"]:
            footer = "\n\n⚠️ Range truncated; read a smaller interval (use `end_line`)."
        return f"{header}\n\n{body}{footer}"


class ListArtifactsTool(Tool):
    name = "list_artifacts"
    description = (
        "Lists the artifacts (saved tool outputs) available for the current task, with their "
        "size in characters and lines. Use it when you are not sure which files exist before "
        "calling `search_artifact` or `read_artifact`."
    )
    inputs = {}
    output_type = "string"

    def __init__(self, store: ArtifactStore | None = None):
        super().__init__()
        self.store = store

    def forward(self) -> str:
        if self.store is None:
            return "No artifacts are available."
        try:
            rows = self.store.list_artifacts()
        except Exception as exc:  # noqa: BLE001
            return f"Could not list the artifacts ({type(exc).__name__}: {exc})."
        if not rows:
            return "No artifacts saved for this task."
        lines = ["## Artifacts"]
        for row in rows:
            lines.append(f"- **{row['path']}** — {row['chars']} characters, {row['lines']} lines")
        return "\n".join(lines)
