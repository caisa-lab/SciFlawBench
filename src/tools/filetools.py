import csv
import io
import json
from pathlib import Path

from smolagents import Tool

# Refuse to read anything larger than this, so a stray path cannot blow up a run.
MAX_FILE_BYTES = 8 * 1024 * 1024

# How much of a file to inspect when sniffing a delimiter or binary content.
SNIFF_BYTES = 8192

DEFAULT_MAX_ROWS = 50
DEFAULT_MAX_CHARS = 20_000

# Longest single table cell before it gets elided.
MAX_CELL_WIDTH = 60

# Suffix -> delimiter for the formats we can render as a table.
TABLE_SUFFIX_DELIMITERS = {".csv": ",", ".tsv": "\t", ".tab": "\t", ".psv": "|"}

# Suffixes whose delimiter is fixed by the extension. `.csv` is deliberately absent:
# CSVs written by European tooling are often ';' separated, so the delimiter is sniffed.
FIXED_DELIMITER_SUFFIXES = {".tsv": "\t", ".tab": "\t", ".psv": "|"}

JSONL_SUFFIXES = {".jsonl", ".ndjson"}

# Escapes the model may pass through as a literal two-character sequence.
_DELIMITER_ESCAPES = {"\\t": "\t", "\\n": "\n", "\\r": "\r", "tab": "\t", "comma": ",", "pipe": "|"}


def looks_binary(data: bytes) -> bool:
    """Cheap binary sniff: NUL bytes, or a low share of printable characters."""
    if b"\x00" in data:
        return True
    sample = data[:SNIFF_BYTES]
    if not sample:
        return False
    printable = sum(byte in b"\t\n\r\f\b" or 32 <= byte < 127 or byte > 127 for byte in sample)
    return printable / len(sample) < 0.9


def decode_text(data: bytes) -> tuple[str, str]:
    """Decode bytes to text, returning `(text, encoding_used)`."""
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return data.decode("latin-1"), "latin-1"


def describe_json(data: object) -> str:
    """A one-line shape summary for a parsed JSON value."""
    if isinstance(data, dict):
        keys = ", ".join(str(key) for key in list(data)[:8])
        more = "" if len(data) <= 8 else f", ... (+{len(data) - 8} more)"
        return f"object with {len(data)} key(s): {keys}{more}"
    if isinstance(data, list):
        detail = f"; first item: {describe_json(data[0])}" if data else ""
        return f"array with {len(data)} item(s){detail}"
    return type(data).__name__


class ReadFileTool(Tool):
    name = "read_file"

    description = """Reads a local file and returns its contents formatted as text.
    CSV/TSV/PSV files are rendered as aligned tables, JSON is pretty-printed, JSON Lines are
    summarised per record, and every other text file (txt, md, yaml, xml, source code, ...)
    is returned as-is. Directories are listed so you can find files. Use this whenever a task
    points at a local data file. Large files are truncated: raise max_rows/max_chars to see
    more."""

    inputs = {
        "path": {
            "type": "string",
            "description": "Path to the file (or directory) to read, absolute or relative to the working directory.",
        },
        "max_rows": {
            "type": "integer",
            "description": (
                f"Maximum number of table rows / JSON Lines / records to show. Defaults to {DEFAULT_MAX_ROWS}."
            ),
            "nullable": True,
        },
        "max_chars": {
            "type": "integer",
            "description": f"Maximum number of characters to return. Defaults to {DEFAULT_MAX_CHARS}.",
            "nullable": True,
        },
        "delimiter": {
            "type": "string",
            "description": (
                "Override the delimiter for delimited files, e.g. ',', '\\t' or ';'. Auto-detected by default."
            ),
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, max_rows: int = DEFAULT_MAX_ROWS, max_chars: int = DEFAULT_MAX_CHARS):
        super().__init__()
        self.max_rows = max_rows
        self.max_chars = max_chars

    def forward(
        self,
        path: str,
        max_rows: int | None = None,
        max_chars: int | None = None,
        delimiter: str | None = None,
    ) -> str:
        max_rows = max_rows or self.max_rows
        max_chars = max_chars or self.max_chars

        try:
            return self._read(path, max_rows, max_chars, delimiter)
        except Exception as exc:  # never let a bad path/format kill the agent step
            return f"Error reading {path!r}: {type(exc).__name__}: {exc}"

    def _read(self, path: str, max_rows: int, max_chars: int, delimiter: str | None) -> str:
        target = Path(path).expanduser()
        if not target.exists():
            return f"Error: no such file or directory: {path}"
        if target.is_dir():
            return self._list_directory(target)

        size = target.stat().st_size
        if size > MAX_FILE_BYTES:
            return f"Error: {target.name} is {size} bytes, which exceeds the {MAX_FILE_BYTES} byte limit for this tool."

        data = target.read_bytes()
        if looks_binary(data):
            return f"Error: {target.name} looks like binary data ({size} bytes); only text files can be read."

        text, encoding = decode_text(data)
        note = "" if encoding == "utf-8" else f", decoded as {encoding}"

        suffix = target.suffix.lower()
        if suffix in TABLE_SUFFIX_DELIMITERS:
            rendered = self._format_table(text, target.name, suffix, max_rows, delimiter, note)
        elif suffix in JSONL_SUFFIXES:
            rendered = self._format_jsonl(text, target.name, max_rows, note)
        elif suffix == ".json":
            rendered = self._format_json(text, target.name, max_chars, note)
        else:
            rendered = self._format_text(text, target.name, note)

        return self._truncate(rendered, max_chars)

    def _list_directory(self, directory: Path) -> str:
        entries = sorted(directory.iterdir(), key=lambda entry: (entry.is_file(), entry.name.lower()))
        if not entries:
            return f"{directory} is an empty directory."

        lines = [f"{directory}: {len(entries)} entry(ies)"]
        for entry in entries[: self.max_rows]:
            if entry.is_dir():
                lines.append(f"  d  {entry.name}/")
            else:
                lines.append(f"  f  {entry.name}  ({entry.stat().st_size} bytes)")
        if len(entries) > self.max_rows:
            lines.append(f"  ... {len(entries) - self.max_rows} more entrie(s) not shown")
        return "\n".join(lines)

    def _format_table(self, text: str, name: str, suffix: str, max_rows: int, delimiter: str | None, note: str) -> str:
        delim = _normalize_delimiter(delimiter) if delimiter else self._sniff_delimiter(text, suffix)
        rows = [row for row in csv.reader(io.StringIO(text), delimiter=delim) if any(cell.strip() for cell in row)]
        if not rows:
            return f"{name}: empty delimited file{note}."

        header, body = rows[0], rows[1:]
        shown = body[:max_rows]
        ncols = max(len(row) for row in rows)

        table = [[*header, *[""] * (ncols - len(header))]]
        table += [[*row, *[""] * (ncols - len(row))] for row in shown]
        table = [[_elide(str(cell)) for cell in row] for row in table]
        widths = [max(len(cell) for cell in column) for column in zip(*table, strict=False)]

        rendered = [" | ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=False)) for row in table]
        rendered.insert(1, "-+-".join("-" * width for width in widths))

        summary = (
            f"{name}: {ncols} column(s) x {len(body)} row(s), delimiter {delim!r}{note}.\n"
            f"The first row is treated as the header. Showing 1-{len(shown)} of {len(body)} data row(s)."
        )
        if len(shown) < len(body):
            summary += f" Raise max_rows to see the remaining {len(body) - len(shown)}."
        return f"{summary}\n\n" + "\n".join(rendered)

    def _format_json(self, text: str, name: str, max_chars: int, note: str) -> str:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            return f"{name}: not valid JSON ({exc}){note}. Raw contents:\n\n{self._truncate(text, max_chars)}"
        pretty = json.dumps(data, indent=2, ensure_ascii=False)
        return f"{name}: JSON {describe_json(data)}{note}.\n\n{pretty}"

    def _format_jsonl(self, text: str, name: str, max_rows: int, note: str) -> str:
        lines = [line for line in text.splitlines() if line.strip()]
        shown = lines[:max_rows]

        rendered = []
        for index, line in enumerate(shown, start=1):
            try:
                rendered.append(f"{index}. {json.dumps(json.loads(line), ensure_ascii=False)}")
            except json.JSONDecodeError:
                rendered.append(f"{index}. {line}   (not valid JSON)")

        summary = f"{name}: JSON Lines with {len(lines)} record(s){note}. Showing 1-{len(shown)}."
        if len(shown) < len(lines):
            summary += f" Raise max_rows to see the remaining {len(lines) - len(shown)}."
        return f"{summary}\n\n" + "\n".join(rendered)

    def _format_text(self, text: str, name: str, note: str) -> str:
        return f"{name}: plain text, {len(text.splitlines())} line(s), {len(text)} character(s){note}.\n\n{text}"

    @staticmethod
    def _sniff_delimiter(text: str, suffix: str) -> str:
        if suffix in FIXED_DELIMITER_SUFFIXES:
            return FIXED_DELIMITER_SUFFIXES[suffix]

        # Prefer the delimiter that appears a consistent, non-zero number of times on
        # every line: csv.Sniffer is easily fooled by small samples (it picks ',' over
        # ';' for a three-line semicolon file).
        lines = [line for line in text[:SNIFF_BYTES].splitlines() if line.strip()][:20]
        best_delimiter, best_score = ",", (0, 0)
        for candidate in (",", ";", "\t", "|"):
            counts = [line.count(candidate) for line in lines]
            if not counts or min(counts) == 0:
                continue
            score = (min(counts), sum(counts))
            if score > best_score:
                best_delimiter, best_score = candidate, score
        return best_delimiter

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        remaining = len(text) - max_chars
        return f"{text[:max_chars]}\n\n... [truncated: {remaining} more character(s); raise max_chars to see more]"


def _elide(cell: str) -> str:
    """Shorten a single cell so one long value cannot wreck the table layout."""
    cell = cell.replace("\n", " ").replace("\r", " ")
    if len(cell) <= MAX_CELL_WIDTH:
        return cell
    return cell[: MAX_CELL_WIDTH - 1] + "…"


def _normalize_delimiter(delimiter: str) -> str:
    """Accept real delimiters as well as the escapes a model is likely to emit."""
    return _DELIMITER_ESCAPES.get(delimiter.strip().lower(), delimiter)
