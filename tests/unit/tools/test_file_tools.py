from pathlib import Path

import pytest

from tools.filetools import ReadFileTool


@pytest.fixture
def tool():
    return ReadFileTool()


def test_csv_is_rendered_as_an_aligned_table(tmp_path, tool):
    path = tmp_path / "measurements.csv"
    path.write_text("sample_id,temperature,value\nS1,20.1,12.0\nS2,20.4,15.5\n")

    out = tool.forward(str(path))

    assert "measurements.csv" in out
    assert "3 column(s) x 2 row(s)" in out
    assert "delimiter ','" in out
    assert "sample_id" in out and "temperature" in out and "value" in out
    assert "S1" in out and "15.5" in out


def test_tsv_uses_a_tab_delimiter(tmp_path, tool):
    path = tmp_path / "data.tsv"
    path.write_text("a\tb\n1\t2\n")

    out = tool.forward(str(path))

    assert "delimiter '\\t'" in out
    assert "2 column(s) x 1 row(s)" in out


def test_semicolon_delimiter_is_sniffed(tmp_path, tool):
    path = tmp_path / "euro.csv"
    path.write_text("a;b;c\n1;2;3\n4;5;6\n")

    out = tool.forward(str(path))

    assert "delimiter ';'" in out
    assert "3 column(s) x 2 row(s)" in out


def test_explicit_delimiter_overrides_sniffing(tmp_path, tool):
    path = tmp_path / "odd.csv"
    path.write_text("a|b\n1|2\n")

    out = tool.forward(str(path), delimiter="|")

    assert "delimiter '|'" in out


def test_tab_escape_is_accepted_as_a_delimiter(tmp_path, tool):
    path = tmp_path / "odd.csv"
    path.write_text("a\tb\n1\t2\n")

    out = tool.forward(str(path), delimiter="\\t")

    assert "delimiter '\\t'" in out


def test_max_rows_limits_a_table(tmp_path, tool):
    rows = "\n".join(f"r{i},{i}" for i in range(10))
    path = tmp_path / "many.csv"
    path.write_text("name,value\n" + rows + "\n")

    out = tool.forward(str(path), max_rows=3)

    assert "Showing 1-3 of 10 data row(s)" in out
    assert "remaining 7" in out
    assert "r9" not in out


def test_json_is_pretty_printed_with_a_shape_summary(tmp_path, tool):
    path = tmp_path / "config.json"
    path.write_text('{"b": 1, "a": [1, 2]}')

    out = tool.forward(str(path))

    assert "JSON object with 2 key(s)" in out
    assert '"a": [' in out


def test_json_array_shape_is_described(tmp_path, tool):
    path = tmp_path / "list.json"
    path.write_text('[{"x": 1}, {"x": 2}]')

    out = tool.forward(str(path))

    assert "array with 2 item(s)" in out
    assert "first item: object with 1 key(s)" in out


def test_invalid_json_falls_back_to_raw_contents(tmp_path, tool):
    path = tmp_path / "broken.json"
    path.write_text("{nope}")

    out = tool.forward(str(path))

    assert "not valid JSON" in out
    assert "{nope}" in out


def test_jsonl_is_summarised_per_record(tmp_path, tool):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}\n')

    out = tool.forward(str(path))

    assert "JSON Lines with 2 record(s)" in out
    assert '1. {"a": 1}' in out
    assert '2. {"a": 2}' in out


def test_jsonl_marks_unparsable_lines(tmp_path, tool):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\nnot json\n')

    out = tool.forward(str(path))

    assert "not valid JSON" in out


def test_plain_text_is_returned_with_line_counts(tmp_path, tool):
    path = tmp_path / "notes.md"
    path.write_text("# Title\n\nbody\n")

    out = tool.forward(str(path))

    assert "plain text, 3 line(s)" in out
    assert "# Title" in out


def test_directory_is_listed_directories_first(tmp_path, tool):
    (tmp_path / "sub").mkdir()
    (tmp_path / "f.txt").write_text("hi")

    out = tool.forward(str(tmp_path))

    assert "2 entry(ies)" in out
    assert "d  sub/" in out
    assert "f  f.txt  (2 bytes)" in out
    assert out.index("sub/") < out.index("f.txt")


def test_missing_file_reports_an_error(tmp_path, tool):
    out = tool.forward(str(tmp_path / "nope.csv"))

    assert "no such file or directory" in out


def test_binary_files_are_refused(tmp_path, tool):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"\x00\x01\x02binary")

    out = tool.forward(str(path))

    assert "binary data" in out


def test_non_utf8_text_is_decoded_with_a_note(tmp_path, tool):
    path = tmp_path / "latin.txt"
    path.write_bytes("café".encode("latin-1"))

    out = tool.forward(str(path))

    assert "decoded as latin-1" in out
    assert "café" in out


def test_max_chars_truncates_the_output(tmp_path, tool):
    path = tmp_path / "big.txt"
    path.write_text("x" * 500)

    out = tool.forward(str(path), max_chars=100)

    assert "truncated:" in out
    assert len(out) < 500


def test_unexpected_input_is_returned_as_text_not_raised(tool):
    # a model can always pass something silly (a null byte path, None, ...); the tool
    # must degrade to an error string rather than blowing up the agent step
    assert tool.forward(None).startswith("Error reading")  # type: ignore[arg-type]
    assert tool.forward("\0invalid").startswith("Error")


def test_reads_the_shipped_example_asset():
    root = Path(__file__).resolve().parents[3]
    assets = sorted((root / "data" / "tasks" / "assets").glob("*/measurements.csv"))
    assert assets, "expected the bundled example CSV asset to exist"

    out = ReadFileTool().forward(str(assets[0]))

    assert "value" in out
    assert "18.0" in out


def test_tool_is_registered_and_opt_in():
    from agents.definitions import DEFAULT_CODING_AGENT, DEFAULT_TOOL_AGENT
    from tools.definitions import CLOSED_BOOK_ALLOWED_TOOLS, tool_registry

    assert tool_registry.get("read_file") is not None

    # closed-book keeps no harness tools at all, so `read_file` is unavailable there
    assert "read_file" not in CLOSED_BOOK_ALLOWED_TOOLS

    # opt-in: it is not wired into the default agent definitions
    assert all(tool.tool_name != "read_file" for tool in DEFAULT_CODING_AGENT.tools)
    assert all(tool.tool_name != "read_file" for tool in DEFAULT_TOOL_AGENT.tools)
