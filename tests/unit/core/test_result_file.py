import json
import multiprocessing as mp

from core.tasks import append_result_line

WRITERS, LINES_EACH, RECORD_CHARS = 4, 25, 100_000


def _write_records(path, writer):
    for repetition in range(LINES_EACH):
        append_result_line(path, {"writer": writer, "repetition": repetition, "trace": "x" * RECORD_CHARS})


def test_parallel_appends_keep_every_line_whole(tmp_path):
    # repetitions of a task can finish at the same moment and append large records to the same file
    path = tmp_path / "001.jsonl"
    context = mp.get_context("fork")
    writers = [context.Process(target=_write_records, args=(path, writer)) for writer in range(WRITERS)]
    for process in writers:
        process.start()
    for process in writers:
        process.join(timeout=60)
        assert process.exitcode == 0

    lines = path.read_text().splitlines()
    assert len(lines) == WRITERS * LINES_EACH
    records = [json.loads(line) for line in lines]  # an interleaved line fails to parse
    assert sorted((r["writer"], r["repetition"]) for r in records) == [
        (writer, repetition) for writer in range(WRITERS) for repetition in range(LINES_EACH)
    ]
