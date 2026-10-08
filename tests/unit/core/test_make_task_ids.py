import json

from core.task_ids import TASK_ID_PLACEHOLDER, authored_template, hash_prompt, recompute_task_ids, render_task

STALE = "0" * 32


def _line(task_id: str, task: str, **extra) -> str:
    return json.dumps({"task_id": task_id, "task": task, **extra})


def test_hash_render_and_authored_roundtrip():
    template = f"read `data/tasks/assets/{TASK_ID_PLACEHOLDER}/x.csv`"
    tid = hash_prompt(template)

    rendered = render_task(template, tid)

    assert rendered == f"read `data/tasks/assets/{tid}/x.csv`"
    assert authored_template(rendered, tid) == template


def test_recompute_rewrites_id_and_keeps_placeholder(tmp_path):
    template = f"read `data/tasks/assets/{TASK_ID_PLACEHOLDER}/x.csv`"
    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(_line(STALE, template) + "\n")

    report = recompute_task_ids(task_file)

    assert report.changed
    assert report.applied
    obj = json.loads(task_file.read_text().strip())
    assert obj["task_id"] == hash_prompt(template)
    assert obj["task"] == template


def test_recompute_migrates_prompt_with_substituted_id(tmp_path):
    template = f"read `data/tasks/assets/{TASK_ID_PLACEHOLDER}/x.csv`"
    tid = hash_prompt(template)
    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(_line(tid, render_task(template, tid)) + "\n")

    report = recompute_task_ids(task_file)

    assert not report.changed
    obj = json.loads(task_file.read_text().strip())
    assert obj["task"] == template
    assert obj["task_id"] == tid


def test_placeholder_free_task_just_hashes_its_text(tmp_path):
    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(_line(STALE, "say hi") + "\n")

    recompute_task_ids(task_file)

    assert json.loads(task_file.read_text().strip())["task_id"] == hash_prompt("say hi")


def test_check_mode_reports_without_writing(tmp_path):
    task_file = tmp_path / "tasks.jsonl"
    original = _line(STALE, "say hi") + "\n"
    task_file.write_text(original)

    report = recompute_task_ids(task_file, apply=False)

    assert report.changed
    assert not report.applied
    assert task_file.read_text() == original


def test_assets_folder_is_renamed(tmp_path):
    template = f"read `data/tasks/assets/{TASK_ID_PLACEHOLDER}/x.csv`"
    new_id = hash_prompt(template)
    assets_dir = tmp_path / "assets"
    (assets_dir / STALE).mkdir(parents=True)
    (assets_dir / STALE / "x.csv").write_text("a,b\n")
    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(_line(STALE, template) + "\n")

    report = recompute_task_ids(task_file, assets_dir)

    assert report.changes[0].renamed is not None
    assert (assets_dir / new_id / "x.csv").is_file()
    assert not (assets_dir / STALE).exists()


def test_invalid_json_is_reported_and_nothing_written(tmp_path):
    task_file = tmp_path / "tasks.jsonl"
    original = "{not json}\n"
    task_file.write_text(original)

    report = recompute_task_ids(task_file)

    assert report.errors
    assert not report.applied
    assert task_file.read_text() == original


def test_rename_collision_aborts_without_writing(tmp_path):
    template = f"read `data/tasks/assets/{TASK_ID_PLACEHOLDER}/x.csv`"
    new_id = hash_prompt(template)
    assets_dir = tmp_path / "assets"
    (assets_dir / STALE).mkdir(parents=True)
    (assets_dir / new_id).mkdir(parents=True)  # destination already taken
    task_file = tmp_path / "tasks.jsonl"
    original = _line(STALE, template) + "\n"
    task_file.write_text(original)

    report = recompute_task_ids(task_file, assets_dir)

    assert report.errors
    assert not report.applied
    assert task_file.read_text() == original
    assert (assets_dir / STALE).is_dir()
