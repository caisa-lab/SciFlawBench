import json

import pytest

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


def test_fixed_id_keeps_its_id_and_folder_and_gets_a_new_prompt_md5(tmp_path):
    template = f"read `data/tasks/assets/{TASK_ID_PLACEHOLDER}/x.csv` again"
    assets_dir = tmp_path / "assets"
    (assets_dir / "SFB-0042").mkdir(parents=True)
    (assets_dir / "SFB-0042" / "x.csv").write_text("a,b\n")
    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(_line("SFB-0042", template, prompt_md5=STALE) + "\n")

    report = recompute_task_ids(task_file, assets_dir)

    assert [(c.field, c.old_id, c.new_id, c.renamed) for c in report.changes] == [
        ("prompt_md5", STALE, hash_prompt(template), None)
    ]
    obj = json.loads(task_file.read_text().strip())
    assert obj["task_id"] == "SFB-0042"
    assert obj["prompt_md5"] == hash_prompt(template)
    assert (assets_dir / "SFB-0042" / "x.csv").is_file()


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


@pytest.mark.parametrize("stale_id", ["../../../.git", "..", "sub/folder", "/an/absolute/path", "back\\slash"])
def test_rename_refuses_an_id_that_is_not_a_plain_folder_name(tmp_path, stale_id):
    """A stale id from the task file must never rename a folder outside the assets folder."""
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    assets_dir = repo / "data" / "tasks" / "assets"
    (assets_dir / "sub" / "folder").mkdir(parents=True)
    task_file = repo / "tasks.jsonl"
    original = _line(stale_id, "a task that reads no files") + "\n"
    task_file.write_text(original)

    report = recompute_task_ids(task_file, assets_dir)

    assert any("not a plain folder name" in error for error in report.errors)
    assert not report.applied
    assert task_file.read_text() == original
    assert (repo / ".git" / "HEAD").is_file()
    assert (assets_dir / "sub" / "folder").is_dir()
    assert sorted(p.name for p in assets_dir.iterdir()) == ["sub"]


def test_an_id_that_is_not_a_folder_name_is_fine_without_assets(tmp_path):
    """With --skip-assets nothing is renamed, so such an id is simply replaced by its md5."""
    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(_line("../odd", "a task that reads no files") + "\n")

    report = recompute_task_ids(task_file)

    assert report.applied and not report.errors
    assert report.changes[0].new_id == hash_prompt("a task that reads no files")
