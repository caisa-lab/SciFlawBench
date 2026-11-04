import csv
import json
import os
import re
import subprocess
import sys

import pytest

import utils.build_task as build_task_module
from core.task_ids import hash_prompt
from core.tasks import TaskDef
from utils.build_task import FAILURE_MODES, REPO_ROOT, VALIDATORS, read_tasks
from utils.import_form import (
    ASSETS_DIR,
    CHECK_BOXES,
    MAX_CHECKS,
    MAX_SOURCES,
    MAX_TASK_BYTES,
    REJECTED_FILE,
    SCOPES,
    SOURCE_BOXES,
    TASK_FILE,
    VALIDATOR_BOXES,
    check_files,
    check_kwargs,
    clean_file_name,
    contains_personal_detail,
    drive_fetch,
    fetch_files,
    import_rows,
    local_source,
    main,
    name_parts,
    new_task_lines,
    parse_rows,
    personal_terms,
    prepare_branch,
    rejected_ids,
    required_columns,
    row_to_task,
    staged_problems,
)

SCRIPT = REPO_ROOT / "utils" / "form_script.gs"
PROMPT_WITH_FILE = "Read data/tasks/assets/{TASK_ID}/data.csv and give the mean."
PERSON = {
    "Full name": "Jane Roe",
    "Email": "jane.roe@example.org",
    "Institution": "CERN",
    "ORCID": "0000-0002-1825-0097",
    "OpenReview": "https://openreview.net/profile?id=~Jane_Roe1",
}
OTHER_PERSON = {
    "Full name": "Müller Şahin",
    "Email": "ms@example.org",
    "Institution": "DESY",
    "ORCID": "0000-0001-5109-3700",
    "OpenReview": "none",
}


def _row(**overrides):
    """A valid submission with one numeric check, with any column replaced."""
    row = {
        **PERSON,
        "Task ID": "SFB-0052",
        "Task description": "What is 1 + 1?",
        "Check 1: failure mode": "correctness",
        "Check 1: validator": "content:numeric_match",
        "Check 1: correct answer": "2",
    }
    row.update(overrides)
    return row


def _row_with_file(**overrides):
    return _row(
        **{"Task description": PROMPT_WITH_FILE, "Files": "https://drive.google.com/open?id=abc-1", **overrides}
    )


def _fetch_as(*uploads):
    return lambda row: list(uploads)


_fetch = _fetch_as(("data - jane roe.csv", b"a,b\n1,2\n"))


# The form and the importer agree.


def _script_list(name):
    match = re.search(rf"const {name} = (\[.*?\]);", SCRIPT.read_text(encoding="utf-8"), re.DOTALL)
    assert match
    return json.loads(match.group(1))


def test_the_script_offers_what_the_harness_has():
    assert sorted(_script_list("FAILURE_MODES")) == sorted(FAILURE_MODES)
    assert sorted(_script_list("VALIDATORS")) == sorted(VALIDATOR_BOXES)


def test_the_script_asks_what_the_importer_reads():
    source = SCRIPT.read_text(encoding="utf-8")

    for box in CHECK_BOXES:
        assert f"`Check ${{n}}: {box}`" in source
    for box in SOURCE_BOXES:
        assert f"`Source ${{n}}: {box}`" in source
    for title in ("Full name", "Email", "Institution", "ORCID", "OpenReview", "Task description"):
        assert f'"{title}"' in source
    assert '"Task ID"' in source
    assert f"const MAX_CHECKS = {MAX_CHECKS};" in source
    assert f"const MAX_SOURCES = {MAX_SOURCES};" in source


def test_every_harness_validator_can_be_built():
    # other tests can register fake verifiers, which never carry the harness's prefixes
    harness = {name for name in VALIDATORS if name.split(":")[0] in {"content", "format", "judge"}}

    assert set(VALIDATOR_BOXES) == harness


def test_the_scopes_are_read_only_and_stated_in_the_docstring():
    source = (REPO_ROOT / "utils" / "import_form.py").read_text(encoding="utf-8")

    assert SCOPES == [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    assert "`spreadsheets.readonly` and `drive.readonly`" in source


# Reading the sheet.


def test_parse_rows_fills_short_rows_by_title():
    rows = parse_rows([required_columns(), [PERSON["Full name"]]])

    assert rows[0]["Full name"] == "Jane Roe"
    assert "Email" not in rows[0]


def test_parse_rows_stops_on_a_renamed_question():
    with pytest.raises(SystemExit, match="Check 1: rubric"):
        parse_rows([[column for column in required_columns() if column != "Check 1: rubric"]])


def test_parse_rows_stops_on_an_empty_sheet():
    with pytest.raises(SystemExit, match="empty"):
        parse_rows([])


# The answer boxes of a check.


@pytest.mark.parametrize(
    ("validator", "boxes", "expected"),
    [
        ("judge:rubric", ("", "", "Must say X"), {"rubric": "Must say X"}),
        ("content:numeric_match", ("18.40", "0.05", ""), {"expected": "18.40", "tol": 0.05}),
        ("content:numeric_match", ("18.4", "", ""), {"expected": "18.4", "tol": 0.0}),
        ("content:numeric_within_range", ("62, 65.5", "", ""), {"minimum": 62.0, "maximum": 65.5}),
        ("content:scientific_notation", ("6.022e23", "0.02", ""), {"expected": "6.022e23", "rel_tol": 0.02}),
        ("content:scientific_notation", ("6.022e23", "", ""), {"expected": "6.022e23"}),
        ("content:json_output", ('{"a": 1}', "", ""), {"expected": {"a": 1}}),
        (
            "content:paper_json_match",
            ("2602.02275, arXiv:hep-th/9901001", "", ""),
            {"expected": [{"arxiv_id": "2602.02275"}, {"arxiv_id": "hep-th/9901001"}]},
        ),
        ("content:exact_str_match", ("yes", "", ""), {"expected": "yes"}),
        ("content:contains_str", ("yes", "", ""), {"expected": "yes"}),
        ("format:json_output", ("", "", ""), {}),
    ],
)
def test_check_kwargs_builds_what_the_verifier_takes(validator, boxes, expected):
    assert check_kwargs(validator, *boxes) == expected


@pytest.mark.parametrize(
    ("validator", "boxes", "message"),
    [
        ("content:nope", ("1", "", ""), "unknown validator"),
        ("content:numeric_match", ("about 3", "", ""), "not a number"),
        ("content:numeric_match", ("1,5", "", ""), "not a number"),
        ("content:numeric_match", ("nan", "", ""), "not a number"),
        ("content:numeric_match", ("__import__('os')", "", ""), "not a number"),
        ("content:numeric_match", ("", "", ""), "needs the correct answer"),
        ("content:numeric_match", ("1", "-1", ""), "must not be negative"),
        ("content:numeric_match", ("1", "", "A rubric"), "does not use it"),
        ("judge:rubric", ("1", "", "A rubric"), "does not use it"),
        ("judge:rubric", ("", "", ""), "needs the rubric"),
        ("content:numeric_within_range", ("62", "", ""), "min, max"),
        ("content:numeric_within_range", ("65, 62", "", ""), "above its maximum"),
        ("content:json_output", ("{a: 1}", "", ""), "not valid JSON"),
        ("content:json_output", ("[1]", "", ""), "JSON object"),
        ("content:paper_json_match", ("2602.02275v2", "", ""), "arXiv id"),
        ("format:json_output", ("1", "", ""), "does not use it"),
    ],
)
def test_check_kwargs_reports_wrong_boxes(validator, boxes, message):
    with pytest.raises(ValueError, match=message):
        check_kwargs(validator, *boxes)


# The task line.


def test_a_submission_becomes_a_task_the_harness_loads():
    task, problems = row_to_task(_row())

    assert problems == []
    assert task["task_id"] == "SFB-0052"
    assert task["prompt_md5"] == hash_prompt("What is 1 + 1?")
    assert task["agent_id"] == "code_agent"
    assert task["extra_tools"] == []
    TaskDef.model_validate(task)


def test_quantitative_checks_come_first_whatever_the_order_on_the_form():
    task, problems = row_to_task(
        _row(
            **{
                "Check 1: failure mode": "sycophancy",
                "Check 1: validator": "judge:rubric",
                "Check 1: correct answer": "",
                "Check 1: rubric": "Rejects the false claim.",
                "Check 2: failure mode": "correctness",
                "Check 2: validator": "content:numeric_match",
                "Check 2: correct answer": "2",
            }
        )
    )

    assert problems == []
    assert task["failure_modes"] == {"quantitative": ["correctness"], "qualitative": ["sycophancy"]}
    assert [check["failure_mode"] for check in task["validators"]] == ["correctness", "sycophancy"]
    TaskDef.model_validate(task)


def test_one_mode_with_two_checks_is_reported():
    _, problems = row_to_task(
        _row(
            **{
                "Check 2: failure mode": "correctness",
                "Check 2: validator": "content:numeric_match",
                "Check 2: correct answer": "2",
            }
        )
    )

    assert any("several" in problem for problem in problems)


def test_a_half_filled_check_is_reported():
    _, problems = row_to_task(_row(**{"Check 2: rubric": "Something"}))

    assert any(problem.startswith("check 2: choose both") for problem in problems)


def test_personal_columns_stay_out_of_the_task():
    task, _ = row_to_task(_row())
    line = json.dumps(task)

    assert "Jane" not in line
    assert "CERN" not in line


def test_sources_keep_their_notes_and_empty_ones_are_dropped():
    task, _ = row_to_task(_row(**{"Source 2: URL": "https://arxiv.org/abs/1", "Source 2: note": "Table 2"}))

    assert task["sources"] == [{"url": "https://arxiv.org/abs/1", "note": "Table 2"}]


def test_uploaded_files_add_read_file_and_no_other_tool():
    task, _ = row_to_task(_row_with_file())

    assert task["extra_tools"] == [{"tool_name": "read_file"}]


# File names.


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("data - jane roe.csv", ("data.csv", "jane roe")),
        ("table - jane roe.dat", ("table.dat", "jane roe")),
        ("script - jane roe.py", ("script.py", "jane roe")),
        ("my - data - J. Roe.csv", ("my - data.csv", "J. Roe")),
        ("data.csv", ("data.csv", "")),
    ],
)
def test_clean_file_name_cuts_after_the_last_separator(name, expected):
    assert clean_file_name(name) == expected


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("data.csv", "cannot be cleaned"),
        ("my data - Jane Roe.csv", "file name"),
        ("../../data - Jane Roe.csv", "file name"),
        ("sub/data - Jane Roe.csv", "file name"),
        (".hidden - Jane Roe.csv", "file name"),
        ("data - Jean - Luc.csv", "file name"),
    ],
)
def test_fetch_files_reports_names_it_cannot_use(name, message):
    files, _, problems = fetch_files(_row_with_file(), _fetch_as((name, b"x")))

    assert files == {}
    assert message in problems[0]


def test_fetch_files_cleans_every_name_and_finds_the_uploaders():
    fetch = _fetch_as(("data - Jane R.csv", b"a"), ("script - Jane R.py", b"b"))

    files, uploaders, problems = fetch_files(_row_with_file(), fetch)

    assert files == {"data.csv": b"a", "script.py": b"b"}
    assert uploaders == ["Jane R", "Jane R"]
    assert problems == []


def test_a_file_uploaded_twice_is_reported():
    _, _, problems = fetch_files(_row_with_file(), _fetch_as(("a - J R.csv", b"1"), ("a - K L.csv", b"2")))

    assert "uploaded twice" in problems[0]


def test_files_over_the_limit_are_reported():
    _, _, problems = fetch_files(_row_with_file(), _fetch_as(("a - J R.csv", b"x" * (MAX_TASK_BYTES + 1))))

    assert "over the limit" in problems[0]


def _unreadable(row):
    raise RuntimeError("Drive is down")


@pytest.mark.parametrize(
    ("fetch", "message"), [(_unreadable, "could not be read"), (_fetch_as(), "none could be found")]
)
def test_files_that_cannot_be_read_stop_the_task(fetch, message):
    _, _, problems = fetch_files(_row_with_file(), fetch)

    assert message in problems[0]


def test_nothing_is_fetched_without_uploads():
    assert fetch_files(_row(), _unreadable) == ({}, [], [])


# The rules for files.


def test_files_named_in_the_prompt_pass():
    assert check_files("See data/tasks/assets/{TASK_ID}/data.csv.", {"data.csv": b"a,b"}) == []


@pytest.mark.parametrize(
    ("prompt", "files", "message"),
    [
        ("Give the mean of data.csv.", {"data.csv": b"a"}, "does not refer"),
        ("See data/tasks/assets/{TASK_ID}/data.csv", {}, "no file was uploaded"),
        ("See data/tasks/assets/{TASK_ID}/data.csv", {"data.csv": b"a", "other.csv": b"b"}, "other.csv"),
        ("See data/tasks/assets/{TASK_ID}/mydata.csv", {"data.csv": b"a"}, "does not mention"),
        ("See data/tasks/assets/{TASK_ID}/a.pdf", {"a.pdf": b"%PDF"}, "file types"),
        ("See data/tasks/assets/{TASK_ID}/a.png", {"a.png": b"x"}, "file types"),
        ("See data/tasks/assets/{TASK_ID}/a.csv", {"a.csv": b"\xff\xfe"}, "UTF-8"),
        ("See data/tasks/assets/{TASK_ID}/a.csv and {TASK_ID}/b.csv", {"a.csv": b"a"}, "b.csv"),
    ],
)
def test_check_files_reports_each_problem(prompt, files, message):
    assert any(message in problem for problem in check_files(prompt, files))


# Personal details.


@pytest.mark.parametrize(
    "text",
    [
        "Written by JANE  ROE.",
        "Contact jane.roe@example.org",
        "orcid.org/0000-0002-1825-0097",
        "see ~Jane_Roe1 on OpenReview",
        "uploaded by j roe",
    ],
)
def test_personal_details_are_found_whatever_their_case(text):
    assert contains_personal_detail(text, personal_terms(_row(), ["J Roe"]))


def test_accents_do_not_hide_a_name():
    terms = personal_terms(OTHER_PERSON, [])

    assert contains_personal_detail("by Muller Sahin", terms)
    assert contains_personal_detail("by MÜLLER ŞAHİN", terms)


def test_an_orcid_given_as_a_link_is_searched_as_the_bare_number():
    terms = personal_terms(_row(ORCID="https://orcid.org/0000-0002-1825-0097"), [])

    assert contains_personal_detail("0000-0002-1825-0097", terms)


def test_ordinary_text_and_the_institution_are_not_personal_details():
    terms = personal_terms(_row(), [])

    assert not contains_personal_detail("The likelihood at CERN of a profile", terms)


@pytest.mark.parametrize("empty", ["none", "None", "n/a", "-", ""])
def test_an_openreview_answer_meaning_none_is_not_searched_for(empty):
    terms = personal_terms(_row(OpenReview=empty), [])

    assert not contains_personal_detail("There is none in this - n/a", terms)


def test_a_one_word_name_only_raises_a_warning_and_needs_a_whole_word():
    row = _row(**{"Full name": "Li"})

    assert not contains_personal_detail("Written by Li.", personal_terms(row, []))
    assert contains_personal_detail("Written by Li.", name_parts(row, []))
    assert not contains_personal_detail("The likelihood", name_parts(row, []))


# The whole import, on a real git repo.


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repo on `new-tasks`, started from a `main` holding one task."""
    monkeypatch.setattr(build_task_module, "REPO_ROOT", tmp_path)
    _git(tmp_path, "init", "--quiet", "--initial-branch=main")
    _git(tmp_path, "config", "user.email", "maintainer@example.org")
    _git(tmp_path, "config", "user.name", "Maintainer")
    (tmp_path / TASK_FILE).parent.mkdir(parents=True)
    existing, _ = row_to_task(_row(**{"Task ID": "SFB-0001", "Task description": "What is 2 + 2?"}))
    (tmp_path / TASK_FILE).write_text(json.dumps(existing) + "\n", encoding="utf-8")
    (tmp_path / REJECTED_FILE).write_text("# Ids only.\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "--quiet", "-m", "Start")
    prepare_branch(tmp_path, "new-tasks", "main")
    return tmp_path


def _head_subject(repo):
    return _git(repo, "log", "-1", "--format=%s").strip()


def test_a_good_submission_becomes_one_commit_with_one_line(repo):
    imported, reports, warnings = import_rows([_row()], _fetch, repo)

    assert (imported, reports, warnings) == (["SFB-0052"], {}, {})
    assert _head_subject(repo) == "Add task SFB-0052"
    assert _git(repo, "show", "--name-only", "--format=", "HEAD").split() == [TASK_FILE]
    assert [task["task_id"] for task in read_tasks(repo / TASK_FILE)] == ["SFB-0001", "SFB-0052"]
    assert not (repo / ASSETS_DIR / "SFB-0052").exists()


def test_files_land_under_their_cleaned_names(repo):
    imported, reports, _ = import_rows([_row_with_file()], _fetch, repo)

    assert (imported, reports) == (["SFB-0052"], {})
    assert (repo / ASSETS_DIR / "SFB-0052" / "data.csv").read_bytes() == b"a,b\n1,2\n"
    committed = _git(repo, "show", "--name-only", "--format=", "HEAD").split()
    assert sorted(committed) == sorted([TASK_FILE, f"{ASSETS_DIR}/SFB-0052/data.csv"])


def test_no_personal_detail_reaches_the_commit(repo):
    import_rows([_row_with_file()], _fetch, repo)

    commit = _git(repo, "show", "--format=%an %ae %cn %ce %s %b", "HEAD").casefold()
    for detail in ("jane", "roe@", "0000-0002", "cern", "maintainer"):
        assert detail not in commit


def test_each_good_submission_gets_its_own_commit(repo):
    second = _row(**{"Task ID": "SFB-0053", "Task description": "What is 3 + 3?"})

    imported, _, _ = import_rows([_row(), second], _fetch, repo)

    assert imported == ["SFB-0052", "SFB-0053"]
    assert _git(repo, "log", "-2", "--format=%s").split("\n")[:2] == ["Add task SFB-0053", "Add task SFB-0052"]


def test_running_it_again_imports_nothing_new(repo):
    import_rows([_row()], _fetch, repo)
    head = _git(repo, "rev-parse", "HEAD")

    assert import_rows([_row()], _fetch, repo) == ([], {}, {})
    assert _git(repo, "rev-parse", "HEAD") == head


def test_a_response_edited_after_the_import_is_a_conflict_and_changes_nothing(repo):
    import_rows([_row()], _fetch, repo)
    head = _git(repo, "rev-parse", "HEAD")

    _, reports, _ = import_rows([_row(**{"Task description": "What is 1 + 2?"})], _fetch, repo)

    assert "different prompt" in reports["SFB-0052"][0]
    assert _git(repo, "rev-parse", "HEAD") == head


def test_a_rejected_id_is_skipped_and_stays_skipped(repo):
    (repo / REJECTED_FILE).write_text("# Ids only.\nSFB-0052  # out of scope\n", encoding="utf-8")

    assert rejected_ids(repo) == {"SFB-0052"}
    assert import_rows([_row()], _fetch, repo) == ([], {}, {})
    assert len(read_tasks(repo / TASK_FILE)) == 1


def test_the_rejected_file_holds_ids_only():
    lines = (REPO_ROOT / REJECTED_FILE).read_text(encoding="utf-8").splitlines()

    assert all(line.startswith("#") for line in lines)


def test_a_row_the_script_missed_is_reported_by_its_sheet_row(repo):
    _, reports, _ = import_rows([_row(), _row(**{"Task ID": ""})], _fetch, repo)

    assert list(reports) == ["sheet row 3"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"Check 1: correct answer": "about 2"}, "not a number"),
        ({"Task description": "Written by Jane Roe: what is 1 + 1?"}, "personal detail"),
        ({"Source 1: URL": "arxiv.org/abs/1"}, "must start with http"),
        ({"Task description": "What is 2 + 2?"}, "already in"),
        ({"Task ID": "../../evil"}, "not a valid task id"),
        ({"Task description": "See data/tasks/assets/{TASK_ID}/data.csv"}, "no file was uploaded"),
    ],
)
def test_a_bad_submission_is_reported_and_nothing_is_written(repo, overrides, message):
    head = _git(repo, "rev-parse", "HEAD")

    imported, reports, _ = import_rows([_row(**overrides)], _fetch, repo)

    assert imported == []
    assert any(message in problem for problem in next(iter(reports.values()))), reports
    assert _git(repo, "rev-parse", "HEAD") == head
    assert _git(repo, "status", "--porcelain") == ""


def test_a_personal_detail_inside_a_file_stops_it(repo):
    fetch = _fetch_as(("data - jane r.csv", b"# by jane.roe@example.org\n1,2\n"))

    _, reports, _ = import_rows([_row_with_file()], fetch, repo)

    assert "appears in data.csv" in reports["SFB-0052"][0]
    assert not (repo / ASSETS_DIR / "SFB-0052").exists()


def test_a_personal_detail_in_a_file_name_stops_it(repo):
    row = _row_with_file(**{"Task description": "Read data/tasks/assets/{TASK_ID}/jane.roe@example.org.csv"})

    _, reports, _ = import_rows([row], _fetch_as(("jane.roe@example.org - J R.csv", b"1")), repo)

    assert any("personal detail" in problem or "file name" in problem for problem in reports["SFB-0052"])


def test_the_account_name_is_searched_for_even_when_it_differs_from_the_typed_name(repo):
    fetch = _fetch_as(("data - Janey R.csv", b"# by janey r\n1,2\n"))

    _, reports, _ = import_rows([_row_with_file()], fetch, repo)

    assert "personal detail" in reports["SFB-0052"][0]


def test_another_contributors_details_stop_a_task_too(repo):
    mention = _row(**{"Task description": "As Muller Sahin showed, what is 1 + 1?"})

    _, reports, _ = import_rows([mention, {**OTHER_PERSON, "Task ID": "SFB-0053"}], _fetch, repo)

    assert "personal detail" in reports["SFB-0052"][0]


def test_part_of_a_name_only_warns_and_the_task_is_imported(repo):
    row = _row(**{"Task description": "Roe et al. found what value of 1 + 1?"})

    imported, reports, warnings = import_rows([row], _fetch, repo)

    assert imported == ["SFB-0052"]
    assert reports == {}
    assert "part of a contributor's name" in warnings["SFB-0052"][0]


def test_problems_never_quote_a_personal_detail(repo):
    row = _row_with_file(**{"Task description": PROMPT_WITH_FILE + " By Jane Roe."})

    _, reports, _ = import_rows([row], _fetch_as(("data.csv - Jane Roe", b"jane.roe@example.org")), repo)

    text = json.dumps(reports).casefold()
    assert reports
    for detail in ("jane", "roe", "0000-0002"):
        assert detail not in text


def test_the_staged_scan_finds_details_links_hidden_files_and_paths_outside(repo):
    assets = repo / ASSETS_DIR / "SFB-0052"
    assets.mkdir(parents=True)
    (assets / "a.csv").write_text("by jane roe\n", encoding="utf-8")
    (assets / ".hidden").write_text("x\n", encoding="utf-8")
    os.symlink("/etc/hosts", assets / "link.csv")
    (repo / "outside.txt").write_text("x\n", encoding="utf-8")
    _git(repo, "add", ".")

    problems = staged_problems(repo, ["Jane Roe"])

    assert any("personal detail" in problem for problem in problems)
    assert any("link.csv is a symbolic link" in problem for problem in problems)
    assert any(".hidden is a hidden file" in problem for problem in problems)
    assert any("outside.txt" in problem for problem in problems)


def test_the_staged_scan_passes_a_plain_task(repo):
    (repo / TASK_FILE).open("a", encoding="utf-8").write('{"task_id": "x"}\n')
    _git(repo, "add", ".")

    assert staged_problems(repo, ["Jane Roe"]) == []


# The branch.


def test_the_branch_restarts_from_main_once_all_it_holds_is_merged(repo):
    import_rows([_row_with_file()], _fetch, repo)
    _git(repo, "switch", "--quiet", "main")
    _git(repo, "merge", "--quiet", "--squash", "new-tasks")
    _git(repo, "commit", "--quiet", "-m", "Squashed")
    (repo / "other.txt").write_text("main moved on\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "--quiet", "-m", "Main moved on")

    prepare_branch(repo, "new-tasks", "main")

    assert _git(repo, "rev-parse", "new-tasks") == _git(repo, "rev-parse", "main")
    assert _git(repo, "branch", "--show-current").strip() == "new-tasks"


def test_the_branch_is_kept_while_it_holds_unmerged_tasks(repo):
    import_rows([_row()], _fetch, repo)
    head = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "--quiet", "main")

    prepare_branch(repo, "new-tasks", "main")

    assert _git(repo, "rev-parse", "new-tasks") == head
    assert _git(repo, "branch", "--show-current").strip() == "new-tasks"


def test_the_branch_is_kept_while_it_holds_an_unmerged_rejection(repo):
    (repo / REJECTED_FILE).write_text("# Ids only.\nSFB-0052\n", encoding="utf-8")
    _git(repo, "commit", "--quiet", "-am", "Reject")
    head = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "--quiet", "main")

    prepare_branch(repo, "new-tasks", "main")

    assert _git(repo, "rev-parse", "new-tasks") == head


def test_it_refuses_to_switch_over_uncommitted_changes(repo):
    (repo / TASK_FILE).write_text("", encoding="utf-8")

    with pytest.raises(SystemExit, match="Commit or set aside"):
        prepare_branch(repo, "new-tasks", "main")


def test_it_refuses_to_switch_over_untracked_task_files(repo):
    (repo / ASSETS_DIR / "SFB-0099").mkdir(parents=True)
    (repo / ASSETS_DIR / "SFB-0099" / "a.csv").write_text("x", encoding="utf-8")

    with pytest.raises(SystemExit, match="Commit or set aside"):
        prepare_branch(repo, "new-tasks", "main")


def test_new_task_lines_holds_only_what_main_lacks(repo):
    import_rows([_row()], _fetch, repo)

    lines = new_task_lines(repo, "new-tasks", "main")

    assert [json.loads(line)["task_id"] for line in lines] == ["SFB-0052"]


# Without Google: a CSV export and a folder of files.


def _export(path, rows):
    columns = required_columns()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows([[row.get(column, "") for column in columns] for row in rows])


def _uploads(tmp_path):
    folder = tmp_path / "uploads" / "SFB-0052"
    folder.mkdir(parents=True)
    (folder / "data - jane roe.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    return tmp_path / "uploads"


def test_the_whole_pipeline_runs_from_a_csv_export(repo, tmp_path_factory, capsys):
    outside = tmp_path_factory.mktemp("outside")
    _export(outside / "sheet.csv", [_row_with_file()])
    args = ["--from-csv", str(outside / "sheet.csv"), "--files-dir", str(_uploads(outside)), "--base", "main"]

    assert main([*args, "--export", str(outside / "new.jsonl")], repo=repo) == 0

    assert "Imported 1 task(s)" in capsys.readouterr().out
    assert _head_subject(repo) == "Add task SFB-0052"
    assert (repo / ASSETS_DIR / "SFB-0052" / "data.csv").exists()
    assert [task["task_id"] for task in read_tasks(outside / "new.jsonl")] == ["SFB-0052"]


def test_a_dry_run_reports_and_touches_nothing(repo, tmp_path_factory, capsys):
    outside = tmp_path_factory.mktemp("outside")
    _export(outside / "sheet.csv", [_row_with_file()])
    head = _git(repo, "rev-parse", "HEAD")
    args = ["--from-csv", str(outside / "sheet.csv"), "--files-dir", str(_uploads(outside)), "--dry-run"]

    assert main(args, repo=repo) == 0

    assert "Would import 1 task(s): ['SFB-0052']" in capsys.readouterr().out
    assert _git(repo, "rev-parse", "HEAD") == head
    assert _git(repo, "status", "--porcelain") == ""
    assert not (repo / ASSETS_DIR / "SFB-0052").exists()


def test_a_bad_row_gives_a_failing_exit_and_a_report(repo, tmp_path_factory, capsys):
    outside = tmp_path_factory.mktemp("outside")
    _export(outside / "sheet.csv", [_row(**{"Check 1: correct answer": "about 2"})])

    assert main(["--from-csv", str(outside / "sheet.csv"), "--base", "main"], repo=repo) == 1

    assert "SFB-0052 was not written" in capsys.readouterr().out


def test_the_local_source_refuses_links_in_the_files_folder(tmp_path):
    _export(tmp_path / "sheet.csv", [_row_with_file()])
    folder = _uploads(tmp_path)
    os.symlink("/etc/hosts", folder / "SFB-0052" / "hosts - jane roe.csv")

    rows, fetch = local_source(tmp_path / "sheet.csv", folder)
    _, _, problems = fetch_files(rows[0], fetch)

    assert "could not be read" in problems[0]


# Google.


class _Request:
    def __init__(self, value):
        self.value = value

    def execute(self):
        return self.value


class _FakeDrive:
    def files(self):
        return self

    def get(self, fileId, fields):  # noqa: N803
        return _Request({"name": f"{fileId} - jane roe.csv"})

    def get_media(self, fileId):  # noqa: N803
        return _Request(fileId.encode())


def test_every_drive_link_of_a_row_is_downloaded():
    row = _row(Files="https://drive.google.com/open?id=a-1, https://drive.google.com/open?id=b_2")

    assert drive_fetch(_FakeDrive())(row) == [("a-1 - jane roe.csv", b"a-1"), ("b_2 - jane roe.csv", b"b_2")]


def test_the_importer_loads_without_the_google_libraries():
    code = (
        "import sys; sys.modules['googleapiclient'] = None; sys.modules['google_auth_oauthlib'] = None; "
        "import utils.import_form"
    )

    subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True, capture_output=True)
