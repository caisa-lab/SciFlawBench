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
from utils.build_task import FAILURE_MODES, REPO_ROOT, read_tasks
from utils.import_form import (
    ASSETS_DIR,
    FILES_COLUMN,
    MAX_TASK_BYTES,
    PERSONAL_COLUMNS,
    REJECTED_FILE,
    SCOPES,
    TASK_FILE,
    check_files,
    clean_file_name,
    contains_personal_detail,
    drive_fetch,
    failure_mode_name,
    fetch_files,
    import_rows,
    local_source,
    main,
    name_parts,
    new_task_lines,
    parse_rows,
    parse_sources,
    parse_validation,
    personal_terms,
    prepare_branch,
    rejected_ids,
    required_columns,
    row_to_task,
    staged_problems,
    ticked_modes,
)

SCRIPT = REPO_ROOT / "utils" / "form_script.gs"
PROMPT_WITH_FILE = "Read data/tasks/assets/{TASK_ID}/data.csv and give the mean."
PERSON = {
    "Full Name": "Jane Roe",
    "Email": "jane.roe@example.org",
    "Institution": "CERN",
    "ORCID": "0000-0002-1825-0097",
    "OpenReview": "https://openreview.net/profile?id=~Jane_Roe1",
}
OTHER_PERSON = {
    "Full Name": "Müller Şahin",
    "Email": "ms@example.org",
    "Institution": "DESY",
    "ORCID": "0000-0001-5109-3700",
    "OpenReview": "none",
}


def _row(**overrides):
    """A valid submission testing correctness, with any column replaced."""
    row = {
        **PERSON,
        "Task ID": "SFB-0052",
        "Task Description": "What is 1 + 1?",
        "Failure Modes": "correctness",
        "Validation": "# Correctness\nThe answer is 2.",
        "Sources": "* https://en.wikipedia.org/wiki/Addition\n  - Defines 1 + 1 = 2.",
    }
    row.update(overrides)
    return row


def _row_with_file(**overrides):
    return _row(
        **{"Task Description": PROMPT_WITH_FILE, FILES_COLUMN: "https://drive.google.com/open?id=abc-1", **overrides}
    )


def _fetch_as(*uploads):
    return lambda row: list(uploads)


_fetch = _fetch_as(("data - jane roe.csv", b"a,b\n1,2\n"))


# The form, the script and the importer agree.


def _script_list(name):
    match = re.search(rf"const {name} = (\[.*?\]);", SCRIPT.read_text(encoding="utf-8"), re.DOTALL)
    assert match
    return json.loads(match.group(1))


def test_the_script_records_the_details_the_importer_keeps_out_of_tasks():
    assert _script_list("AUTHOR_COLUMNS") == ["Task ID", *PERSONAL_COLUMNS]


def test_the_script_adds_the_task_id_column_the_importer_reads():
    assert 'const TASK_ID = "Task ID";' in SCRIPT.read_text(encoding="utf-8")
    assert "Task ID" in required_columns()


@pytest.mark.parametrize("mode", FAILURE_MODES)
def test_every_failure_mode_can_be_ticked_as_the_form_writes_it(mode):
    assert failure_mode_name(mode.replace("_", " ")) == mode
    assert failure_mode_name(mode.replace("_", " ").title()) == mode


def test_the_scopes_are_read_only_and_stated_in_the_docstring():
    source = (REPO_ROOT / "utils" / "import_form.py").read_text(encoding="utf-8")

    assert SCOPES == [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    assert "`spreadsheets.readonly` and `drive.readonly`" in source


# Reading the sheet.


def test_parse_rows_fills_short_rows_by_title():
    rows = parse_rows([required_columns(), [PERSON["Full Name"]]])

    assert rows[0]["Full Name"] == "Jane Roe"
    assert "Email" not in rows[0]


def test_parse_rows_ignores_case_and_spacing_and_keeps_other_columns():
    header = ["Timestamp", *[f" {column.upper()}  " for column in required_columns()]]

    rows = parse_rows([header, ["13/11/2026 10:00:00", "Jane Roe"]])

    assert rows[0] == {"Timestamp": "13/11/2026 10:00:00", "Full Name": "Jane Roe"}


def test_parse_rows_stops_on_a_renamed_question():
    with pytest.raises(SystemExit, match="Validation"):
        parse_rows([[column for column in required_columns() if column != "Validation"]])


def test_parse_rows_stops_on_two_columns_for_one_question():
    with pytest.raises(SystemExit, match="more than one column"):
        parse_rows([[*required_columns(), "task description"]])


def test_parse_rows_stops_on_an_empty_sheet():
    with pytest.raises(SystemExit, match="empty"):
        parse_rows([])


# The failure modes, the validation and the sources.


def test_ticked_modes_get_the_harness_names_quantitative_first():
    modes, problems = ticked_modes("sycophancy, correct tool calls, robustness against adversarial inputs")

    assert modes == ["correct_tool_calls", "robustness_against_adversarial_inputs", "sycophancy"]
    assert problems == []


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        ("", "between 1 and 4"),
        ("correctness, reasoning, planning, sycophancy, uncertainty awareness", "between 1 and 4"),
        ("correctness, telepathy", "not ones the harness knows"),
    ],
)
def test_ticked_modes_report_what_the_harness_cannot_take(answer, message):
    _, problems = ticked_modes(answer)

    assert any(message in problem for problem in problems)


def test_each_validation_section_becomes_the_description_of_its_mode():
    answer = "# Correctness\nThe answer is 42.\n\n## correct tool calls:\nIt searches\nthe web.\n"

    assert parse_validation(answer) == (
        {"correctness": "The answer is 42.", "correct_tool_calls": "It searches\nthe web."},
        [],
    )


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        ("The answer is 42.", "start each failure mode"),
        ("# Correctness", "correctness has no description"),
        ("# Correctness\n42\n# correctness\n43", "described twice"),
        ("# Correctness\n42\n# Jane Roe\nx", "section 2 is not a failure mode"),
    ],
)
def test_validation_problems_are_reported_without_quoting_it(answer, message):
    _, problems = parse_validation(answer)

    assert any(message in problem for problem in problems)
    assert "jane" not in json.dumps(problems).casefold()


def test_sources_are_read_in_the_forms_format():
    answer = (
        "* https://arxiv.org/abs/1807.06209\n"
        "  - Planck 2018, Table 2.\n"
        "  - Gives H0.\n"
        "* Hubble's law, https://en.wikipedia.org/wiki/Hubble%27s_law.\n"
    )

    assert parse_sources(answer) == (
        [
            {"url": "https://arxiv.org/abs/1807.06209", "note": "Planck 2018, Table 2. Gives H0."},
            {"url": "https://en.wikipedia.org/wiki/Hubble%27s_law", "note": "Hubble's law"},
        ],
        [],
    )


def test_a_plain_list_of_links_is_read_as_sources():
    sources, problems = parse_sources("https://a.org/x\n(see https://b.org/Galaxy_(novel))")

    assert sources == [{"url": "https://a.org/x", "note": ""}, {"url": "https://b.org/Galaxy_(novel)", "note": "see"}]
    assert problems == []


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        ("", "at least one source"),
        ("* Planck 2018 results\n  - Table 2", "source 1 has no link"),
        ("Table 2 of\nhttps://a.org", "before the first source"),
    ],
)
def test_source_problems_are_reported(answer, message):
    _, problems = parse_sources(answer)

    assert any(message in problem for problem in problems)


# The task line.


def test_a_submission_becomes_a_task_the_harness_loads():
    task, problems = row_to_task(_row())

    assert problems == []
    assert task["task_id"] == "SFB-0052"
    assert task["prompt_md5"] == hash_prompt("What is 1 + 1?")
    assert task["agent_id"] == "code_agent"
    assert task["extra_tools"] == []
    assert task["validators"] == [
        {"name": "judge:rubric", "kwargs": {"rubric": "The answer is 2."}, "failure_mode": "correctness"}
    ]
    assert task["sources"] == [{"url": "https://en.wikipedia.org/wiki/Addition", "note": "Defines 1 + 1 = 2."}]
    TaskDef.model_validate(task)


def test_the_forms_own_example_is_imported():
    task, problems = row_to_task(
        _row(
            **{
                "Task Description": (
                    "What is the answer to the ultimate question of life, the universe, and everything?"
                ),
                "Failure Modes": "correctness, reasoning",
                "Validation": (
                    "# Correctness\n"
                    "The answer, according to the The Hitchhiker's Guide to the Galaxy, is 42.\n"
                    "\n"
                    "# Reasoning\n"
                    "The agent should understand that this is a reference to The Hitchhiker's Guide to the Galaxy "
                    "and that it is not asked to write an elaborate philosophical essay."
                ),
                "Sources": (
                    "* https://en.wikipedia.org/wiki/The_Hitchhiker's_Guide_to_the_Galaxy\n"
                    "  - The answer to the ultimate question of life, the universe, and everything is, according to "
                    "Deep Thought, 42."
                ),
            }
        )
    )

    assert problems == []
    assert task["failure_modes"] == {"quantitative": ["correctness"], "qualitative": ["reasoning"]}
    assert [validator["failure_mode"] for validator in task["validators"]] == ["correctness", "reasoning"]
    assert task["sources"][0]["url"] == "https://en.wikipedia.org/wiki/The_Hitchhiker's_Guide_to_the_Galaxy"
    TaskDef.model_validate(task)


def test_quantitative_validators_come_first_whatever_the_order_on_the_form():
    task, problems = row_to_task(
        _row(
            **{
                "Failure Modes": "sycophancy, correctness",
                "Validation": "# Sycophancy\nRejects the false claim.\n# Correctness\nThe answer is 2.",
            }
        )
    )

    assert problems == []
    assert task["failure_modes"] == {"quantitative": ["correctness"], "qualitative": ["sycophancy"]}
    assert [validator["failure_mode"] for validator in task["validators"]] == ["correctness", "sycophancy"]
    TaskDef.model_validate(task)


def test_a_ticked_mode_without_validation_is_reported():
    _, problems = row_to_task(_row(**{"Failure Modes": "correctness, reasoning"}))

    assert any("have no validation: ['reasoning']" in problem for problem in problems)


def test_a_validation_for_a_mode_not_ticked_is_reported():
    _, problems = row_to_task(_row(Validation="# Correctness\nThe answer is 2.\n# Reasoning\nShows the steps."))

    assert any("not ticked: ['reasoning']" in problem for problem in problems)


def test_personal_columns_stay_out_of_the_task():
    task, _ = row_to_task(_row())
    line = json.dumps(task)

    assert "Jane" not in line
    assert "CERN" not in line


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
    row = _row(**{"Full Name": "Li"})

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
    existing, _ = row_to_task(_row(**{"Task ID": "SFB-0001", "Task Description": "What is 2 + 2?"}))
    (tmp_path / TASK_FILE).write_text(json.dumps(existing) + "\n", encoding="utf-8")
    (tmp_path / REJECTED_FILE).write_text("# Ids only.\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "--quiet", "-m", "Start")
    prepare_branch(tmp_path, "new-tasks", "main")
    return tmp_path


def _head_subject(repo):
    return _git(repo, "log", "-1", "--format=%s").strip()


def test_a_valid_submission_becomes_one_commit_with_one_line(repo):
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


def test_each_valid_submission_gets_its_own_commit(repo):
    second = _row(**{"Task ID": "SFB-0053", "Task Description": "What is 3 + 3?"})

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

    _, reports, _ = import_rows([_row(**{"Task Description": "What is 1 + 2?"})], _fetch, repo)

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
        ({"Validation": "# Reasoning\nShows the steps."}, "have no validation"),
        ({"Task Description": "Written by Jane Roe: what is 1 + 1?"}, "personal detail"),
        ({"Sources": "* arxiv.org/abs/1\n  - the paper"}, "has no link"),
        ({"Task Description": "What is 2 + 2?"}, "already in"),
        ({"Task ID": "../../evil"}, "not a valid task id"),
        ({"Task Description": "See data/tasks/assets/{TASK_ID}/data.csv"}, "no file was uploaded"),
    ],
)
def test_an_invalid_submission_is_reported_and_nothing_is_written(repo, overrides, message):
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
    row = _row_with_file(**{"Task Description": "Read data/tasks/assets/{TASK_ID}/jane.roe@example.org.csv"})

    _, reports, _ = import_rows([row], _fetch_as(("jane.roe@example.org - J R.csv", b"1")), repo)

    assert any("personal detail" in problem or "file name" in problem for problem in reports["SFB-0052"])


def test_the_account_name_is_searched_for_even_when_it_differs_from_the_typed_name(repo):
    fetch = _fetch_as(("data - Janey R.csv", b"# by janey r\n1,2\n"))

    _, reports, _ = import_rows([_row_with_file()], fetch, repo)

    assert "personal detail" in reports["SFB-0052"][0]


def test_another_contributors_details_stop_a_task_too(repo):
    mention = _row(**{"Task Description": "As Muller Sahin showed, what is 1 + 1?"})

    _, reports, _ = import_rows([mention, {**OTHER_PERSON, "Task ID": "SFB-0053"}], _fetch, repo)

    assert "personal detail" in reports["SFB-0052"][0]


def test_part_of_a_name_only_warns_and_the_task_is_imported(repo):
    row = _row(**{"Task Description": "Roe et al. found what value of 1 + 1?"})

    imported, reports, warnings = import_rows([row], _fetch, repo)

    assert imported == ["SFB-0052"]
    assert reports == {}
    assert "part of a contributor's name" in warnings["SFB-0052"][0]


def test_problems_never_quote_a_personal_detail(repo):
    row = _row_with_file(**{"Task Description": PROMPT_WITH_FILE + " By Jane Roe."})

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


@pytest.fixture
def team(repo, tmp_path_factory):
    """Two maintainers' clones of a shared remote that only has `main`."""
    root = tmp_path_factory.mktemp("team")
    _git(root, "clone", "--quiet", "--bare", str(repo), "origin.git")
    _git(root / "origin.git", "symbolic-ref", "HEAD", "refs/heads/main")
    _git(root / "origin.git", "branch", "--quiet", "-D", "new-tasks")
    for name in ("a", "b"):
        _git(root, "clone", "--quiet", "origin.git", name)
        _git(root / name, "config", "user.email", f"{name}@example.org")
        _git(root / name, "config", "user.name", name)
    return root / "a", root / "b"


def _import_and_push(clone, *rows):
    _git(clone, "fetch", "--quiet", "origin")
    prepare_branch(clone, "new-tasks", "origin/main")
    imported, _, _ = import_rows(list(rows), _fetch, clone)
    _git(clone, "push", "--quiet", "origin", "new-tasks")
    return imported


SECOND_ROW = {"Task ID": "SFB-0053", "Task Description": "What is 3 + 3?"}
THIRD_ROW = {"Task ID": "SFB-0054", "Task Description": "What is 4 + 4?"}


def test_the_next_maintainer_continues_from_the_pushed_branch(team):
    first, second = team
    assert _import_and_push(first, _row()) == ["SFB-0052"]

    assert _import_and_push(second, _row(), _row(**SECOND_ROW)) == ["SFB-0053"]

    assert [task["task_id"] for task in read_tasks(second / TASK_FILE)] == ["SFB-0001", "SFB-0052", "SFB-0053"]
    assert _git(second, "log", "-2", "--format=%s").split("\n")[:2] == ["Add task SFB-0053", "Add task SFB-0052"]


def test_a_local_branch_behind_the_pushed_one_is_brought_up_to_date(team):
    first, second = team
    _import_and_push(first, _row())
    _import_and_push(second, _row(), _row(**SECOND_ROW))

    _git(first, "fetch", "--quiet", "origin")
    prepare_branch(first, "new-tasks", "origin/main")

    assert _git(first, "rev-parse", "HEAD") == _git(first, "rev-parse", "origin/new-tasks")


def test_unpushed_local_commits_are_kept(team):
    first, _ = team
    _import_and_push(first, _row())
    import_rows([_row(), _row(**SECOND_ROW)], _fetch, first)
    head = _git(first, "rev-parse", "HEAD")

    _git(first, "fetch", "--quiet", "origin")
    prepare_branch(first, "new-tasks", "origin/main")

    assert _git(first, "rev-parse", "HEAD") == head


def test_it_stops_when_the_local_and_pushed_branches_have_both_changed(team):
    first, second = team
    _import_and_push(first, _row())
    _import_and_push(second, _row(), _row(**SECOND_ROW))
    import_rows([_row(), _row(**THIRD_ROW)], _fetch, first)
    head = _git(first, "rev-parse", "HEAD")

    _git(first, "fetch", "--quiet", "origin")
    with pytest.raises(SystemExit, match="have both changed"):
        prepare_branch(first, "new-tasks", "origin/main")

    assert _git(first, "rev-parse", "HEAD") == head


def test_a_pushed_branch_already_merged_is_not_continued(team):
    first, second = team
    _import_and_push(first, _row())
    _git(first, "switch", "--quiet", "main")
    _git(first, "merge", "--quiet", "--squash", "new-tasks")
    _git(first, "commit", "--quiet", "-m", "Squashed")
    _git(first, "push", "--quiet", "origin", "main")

    _git(second, "fetch", "--quiet", "origin")
    prepare_branch(second, "new-tasks", "origin/main")

    assert _git(second, "rev-parse", "HEAD") == _git(second, "rev-parse", "origin/main")
    assert import_rows([_row()], _fetch, second) == ([], {}, {})


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


def test_an_invalid_row_gives_a_failing_exit_and_a_report(repo, tmp_path_factory, capsys):
    outside = tmp_path_factory.mktemp("outside")
    _export(outside / "sheet.csv", [_row(Validation="# Reasoning\nShows the steps.")])

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
    row = _row(**{FILES_COLUMN: "https://drive.google.com/open?id=a-1, https://drive.google.com/open?id=b_2"})

    assert drive_fetch(_FakeDrive())(row) == [("a-1 - jane roe.csv", b"a-1"), ("b_2 - jane roe.csv", b"b_2")]


def test_the_importer_loads_without_the_google_libraries():
    code = (
        "import sys; sys.modules['googleapiclient'] = None; sys.modules['google_auth_oauthlib'] = None; "
        "import utils.import_form"
    )

    subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True, capture_output=True)
