import pytest
from smolagents.local_python_executor import InterpreterError

from agents.base import build_agent
from core.config import ModelConfig
from core.events import EventWatcher


def _build(monkeypatch, agent_id="code_agent", extra_tools=()):
    monkeypatch.setenv("FAKE_KEY", "x")
    model = ModelConfig(
        provider="fake_model", model_id="fake", api_key_env="FAKE_KEY", extra_kwargs={"responses": ["ok"]}
    )
    return build_agent(agent_id, model, EventWatcher(task_id=1, sink=lambda event: None), {}, list(extra_tools)).agent


def _run(agent, code):
    """Run `code` in the agent's own executor, with its tools loaded, and return the value of the last line."""
    agent.python_executor.send_tools(agent.tools)
    return agent.python_executor(code).output


def test_code_agent_can_import_json(monkeypatch):
    agent = _build(monkeypatch)

    assert "json" in agent.additional_authorized_imports
    assert _run(agent, 'import json\njson.loads(\'{"a": [1, 2]}\')["a"][1]') == 2
    assert _run(agent, 'import json\njson.dumps({"b": 1})') == '{"b": 1}'


def test_the_system_prompt_lists_json_among_the_allowed_imports(monkeypatch):
    agent = _build(monkeypatch)

    assert "'json'" in agent.system_prompt


def test_code_agent_can_parse_a_json_tool_output(monkeypatch):
    # the motivating case: chaining calls needs the previous tool's JSON result inside the code
    agent = _build(monkeypatch, extra_tools=["numerics"])

    code = (
        "import json\n"
        'out = numerics(operation="solve", arguments={"matrix": [[2, 0], [0, 4]], "vector": [2, 4]})\n'
        'json.loads(out)["solution"]'
    )
    assert _run(agent, code) == [1.0, 1.0]


def test_files_stay_out_of_reach_with_json(monkeypatch, tmp_path):
    agent = _build(monkeypatch)
    target = tmp_path / "out.json"

    with pytest.raises(InterpreterError, match="'open'"):
        _run(agent, 'import json\njson.load(open("secrets.json"))')
    with pytest.raises(InterpreterError, match="'open'"):
        _run(agent, f'import json\nwith open("{target}", "w") as f:\n    json.dump({{}}, f)')
    assert not target.exists()


@pytest.mark.parametrize("module", ["os", "io", "pathlib", "subprocess", "shutil"])
def test_other_modules_stay_blocked(monkeypatch, module):
    agent = _build(monkeypatch)

    with pytest.raises(InterpreterError, match="not allowed"):
        _run(agent, f"import {module}")


@pytest.mark.parametrize("attribute", ["codecs", "decoder", "encoder", "scanner"])
def test_modules_imported_inside_json_stay_out_of_reach(monkeypatch, attribute):
    # json imports codecs (which can open files); the executor refuses any module that is not allowed
    agent = _build(monkeypatch)

    with pytest.raises(InterpreterError, match="Forbidden access to module"):
        _run(agent, f"import json\njson.{attribute}")
