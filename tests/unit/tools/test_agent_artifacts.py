from agents.base import build_agent
from core.config import ModelConfig
from core.events import EventWatcher
from tools.artifacts import ArtifactStore

ARTIFACT_TOOLS = {"search_artifact", "read_artifact", "list_artifacts"}


def _model():
    return ModelConfig(
        provider="fake_model", model_id="fake", api_key_env="FAKE_KEY", extra_kwargs={"responses": ["ok"]}
    )


def _build(monkeypatch, store):
    monkeypatch.setenv("FAKE_KEY", "x")
    return build_agent(
        "code_agent",
        _model(),
        EventWatcher(task_id=1, sink=lambda event: None),
        {},
        [],
        artifact_store=store,
    )


def test_agent_without_store_has_no_artifact_tools(tmp_path, monkeypatch):
    agent = _build(monkeypatch, None)
    assert ARTIFACT_TOOLS.isdisjoint(set(agent.agent.tools))


def test_agent_with_store_gains_artifact_tools(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "artifacts")
    agent = _build(monkeypatch, store)
    assert set(agent.agent.tools) >= ARTIFACT_TOOLS


def test_agent_content_tools_are_output_capped(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "artifacts")
    agent = _build(monkeypatch, store)
    # the cap is installed on the *inner* tool so the watcher records the capped observation
    assert agent.agent.tools["web_search"]._wrapped._sfb_output_cap == 20_000
    assert agent.agent.tools["wikipedia_search"]._wrapped._sfb_output_cap == 50_000


def test_disabled_store_does_not_add_artifact_tools(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "artifacts", enabled=False)
    agent = _build(monkeypatch, store)
    assert ARTIFACT_TOOLS.isdisjoint(set(agent.agent.tools))
