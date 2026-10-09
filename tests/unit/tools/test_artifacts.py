import pytest

from tools import artifacts
from tools.artifacttools import ListArtifactsTool, ReadArtifactTool, SearchArtifactTool


@pytest.fixture()
def store(tmp_path):
    st = artifacts.ArtifactStore(tmp_path / "artifacts")
    st.set_scope("task-1-a1")
    return st


def test_cap_output_passthrough_when_small(store):
    text = "short output"
    assert artifacts.cap_output(text, tool_name="web_search", store=store, cap_chars=1000) == text


def test_cap_output_spills_and_banners(store):
    text = "linha\n" * 500  # ~3000 chars
    out = artifacts.cap_output(text, tool_name="wikipedia_search", store=store, cap_chars=500)
    assert len(out) < len(text)
    assert "search_artifact" in out and "read_artifact" in out
    assert "wikipedia_search_" in out
    # The full text is on disk and searchable.
    result = store.search("linha")
    assert result["total"] == 500


def test_cap_output_first_line_kept_in_preview(store):
    text = "RESPOSTA 50068\n" + "ruido\n" * 1000
    out = artifacts.cap_output(text, tool_name="wikipedia_search", store=store, cap_chars=500)
    assert out.startswith("RESPOSTA 50068")


def test_cap_output_plain_truncation_without_store():
    text = "x" * 5000
    out = artifacts.cap_output(text, tool_name="web_search", store=None, cap_chars=1000)
    assert "truncated" in out.lower()
    assert "search_artifact" not in out


def test_cap_output_dedup(store):
    text = "abc\n" * 1000
    first = artifacts.cap_output(text, tool_name="web_search", store=store, cap_chars=100)
    second = artifacts.cap_output(text, tool_name="web_search", store=store, cap_chars=100)
    rows = store.list_artifacts()
    assert len(rows) == 1
    rel = rows[0]["path"]
    assert rel in first and rel in second


def test_search_literal_with_context(store):
    store.save("alfa\nbeta\nGAMA\ndelta\n", tool_name="web_search")
    result = store.search("GAMA")
    assert result["total"] == 1
    match = result["matches"][0]
    assert match["text"] == "GAMA"
    assert match["before"] == ["alfa", "beta"]
    assert match["after"] == ["delta"]


def test_search_ignore_case_and_regex(store):
    store.save("Porto Alegre\nporto seguro\n", tool_name="web_search")
    assert store.search("porto")["total"] == 2
    assert store.search("porto", ignore_case=False)["total"] == 1
    assert store.search(r"P\w+to", regex=True)["total"] == 2


def test_search_max_matches(store):
    store.save("\n".join(["x"] * 100), tool_name="web_search")
    result = store.search("x", max_matches=10)
    assert result["total"] == 100
    assert result["shown"] == 10
    assert result["truncated"] is True


def test_search_across_all_artifacts(store):
    store.save("segredo-alfa\n", tool_name="web_search")
    store.save("segredo-beta\n", tool_name="wikipedia_search")
    result = store.search("segredo")
    assert result["total"] == 2


def test_search_no_matches(store):
    store.save("nada\n", tool_name="web_search")
    assert store.search("inexistente")["total"] == 0


def test_search_invalid_regex(store):
    store.save("x\n", tool_name="web_search")
    with pytest.raises(artifacts.ArtifactError):
        store.search("([", regex=True)


def test_read_range(store):
    store.save("\n".join(f"linha {i}" for i in range(1, 21)), tool_name="web_search")
    result = store.read(store.list_artifacts()[0]["path"], start_line=3, end_line=5)
    assert result["text"] == "linha 3\nlinha 4\nlinha 5"
    assert result["total_lines"] == 20


def test_read_max_chars_truncates(store):
    store.save("y" * 5000, tool_name="web_search")
    result = store.read(store.list_artifacts()[0]["path"], max_chars=100)
    assert result["truncated"] is True
    assert len(result["text"]) == 100


def test_list_artifacts(store):
    assert store.list_artifacts() == []
    store.save("conteudo\n", tool_name="web_search")
    rows = store.list_artifacts()
    assert len(rows) == 1
    assert rows[0]["path"].startswith("task-1-a1/")
    assert rows[0]["lines"] == 1


def test_resolve_rejects_escape(store):
    with pytest.raises(artifacts.ArtifactError):
        store.resolve("../../etc/passwd")


def test_resolve_accepts_bare_filename_and_redundant_prefix(store):
    ref = store.save("dados\n", tool_name="example_tool")
    name = ref.rel_path.split("/")[-1]
    assert store.resolve(name).name == name  # bare filename
    assert store.resolve(f"artifacts/{ref.rel_path}").name == name  # redundant root prefix


class _FakeTool:
    def __init__(self, name):
        self.name = name

    def forward(self, query):
        return f"{query}:" + "Z" * 3000


class _FakeWrappedTool:
    """Mimics tools.base.WrappedTool, whose forward delegates to an inner tool."""

    def __init__(self, inner):
        self._wrapped = inner
        self.name = inner.name


def test_install_output_cap_wraps_and_caps(store):
    fake = _FakeTool("web_search")
    artifacts.install_output_cap([fake], store, default_cap=500)
    out = fake.forward("q")
    assert "search_artifact" in out
    assert fake._sfb_output_cap == 500


def test_install_output_cap_is_idempotent(store):
    fake = _FakeTool("web_search")
    artifacts.install_output_cap([fake], store, default_cap=500)
    wrapped = fake.forward
    artifacts.install_output_cap([fake], store, default_cap=500)
    assert fake.forward is wrapped


def test_install_output_cap_unwraps_harness_tools(store):
    inner = _FakeTool("wikipedia_search")
    wrapped = _FakeWrappedTool(inner)
    artifacts.install_output_cap([wrapped], store, default_cap=500)
    # the *inner* tool is wrapped, so the watcher records the capped observation
    out = inner.forward("q")
    assert "search_artifact" in out
    assert inner._sfb_output_cap == 500
    assert not hasattr(wrapped, "_sfb_output_cap")


def test_install_output_cap_override_and_skip(store):
    small = _FakeTool("web_search")
    skipped = _FakeTool("final_answer")
    artifacts.install_output_cap([small, skipped], store, default_cap=0, overrides={"web_search": 500})
    assert "search_artifact" in small.forward("q")  # override applies
    assert skipped.forward("q").endswith("Z" * 10)  # cap<=0 -> untouched


def test_artifact_tools_roundtrip(store):
    store.save("alfa\ndado 50068\nomega\n", tool_name="wikipedia_search")
    search = SearchArtifactTool(store)
    read = ReadArtifactTool(store)
    listing = ListArtifactsTool(store)

    found = search.forward("50068")
    assert "50068" in found and "occurrence" in found
    assert "artifacts" in listing.forward().lower()
    path = listing.forward().split("**")[1]
    assert "dado 50068" in read.forward(path, start_line=1, end_line=3)


def test_artifact_tools_handle_missing_store():
    assert "No artifacts" in SearchArtifactTool(None).forward("x")
    assert "No artifacts" in ReadArtifactTool(None).forward("some/path")
    assert "No artifacts" in ListArtifactsTool(None).forward()
