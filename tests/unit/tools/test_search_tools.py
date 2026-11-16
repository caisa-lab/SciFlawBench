import os
import socket
import time

import pytest
import requests

from tools.definitions import tool_registry
from tools.searchtools import (
    ArxivSearchTool,
    ImprovedWikipediaSearchTool,
    RetryingDuckDuckGoSearchTool,
    SerpAPISearchTool,
)


class _FakePage:
    def __init__(self, title, text="", exists=True):
        self.title = title
        self.fullurl = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        self._text = text
        self._exists = exists

    def exists(self):
        return self._exists

    @property
    def text(self):
        return self._text

    @property
    def summary(self):
        return self._text


class _FakeWiki:
    def __init__(self, pages=None):
        self._pages = pages or {}

    def page(self, title):
        return self._pages.get(title, _FakePage(title, exists=False))


def _offline_tool(pages, suggestions):
    """A Wikipedia tool wired to a fake wiki/search so the test never hits the network."""
    tool = ImprovedWikipediaSearchTool(operator="test_suite")
    tool.wiki = _FakeWiki(pages)
    tool._suggest_titles = lambda query, limit=8: list(suggestions)
    return tool


def test_wiki_exact_title_is_returned_directly():
    tool = _offline_tool({"Anthropic": _FakePage("Anthropic", "Anthropic is an AI company.")}, [])
    res = tool.forward("Anthropic")
    assert "**Wikipedia Page:** Anthropic" in res
    assert "Anthropic is an AI company." in res
    assert "exact title" not in res  # no auto-correction note for an exact hit


def test_wiki_query_falls_back_to_closest_match():
    tool = _offline_tool({"AI alignment": _FakePage("AI alignment", "Alignment research text.")}, ["AI alignment"])
    res = tool.forward("LLM alignment")
    assert "**Wikipedia Page:** AI alignment" in res
    assert "Alignment research text." in res
    assert '*(exact title: "AI alignment")*' in res


def test_wiki_empty_exact_page_falls_back():
    tool = _offline_tool(
        {"LLM alignment": _FakePage("LLM alignment", "   "), "AI alignment": _FakePage("AI alignment", "real text")},
        ["AI alignment"],
    )
    res = tool.forward("LLM alignment")
    assert "AI alignment" in res
    assert "real text" in res


def test_wiki_no_page_and_no_suggestions():
    tool = _offline_tool({}, [])
    res = tool.forward("zzzz nonsense qqq")
    assert "No wikipedia pages found" in res
    assert "short, exact article title" in res


def test_wiki_no_page_lists_suggestions():
    tool = _offline_tool({}, ["AI alignment", "Large language model"])
    res = tool.forward("something odd")
    assert "No wikipedia pages found" in res
    assert "AI alignment" in res


def test_wiki_invalid_content_type():
    tool = _offline_tool({}, [])
    tool.content_type = "bogus"
    assert "Invalid content type" in tool.forward("Anthropic")


def test_simple_wiki_query():
    search_tool = ImprovedWikipediaSearchTool(operator="test_suite")
    res = search_tool.forward(query="Anthropic")
    print(res)
    assert "Anthropic" in res
    assert "Amodei" in res
    assert "founded in January 2021" in res


def test_failed_wiki_query():
    """A non-title query auto-resolves to the closest article instead of dead-ending."""
    search_tool = ImprovedWikipediaSearchTool(operator="test_suite")
    res = search_tool.forward(query="LLM alignment")
    print(res)
    assert "**Wikipedia Page:**" in res
    assert "No wikipedia pages found" not in res
    assert "exact title" in res  # the resolver notes the auto-corrected title


def test_simple_arxiv_query(monkeypatch):
    payload = {
        "hits": [
            {
                "publishedAt": "2024-01-01T00:00:00Z",
                "arxivId": "2401.00001",
                "title": "A Study of LLM Alignment",
                "abstract": "We study alignment of large language models.",
                "links": {"markdown": "https://example.com/article.md"},
            }
        ]
    }

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    monkeypatch.setattr("tools.searchtools.requests.get", lambda *args, **kwargs: _FakeResponse())

    search_tool = ArxivSearchTool(operator="test_suite")
    res = search_tool.forward(query="LLM alignment", sort="relevance", fromYear=2020)
    print(res)
    assert "article markdown" in res
    assert "**title:**" in res
    assert "id:" in res


def test_arxiv_query_sets_a_timeout(monkeypatch):
    """The request carries a timeout, so a slow server cannot stall a run."""
    seen = {}

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"hits": []}

    def _fake_get(*args, **kwargs):
        seen.update(kwargs)
        return _FakeResponse()

    monkeypatch.setattr("tools.searchtools.requests.get", _fake_get)

    ArxivSearchTool(operator="test_suite").forward(query="LLM alignment", sort="relevance", fromYear=2020)
    assert seen["timeout"] == ArxivSearchTool.TIMEOUT_S == 10


def test_arxiv_timeout_returns_an_error(monkeypatch):
    """A timeout reaches the agent as an error message instead of raising."""

    def _timeout(*args, **kwargs):
        raise requests.exceptions.Timeout("read timed out")

    monkeypatch.setattr("tools.searchtools.requests.get", _timeout)

    res = ArxivSearchTool(operator="test_suite").forward(query="LLM alignment", sort="relevance", fromYear=2020)
    assert "Error searching for Arxiv papers" in res
    assert "read timed out" in res


def test_arxiv_gives_up_on_a_server_that_never_answers(monkeypatch):
    """End to end: a local server that accepts the connection but never replies is abandoned after the timeout."""
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        host, port = server.getsockname()
        monkeypatch.setattr(ArxivSearchTool, "BASE_URL", f"http://{host}:{port}/api/agent/search")
        monkeypatch.setattr(ArxivSearchTool, "TIMEOUT_S", 0.5)

        start = time.monotonic()
        res = ArxivSearchTool(operator="test_suite").forward(query="LLM alignment", sort="relevance", fromYear=2020)
        elapsed = time.monotonic() - start

    assert "Error searching for Arxiv papers" in res
    assert elapsed < 5


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("SERPAPI_KEY"), reason="an api key is required to hit the real api")
def test_serp_search():
    key = os.environ["SERPAPI_KEY"]
    assert key is not None

    search_tool = SerpAPISearchTool(api_key=key)
    res = search_tool.forward(query="LLM alignment")
    print(res)
    assert "**Search Results:**" in res


def _flaky_search(monkeypatch, fails, attempts=3, retry_wait_s=0):
    """A web search whose first `fails` searches raise and whose next one answers, without the network."""
    tool = RetryingDuckDuckGoSearchTool(attempts=attempts, retry_wait_s=retry_wait_s, rate_limit=None)
    queries = []

    def fake_text(query, max_results):
        queries.append(query)
        if len(queries) <= fails:
            raise RuntimeError("No results found.")
        return [{"title": "GW190521", "href": "https://example.org/gw190521", "body": "A heavy black-hole merger."}]

    monkeypatch.setattr(tool.ddgs, "text", fake_text)
    return tool, queries


def test_web_search_tries_a_failed_search_again(monkeypatch):
    tool, queries = _flaky_search(monkeypatch, fails=2)
    res = tool.forward("GW190521 masses")
    assert "[GW190521](https://example.org/gw190521)" in res
    assert queries == ["GW190521 masses"] * 3


def test_web_search_raises_the_last_error_when_every_try_fails(monkeypatch):
    tool, queries = _flaky_search(monkeypatch, fails=5)
    with pytest.raises(RuntimeError, match="No results found"):
        tool.forward("GW190521 masses")
    assert len(queries) == 3


def test_web_search_with_one_attempt_does_not_try_again(monkeypatch):
    tool, queries = _flaky_search(monkeypatch, fails=1, attempts=1)
    with pytest.raises(RuntimeError):
        tool.forward("GW190521 masses")
    assert len(queries) == 1


def test_web_search_waits_between_tries(monkeypatch):
    waits = []
    monkeypatch.setattr("tools.searchtools.time.sleep", waits.append)
    tool, _ = _flaky_search(monkeypatch, fails=2, retry_wait_s=3.0)
    tool.forward("GW190521 masses")
    assert waits == [3.0, 3.0]


def test_web_search_needs_at_least_one_attempt():
    with pytest.raises(ValueError, match="attempts"):
        RetryingDuckDuckGoSearchTool(attempts=0)


def test_web_search_options_reach_the_tool(monkeypatch):
    monkeypatch.delenv("SERPAPI_KEY", raising=False)
    wrapped = tool_registry.get("web_search")(watcher=None, attempts=5, retry_wait_s=0.5)
    assert isinstance(wrapped._wrapped, RetryingDuckDuckGoSearchTool)
    assert (wrapped._wrapped.attempts, wrapped._wrapped.retry_wait_s) == (5, 0.5)
    default = tool_registry.get("web_search")(watcher=None)._wrapped
    assert (default.attempts, default.retry_wait_s, default.max_results) == (3, 3.0, 8)
