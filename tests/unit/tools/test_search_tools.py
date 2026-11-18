import os
import socket
import tempfile
import time
from pathlib import Path

import pytest
import requests

from tools.definitions import tool_registry
from tools.searchtools import (
    ArxivSearchTool,
    ImprovedWikipediaSearchTool,
    RetryingDuckDuckGoSearchTool,
    SerpAPISearchTool,
    TavilySearchError,
    TavilySearchTool,
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


def _feed(*papers):
    """An arXiv API reply (Atom) listing `papers`, each (arXiv id with version, title, abstract, published)."""
    entries = "".join(
        f"<entry><id>http://arxiv.org/abs/{arxiv_id}</id><published>{published}</published>"
        f"<title>{title}</title><summary>{abstract}</summary></entry>"
        for arxiv_id, title, abstract, published in papers
    )
    return f'<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom">{entries}</feed>'


RAITHEL = (
    "1803.07687v1",
    "Tidal deformability from GW170817 as a direct probe of the neutron star radius",
    "We show that the tidal deformability depends on the radius.",
    "2018-03-20T18:00:01Z",
)


class _FakeReply:
    def __init__(self, text="", status_code=200):
        self.text, self.status_code = text, status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


def _offline_arxiv(monkeypatch, *replies):
    """An arxiv_search tool whose requests get `replies` in turn, with no waiting; returns it and the requests made."""
    monkeypatch.setattr(ArxivSearchTool, "LOCK_FILE", Path(tempfile.mkdtemp()) / "arxiv.lock")
    monkeypatch.setattr(ArxivSearchTool, "MIN_INTERVAL_S", 0)
    monkeypatch.setattr(ArxivSearchTool, "RETRY_WAIT_S", 0)
    requests_made, queue = [], list(replies)

    def fake_get(url, params, headers, timeout):
        requests_made.append({"url": url, "timeout": timeout, **params})
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr("tools.searchtools.requests.get", fake_get)
    return ArxivSearchTool(operator="test_suite"), requests_made


def test_arxiv_search_formats_each_paper(monkeypatch):
    tool, requests_made = _offline_arxiv(monkeypatch, _FakeReply(_feed(RAITHEL)))
    res = tool.forward(query="Raithel tidal deformability", sort="relevance", fromYear=2018)
    assert "2018-03-20T18:00:01Z - id: 1803.07687 - **title:** Tidal deformability from GW170817" in res
    assert "**abstract:**We show that the tidal deformability depends on the radius." in res
    assert "link: https://arxiv.org/abs/1803.07687" in res
    assert requests_made[0]["url"] == "https://export.arxiv.org/api/query"


def test_arxiv_search_looks_up_an_id_in_the_query(monkeypatch):
    tool, requests_made = _offline_arxiv(monkeypatch, _FakeReply(_feed(RAITHEL)))
    res = tool.forward(query="Raithel et al., arXiv:1803.07687v2", sort="relevance", fromYear=1991)
    assert "id: 1803.07687" in res
    assert requests_made[0]["id_list"] == "1803.07687"
    assert "search_query" not in requests_made[0]


def test_arxiv_search_wants_all_the_words_first(monkeypatch):
    tool, requests_made = _offline_arxiv(monkeypatch, _FakeReply(_feed(RAITHEL)))
    tool.forward(query="Raithel Özel Psaltis 2018", sort="newest", fromYear=2018)
    [request] = requests_made
    assert request["search_query"] == (
        "(all:Raithel AND all:Özel AND all:Psaltis AND all:2018) AND submittedDate:[201801010000 TO 299912312359]"
    )
    assert (request["sortBy"], request["sortOrder"]) == ("submittedDate", "descending")


def test_arxiv_search_falls_back_to_any_word_by_relevance(monkeypatch):
    tool, requests_made = _offline_arxiv(monkeypatch, _FakeReply(_feed()), _FakeReply(_feed(RAITHEL)))
    res = tool.forward(query="Raithel deformability", sort="newest", fromYear=1991)
    assert "id: 1803.07687" in res
    assert [r["search_query"] for r in requests_made] == [
        "(all:Raithel AND all:deformability)",
        "(all:Raithel OR all:deformability)",
    ]
    assert (requests_made[1]["sortBy"], requests_made[1]["sortOrder"]) == ("relevance", "descending")


def test_arxiv_search_with_no_match_returns_an_empty_list(monkeypatch):
    tool, requests_made = _offline_arxiv(monkeypatch, _FakeReply(_feed()))
    assert tool.forward(query="zzzqqq", sort="relevance", fromYear=1991) == "**Results:** \n[\n\n]"
    assert len(requests_made) == 1  # one word: nothing to fall back to


def test_arxiv_search_waits_and_tries_again_while_arxiv_limits_requests(monkeypatch):
    refused = requests.ConnectionError("Connection refused")
    tool, requests_made = _offline_arxiv(monkeypatch, _FakeReply(status_code=429), refused, _FakeReply(_feed(RAITHEL)))
    res = tool.forward(query="1803.07687", sort="relevance", fromYear=1991)
    assert "id: 1803.07687" in res
    assert len(requests_made) == 3


def test_arxiv_search_reports_an_error_when_every_try_fails(monkeypatch):
    tool, requests_made = _offline_arxiv(monkeypatch, *[_FakeReply(status_code=503)] * 3)
    res = tool.forward(query="1803.07687", sort="relevance", fromYear=1991)
    assert "Error searching for Arxiv papers: 503 from the arXiv API" in res
    assert len(requests_made) == ArxivSearchTool.ATTEMPTS == 3


def test_arxiv_search_reports_a_query_arxiv_rejects(monkeypatch):
    error = ("http://arxiv.org/api/errors#incorrect_id_format", "Error", "incorrect id format for 1803.0", "2026")
    tool, _ = _offline_arxiv(monkeypatch, _FakeReply(_feed(error)))
    res = tool.forward(query="1803.07687", sort="relevance", fromYear=1991)
    assert "Error searching for Arxiv papers: incorrect id format" in res


def test_arxiv_search_keeps_three_seconds_between_the_requests_of_all_tools(monkeypatch, tmp_path):
    """Every task runs in its own process, so the gap is kept through a shared lock file, not in memory."""
    clock, waits = [1000.0], []
    monkeypatch.setattr("tools.searchtools.time.time", lambda: clock[0])
    monkeypatch.setattr("tools.searchtools.time.sleep", waits.append)
    monkeypatch.setattr(ArxivSearchTool, "LOCK_FILE", tmp_path / "arxiv.lock")
    first, second = ArxivSearchTool(operator="test_suite"), ArxivSearchTool(operator="test_suite")
    first._keep_interval()
    clock[0] += 1.0
    second._keep_interval()  # another tool, as in another task's process
    clock[0] += 10.0
    first._keep_interval()  # more than 3 s later: no wait
    assert waits == [2.0]
    assert float((tmp_path / "arxiv.lock").read_text()) == 1011.0


def test_arxiv_query_sets_a_timeout(monkeypatch):
    """The request carries a timeout, so a slow server cannot stall a run."""
    tool, requests_made = _offline_arxiv(monkeypatch, _FakeReply(_feed()))
    tool.forward(query="LLM", sort="relevance", fromYear=2020)
    assert requests_made[0]["timeout"] == ArxivSearchTool.TIMEOUT_S == 30


def test_arxiv_search_tries_a_request_that_timed_out_again(monkeypatch):
    tool, requests_made = _offline_arxiv(
        monkeypatch, requests.exceptions.ReadTimeout("read timed out"), _FakeReply(_feed(RAITHEL))
    )
    res = tool.forward(query="1803.07687", sort="relevance", fromYear=1991)
    assert "id: 1803.07687" in res
    assert len(requests_made) == 2


def test_arxiv_timeout_returns_an_error(monkeypatch):
    """When every try times out, the agent gets an error message instead of an exception."""
    tool, requests_made = _offline_arxiv(monkeypatch, *[requests.exceptions.Timeout("read timed out")] * 3)
    res = tool.forward(query="LLM alignment", sort="relevance", fromYear=2020)
    assert "Error searching for Arxiv papers" in res
    assert "read timed out" in res
    assert len(requests_made) == 3


def test_arxiv_gives_up_on_a_server_that_never_answers(monkeypatch):
    """End to end: a local server that accepts the connection but never replies is abandoned after the timeout."""
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        host, port = server.getsockname()
        monkeypatch.setattr(ArxivSearchTool, "BASE_URL", f"http://{host}:{port}/api/query")
        monkeypatch.setattr(ArxivSearchTool, "TIMEOUT_S", 0.5)
        monkeypatch.setattr(ArxivSearchTool, "MIN_INTERVAL_S", 0)
        monkeypatch.setattr(ArxivSearchTool, "LOCK_FILE", Path(tempfile.mkdtemp()) / "arxiv.lock")
        monkeypatch.setattr(ArxivSearchTool, "RETRY_WAIT_S", 0)

        start = time.monotonic()
        res = ArxivSearchTool(operator="test_suite").forward(query="LLM alignment", sort="relevance", fromYear=2020)
        elapsed = time.monotonic() - start

    assert "Error searching for Arxiv papers" in res
    assert elapsed < 5  # three tries of 0.5 s each


@pytest.mark.live
def test_live_arxiv_search_finds_a_paper_by_its_authors():
    res = ArxivSearchTool(operator="test_suite").forward(
        query="Raithel Ozel Psaltis tidal deformability", sort="relevance", fromYear=2018
    )
    assert "id: 1803.07687" in res


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
    monkeypatch.setattr(RetryingDuckDuckGoSearchTool, "LOCK_FILE", Path(tempfile.mkdtemp()) / "web_search.lock")
    tool = RetryingDuckDuckGoSearchTool(attempts=attempts, retry_wait_s=retry_wait_s, rate_limit=None, min_interval_s=0)
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
    wrapped = tool_registry.get("web_search")(watcher=None, attempts=5, retry_wait_s=0.5, min_interval_s=1.0)
    assert isinstance(wrapped._wrapped, RetryingDuckDuckGoSearchTool)
    assert (wrapped._wrapped.attempts, wrapped._wrapped.retry_waits_s, wrapped._wrapped.min_interval_s) == (
        5,
        [0.5],
        1.0,
    )
    default = tool_registry.get("web_search")(watcher=None)._wrapped
    assert (default.attempts, default.retry_waits_s, default.min_interval_s, default.max_results) == (
        4,
        [5.0, 15.0, 30.0],
        2.0,
        8,
    )


def test_web_search_waits_longer_before_each_new_try(monkeypatch):
    waits = []
    monkeypatch.setattr("tools.searchtools.time.sleep", waits.append)
    tool, queries = _flaky_search(monkeypatch, fails=3, attempts=4, retry_wait_s=[5.0, 15.0, 30.0])
    tool.forward("GW190521 masses")
    assert waits == [5.0, 15.0, 30.0]
    assert len(queries) == 4


def test_web_search_keeps_its_last_wait_when_tries_outnumber_the_waits(monkeypatch):
    waits = []
    monkeypatch.setattr("tools.searchtools.time.sleep", waits.append)
    tool, _ = _flaky_search(monkeypatch, fails=3, attempts=4, retry_wait_s=[5.0, 15.0])
    tool.forward("GW190521 masses")
    assert waits == [5.0, 15.0, 15.0]


def test_web_search_needs_at_least_one_wait():
    with pytest.raises(ValueError, match="retry_wait_s"):
        RetryingDuckDuckGoSearchTool(retry_wait_s=[])


def test_web_searches_of_all_agents_keep_one_pace(monkeypatch, tmp_path):
    """Every task runs in its own process, so the gap is kept through a shared lock file, not in memory."""
    clock, waits = [1000.0], []
    monkeypatch.setattr("tools.searchtools.time.time", lambda: clock[0])
    monkeypatch.setattr("tools.searchtools.time.sleep", waits.append)
    monkeypatch.setattr(RetryingDuckDuckGoSearchTool, "LOCK_FILE", tmp_path / "web_search.lock")
    first, second = (RetryingDuckDuckGoSearchTool(rate_limit=None) for _ in range(2))
    answer = [{"title": "GW190521", "href": "https://example.org/gw190521", "body": "A heavy merger."}]
    for tool in (first, second):
        monkeypatch.setattr(tool.ddgs, "text", lambda query, max_results: answer)
    first.forward("GW190521")
    clock[0] += 0.5
    second.forward("GW190521")  # another agent, half a second later: waits for the rest of the 2 s
    assert waits == [1.5]


class _FakeTavilyReply:
    def __init__(self, results=None, status_code=200, detail="", headers=None):
        self.status_code, self.headers = status_code, headers or {}
        self._json = {"results": results or []} if status_code == 200 else {"detail": {"error": detail}}
        self.text = str(self._json)

    def json(self):
        return self._json


GW190521 = {"title": "GW190521", "url": "https://example.org/gw190521", "content": "A heavy black-hole merger."}


def _offline_tavily(monkeypatch, *replies, **options):
    """A Tavily search whose requests get `replies` in turn, without the network or waiting; returns it, the
    requests made and the waits."""
    requests_made, waits, queue = [], [], list(replies)
    monkeypatch.setattr("tools.searchtools.time.sleep", waits.append)

    def fake_post(url, json, headers, timeout):
        requests_made.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr("tools.searchtools.requests.post", fake_post)
    options = {"rate_limit": None, **options}
    return TavilySearchTool("tvly-test-key", **options), requests_made, waits


def test_tavily_search_formats_results_like_the_duckduckgo_search(monkeypatch):
    tool, _, _ = _offline_tavily(monkeypatch, _FakeTavilyReply([GW190521]))
    assert tool.forward("GW190521 masses") == (
        "## Search Results\n\n[GW190521](https://example.org/gw190521)\nA heavy black-hole merger."
    )


def test_tavily_search_asks_for_ranked_pages_only(monkeypatch):
    tool, requests_made, _ = _offline_tavily(monkeypatch, _FakeTavilyReply([GW190521]), max_results=8)
    tool.forward("GW190521 masses")
    sent = requests_made[0]
    assert sent["url"] == "https://api.tavily.com/search"
    assert sent["headers"] == {"Authorization": "Bearer tvly-test-key"}
    assert sent["timeout"] == 30
    assert sent["json"] == {
        "query": "GW190521 masses",
        "search_depth": "basic",
        "max_results": 13,  # 8 plus 5 spare, to replace results from blocked URLs
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
    }


def test_tavily_search_drops_results_from_the_benchmark_itself(monkeypatch):
    leaks = [
        {
            "title": "tasks.jsonl",
            "url": "https://github.com/caisa-lab/SciFlawBench/blob/main/data/tasks/v0/tasks.jsonl",
        },
        {"title": "SciFlawBench", "url": "https://caisa-lab.github.io/SciFlawBench/"},
        {"title": "a copy", "url": "https://github.com/someone/SciFlawBenchHarness-copy"},
    ]
    tool, _, _ = _offline_tavily(monkeypatch, _FakeTavilyReply([leaks[0], GW190521, *leaks[1:]]))
    res = tool.forward("GW190521 masses")
    assert "[GW190521](https://example.org/gw190521)" in res
    assert "SciFlawBench" not in res and "tasks.jsonl" not in res


def test_tavily_search_blocks_the_urls_it_is_given(monkeypatch):
    other = {"title": "Wikipedia", "url": "https://en.wikipedia.org/wiki/GW190521", "content": "An event."}
    tool, _, _ = _offline_tavily(monkeypatch, _FakeTavilyReply([other, GW190521]), blocked_urls=["Wikipedia.org"])
    res = tool.forward("GW190521")
    assert "wikipedia" not in res and "example.org/gw190521" in res


def test_tavily_search_keeps_at_most_max_results(monkeypatch):
    pages = [{"title": f"page {i}", "url": f"https://example.org/{i}", "content": ""} for i in range(10)]
    tool, _, _ = _offline_tavily(monkeypatch, _FakeTavilyReply(pages), max_results=3)
    res = tool.forward("pages")
    assert res.count("](https://example.org/") == 3


def test_tavily_search_with_no_results_says_so(monkeypatch):
    tool, _, _ = _offline_tavily(monkeypatch, _FakeTavilyReply([]))
    assert tool.forward("a query that finds nothing").startswith("No results found")


def test_tavily_search_tries_a_rate_limit_or_server_error_again(monkeypatch):
    tool, requests_made, waits = _offline_tavily(
        monkeypatch, _FakeTavilyReply(status_code=429), _FakeTavilyReply(status_code=502), _FakeTavilyReply([GW190521])
    )
    assert "GW190521" in tool.forward("GW190521 masses")
    assert len(requests_made) == 3
    assert waits == [5.0, 15.0]


def test_tavily_search_waits_as_long_as_tavily_asks(monkeypatch):
    tool, _, waits = _offline_tavily(
        monkeypatch, _FakeTavilyReply(status_code=429, headers={"Retry-After": "40"}), _FakeTavilyReply([GW190521])
    )
    tool.forward("GW190521 masses")
    assert waits == [40.0]


def test_tavily_search_tries_a_timed_out_request_again(monkeypatch):
    tool, requests_made, _ = _offline_tavily(
        monkeypatch, requests.Timeout("read timed out"), requests.ConnectionError("reset"), _FakeTavilyReply([GW190521])
    )
    assert "GW190521" in tool.forward("GW190521 masses")
    assert len(requests_made) == 3


def test_tavily_search_raises_the_last_error_when_every_try_fails(monkeypatch):
    tool, requests_made, _ = _offline_tavily(monkeypatch, *[_FakeTavilyReply(status_code=503)] * 4)
    with pytest.raises(TavilySearchError, match="HTTP 503"):
        tool.forward("GW190521 masses")
    assert len(requests_made) == 4


@pytest.mark.parametrize("status_code", [400, 401, 432, 433])
def test_tavily_search_fails_at_once_when_tavily_refuses_it(monkeypatch, status_code):
    tool, requests_made, _ = _offline_tavily(monkeypatch, _FakeTavilyReply(status_code=status_code, detail="refused"))
    with pytest.raises(TavilySearchError, match=f"HTTP {status_code}: .*refused"):
        tool.forward("GW190521 masses")
    assert len(requests_made) == 1


def test_tavily_search_errors_never_show_the_key(monkeypatch):
    tool, _, _ = _offline_tavily(monkeypatch, *[requests.Timeout("timed out")] * 4)
    with pytest.raises(TavilySearchError) as error:
        tool.forward("GW190521 masses")
    assert "tvly-test-key" not in str(error.value)


def test_tavily_search_needs_a_key_and_an_attempt():
    with pytest.raises(ValueError, match="API key"):
        TavilySearchTool("")
    with pytest.raises(ValueError, match="attempts"):
        TavilySearchTool("tvly-test-key", attempts=0)


def test_web_search_uses_tavily_when_asked(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    monkeypatch.setenv("SERPAPI_KEY", "serp-test-key")  # Tavily is chosen even when a SerpAPI key is set
    wrapped = tool_registry.get("web_search")(
        watcher=None, engine="tavily", max_results=5, attempts=2, retry_wait_s=[1.0], blocked_urls=["example.org"]
    )
    tool = wrapped._wrapped
    assert isinstance(tool, TavilySearchTool)
    assert (tool.max_results, tool.attempts, tool.retry_waits_s, tool.blocked_urls) == (5, 2, [1.0], ["example.org"])
    assert tool_registry.get("web_search")(watcher=None, engine="tavily")._wrapped.blocked_urls == ["sciflawbench"]


def test_web_search_with_tavily_needs_its_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        tool_registry.get("web_search")(watcher=None, engine="tavily")


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("TAVILY_API_KEY"), reason="an api key is required to hit the real api")
def test_live_tavily_search():
    res = TavilySearchTool(os.environ["TAVILY_API_KEY"]).forward("GW190521 binary black hole merger masses")
    assert res.startswith("## Search Results")
    assert "sciflawbench" not in res.lower()
