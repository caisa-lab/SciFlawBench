import os

import pytest

from tools.searchtools import (
    ArxivSearchTool,
    ImprovedWikipediaSearchTool,
    SerpAPISearchTool,
)


def test_simple_wiki_query():
    search_tool = ImprovedWikipediaSearchTool(operator="test_suite")
    res = search_tool.forward(query="Anthropic")
    print(res)
    assert "Anthropic" in res
    assert "Amodei" in res
    assert "founded in January 2021" in res


def test_failed_wiki_query():
    search_tool = ImprovedWikipediaSearchTool(operator="test_suite")
    res = search_tool.forward(query="LLM alignment")
    print(res)
    assert "Try the following instead" in res
    assert "Title:" in res
    assert "AI alignment" in res


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


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("SERPAPI_KEY"), reason="an api key is required to hit the real api")
def test_serp_search():
    key = os.environ["SERPAPI_KEY"]
    assert key is not None

    search_tool = SerpAPISearchTool(api_key=key)
    res = search_tool.forward(query="LLM alignment")
    print(res)
    assert "**Search Results:**" in res
