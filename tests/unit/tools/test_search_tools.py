import os

import pytest

from tools.searchtools import (
    ArxivSearchTool,
    ImprovedWikipediaSearchTool,
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


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("SERPAPI_KEY"), reason="an api key is required to hit the real api")
def test_serp_search():
    key = os.environ["SERPAPI_KEY"]
    assert key is not None

    search_tool = SerpAPISearchTool(api_key=key)
    res = search_tool.forward(query="LLM alignment")
    print(res)
    assert "**Search Results:**" in res
