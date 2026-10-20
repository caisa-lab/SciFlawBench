import re
import time
import unicodedata
from typing import Literal

import requests
from serpapi import Client
from smolagents import Tool, WikipediaSearchTool


class SerpAPISearchTool(Tool):
    name = "web_search"
    description = """Performs a web search based on your query (think a Google search) then returns the top
    search results."""
    inputs = {"query": {"type": "string", "description": "The search query to perform."}}
    output_type = "string"

    def __init__(self, max_results: int = 8, rate_limit: float | None = 1.0, **kwargs):
        super().__init__()
        self.max_results = max_results
        self.rate_limit = rate_limit
        self._min_interval = 1.0 / rate_limit if rate_limit else 0.0
        self._last_request_time = 0.0

        self.engine = kwargs.pop("engine", "google")
        self.client = Client(**kwargs)

    def forward(self, query: str):
        self._enforce_rate_limit()
        parts = []

        result = self.client.search(q=query, engine=self.engine, num=self.max_results, safe="active")

        if answer := result.get("answer_box"):
            text = answer.get("answer") or answer.get("snippet") or ""
            if text:
                parts.append(f"**direct answer:** {text}")

        if kg := result.get("knowledge_graph"):
            title = kg.get("title")
            desc = kg.get("desc")
            if title or desc:
                parts.append(f"**Knowledge Graph:** {title} - {desc}")

        organic_results = result.get("organic_results", [])[: self.max_results]
        if organic_results:
            lines = ["**Search Results:**"]

            for res in organic_results:
                title = res.get("title", "")
                link = res.get("link", "")
                snippet = res.get("snippet", "")
                lines.append(f"{res.get('position', '')}. [{title}] ({link}) \n {snippet}")
            parts.append("\n".join(lines))

        return "\n\n".join(parts) if parts else "No Results Found."

    def _enforce_rate_limit(self) -> None:
        # No rate limit enforced
        if not self.rate_limit:
            return

        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()


class ImprovedWikipediaSearchTool(WikipediaSearchTool):
    name = "wikipedia_search"
    description = (
        "Fetches a Wikipedia article by title and returns its text (or summary) together with its URL. The query is "
        "matched against article titles, so pass a short, title-like phrase (e.g. 'Ayrton Senna' or 'Great Pyramid "
        "of Giza'), not a sentence or a question. If the exact title is not found, the closest matching article is "
        "fetched automatically, so partial or slightly misspelled titles still work."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": (
                "The Wikipedia article title to fetch. Use a short, title-like phrase; if the exact title is not "
                "found, the closest matching article is fetched automatically."
            ),
        },
    }

    SEARCH_API_URL = "https://{language}.wikipedia.org/w/api.php"
    USER_AGENT_BASE = "AgenticWikipediaSearch/0.1 (https://github.com/ivzx04/SciFlawBench; "

    def __init__(self, operator: str, content_type: str = "text", extract_format: str = "WIKI", language: str = "en"):
        user_agent = self.USER_AGENT_BASE + f"{operator})"

        super().__init__(
            user_agent=user_agent, language=language, content_type=content_type, extract_format=extract_format
        )

    def forward(self, query: str) -> str:
        if self.content_type not in ("summary", "text"):
            return "Invalid content type. Use either 'summary' or 'text'"

        try:
            resolved = self._resolve(query)
            if resolved is None:
                return self._none_found(query)

            page, text = resolved
            title = page.title
            url = page.fullurl

            resolved_note = ""
            if self._norm_title(title) != self._norm_title(query):
                resolved_note = f'*(exact title: "{title}")*\n\n'

            return f"**Wikipedia Page:** {title}\n\n{resolved_note}**Content:** {text}\n\n**Read More:** {url}"
        except Exception as e:
            return f"Error searching wikipedia page: {str(e)}"

    def _resolve(self, query: str):
        """Return `(page, content)` for the exact title or the closest matching article.

        Tries the exact title first; on a miss (or an empty page) it falls back to
        Wikipedia's full-text search and returns the first candidate that exists, so
        near-miss titles (e.g. "LLM alignment" -> "AI alignment") resolve in a single call
        instead of costing the agent another step.
        """
        exact = self.wiki.page(query)
        if exact.exists():
            text = self._content(exact)
            if str(text or "").strip():
                return exact, text

        for title in self._suggest_titles(query, limit=5):
            candidate = self.wiki.page(title)
            if not candidate.exists():
                continue
            text = self._content(candidate)
            if str(text or "").strip():
                return candidate, text
        return None

    def _content(self, page) -> str:
        """The page's summary or full text, per `content_type`."""
        return page.summary if self.content_type == "summary" else page.text

    @staticmethod
    def _norm_title(text: str) -> str:
        """Normalize a title for equality comparison (case/accents/whitespace)."""
        decomposed = unicodedata.normalize("NFKD", str(text))
        ascii_ish = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
        return re.sub(r"[^a-z0-9]+", " ", ascii_ish.lower()).strip()

    def _suggest_titles(self, query: str, limit: int = 8) -> list[str]:
        """Return up to `limit` Wikipedia titles ranked by full-text relevance for `query`.

        Uses the MediaWiki search API (`list=search`), which performs a *full-text* search
        (not just a title prefix), so descriptive queries such as "LLM alignment" still
        surface a genuine article ("AI alignment") to fall back on.
        """
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        }
        try:
            resp = requests.get(
                self.SEARCH_API_URL.format(language=self.language),
                params=params,
                headers={"User-Agent": self.user_agent},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []

        hits = data.get("query", {}).get("search") if isinstance(data, dict) else None
        if not isinstance(hits, list):
            return []
        return [hit["title"] for hit in hits if isinstance(hit, dict) and hit.get("title")]

    def _none_found(self, query: str) -> str:
        """Not-found message, with title suggestions when the search returns any."""
        suggestions = self._suggest_titles(query)
        if suggestions:
            joined = ", ".join(repr(title) for title in suggestions)
            return f"No wikipedia pages found matching '{query}'. Try one of these exact article titles: {joined}."
        return (
            f"No wikipedia pages found matching '{query}'. "
            "Use a short, exact article title (e.g. 'Ayrton Senna'), not a sentence."
        )


class ArxivSearchTool(Tool):
    name = "arxiv_search"
    description = (
        "Tool which searches arxiv for papers that have abstracts or titles that fuzzily match the provided query."
        "returns a list of possible article titles along with their time of publication, arXiv ID, abstract, and a link"
        "to where the markdown of the article is hosted if it exists."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "",
        },
        "sort": {
            "type": "string",
            "description": "Sorting order of results. needs to be one of the following: relevance|newest|oldest",
        },
        "fromYear": {
            "type": "integer",
            "description": "lower bound for the year of release of articles in the results. Minimum value: 1991",
        },
    }
    output_type = "string"

    BASE_URL = "https://arcxiv.org/api/agent/search"
    USER_AGENT_BASE = "AgenticArXivSearch/0.1 (https://github.com/ivzx04/SciFlawBench; "

    def __init__(self, operator: str):
        super().__init__()
        self.user_agent = self.USER_AGENT_BASE + f"{operator})"

    def forward(self, query: str, sort: Literal["relevance", "newest", "oldest"], fromYear: int) -> str:
        data = self._make_request(query, sort, fromYear)
        return "**Results:** \n[\n" + ", \n\n".join(data) + "\n]"

    def _make_request(self, query: str, sortType: Literal["relevance", "newest", "oldest"], fromYear: int) -> list[str]:

        params = {
            "q": query,
            "sort": sortType,
            "fromYear": fromYear,
            "pageSize": 5,  # approximately how many results actually show up
        }

        try:
            resp = requests.get(self.BASE_URL, params=params, headers={"User-Agent": self.user_agent})
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return [f"Error searching for Arxiv papers: {e}"]

        if "hits" not in data:
            return []

        data_list = [
            f"{item['publishedAt']} - id: {item['arxivId']} - **title:** {item['title']}  \n **abstract:**"
            f"{item['abstract']} \n article markdown: {item['links'].get('markdown', '**NOT AVAILABLE**')}"
            for item in data["hits"]
        ]

        return data_list
