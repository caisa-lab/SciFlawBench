import time
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
        "Searches Wikipedia and returns the full text (or summary) of the requested article along with its URL. "
        "IMPORTANT: the query is matched against Wikipedia article titles, so pass a short title-like query (e.g. "
        "'Ayrton Senna' or 'Great Pyramid of Giza'), not a sentence or a question. If no title is found, returns "
        "suggested titles present in wikipedia."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The wikipedia article title to getch. Use a short title-lilke phrase, if its not found,"
            "close matches will be suggested",
        },
    }

    BASE_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"
    USER_AGENT_BASE = "AgenticWikipediaSearch/0.1 (https://github.com/ivzx04/SciFlawBench; "

    def __init__(self, operator: str, content_type: str = "text", extract_format: str = "WIKI"):

        user_agent = self.USER_AGENT_BASE + f"{operator})"

        super().__init__(user_agent=user_agent, content_type=content_type, extract_format=extract_format)

    def forward(self, query: str) -> str:
        try:
            page = self.wiki.page(query)

            if not page.exists():
                return self._none_found(query)

            title = page.title
            url = page.fullurl

            match self.content_type:
                case "summary":
                    text = page.summary
                case "text":
                    text = page.text
                case _:
                    return "Invalid content type. Use either 'summary' or 'text'"

            return f" **Wikipedia Page:** {title}\n\n**Content:** {text}\n\n**Read More:** {url}"
        except Exception as e:
            return f"Error searching wikipedia page: {str(e)}"

    def _none_found(self, query: str) -> str:
        suggestions = self._find_suggestions(query)

        if suggestions:
            joined = "\n[" + ", \n".join(title for title in suggestions) + "]"
            return f"No wikipedia pages found matching {query}. Try the following instead: {joined}"

        return f"No wikipedia Pages found matching {query} and no suggestions matched either..."

    "Try using specific title names instead of natural language queries."

    def _find_suggestions(self, query: str) -> list[str]:
        params = {
            "q": query,
            "limit": 8,
        }
        try:
            resp = requests.get(self.BASE_URL, params=params, headers={"User-Agent": self.user_agent})
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []

        if "pages" not in data:
            return []

        return [f"Title: [{page['title']}] short description - {page['description']}" for page in data["pages"]]


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
        except Exception:
            return ["Error searching for Arxiv papers: {str(e)}]"]

        data = resp.json()

        if "hits" not in data:
            return []

        data_list = [
            f"{item['publishedAt']} - id: {item['arxivId']} - **title:** {item['title']}  \n **abstract:**"
            f"{item['abstract']} \n article markdown: {item['links'].get('markdown', '**NOT AVAILABLE**')}"
            for item in data["hits"]
        ]

        return data_list
