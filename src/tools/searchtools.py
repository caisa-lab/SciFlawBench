import fcntl
import re
import tempfile
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
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
        "Searches arXiv for papers whose title, authors or abstract match the query, or looks papers up by their "
        "arXiv ID (e.g. 1803.07687). Returns up to five papers with their time of publication, arXiv ID, title, "
        "abstract and a link to the paper's arXiv page."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "Words from a title, author names or a topic, or one or more arXiv IDs",
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

    # arXiv's own API, which answers in Atom XML (https://info.arxiv.org/help/api/user-manual.html)
    BASE_URL = "https://export.arxiv.org/api/query"
    USER_AGENT_BASE = "AgenticArXivSearch/0.1 (https://github.com/caisa-lab/SciFlawBench; "
    # Seconds to wait for a response, so a slow or unreachable server cannot stall a run.
    TIMEOUT_S = 10
    MAX_RESULTS = 5
    # arXiv asks API clients for at most one request every 3 seconds. When asked more often it answers 429 (or 503),
    # refuses the connection, or answers too slowly, so such a request is tried again after a pause, up to ATTEMPTS
    # times in all: at worst about ATTEMPTS * TIMEOUT_S + (ATTEMPTS - 1) * RETRY_WAIT_S seconds, 40 s by default.
    MIN_INTERVAL_S = 3.0
    ATTEMPTS = 3
    RETRY_WAIT_S = 5.0
    # Every task runs in its own process, so the 3 s are kept across processes: the time of the last request is kept
    # in this file, which a request locks while it waits for its turn.
    LOCK_FILE = Path(tempfile.gettempdir()) / "sciflawbench_arxiv_search.lock"

    _ARXIV_ID = re.compile(r"(?<![\w.])(\d{4}\.\d{4,5}|[a-z][a-z-]*(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?(?![\w.])")
    _WORD = re.compile(r"\w[\w.-]*")
    _SORTS = {
        "relevance": ("relevance", "descending"),
        "newest": ("submittedDate", "descending"),
        "oldest": ("submittedDate", "ascending"),
    }
    _ATOM = "{http://www.w3.org/2005/Atom}"
    _RETRY_STATUSES = (429, 500, 502, 503, 504)

    def __init__(self, operator: str):
        super().__init__()
        self.user_agent = self.USER_AGENT_BASE + f"{operator})"

    def forward(self, query: str, sort: Literal["relevance", "newest", "oldest"], fromYear: int) -> str:
        data = self._make_request(query, sort, fromYear)
        return "**Results:** \n[\n" + ", \n\n".join(data) + "\n]"

    def _make_request(self, query: str, sortType: Literal["relevance", "newest", "oldest"], fromYear: int) -> list[str]:
        """The papers for a query, formatted; an arXiv ID in the query is looked up, other words are searched."""
        try:
            ids = self._ARXIV_ID.findall(query)
            if ids:
                entries = self._fetch({"id_list": ",".join(ids), "max_results": self.MAX_RESULTS})
            else:
                entries = self._search(query, sortType, fromYear)
        except Exception as e:
            return [f"Error searching for Arxiv papers: {e}"]
        return [self._format(entry) for entry in entries]

    def _search(self, query: str, sortType: str, fromYear: int) -> list[ET.Element]:
        """Papers matching all the words, in the asked order; failing that, any of them, the most relevant first."""
        words = [word.strip(".-") for word in self._WORD.findall(query)]
        words = [word for word in words if len(word) > 1]
        if not words:
            return []
        try:
            year = int(fromYear)
        except (TypeError, ValueError):
            year = 1991
        since = f" AND submittedDate:[{year}01010000 TO 299912312359]" if year > 1991 else ""
        sort_by, order = self._SORTS.get(sortType, self._SORTS["relevance"])
        tries = [(" AND ", sort_by, order)]
        if len(words) > 1:
            # sorted by relevance: sorted by date, papers matching one common word would come first
            tries.append((" OR ", "relevance", "descending"))
        for joiner, sort_by, order in tries:
            terms = joiner.join(f"all:{word}" for word in words)
            params = {"search_query": f"({terms}){since}", "sortBy": sort_by, "sortOrder": order}
            entries = self._fetch({**params, "max_results": self.MAX_RESULTS})
            if entries:
                return entries
        return []

    def _fetch(self, params: dict) -> list[ET.Element]:
        """The entries of one API request, tried again while arXiv is busy, refuses it or does not answer in time."""
        error: Exception = RuntimeError("no request made")
        for attempt in range(self.ATTEMPTS):
            if attempt:
                time.sleep(self.RETRY_WAIT_S)
            self._keep_interval()
            try:
                resp = requests.get(
                    self.BASE_URL, params=params, headers={"User-Agent": self.user_agent}, timeout=self.TIMEOUT_S
                )
            except (requests.ConnectionError, requests.Timeout) as exc:  # a refused connection, or no answer in time
                error = exc
                continue
            if resp.status_code in self._RETRY_STATUSES:
                error = requests.HTTPError(f"{resp.status_code} from the arXiv API (it is busy or limiting requests)")
                continue
            resp.raise_for_status()
            return self._entries(resp.text)
        raise error

    def _keep_interval(self) -> None:
        """Wait until MIN_INTERVAL_S have passed since the last request of any arxiv_search tool on this machine."""
        with open(self.LOCK_FILE, "a+", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)  # released when the file is closed
            lock.seek(0)
            try:
                last = float(lock.read().strip())
            except ValueError:  # a new or unreadable file: no request yet
                last = float("-inf")
            wait = self.MIN_INTERVAL_S - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            lock.seek(0)
            lock.truncate()
            lock.write(repr(time.time()))

    def _entries(self, feed: str) -> list[ET.Element]:
        entries = ET.fromstring(feed).findall(self._ATOM + "entry")
        for entry in entries:
            if "/api/errors" in self._text(entry, "id"):  # arXiv reports a malformed query as an entry
                raise ValueError(self._text(entry, "summary") or "the arXiv API rejected the query")
        return entries

    def _text(self, entry: ET.Element, tag: str) -> str:
        node = entry.find(self._ATOM + tag)
        return " ".join((node.text or "").split()) if node is not None else ""

    def _format(self, entry: ET.Element) -> str:
        arxiv_id = re.sub(r"v\d+$", "", self._text(entry, "id").rsplit("/abs/", 1)[-1])
        return (
            f"{self._text(entry, 'published')} - id: {arxiv_id} - **title:** {self._text(entry, 'title')}  \n"
            f" **abstract:**{self._text(entry, 'summary')} \n link: https://arxiv.org/abs/{arxiv_id}"
        )
