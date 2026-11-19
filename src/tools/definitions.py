import logging
import os
from collections.abc import Iterable
from typing import Any

from core.registry import Registry
from tools.base import WrappedTool
from tools.cosmology import CosmologyTool
from tools.filetools import DEFAULT_MAX_CHARS, DEFAULT_MAX_ROWS, ReadFileTool
from tools.misc import CalculatorTool, CurrentTimeTool, JsonFinalAnswerTool
from tools.numerics import NumericsTool
from tools.searchtools import (
    ArxivSearchTool,
    FallbackSearchTool,
    ImprovedWikipediaSearchTool,
    RetryingDuckDuckGoSearchTool,
    SerpAPISearchTool,
    TavilySearchTool,
)
from tools.significance import SignificanceTool
from tools.sky import SkyTool
from tools.webpage import PublicVisitWebpageTool

# tool registry global object that will keep track of string -> factory mappings for building our tools
tool_registry = Registry("Tool")

logger = logging.getLogger()

# `VisitWebpageTool` truncates internally at 40 000 characters.  We disable that (by raising
# the limit far above any real page) so the artifact output cap sees the *full* text and can
# spill it to disk; the cap still truncates for the agent's context.
_VISIT_WEBPAGE_MAX_OUTPUT_LENGTH = 10**9


# Tools kept in a closed-book run. Closed-book means the model may only draw on its own
# knowledge: no web search, no local file reads, no calculator, no task-specific tools.
# The agent's built-in Python execution and the `final_answer` tool are supplied by
# smolagents itself and are not affected by this list.
CLOSED_BOOK_ALLOWED_TOOLS: frozenset[str] = frozenset()


@tool_registry.register("read_file")
def make_read_file_tool(watcher, max_rows: int = DEFAULT_MAX_ROWS, max_chars: int = DEFAULT_MAX_CHARS) -> WrappedTool:
    return WrappedTool(wrapped_tool=ReadFileTool(max_rows=max_rows, max_chars=max_chars), watcher=watcher)


#: The search engine of `web_search` when a config names none.
DEFAULT_SEARCH_ENGINE = "tavily"


@tool_registry.register("web_search")
def make_web_search_tool(
    watcher,
    max_results: int = 8,
    rate_limit: float = 1.0,
    engine: str = DEFAULT_SEARCH_ENGINE,
    attempts: int = 4,
    retry_wait_s: float | list[float] = (5.0, 15.0, 30.0),
    min_interval_s: float = 2.0,
    blocked_urls: list[str] | tuple[str, ...] = TavilySearchTool.BLOCKED_URLS,
    fallback: bool = True,
) -> WrappedTool:
    # Tavily (issue #66) is the default: it uses an API key instead of scraping search engines, which block the
    # DuckDuckGo search often, more so when several agents search at once (issues #59, #60). With `fallback` on,
    # DuckDuckGo is the backup: it answers a query whose Tavily search fails, the rest of the run once Tavily refuses
    # the key or its credits are used up, and every query when TAVILY_API_KEY is not set. Both searches try a failed
    # search again (`attempts`, `retry_wait_s`) and drop results whose URL contains one of `blocked_urls`; DuckDuckGo
    # searches of all agents keep one pace (`min_interval_s`).
    def duckduckgo() -> RetryingDuckDuckGoSearchTool:
        return RetryingDuckDuckGoSearchTool(
            max_results,
            rate_limit,
            attempts=attempts,
            retry_wait_s=retry_wait_s,
            min_interval_s=min_interval_s,
            blocked_urls=blocked_urls,
        )

    if engine == "tavily":
        key = os.environ.get("TAVILY_API_KEY")
        if key:
            tavily = TavilySearchTool(
                key, max_results, rate_limit, attempts=attempts, retry_wait_s=retry_wait_s, blocked_urls=blocked_urls
            )
            search = FallbackSearchTool(tavily, duckduckgo()) if fallback else tavily
            return WrappedTool(wrapped_tool=search, watcher=watcher)
        if not fallback:
            raise ValueError(
                "web_search with engine 'tavily' and no fallback needs the TAVILY_API_KEY environment variable"
            )
        logger.warning("TAVILY_API_KEY is not set, so web_search uses DuckDuckGo")
        return WrappedTool(wrapped_tool=duckduckgo(), watcher=watcher)

    if os.environ.get("SERPAPI_KEY", None) is not None and engine != "duckduckgo":
        key = os.environ["SERPAPI_KEY"]
        kwargs = {"max_results": 8, "rate_limit": 1.0, "engine": engine, "api_key": key}
        return WrappedTool(wrapped_tool=SerpAPISearchTool(**kwargs), watcher=watcher)

    return WrappedTool(wrapped_tool=duckduckgo(), watcher=watcher)


def resolved_web_search(tool_configs: Iterable[Any], closed_book: bool = False) -> dict[str, Any] | None:
    """
    Which search `web_search` will use in a run: its engine and backup, as `make_web_search_tool` chooses them.

    It depends on the config and on which API keys are set, so it is recorded in the run signature: a run resumed
    with a different key setup could otherwise switch engines without notice. None for a closed-book run.
    """
    if closed_book:
        return None
    options = next((tool.kwargs for tool in tool_configs if tool.tool_name == "web_search"), {})
    engine = options.get("engine", DEFAULT_SEARCH_ENGINE)
    fallback = options.get("fallback", True)
    if engine == "tavily":
        if os.environ.get("TAVILY_API_KEY"):
            return {"engine": "tavily", "backup": "duckduckgo" if fallback else None}
        return {"engine": "duckduckgo" if fallback else "tavily", "backup": None}
    if engine != "duckduckgo" and os.environ.get("SERPAPI_KEY") is not None:
        return {"engine": f"serpapi:{engine}", "backup": None}
    return {"engine": "duckduckgo", "backup": None}


@tool_registry.register("wikipedia_search")
def make_wiki_search_tool(watcher, operator: str = "OPERATOR EMAIL NOT SET") -> WrappedTool:
    logger.info("api operator email was not set. This should be done for best compliance with wikipedia api rules")
    return WrappedTool(wrapped_tool=ImprovedWikipediaSearchTool(operator), watcher=watcher)


@tool_registry.register("arxiv_search")
def make_arxiv_search_tool(watcher, operator: str = "OPERATOR EMAIL NOT SET") -> WrappedTool:
    return WrappedTool(wrapped_tool=ArxivSearchTool(operator), watcher=watcher)


@tool_registry.register("visit_webpage")
def make_visit_webpage_tool(watcher, blocked_networks: list[str] | tuple[str, ...] = ()) -> WrappedTool:
    # Only public web addresses: no loopback, private, link-local (cloud metadata) or other non-public address, and
    # none in `blocked_networks` (CIDR ranges, for internal services on public addresses), checked on every redirect.
    tool = PublicVisitWebpageTool(max_output_length=_VISIT_WEBPAGE_MAX_OUTPUT_LENGTH, blocked_networks=blocked_networks)
    return WrappedTool(wrapped_tool=tool, watcher=watcher)


@tool_registry.register("calculator")
def make_calculator_tool(watcher) -> WrappedTool:
    return WrappedTool(wrapped_tool=CalculatorTool(), watcher=watcher)


@tool_registry.register("numerics")
def make_numerics_tool(watcher) -> WrappedTool:
    return WrappedTool(wrapped_tool=NumericsTool(), watcher=watcher)


@tool_registry.register("significance")
def make_significance_tool(watcher) -> WrappedTool:
    return WrappedTool(wrapped_tool=SignificanceTool(), watcher=watcher)


@tool_registry.register("cosmology")
def make_cosmology_tool(watcher) -> WrappedTool:
    return WrappedTool(wrapped_tool=CosmologyTool(), watcher=watcher)


@tool_registry.register("sky")
def make_sky_tool(watcher) -> WrappedTool:
    return WrappedTool(wrapped_tool=SkyTool(), watcher=watcher)


@tool_registry.register("json_answer_tool")
def make_json_formatting_tool(watcher, required_fields: list[str] | None = None) -> WrappedTool:
    required_fields = required_fields or []
    return WrappedTool(wrapped_tool=JsonFinalAnswerTool(required_keys=required_fields), watcher=watcher)


@tool_registry.register("current_time")
def make_current_time_tool(watcher) -> WrappedTool:
    return WrappedTool(wrapped_tool=CurrentTimeTool(), watcher=watcher)
