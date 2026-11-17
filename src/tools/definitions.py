import logging
import os

from smolagents import VisitWebpageTool

from core.registry import Registry
from tools.base import WrappedTool
from tools.cosmology import CosmologyTool
from tools.filetools import DEFAULT_MAX_CHARS, DEFAULT_MAX_ROWS, ReadFileTool
from tools.misc import CalculatorTool, CurrentTimeTool, JsonFinalAnswerTool
from tools.numerics import NumericsTool
from tools.searchtools import (
    ArxivSearchTool,
    ImprovedWikipediaSearchTool,
    RetryingDuckDuckGoSearchTool,
    SerpAPISearchTool,
)
from tools.significance import SignificanceTool
from tools.sky import SkyTool

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


@tool_registry.register("web_search")
def make_web_search_tool(
    watcher,
    max_results: int = 8,
    rate_limit: float = 1.0,
    engine: str = "duckduckgo",
    attempts: int = 4,
    retry_wait_s: float | list[float] = (5.0, 15.0, 30.0),
    min_interval_s: float = 2.0,
) -> WrappedTool:
    # `attempts`, `retry_wait_s` and `min_interval_s` apply to the DuckDuckGo search only: it fails often at random,
    # more so when several agents search at once (issues #59, #60), so all agents keep one pace and a failed search is
    # tried again, with growing waits, before the agent sees an error.
    if os.environ.get("SERPAPI_KEY", None) is not None and engine != "duckduckgo":
        key = os.environ["SERPAPI_KEY"]
        kwargs = {"max_results": 8, "rate_limit": 1.0, "engine": engine, "api_key": key}
        return WrappedTool(wrapped_tool=SerpAPISearchTool(**kwargs), watcher=watcher)

    logger.info(
        "web_search tool created with DuckDuckGo search, which has worse results than engines accessible via \
                serpapi "
    )

    search = RetryingDuckDuckGoSearchTool(
        max_results, rate_limit, attempts=attempts, retry_wait_s=retry_wait_s, min_interval_s=min_interval_s
    )
    return WrappedTool(wrapped_tool=search, watcher=watcher)


@tool_registry.register("wikipedia_search")
def make_wiki_search_tool(watcher, operator: str = "OPERATOR EMAIL NOT SET") -> WrappedTool:
    logger.info("api operator email was not set. This should be done for best compliance with wikipedia api rules")
    return WrappedTool(wrapped_tool=ImprovedWikipediaSearchTool(operator), watcher=watcher)


@tool_registry.register("arxiv_search")
def make_arxiv_search_tool(watcher, operator: str = "OPERATOR EMAIL NOT SET") -> WrappedTool:
    return WrappedTool(wrapped_tool=ArxivSearchTool(operator), watcher=watcher)


@tool_registry.register("visit_webpage")
def make_visit_webpage_tool(watcher) -> WrappedTool:
    return WrappedTool(
        wrapped_tool=VisitWebpageTool(max_output_length=_VISIT_WEBPAGE_MAX_OUTPUT_LENGTH), watcher=watcher
    )


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
