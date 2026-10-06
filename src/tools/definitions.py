import logging
import os

from smolagents import DuckDuckGoSearchTool, VisitWebpageTool

from core.registry import Registry
from tools.base import WrappedTool
from tools.misc import CalculatorTool, CurrentTimeTool, JsonFinalAnswerTool
from tools.searchtools import (
    ArxivSearchTool,
    ImprovedWikipediaSearchTool,
    SerpAPISearchTool,
)

# tool registry global object that will keep track of string -> factory mappings for building our tools
tool_registry = Registry("Tool")

logger = logging.getLogger()


WEB_TOOL_NAMES = {
    "web_search",
    "visit_webpage",
    "wikipedia_search",
    "arxiv_search",
}

@tool_registry.register("web_search")
def make_web_search_tool(
    watcher, max_results: int = 8, rate_limit: float = 1.0, engine: str = "duckduckgo"
) -> WrappedTool:
    if os.environ.get("SERPAPI_KEY", None) is not None and engine != "duckduckgo":
        key = os.environ["SERPAPI_KEY"]
        kwargs = {"max_results": 8, "rate_limit": 1.0, "engine": engine, "api_key": key}
        return WrappedTool(wrapped_tool=SerpAPISearchTool(**kwargs), watcher=watcher)

    logger.info("web_search tool created with DuckDuckGo search, which has worse results than engines accessible via \
                serpapi ")

    return WrappedTool(wrapped_tool=DuckDuckGoSearchTool(max_results, rate_limit), watcher=watcher)


@tool_registry.register("wikipedia_search")
def make_wiki_search_tool(watcher, operator: str = "OPERATOR EMAIL NOT SET") -> WrappedTool:
    logger.info("api operator email was not set. This should be done for best compliance with wikipedia api rules")
    return WrappedTool(wrapped_tool=ImprovedWikipediaSearchTool(operator), watcher=watcher)


@tool_registry.register("arxiv_search")
def make_arxiv_search_tool(watcher, operator: str = "OPERATOR EMAIL NOT SET") -> WrappedTool:
    return WrappedTool(wrapped_tool=ArxivSearchTool(operator), watcher=watcher)


@tool_registry.register("visit_webpage")
def make_visit_webpage_tool(watcher) -> WrappedTool:
    return WrappedTool(wrapped_tool=VisitWebpageTool(), watcher=watcher)


@tool_registry.register("calculator")
def make_calculator_tool(watcher) -> WrappedTool:
    return WrappedTool(wrapped_tool=CalculatorTool(), watcher=watcher)


@tool_registry.register("json_answer_tool")
def make_json_formatting_tool(watcher, required_fields: list[str] | None = None) -> WrappedTool:
    required_fields = required_fields or []
    return WrappedTool(wrapped_tool=JsonFinalAnswerTool(required_keys=required_fields), watcher=watcher)


@tool_registry.register("current_time")
def make_current_time_tool(watcher) -> WrappedTool:
    return WrappedTool(wrapped_tool=CurrentTimeTool(), watcher=watcher)
