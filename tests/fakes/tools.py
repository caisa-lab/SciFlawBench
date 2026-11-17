from smolagents import Tool

from tools.base import WrappedTool
from tools.definitions import tool_registry


class FakeSearchTool(Tool):
    name = "fake_search"
    description = "deterministic fake search for tests"
    inputs = {"query": {"type": "string", "description": "search query"}}
    output_type = "string"

    def forward(self, query: str) -> str:
        return f"fake results for: {query}"


tool_registry.register("fake_search")(lambda watcher: WrappedTool(wrapped_tool=FakeSearchTool(), watcher=watcher))


class FakeBrokenSearchTool(Tool):
    name = "fake_broken_search"
    description = "a fake search that always fails, like a blocked web search"
    inputs = {"query": {"type": "string", "description": "search query"}}
    output_type = "string"

    def forward(self, query: str) -> str:
        raise RuntimeError("No results found.")


tool_registry.register("fake_broken_search")(
    lambda watcher: WrappedTool(wrapped_tool=FakeBrokenSearchTool(), watcher=watcher)
)
