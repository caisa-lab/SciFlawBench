from dataclasses import dataclass

from smolagents import CodeAgent, LogLevel, ToolCallingAgent

from agents.definitions import AgentDef, agent_registry
from agents.prompts import load_prompt_templates
from core.config import ModelConfig
from core.events import EventWatcher
from models.base import build_model
from models.memory import MemoryOffloader
from tools.artifacts import (
    DEFAULT_TOOL_OUTPUT_MAX_CHARS,
    WIKIPEDIA_OUTPUT_CAP,
    ArtifactStore,
    install_output_cap,
)
from tools.artifacttools import ListArtifactsTool, ReadArtifactTool, SearchArtifactTool
from tools.base import ToolDef, WrappedTool, resolve_tools
from tools.definitions import CLOSED_BOOK_ALLOWED_TOOLS, tool_registry


@dataclass
class BuiltAgent:
    """
    class that just holds all of the information about a provisioned agent / agentic system for ease of passing around
    later
    """

    watcher: EventWatcher
    agent: CodeAgent | ToolCallingAgent
    definition: AgentDef


# Extra Python packages a task may let the code agent import. smolagents authorises submodules one by
# one, so a package expands to the submodules a scientific computation needs; anything not listed
# (e.g. numpy.f2py, numpy.testing, numpy.distutils, which can call external programs) stays forbidden.
AUTHORIZED_IMPORT_EXPANSIONS: dict[str, list[str]] = {
    "numpy": ["numpy", "numpy.linalg", "numpy.random", "numpy.fft", "numpy.polynomial"],
}

# The executor blocks open(), so numpy's file readers/writers must not reopen that door (np.load with
# allow_pickle=True can also execute arbitrary code). smolagents' DANGEROUS_FUNCTIONS check does not
# cover method-style calls such as np.load(...), so the functions themselves are replaced with stubs
# in the task's process. Not covered: ndarray methods (arr.tofile, arr.dump) on numpy's C type.
NUMPY_FILE_IO_FUNCTIONS = [
    "load",
    "save",
    "savez",
    "savez_compressed",
    "loadtxt",
    "savetxt",
    "genfromtxt",
    "fromfile",
    "fromregex",
    "memmap",
]


def expand_authorized_imports(requested: list[str] | None) -> list[str]:
    """Expand a task's requested packages into the module list smolagents should authorise."""
    expanded: list[str] = []
    for name in requested or []:
        for module in AUTHORIZED_IMPORT_EXPANSIONS.get(name, [name]):
            if module not in expanded:
                expanded.append(module)
    return expanded


def block_numpy_file_io() -> None:
    """Replace numpy's module-level file I/O functions with stubs that refuse to run (idempotent)."""
    import numpy

    for name in NUMPY_FILE_IO_FUNCTIONS:
        original = getattr(numpy, name, None)
        if original is None or getattr(original, "_file_io_blocked", False):
            continue

        def _blocked(*args, _name=name, **kwargs):
            raise PermissionError(f"numpy.{_name} is disabled for the agent: no file access")

        _blocked._file_io_blocked = True
        setattr(numpy, name, _blocked)


# TODO: make this thing work for multi agent setups via specifying children and parents
def build_agent(
    agent_id: str,
    model_conf: ModelConfig,
    watcher: EventWatcher,
    tool_overrides: dict[str, ToolDef],
    extra_tools: list[ToolDef | str],
    closed_book: bool = False,
    artifact_store: ArtifactStore | None = None,
    tool_output_max_chars: int = DEFAULT_TOOL_OUTPUT_MAX_CHARS,
    memory_offloader: MemoryOffloader | None = None,
    authorized_imports: list[str] | None = None,
) -> BuiltAgent:
    """
    builds an agent from the specified agent_id and model conf along with the associated watcher class
    note this builds all of its tools and its model configuration here

    Args:
        agent_id (str): string specifying the agent / agentic setup to be built
        model_conf (ModelConfig): description of what is needed to build the model associated with this agent
        watcher (EventWatcher): the watcher associated with this agent, its model instance and its tools
        TODO: add tool_overrides to docstring with explanation
        extra_tools (List[ToolDef | str]): definition of extra_tools to be passed on a task basis
        closed_book (bool): give the agent no harness tools at all (its own knowledge only)
        artifact_store (ArtifactStore | None): per-task store for oversized tool outputs. When
            given, every content tool's output is capped (spilling the overflow to the store)
            and the `search_artifact`/`read_artifact`/`list_artifacts` tools are added. When
            None, caps still apply but overflow is truncated in place instead of spilled.
        tool_output_max_chars (int): characters a tool may return before its output is capped.
        memory_offloader (MemoryOffloader | None): spills context evicted by the trimmer to the task's memory
            file (read back through the artifact tools); None just replaces evicted text with a note.
        authorized_imports (list[str] | None): extra packages the code agent may import (e.g. ["numpy"]);
            see AUTHORIZED_IMPORT_EXPANSIONS. Ignored by the tool-calling agent.

    """
    definition = agent_registry.create(agent_id)
    prompts = load_prompt_templates(definition.prompt_path)
    overrides = resolve_tools(definition.tools, tool_overrides)

    all_tools = overrides + [t if isinstance(t, ToolDef) else ToolDef(tool_name=t) for t in extra_tools]
    if closed_book:
        all_tools = [t for t in all_tools if t.tool_name in CLOSED_BOOK_ALLOWED_TOOLS]
    tools = [tool_registry.create(t.tool_name, watcher=watcher, **t.kwargs) for t in all_tools]

    # Context hygiene for large outputs: cap every content tool's observation, spilling the
    # overflow to the per-task artifact store when one is available. Closed-book runs get no
    # tools and therefore nothing to cap.
    store = artifact_store if getattr(artifact_store, "enabled", False) else None
    if tools:
        install_output_cap(
            tools,
            store,
            default_cap=tool_output_max_chars,
            overrides={"wikipedia_search": WIKIPEDIA_OUTPUT_CAP},
        )
        if store is not None:
            tools = tools + [
                WrappedTool(wrapped_tool=SearchArtifactTool(store), watcher=watcher),
                WrappedTool(wrapped_tool=ReadArtifactTool(store), watcher=watcher),
                WrappedTool(wrapped_tool=ListArtifactsTool(store), watcher=watcher),
            ]

    # Evicted context is only spilled when the agent has the retrieval tools to read it back.
    retrievable = bool(tools) and store is not None
    model = build_model(model_conf, watcher, offloader=memory_offloader if retrievable else None)

    # TODO: for indentations, we can also use pre-commit with black, I can set that up
    # TODO: make this an elif with else for raise ValueError in case agent_type is neither code nor search
    if definition.agent_type == "code":
        kwargs = {}
        if model_conf.code_block_tags is not None:
            kwargs["code_block_tags"] = model_conf.code_block_tags
        extra_imports = expand_authorized_imports(authorized_imports)
        if extra_imports:
            kwargs["additional_authorized_imports"] = extra_imports
            if "numpy" in extra_imports:
                block_numpy_file_io()

        agent = CodeAgent(
            tools=tools,
            model=model,
            prompt_templates=prompts,
            max_steps=definition.max_steps,
            verbosity_level=LogLevel.OFF,
            **kwargs,
        )
    else:
        agent = ToolCallingAgent(
            tools=tools,
            model=model,
            prompt_templates=prompts,
            max_steps=definition.max_steps,
            verbosity_level=LogLevel.OFF,
        )

    return BuiltAgent(
        watcher=watcher,
        agent=agent,
        definition=definition,
    )
