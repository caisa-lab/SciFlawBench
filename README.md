# SciFlawBench Harness

## descriptions
This repository is mean to be a harness from which to run the sciflaw benchmark for models within an agentic context at the university of Bonn. It is built on top of smolagents and has a multiprocess architechture to allow for **blazingly** fast running of the benchmark

## code repository overview

```
src/
├── core/
│   ├── config.py               # base types used for parsing the config (w/ pydantic)
│   ├── events.py               # defines event watchers and base event primitives
│   ├── manager.py              # runtime manager code that handles dispatching subprocesses
│   ├── registry.py             # registry code which is reused for agent and tool registry (handles mapping string -> obj factory)
│   └── tasks.py                # definition for task primiteves and contains teh run_task function used for actual task runs
├── agents/
│   ├── prompts/
│   │   ├── code_agent.yaml        # default prompt associated with coding agent
│   │   └── tool_agent.yaml        # default prompt associated with tool calling agent
│   ├── prompts.py              # right now just containes a load prompt file
│   ├── definitions.py          # definitions of base agents to be used in testing and agent registry
│   └── base.py                 # contains build_agent function
├── evaluation/
│   ├── print_report.py         # contains logic pertaining to converting reports into markdown
│   ├── definitions.py          # definitions of verifiers and the verifier registry
│   └── base.py                 # defines basic necessary structures for validation code
├── models/
│   └── base.py                 # defines how to build a model and model wrapper
├── tools/
│   ├── misc.py                 # miscellaneous custom tool classes that get registered
│   ├── definitions.py          # all custom tool definitions and wrapper/registry definiotion for use in pipeline
│   ├── searchtools.py          # contains the searchtools available to the agents: (arxiv, SerpAPI, wikipedia)
│   └── base.py                 # containes wrapper for use on all tools
└── main.py                     # main entrypoint for running testing harness
```

## How to use

### Installation

### Installing through pip
1. clone the repository and cd in
```bash
git clone git@github.com:ivzx04/SciFlawBenchHarness.git && cd SciFlawBenchHarness
```
2. Create a virtual environment for this project and enter the environment (optional)
```bash
python -m venv <name-of-your-venv>  && source <name-of-your-venv>/bin/activate
```
3. pip install the enviornment and dependencies
```bash
pip install .
```

### Installing with uv
1. clone the repository and cd in
```bash
git clone git@github.com:ivzx04/SciFlawBenchHarness.git && cd SciFlawBenchHarness
```
2. install the required packages
```bash
uv sync
```
respectively as a developer run
```bash
uv sync --all-extras
```

### Configuring the benchmark
1. Write the config file in config.json with your specific model access credentials/settings

     - The config struct roughly corresponds to the following:
        a. Configurations for the entire run

        ```python
        class RunConfig(BaseModel):
            """
            class which stores all the information needed to provision a benchmark run (also gets read from the config)
            """
            model: ModelConfig
            task_file: Path
            tool_configs: list[ToolDef] = Field(default_factory=list)
            log_path: Path = Path("logs/")
            max_concurrent: int = 4         # default max concurrent task running processes

            repititions_per_task: int=3
            logging_level: int = 20
            task_timeout_s: int = 60 * 15 # 15 minute timeout for tasks before they get killed by the runtime manager
            restarting: bool | None = None  # if you want to restearting on a specific dir specify the path and set to True
        ```

        b. Configurations for the Model

        ```python

        class ModelConfig(BaseModel):
            """
            class which stores all the information needed to provision a model (gets read from the config)

            also acts as a typing mechanism thoruhg pydantic to verify things were correctly specified
            """
            provider: Literal["litellm", "openai_server", "hf_api", "fake_model"]
            model_id: str
            api_key_env: str
            api_base: str | None = None
            extra_kwargs: dict = Field(default_factory=dict)

            # this is kept in the model config because it generally is a model dependant field to be configured
            code_block_tags: tuple[str,str] | None = None

            _api_key: str = PrivateAttr() # populated via environment using api_key_env
        ```

        c. Tool definitions that modify base tool behaviour for the entire run (kwargs vary by tool, passed in through the tool_configs in [1]):
        ```python
        class ToolDef(BaseModel):
            tool_name: str
            kwargs: dict = Field(default_factory=dict)
        ```

    - tool overrides modify the behaviour of the tool for the entirety of the run for all agents
    - all of these BaseModel classes correspond directly to writeable json which should hopefully help for understanding how things can be expressed

2. export the api key environment variables associated with your model providers

3. run src/main.py with your config path
```bash
python src/main.py --config /path/to/your/config
```

4. (hint) You can see all the configurations for your run without actually running the benchmark by using the --dry flag


## Check out our Google Colab from which you can run this as well

https://colab.research.google.com/drive/1ctDfb7he22O-ipqqhIxSmfM40fEOTWXI?usp=sharing

## FAQ

1. how do I define a test task to try and run?

    For an arbitrary task one only needs to define 3 fields for the harness to run:
        "task_id": int
        "task": str
        "agent_id": Literal_string["code_agent", "tool_agent"]

2. What provider should i use for a given model / how should i configure my settings for this ?

    This largely depends on what kind of API you are hitting where the model is hosted. The most standard provider type
    is the openai_server, which is compatible with many sorts of providers including openai itself, vllm, etc.

    Currently this code base supports two other types of providers as well, those being litellm and huggingfaces own api.

    Almost all configuration options for these specific providers are documented at the following link:
    https://deepwiki.com/huggingface/smolagents/4.2-api-based-models

    and can be specified via the extra_kwargs section in the model config.
