import hashlib
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.config import RunConfig
from core.task_ids import find_repo_root
from evaluation.judge import judge_template_sha256

MANIFEST_FILENAME = "run_manifest.json"
MANIFEST_SHA_FILENAME = "run_manifest.sha256"

#: Agent prompt templates shipped with the harness; changes to these change results.
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "agents" / "prompts"

#: Environment variable that downgrades reproducibility failures to warnings.
OVERRIDE_ENV_VAR = "NO_REPRODUCIBILITY_GUARANTEES"


# Hashing primitives.
def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialisation, used as the input to every hash here."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))


def sha256_hex(data: str | bytes) -> str:
    """SHA-256 hex digest of a string or bytes."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def hash_file(path: str | Path) -> str | None:
    """SHA-256 of a file's bytes, or `None` when the file cannot be read."""
    try:
        return sha256_hex(Path(path).read_bytes())
    except OSError:
        return None


def hash_rows(rows: list[Any]) -> str:
    """Deterministic SHA-256 over parsed task rows (pydantic models or plain dicts)."""
    serialised = [row.model_dump(mode="json") if hasattr(row, "model_dump") else row for row in rows]
    return sha256_hex(canonical_json(serialised))


def hash_directory(directory: Path) -> str | None:
    """Combined SHA-256 over `{relative path: sha256}` for every file under `directory`."""
    directory = Path(directory)
    if not directory.is_dir():
        return None
    entries = {
        path.relative_to(directory).as_posix(): hash_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
    return sha256_hex(canonical_json(entries))


# Environment provenance.
def git_info() -> dict[str, Any]:
    """Git revision info, degrading gracefully outside a git repository."""

    def _run(*cmd: str) -> str | None:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        except Exception:
            return None

    commit = _run("git", "rev-parse", "HEAD")
    status = _run("git", "status", "--porcelain") if commit is not None else None
    return {
        "commit": commit,
        "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status.strip()),
    }


def dependency_versions() -> dict[str, str]:
    """Installed distribution name -> version, sorted (the environment fingerprint)."""
    from importlib import metadata

    versions = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name") if dist.metadata else None
        if name:
            versions[name] = dist.version
    return dict(sorted(versions.items()))


def environment_info() -> dict[str, Any]:
    """Platform / interpreter info for the run manifest."""
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
    }


def reproducibility_override() -> bool:
    """Whether the user opted out of the reproducibility guarantees."""
    return os.getenv(OVERRIDE_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


# Run signature.
def prompts_fingerprint() -> str:
    """SHA-256 over the agent prompt templates that ship with the harness."""
    return hash_directory(PROMPTS_DIR) or sha256_hex("")


def assets_fingerprint(task_file: Path) -> tuple[str | None, str | None]:
    """`(assets_dir, sha256)` for the task set's local asset files, or `(None, None)`.

    Local assets change what a task can read, so they are part of reproducibility.
    The directory follows the repo convention (`<repo-root>/data/tasks/assets`).
    """
    root = find_repo_root(Path(task_file).parent)
    if root is None:
        return None, None
    assets_dir = root / "data" / "tasks" / "assets"
    if not assets_dir.is_dir():
        return None, None
    return str(assets_dir), hash_directory(assets_dir)


def judge_fingerprint(conf: RunConfig) -> dict[str, Any] | None:
    """
    The judge's identity and prompt template, or `None` when the run has no judge.

    Nothing secret goes in here: the API key, and even the *name* of the environment variable
    that holds it, are left out (matching how the model configuration is recorded).
    """
    if conf.judge is None:
        return None
    return {
        "model_id": conf.judge.model_id,
        "api_base": conf.judge.api_base,
        "extra_kwargs": conf.judge.extra_kwargs,
        "prompt_sha256": judge_template_sha256(),
    }


def build_run_signature(conf: RunConfig, tasks: list[Any]) -> dict[str, Any]:
    """
    The subset of a run's configuration that must match for a resume to be valid.

    Deliberately excludes knobs that do not affect results (log paths, concurrency,
    log verbosity) and anything secret. Dependencies are recorded in the manifest
    but kept out of the signature so a run can be resumed on a rebuilt environment.
    """
    _, assets_sha = assets_fingerprint(conf.task_file)
    return {
        "task_set_sha256": hash_rows(tasks),
        "n_tasks": len(tasks),
        "task_file_sha256": hash_file(conf.task_file),
        "assets_sha256": assets_sha,
        "model": {
            "provider": conf.model.provider,
            "model_id": conf.model.model_id,
            "api_base": conf.model.api_base,
            "extra_kwargs": conf.model.extra_kwargs,
            "code_block_tags": list(conf.model.code_block_tags) if conf.model.code_block_tags else None,
            "model_max_context": conf.model.model_max_context,
        },
        "judge": judge_fingerprint(conf),
        "tools": sorted((tool.model_dump(mode="json") for tool in conf.tool_configs), key=lambda t: t["tool_name"]),
        "prompts_sha256": prompts_fingerprint(),
        "closed_book": conf.closed_book,
        "artifact_spill": conf.artifact_spill,
        "tool_output_max_chars": conf.tool_output_max_chars,
        "repetitions_per_task": conf.repetitions_per_task,
        "task_timeout_s": conf.task_timeout_s,
    }


# Manifest construction, persistence and validation.
def build_run_manifest(*, conf: RunConfig, tasks: list[Any], run_signature: dict[str, Any]) -> dict[str, Any]:
    """Build the immutable run manifest recorded before a fresh run starts."""
    assets_dir, assets_sha = assets_fingerprint(conf.task_file)
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "reproducibility_override": reproducibility_override(),
        "git": git_info(),
        "environment": environment_info(),
        "task_set": {
            "file": str(conf.task_file),
            "n_tasks": len(tasks),
            "file_sha256": hash_file(conf.task_file),
            "task_set_sha256": run_signature["task_set_sha256"],
            "assets_dir": assets_dir,
            "assets_sha256": assets_sha,
        },
        "model": run_signature["model"],
        "judge": run_signature["judge"],
        "tools": run_signature["tools"],
        "prompts": {"dir": str(PROMPTS_DIR), "sha256": run_signature["prompts_sha256"]},
        "budget": {
            "repetitions_per_task": conf.repetitions_per_task,
            "task_timeout_s": conf.task_timeout_s,
            "max_concurrent": conf.max_concurrent,
        },
        "dependencies": dependency_versions(),
        "run_signature": run_signature,
    }


def manifest_sha256(manifest: dict[str, Any]) -> str:
    """SHA-256 over the canonical form of a manifest."""
    return sha256_hex(canonical_json(manifest))


def _atomic_write_json(filepath: Path, data: Any) -> None:
    """Write JSON atomically via a temporary file + rename."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp = filepath.with_name(filepath.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, filepath)


def save_run_manifest(output_dir: Path, manifest: dict[str, Any]) -> Path:
    """Write the manifest and its checksum once; manifests are immutable."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / MANIFEST_FILENAME
    if filepath.exists():
        raise FileExistsError(f"Refusing to overwrite existing run manifest: {filepath} (manifests are immutable).")
    _atomic_write_json(filepath, manifest)
    (output_dir / MANIFEST_SHA_FILENAME).write_text(manifest_sha256(manifest) + "\n", encoding="utf-8")
    return filepath


def load_run_manifest(output_dir: Path) -> dict[str, Any] | None:
    """Load a previously written run manifest, if present and readable."""
    try:
        with open(Path(output_dir) / MANIFEST_FILENAME, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def load_manifest_sha256(output_dir: Path) -> str | None:
    """Load the checksum recorded alongside the manifest, if present."""
    try:
        return (Path(output_dir) / MANIFEST_SHA_FILENAME).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def validate_resume(output_dir: Path, current_signature: dict[str, Any]) -> list[str]:
    """
    Check that a checkpoint can be resumed reproducibly.

    Returns a list of human-readable problems; an empty list means the checkpoint
    matches the current configuration and is safe to continue.
    """
    problems: list[str] = []

    manifest = load_run_manifest(output_dir)
    if manifest is None:
        return ["checkpoint has no run_manifest.json, so its checksums cannot be verified."]

    recorded_sha = load_manifest_sha256(output_dir)
    if recorded_sha is None:
        problems.append("checkpoint has no run_manifest.sha256, so the manifest cannot be verified.")
    elif recorded_sha != manifest_sha256(manifest):
        problems.append("run_manifest.json does not match its recorded checksum (the manifest was modified).")

    recorded_signature = manifest.get("run_signature")
    if not isinstance(recorded_signature, dict):
        problems.append("manifest has no 'run_signature', so the run cannot be compared.")
        return problems

    differing = sorted(
        key
        for key in set(recorded_signature) | set(current_signature)
        if recorded_signature.get(key) != current_signature.get(key)
    )
    if differing:
        problems.append(f"run signature differs from the manifest (changed: {', '.join(differing)}).")

    return problems
