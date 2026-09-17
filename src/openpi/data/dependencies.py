"""Dependency checks for the optional legacy recording service."""

import importlib.util


def require_legacy_dependencies() -> None:
    missing = [name for name in ("bson", "transforms3d") if importlib.util.find_spec(name) is None]
    if missing:
        raise ImportError(
            f"Legacy data-cloud ingestion needs {', '.join(missing)}. "
            "Install the optional dependencies with `uv sync --extra legacy-data`."
        )
