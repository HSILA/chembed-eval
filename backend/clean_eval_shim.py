"""Pinning shim, extracted verbatim in behavior from
ChEmbed-overlap-audit/clean-eval/03_evaluate.py::_pin_chembed_cached_file.

Patch transformers.utils.hub.cached_file / cached_files so every call
targeting a given repo id is redirected to the pinned revision with the
token from the seeded environment; all other repo ids pass through
untouched.  This is required because ChEmbed's remote modeling code
(modeling_hf_nomic_bert.py) resolves its own weights via cached_file()
without revision or token: tokenless calls 401 on the gated repo, and
even passing calls silently resolve against mutable main.
"""

import os
from typing import Callable

RESOLVED_PATHS: dict[str, list[str]] = {}


def pin_chembed_cached_file(model_name: str, revision: str, token: str) -> None:
    import transformers.utils.hub as hub

    if model_name in RESOLVED_PATHS:
        return  # already pinned in this process

    resolved: list[str] = []
    RESOLVED_PATHS[model_name] = resolved

    def adapt(kwargs: dict) -> None:
        if kwargs.get("revision") in (None, "", "main"):
            kwargs["revision"] = revision
        if not kwargs.get("token"):
            kwargs["token"] = token

    orig_cached_file = hub.cached_file
    orig_cached_files = hub.cached_files

    def patched_cached_file(path_or_repo_id, filename, **kwargs):
        if str(path_or_repo_id) == model_name:
            adapt(kwargs)
            out = orig_cached_file(path_or_repo_id, filename, **kwargs)
            if out:
                resolved.append(str(out))
            return out
        return orig_cached_file(path_or_repo_id, filename, **kwargs)

    def patched_cached_files(path_or_repo_id, **kwargs):
        if str(path_or_repo_id) == model_name:
            adapt(kwargs)
        return orig_cached_files(path_or_repo_id, **kwargs)

    hub.cached_file = patched_cached_file
    hub.cached_files = patched_cached_files


def assert_provenance(model_name: str, revision: str) -> None:
    """Fail loudly unless every weight file resolved for the repo came from
    the pinned snapshot directory (no main-branch fallback)."""
    snapshot = RESOLVED_PATHS.get(model_name)
    assert snapshot, f"{model_name}: no files resolved via pinned cached_file shim"
    for p in snapshot:
        assert f"/snapshots/{revision}/" in p.replace("\\", "/"), (
            f"{model_name}: {p} is not inside the pinned revision snapshot"
        )
