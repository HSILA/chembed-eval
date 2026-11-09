"""Driver-side client for the deployed chembed-eval-models Modal app.

Used by backend/precompute.py to run the resumable corpus precompute and by
later milestones for live query encoding. Requires the app deployed:

    cd backend && modal deploy modal_app.py
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from .protocol import MODELS

APP_NAME = "chembed-eval-models"

_CLASS_NAMES = {"base": "ModelServerBase", "prog": "ModelServerProg"}
_EXPECTED_REVISIONS = {k: v["revision"] for k, v in MODELS.items()}


def remote_server(model_key: str):
    """Hydrate the deployed class handle for this model."""
    import modal

    if model_key not in _CLASS_NAMES:
        raise ValueError(f"unknown model key {model_key!r}")
    cls = modal.Cls.from_name(APP_NAME, _CLASS_NAMES[model_key])
    return cls()


def encode_texts(model_key: str, texts: Sequence[str], kind: str) -> np.ndarray:
    """Encode via the deployed server; verify revision matches the protocol."""
    server = remote_server(model_key)
    revision, emb = server.encode.remote(list(texts), kind=kind)
    if revision != _EXPECTED_REVISIONS[model_key]:
        raise RuntimeError(
            f"deployed {model_key} revision {revision} != protocol {_EXPECTED_REVISIONS[model_key]}"
        )
    vec = np.asarray(emb, dtype=np.float32)
    if vec.shape != (len(texts), 768):
        raise RuntimeError(f"unexpected remote embedding shape {vec.shape}")
    return vec


def health(model_key: str) -> dict:
    return remote_server(model_key).health.remote()
