"""Exact model, corpus, and encoding protocol for the chembed-eval milestone.

All numbers are transcribed from ChEmbed-overlap-audit/clean-eval (03_evaluate.py,
ChEmbedWrapper.py, dataset_manifest.json) so passage preparation and live query
encoding use precisely the evaluated settings.
"""

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel

# Dataset root lives outside this repo, next to it in the workspace.
CLEAN_EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "ChEmbed-overlap-audit" / "clean-eval"
DATASET_DIR = CLEAN_EVAL_DIR / "dataset"
DATASET_MANIFEST = CLEAN_EVAL_DIR / "dataset_manifest.json"
EXCLUDED_PAPERS = CLEAN_EVAL_DIR / "excluded_papers.txt"

MODELS: dict[str, dict[str, str]] = {
    "base": {
        "hf_name": "nomic-ai/nomic-embed-text-v1",
        "revision": "0759316f275aa0cb93a5b830973843ca66babcf5",
    },
    "prog": {
        "hf_name": "BASF-AI/ChEmbed-prog",
        "revision": "2cac1b6e7f9c5a419fcde18dac57efcde0f0da54",
    },
}
MODEL_KEYS: tuple[str, ...] = ("base", "prog")

PROMPTS = {
    "query": "search_query: ",
    "document": "search_document: ",
}

# Original evaluation pipeline settings (clean-eval/03_evaluate.py).
BATCH_SIZE_BASE = 16
BATCH_SIZE_PROG = 8
NUM_PROC = 4
MAX_SEQ_LENGTH = 8192  # from model's sentence_bert_config, Qwen 2048 does not apply
EMBED_DIM = 768
POOLING = "mean"  # 1_Pooling/config.json: mean tokens, not CLS
NORMALIZE = True  # store L2-normalized; cos_sim normalizes both sides anyway
DTYPE = "float32"

PINECONE_INDEX_FOR_KEY = {
    "base": "chembed-eval-base",
    "prog": "chembed-eval-progressive",
}


def dataset_revision_digest(manifest_path: Path, excluded_path: Path) -> str:
    """Content digest of the filtered dataset, replicated from 03_evaluate.py."""
    manifest = json.loads(manifest_path.read_text())
    excluded_hash = hashlib.sha256(excluded_path.read_bytes()).hexdigest()
    digest = hashlib.sha256(
        json.dumps(
            {
                "source_revision": manifest["source_revision"],
                "excluded_sha256": excluded_hash,
                "outputs": manifest["output_files"],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return digest


def corpus_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoder_source_fingerprint() -> str:
    """Fingerprint of the encoder source that determines embedding outputs.

    Hashes backend/encoder.py and backend/clean_eval_shim.py (files shipped to
    Modal via add_local_python_source, so driver and server see identical
    bytes). A missing file is marked "absent" rather than skipped so the
    fingerprint stays deterministic if a file is not present.
    """
    here = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for name in ("encoder.py", "clean_eval_shim.py"):
        p = here / name
        h.update(name.encode())
        h.update(b"\0")
        if p.exists():
            h.update(hashlib.sha256(p.read_bytes()).digest())
        else:
            h.update(b"absent")
    h.update(b"\0")
    return h.hexdigest()


# Bumped whenever the canonical payload below changes shape.
PROTOCOL_ID_VERSION = 2


def compute_protocol_id(corpus_sha256_str: str, dataset_digest_str: str) -> str:
    """Bind checkpoints/uploads to the exact encoding protocol.

    Hashes the canonical model revisions, prompts, max_seq_length (8192),
    pooling, normalization, dtype, encoder source fingerprint, corpus hash,
    and filtered-dataset digest. Any change to model revisions, prompt
    strings, encoding settings, or encoder source yields a new id, cleanly
    separating experiment versions.
    """
    canonical = {
        "version": PROTOCOL_ID_VERSION,
        "models": {key: dict(MODELS[key]) for key in sorted(MODELS)},
        "prompts": dict(PROMPTS),
        "max_seq_length": MAX_SEQ_LENGTH,
        "pooling": POOLING,
        "normalize": NORMALIZE,
        "dtype": DTYPE,
        "encoder_source_sha256": encoder_source_fingerprint(),
    }
    payload = json.dumps(canonical, sort_keys=True).encode()
    digest = hashlib.sha256(payload)
    digest.update(corpus_sha256_str.encode())
    digest.update(b"\0")
    digest.update(dataset_digest_str.encode())
    return digest.hexdigest()[:16]


class Protocol(BaseModel):
    """Versioned config saved into every checkpoint and manifest."""

    models: dict[str, dict[str, str]]
    prompts: dict[str, str]
    max_seq_length: int
    embed_dim: int
    pooling: str
    normalize: bool
    dtype: str
    batch_sizes: dict[str, int]
    corpus_sha256: str
    corpus_num_passages: int
    passage_order: list[str]  # full 66,575 ids, list for readability
    dataset_revision_digest: str
    pinecone_index: dict[str, str]
    protocol_version: int = 1


def build_protocol(
    ids: list[str],
    corpus_hash: str,
    dataset_digest: str,
    num_passages: int,
) -> Protocol:
    return Protocol(
        models=MODELS,
        prompts=PROMPTS,
        max_seq_length=MAX_SEQ_LENGTH,
        embed_dim=EMBED_DIM,
        pooling=POOLING,
        normalize=NORMALIZE,
        dtype=DTYPE,
        batch_sizes={"base": BATCH_SIZE_BASE, "prog": BATCH_SIZE_PROG},
        corpus_sha256=corpus_hash,
        corpus_num_passages=num_passages,
        passage_order=list(ids),
        dataset_revision_digest=dataset_digest,
        pinecone_index=dict(PINECONE_INDEX_FOR_KEY),
    )
