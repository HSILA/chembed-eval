"""Load and validate the filtered evaluation corpus (66,575 passages)."""

import json
from dataclasses import dataclass

import pyarrow.parquet as pq

from .protocol import (
    DATASET_DIR,
    DATASET_MANIFEST,
    EXCLUDED_PAPERS,
    corpus_sha256,
    dataset_revision_digest,
)

PASSAGE_COUNT = 66_575


@dataclass(frozen=True)
class Corpus:
    ids: list[str]
    texts: list[str]  # raw passage text (not yet stripped/prefixed)
    sha256: str
    revision_digest: str


def load_corpus(dataset_dir=DATASET_DIR, manifest_path=DATASET_MANIFEST, excluded_path=EXCLUDED_PAPERS) -> Corpus:
    """Validate corpus hash against the stage-2 manifest, then return id/text lists."""
    parquet = dataset_dir / "corpus" / "train-00000-of-00001.parquet"
    expected = json.loads(manifest_path.read_text())["output_files"]["corpus"]["sha256"]
    actual = corpus_sha256(parquet)
    if actual != expected:
        raise ValueError(
            f"corpus hash mismatch: {actual} != manifest {expected}; refusing to embed"
        )

    table = pq.read_table(parquet, columns=["_id", "text"])
    if table.num_rows != PASSAGE_COUNT:
        raise ValueError(
            f"corpus row count {table.num_rows} != expected {PASSAGE_COUNT}"
        )
    ids = table.column("_id").to_pylist()
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate passage ids in corpus")
    texts = table.column("text").to_pylist()
    if any(t is None or t == "" for t in texts):
        raise ValueError("empty passage text in corpus")

    digest = dataset_revision_digest(manifest_path, excluded_path)
    return Corpus(ids=ids, texts=texts, sha256=actual, revision_digest=digest)
