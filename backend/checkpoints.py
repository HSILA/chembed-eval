"""Durable, atomic, resumable embedding checkpoints.

Layout (all under artifacts/, which is git-ignored):
  artifacts/embeddings/{protocol_id}/{model_key}/
      batches/batch_{start:06d}_{end:06d}.npy      float32 rows in corpus order
      batches/batch_{start:06d}_{end:06d}.json     {model_key, revision, rows, sha256, ...}
      done/{start:06d}.done                        written atomically after batch npy verified

A batch is resumable only if its .done marker exists AND the meta says
"complete" AND the npy verifies against the meta (checksum, row bounds,
protocol id, model revision, shape, norm). All three artifacts are written
in that order (npy → meta → done) so a crash between meta and done leaves a
batch that is NOT resumable; on retry save_batch rewrites the stale files.

load/completed_starts/merge_all reject corruption instead of silently
skipping: a present meta + npy + done that fails verification raises
ValueError (only a literally missing does-marker means "not done yet").
merge_all additionally refuses batches lacking a done marker so a stale
meta/npy pair from a crashed write can never pollute a merge.
"""

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .protocol import MODELS, compute_protocol_id

ARTIFACTS_DIR = Path(os.environ.get("CHEMBED_ARTIFACTS", "artifacts"))

protocol_id = compute_protocol_id  # single shared implementation


def checkpoint_dir(model_key: str, corpus_sha256: str, dataset_digest: str) -> Path:
    return ARTIFACTS_DIR / "embeddings" / protocol_id(corpus_sha256, dataset_digest) / model_key


def batch_npy_path(d: Path, start: int, end: int) -> Path:
    return d / "batches" / f"batch_{start:06d}_{end:06d}.npy"


def batch_meta_path(d: Path, start: int, end: int) -> Path:
    return d / "batches" / f"batch_{start:06d}_{end:06d}.json"


def done_path(d: Path, start: int) -> Path:
    return d / "done" / f"{start:06d}.done"


def invalidate_done(d: Path, start: int) -> None:
    """Remove the done marker BEFORE overwriting a batch's artifacts."""
    done_path(d, start).unlink(missing_ok=True)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: Path, obj: dict) -> None:
    atomic_write_bytes(path, json.dumps(obj, sort_keys=True, indent=1).encode())


def _vec_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def save_batch(
    d: Path,
    start: int,
    end: int,
    vectors,
    meta: dict,
    embed_dim: int | None = None,
) -> None:
    """Persist one batch atomically: npy + meta(status=complete) + done marker.

    Write order: npy → meta → done marker, so the marker only exists once
    both other artifacts are durable. Overwriting an existing batch requires
    invalidate_done(d, start) first; a leftover done marker across a rerun of
    the save itself is an inconsistency and raises.
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"save_batch: expected 2-D vectors, got shape {vectors.shape}")
    if embed_dim is not None and vectors.shape[1] != embed_dim:
        raise ValueError(f"save_batch: vectors dim {vectors.shape[1]} != expected {embed_dim}")
    if not np.isfinite(vectors).all():
        raise ValueError(f"save_batch: non-finite values in batch [{start},{end})")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(
            f"save_batch: unit-norm check failed for [{start},{end}) (norms {norms.min():.4f}..{norms.max():.4f})"
        )
    if vectors.shape[0] != end - start:
        raise ValueError("save_batch: row count does not match bounds")
    if start < 0 or end <= start or (start, end) != (int(start), int(end)):
        raise ValueError(f"save_batch: invalid row bounds [{start},{end})")

    npy, meta_p, done = batch_npy_path(d, start, end), batch_meta_path(d, start, end), done_path(d, start)
    if done.exists():
        raise ValueError(f"save_batch [{start},{end}): done marker present; invalidate it first")

    bio = io.BytesIO()
    np.save(bio, vectors, allow_pickle=False)
    raw = bio.getvalue()
    checksum = _vec_sha256(raw)

    atomic_write_bytes(npy, raw)
    atomic_write_json(
        meta_p,
        {
            **meta,
            "status": "complete",
            "start": start,
            "end": end,
            "rows": end - start,
            "dim": int(vectors.shape[1]),
            "shape": list(vectors.shape),
            "npy_sha256": checksum,
        },
    )
    atomic_write_bytes(done, b"ok")


def _verify_loaded_batch(
    name: str,
    start: int,
    end: int,
    vectors: np.ndarray,
    meta: dict,
    model_key: str,
    corpus_sha256_str: str,
    pid: str,
    embed_dim: int | None,
    require_done: bool,
    d: Path | None = None,
    expected_revision: str | None = None,
) -> None:
    """Raise ValueError on any mismatch or corruption."""
    for f, want in (
        ("model_key", model_key),
        ("corpus_sha256", corpus_sha256_str),
        ("protocol_id", pid),
    ):
        if meta.get(f) != want:
            raise ValueError(f"checkpoint {name}: {f}={meta.get(f)!r} != expected {want!r}")
    expected_revision = expected_revision or MODELS[model_key]["revision"]
    if vectors.dtype != np.float32 or vectors.shape != (end - start, embed_dim or 768):
        raise ValueError(f"checkpoint {name}: wrong dtype or shape")
    if (meta.get("start"), meta.get("end")) != (start, end):
        raise ValueError(f"checkpoint {name}: wrong row bounds")
    if meta.get("status") != "complete":
        raise ValueError(f"checkpoint {name}: status={meta.get('status')!r} != 'complete'")
    if meta.get("revision") is None and "revision" not in meta:
        raise ValueError(f"checkpoint {name}: meta missing 'revision' field")
    if expected_revision is not None and meta.get("revision") != expected_revision:
        raise ValueError(
            f"checkpoint {name}: revision {meta.get('revision')!r} != expected {expected_revision!r}"
        )
    rows, dim = meta.get("rows"), meta.get("dim")
    if rows != end - start or dim != (embed_dim if embed_dim is not None else 768):
        raise ValueError(
            f"checkpoint {name}: meta rows/dim {rows}/{dim} != expected {end - start}/{embed_dim}"
        )
    if list(meta.get("shape", [])) != list(vectors.shape):
        raise ValueError(f"checkpoint {name}: shape {vectors.shape} != meta {meta.get('shape')}")
    raw = io.BytesIO()
    np.save(raw, vectors, allow_pickle=False)
    checksum = _vec_sha256(raw.getvalue())
    if meta.get("npy_sha256") != checksum:
        raise ValueError(
            f"checkpoint {name}: checksum mismatch (npy_sha256={meta.get('npy_sha256')!r}, recomputed {checksum})"
        )
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(vectors).all() or not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(
            f"checkpoint {name}: unit-norm check failed (norms {norms.min():.4f}..{norms.max():.4f}, finite={bool(np.isfinite(vectors).all())})"
        )
    if require_done:
        assert d is not None
        if not done_path(d, start).exists():
            raise ValueError(f"checkpoint {name}: done marker missing")


def load_batch(
    d: Path,
    start: int,
    end: int,
    model_key: str,
    corpus_sha256_str: str,
    pid: str,
    embed_dim: int | None = None,
    expected_revision: str | None = None,
):
    """Load a completed batch; raise on any mismatch. Returns (vectors, meta).

    All three files (npy, meta, done) must exist; meta + npy must verify
    (fields, checksum, row bounds, shape, unit-norm float32). If
    expected_revision is given, meta's `revision` field must equal it.
    """
    npy, meta_p, done = batch_npy_path(d, start, end), batch_meta_path(d, start, end), done_path(d, start)
    if not (npy.exists() and meta_p.exists() and done.exists()):
        return None
    meta = json.loads(meta_p.read_text())
    vectors = np.load(npy, allow_pickle=False)
    _verify_loaded_batch(
        npy.name, start, end, vectors, meta, model_key, corpus_sha256_str, pid, embed_dim,
        require_done=True, d=d, expected_revision=expected_revision,
    )
    return vectors, meta


def completed_starts(
    d: Path,
    ends: dict[int, int],
    model_key: str,
    corpus_sha256_str: str,
    pid: str,
    embed_dim: int | None = None,
) -> dict[int, int]:
    """Map start -> end for batches whose done marker + verified file set are present.

    Raises on corruption (a files-present batch that fails verification is a
    hard error, NOT silently skipped). Only missing/nonexistent artifacts
    mean "not done".
    """
    out: dict[int, int] = {}
    if not d.exists():
        return out
    for start, end in ends.items():
        try:
            loaded = load_batch(d, start, end, model_key, corpus_sha256_str, pid, embed_dim=embed_dim)
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"completed_starts: corrupt batch [{start},{end}): {exc}") from exc
        if loaded is not None:
            out[start] = end
    return out


# ---------------- upload / final-state markers -----------------------------

def upload_marker_path(d: Path, pid: str) -> Path:
    return d / "uploads" / f"{pid}.json"


def write_upload_marker(
    model_key: str,
    d: Path,
    pid: str,
    index_name: str,
    namespace: str,
    success: bool,
) -> None:
    """Record a completed (or explicitly attempted) upload for this protocol.

    `d` is the checkpoint dir (backend.checkpoints.checkpoint_dir(...)) so
    markers land next to the data they describe, whatever ARTIFACTS_DIR
    was at module import (tests patch checkpoint_dir directly).
    """
    path = upload_marker_path(d, pid)
    atomic_write_json(path, {
        "index_name": index_name,
        "namespace": namespace,
        "success": success,
        "model_key": model_key,
        "protocol_id": pid,
    })


def load_upload_markers(d: Path) -> list[dict]:
    """All upload markers for this checkpoint dir (may be empty)."""
    up = d / "uploads"
    if not up.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(up.glob("*.json"))]


def _success_manifest_path(d: Path, pid: str) -> Path:
    return d / "final" / f"{pid}.json"


def write_success_manifest(
    model_key: str, d: Path, pid: str, index_name: str, namespace: str, extra: dict | None = None
) -> None:
    """Persist manifest only after upload + verify both succeeded (into `d`)."""
    payload = {
        "model_key": model_key,
        "protocol_id": pid,
        "index_name": index_name,
        "namespace": namespace,
        **(extra or {}),
    }
    atomic_write_json(_success_manifest_path(d, pid), payload)


def success_manifest(d: Path, pid: str) -> dict | None:
    p = _success_manifest_path(d, pid)
    return json.loads(p.read_text()) if p.exists() else None


# ---------------- sample-prefix adoption for full runs -----------------------


def sample_checkpoint_dir_parent(d: Path) -> Path:
    """Parent of a sample checkpoint dir (relative to full-run `d`)."""
    return d


def adopt_sample_prefix_into_full_run(
    full_d: Path,
    sample_d: Path,
    ranges: dict[int, int],  # full-run 512-aligned ranges within sample bounds
    model_key: str,
    corpus_sha256_str: str,
    pid: str,
    embed_dim: int = 768,
    sample_total: int = 0,
) -> int:
    """Re-index a completed sample run's checkpoint rows into the full-run grid.

    Sample runs use 512-aligned chunks, but their last batch ends at
    `sample_total`, which may fall mid-chunk relative to the full grid. Rows
    are re-saved into full-run batch files so the full grid stays contiguous.
    Rows already present in the full run are NOT overwritten.
    Returns the number of full-run batches filled from the sample prefix.
    """
    filled = 0
    for start, end in sorted(ranges.items()):
        if end > sample_total:
            break
        npy = batch_npy_path(full_d, start, end)
        done = done_path(full_d, start)
        if npy.exists() and done.exists():
            continue  # full run already has this chunk; never overwrite
        loaded = load_batch(sample_d, start, end, model_key, corpus_sha256_str, pid)
        if loaded is None:
            continue  # sample batch incomplete; full run will recompute it
        vectors, sample_meta = loaded
        full_meta = {
            "model_key": model_key,
            "corpus_sha256": corpus_sha256_str,
            "protocol_id": pid,
            "embed_dim": embed_dim,
            "origin": "sample_prefix",
            "revision": MODELS[model_key]["revision"],
        }
        invalidate_done(full_d, start)
        save_batch(full_d, start, end, vectors, full_meta)
        filled += 1
    return filled


ARTIFACTS_DIR = Path(os.environ.get("CHEMBED_ARTIFACTS", "artifacts"))


def merge_all(
    d: Path,
    total: int,
    model_key: str,
    corpus_sha256_str: str,
    pid: str,
    embed_dim: int | None = None,
    expected_revision: str | None = None,
) -> np.ndarray:
    """Merge completed batches in corpus order; error loudly on gaps, corruption,
    wrong model/protocol/revision, missing done markers, or silent skip attempts."""
    manifests = sorted(d.glob("batches/*.json"))
    used: list[tuple[int, int, Path]] = []
    for meta_p in manifests:
        meta = json.loads(meta_p.read_text())
        if meta.get("status") != "complete":
            raise ValueError(f"{meta_p}: status {meta.get('status')!r}; refusing silently-incomplete batch in merge")
        if meta.get("model_key") != model_key or meta.get("protocol_id") != pid:
            raise ValueError(f"{meta_p}: wrong protocol/model: {meta.get('model_key')}, {meta.get('protocol_id')}")
        if expected_revision is not None and meta.get("revision") != expected_revision:
            raise ValueError(f"{meta_p}: revision {meta.get('revision')!r} != expected {expected_revision!r}")
        start, end = meta["start"], meta["end"]
        if not done_path(d, start).exists():
            raise ValueError(f"{meta_p}: done marker missing for [{start},{end}); run started but not finished")
        used.append((start, end, meta_p))

    used.sort()
    out = np.empty((total, embed_dim), dtype=np.float32)
    filled = np.zeros(total, dtype=bool)
    cursor = 0
    for start, end, meta_p in used:
        if start != cursor:
            raise ValueError(f"checkpoint gap: expected next batch at {cursor}, found {start}")
        vecs, meta = load_batch(
            d, start, end, model_key, corpus_sha256_str, pid, embed_dim=embed_dim,
            expected_revision=expected_revision,
        )
        if vecs is None:
            raise ValueError(f"merge_all: batch [{start},{end}) missing; cannot merge")
        # meta from load_batch has verified fields incl. npy_sha256 vs file on disk,
        # but npy was re-loaded here from scratch: re-verify checksum on this array
        raw = io.BytesIO()
        np.save(raw, vecs, allow_pickle=False)
        if meta.get("npy_sha256") != _vec_sha256(raw.getvalue()):
            raise ValueError(f"merge_all: checksum mismatch for {batch_npy_path(d, start, end).name}")
        out[start:end] = vecs
        filled[start:end] = True
        cursor = end
    if cursor != total:
        raise ValueError(f"checkpoint incomplete: covered 0..{cursor} of {total}")
    assert filled.all()
    return out
