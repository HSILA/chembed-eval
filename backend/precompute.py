"""Resumable precompute driver: stream corpus chunks to Modal GPU servers,
persist durable checkpoints locally, then upload into Pinecone.

Usage:
  python -m backend.precompute plan            # batch plan + estimate
  python -m backend.precompute precompute --model base --sample 200
  python -m backend.precompute precompute --model base            # full
  python -m backend.precompute upload --model base
  python -m backend.precompute status --model base

Sample runs write checkpoints to a separate `sample-<n>` directory under the
model's checkpoint root, so they cannot pollute full-run batch boundaries.
Because every sample chunk is 512-aligned (CHUNK-grid prefix), a later full
run can safely adopt those rows by re-sharding them into the shared grid via
adopt_sample_prefix_into_full_run().

Pinecone namespace ALWAYS equals the protocol_id, isolating experiment
versions (sample runs append "-sample" so they never mix with full-run rows).
"""

import argparse
import time
from pathlib import Path

import numpy as np

from .checkpoints import (
    checkpoint_dir,
    completed_starts,
    merge_all,
    save_batch,
    write_upload_marker,
    write_success_manifest,
)
from .client import remote_server
from .corpus import load_corpus
from .protocol import (
    EMBED_DIM,
    MODELS,
    PINECONE_INDEX_FOR_KEY,
    compute_protocol_id,
    dataset_revision_digest,
)

CHUNK = 512  # passages per GPU call; ~ bounded for 8192-token worst case upload size
UPSERT_BATCH = 200

UPLOAD_MAX_RETRIES = 5


def batch_ranges(total: int, chunk: int = CHUNK) -> list[tuple[int, int]]:
    return [(s, min(s + chunk, total)) for s in range(0, total, chunk)]


def _clean_eval_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "ChEmbed-overlap-audit" / "clean-eval"


def _digest() -> str:
    here = _clean_eval_dir()
    return dataset_revision_digest(here / "dataset_manifest.json", here / "excluded_papers.txt")


def sample_checkpoint_dir(model_key: str, sha: str, digest: str, sample_total: int) -> Path:
    return checkpoint_dir(model_key, sha, digest) / f"sample-{sample_total}"


def _load_env():
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env.local")
    load_dotenv(_clean_eval_dir() / ".env")
    load_dotenv(_clean_eval_dir().parent / ".env")


def _remote_class(model_key: str):
    """Hydrate the deployed Modal class for this model (app must be deployed)."""
    return remote_server(model_key)


def encode_with_resume(model_key: str, sample: int | None, dry_run: bool = False) -> None:
    corpus = load_corpus()
    dataset_digest = _digest()
    sha = corpus.sha256
    pid = compute_protocol_id(sha, dataset_digest)
    full_total = len(corpus.ids)
    is_sample = sample is not None and min(sample, full_total) < full_total
    total = min(sample, full_total) if sample is not None else full_total
    d = (
        sample_checkpoint_dir(model_key, sha, dataset_digest, total)
        if is_sample
        else checkpoint_dir(model_key, sha, dataset_digest)
    )

    ranges = batch_ranges(total)
    done = completed_starts(d, dict(ranges), model_key, sha, pid, embed_dim=EMBED_DIM)
    remaining = [(s, e) for s, e in ranges if s not in done]
    print(f"model={model_key} protocol={pid} sample={is_sample} dir={d}")
    print(f"total {total} passages | done batches {len(done)} | remaining {len(remaining)}")
    if not remaining:
        print("nothing to do")
        return
    if dry_run:
        for s, e in remaining[:3]:
            print(f"  would encode [{s}, {e})")
        return

    remote = _remote_class(model_key)
    texts = corpus.texts
    meta_common = {
        "model_key": model_key,
        "revision": MODELS[model_key]["revision"],
        "protocol_id": pid,
        "corpus_sha256": sha,
        "dataset_digest": dataset_digest,
        "embed_dim": EMBED_DIM,
    }
    t0 = time.time()
    n_done = 0
    # First-loop progress values (populated after each finished batch).
    elapsed_s = 0.0
    rate = 0.0
    for s, e in remaining:
        res = remote.encode_chunk.remote(
            texts=texts[s:e],
            start=s,
            end=e,
            corpus_sha256=sha,
            dataset_digest=dataset_digest,
        )
        if res["model_key"] != model_key or res["protocol_id"] != pid or (res["start"], res["end"]) != (s, e):
            raise RuntimeError(f"modal/chunk mismatch: {dict((k, res[k]) for k in ('model_key', 'protocol_id', 'start', 'end'))} vs {(model_key, pid, s, e)}")
        if res["revision"] != MODELS[model_key]["revision"]:
            raise RuntimeError(
                f"deployed revision {res['revision']} != protocol {MODELS[model_key]['revision']}"
            )
        # provenance assertion for the progressive model's remote-code weight
        # resolution (informational for base): first resolved paths must come
        # from the pinned snapshot dir
        if res.get("resolved"):
            rev = MODELS[model_key]["revision"]
            for p in res["resolved"]:
                assert f"/snapshots/{rev}/" in p.replace("\\", "/"), f"provenance: {p}"
        save_batch(d, s, e, res["vectors"], meta_common, embed_dim=EMBED_DIM)
        n_done += 1
        elapsed_s = time.time() - t0
        rate = n_done / max(elapsed_s, 1e-9)
        remaining_n = len(remaining) - n_done
        eta = remaining_n / max(rate, 1e-9) if remaining_n else 0.0
        print(
            f"batch [{s:6d},{e:6d}) ok | {n_done}/{len(remaining)}"
            f" | {rate:.2f} bat/s | elapsed {elapsed_s:.0f}s | ETA {int(eta/60)}m"
        )
    print(
        f"pipeline complete for {model_key}: {n_done} batches, "
        f"{elapsed_s:.0f}s, mean {rate:.2f} batch/s"
    )


def _upsert_with_retry(index, ns: str, items: list[dict]) -> dict:
    """Bounded exponential-backoff retry up to UPLOAD_MAX_RETRIES."""
    last: Exception | None = None
    for attempt in range(UPLOAD_MAX_RETRIES):
        try:
            return index.upsert(vectors=items, namespace=ns)
        except Exception as exc:  # noqa: BLE001 - Pinecone SDK raises varied types
            last = exc
        sleep_s = min(2**attempt, 30)
        print(f"  upsert retriable error ({type(last).__name__}); backoff {sleep_s}s")
        time.sleep(sleep_s)
    raise RuntimeError(f"upsert failed after {UPLOAD_MAX_RETRIES} attempts") from last


def _wait_for_count(index, ns: str, expected: int, attempts: int = 30, wait_s: int = 10) -> int:
    """Poll describe_index_stats until the namespace count reaches `expected`
    (Pinecone counts are eventually consistent right after upserts)."""
    got = -1
    for attempt in range(attempts):
        stats = index.describe_index_stats()
        got = int(stats["namespaces"].get(ns, {}).get("vector_count", 0))
        if got == expected:
            return got
        print(f"  count {got}/{expected} (attempt {attempt + 1}/{attempts})")
        time.sleep(wait_s)
    raise RuntimeError(f"count never reached {expected} after {attempts} polls; final {got}")


def _load_index(model_key: str):
    import os

    from pinecone import Pinecone

    _load_env()
    index_name = PINECONE_INDEX_FOR_KEY[model_key]
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    description = pc.describe_index(index_name)
    if description.dimension != EMBED_DIM or description.metric != "cosine":
        raise RuntimeError(f"Wrong index dimension or metric: {index_name}")
    return pc.Index(index_name), index_name


def _check_index_config(index, index_name: str) -> None:
    """Fail fast if the index dimension/metric don't match the protocol."""
    try:
        desc = index.describe()
    except AttributeError:
        return  # stub/test double without describe()
    dim, metric = desc.get("dimension"), desc.get("metric")
    if dim is not None and int(dim) != EMBED_DIM:
        raise RuntimeError(f"index {index_name} dimension {dim} != protocol {EMBED_DIM}")
    if metric is not None and metric not in ("cosine", "dotproduct"):
        # normalized vectors make the two equivalent; anything else is wrong
        raise RuntimeError(f"index {index_name} metric {metric!r} != cosine/dotproduct")


def _manifest_common(sha: str, dataset_digest: str, pid: str, total: int, is_sample: bool) -> dict:
    """Nonsecret provenance fields shared by every success manifest."""
    from .protocol import (
        MODELS,
        EMBED_DIM,
        MAX_SEQ_LENGTH,
        NORMALIZE,
        POOLING,
        DTYPE,
        dataset_revision_digest,
        encoder_source_fingerprint,
    )

    return {
        "protocol_id": pid,
        "passage_count": total,
        "sample": total if is_sample else None,
        "corpus_sha256": sha,
        "dataset_revision_digest": dataset_digest,
        "encoding": {
            "models": {k: dict(MODELS[k]) for k in sorted(MODELS)},
            "prompts": {  # protocol PROMPTS, session-copied (kept nonsecret)
                "query": "search_query: ",
                "document": "search_document: ",
            },
            "max_seq_length": MAX_SEQ_LENGTH,
            "pooling": POOLING,
            "normalize": NORMALIZE,
            "dtype": DTYPE,
            "embed_dim": EMBED_DIM,
        },
        "encoder_source_sha256": encoder_source_fingerprint(),
    }


def upload(model_key: str, sample: int | None = None) -> None:
    corpus = load_corpus()
    dataset_digest = _digest()
    sha = corpus.sha256
    pid = compute_protocol_id(sha, dataset_digest)
    full_total = len(corpus.ids)
    is_sample = sample is not None and min(sample, full_total) < full_total
    total = min(sample, full_total) if sample is not None else full_total
    d = (
        sample_checkpoint_dir(model_key, sha, dataset_digest, total)
        if is_sample
        else checkpoint_dir(model_key, sha, dataset_digest)
    )
    # Namespace equals the protocol_id so experiment versions never mix;
    # sample runs isolate under "<pid>-sample".
    ns = f"{pid}-sample" if is_sample else pid

    revision = MODELS[model_key]["revision"]
    vectors = merge_all(d, total, model_key, sha, pid, embed_dim=EMBED_DIM, expected_revision=revision)
    index, index_name = _load_index(model_key)
    _check_index_config(index, index_name)

    stats = index.describe_index_stats()
    already = int(stats["namespaces"].get(ns, {}).get("vector_count", 0))
    if already == total:
        print(f"index {index_name} ns={ns} already has {total} vectors; verifying sampled search then done")
        verify_search(index, ns, corpus, vectors)
        write_upload_marker(model_key, d, pid, index_name, ns, success=True)
        write_success_manifest(
            model_key, d, pid, index_name, ns,
            {**_manifest_common(sha, dataset_digest, pid, total, is_sample),
             "vectors": total},
        )
        print(f"upload verified; manifest in artifacts/embeddings/{pid}/{model_key}/final/{pid}.json")
        return

    ids = corpus.ids
    if vectors.shape != (total, EMBED_DIM):
        raise RuntimeError(f"vector shape {vectors.shape} != ({total}, {EMBED_DIM})")
    metadata = {
        "model": MODELS[model_key]["hf_name"],
        "model_revision": MODELS[model_key]["revision"],
        "protocol_id": pid,
    }
    for i in range(0, total, UPSERT_BATCH):
        batch = vectors[i : i + UPSERT_BATCH]
        items = [
            {"id": ids[i + j], "values": batch[j].tolist(), "metadata": dict(metadata)}
            for j in range(len(batch))
        ]
        resp = _upsert_with_retry(index, ns, items)
        if (i // UPSERT_BATCH) % 20 == 0:
            print(f"upserted {i + len(batch)}/{total}", resp.get("upserted_count"))
    got = _wait_for_count(index, ns, total)
    print(f"uploaded {got} vectors to {index_name}/{ns}")
    verify_search(index, ns, corpus, vectors)
    write_upload_marker(model_key, d, pid, index_name, ns, success=True)
    write_success_manifest(
        model_key, d, pid, index_name, ns,
        {**_manifest_common(sha, dataset_digest, pid, total, is_sample),
         "vectors": got},
    )
    print(f"upload verified; manifest in artifacts/embeddings/{pid}/{model_key}/final/{pid}.json")


def verify_search(index, ns, corpus, vectors) -> None:
    """Sampled exact-vs-Pinecone search check.

    Ties must not fail the check: `vectors @ v` is cosine (rows unit-norm);
    identical passages produce identical vectors, so exact rank order among
    ties is nondeterministic. A hit passes if its similarity matches the best
    exact similarity within tolerance AND its id is inside the exact top-1
    tie set (fall back to strict top-1 for clear-cut winners).
    """
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(vectors), size=min(3, len(vectors)), replace=False)
    ok = True
    for i in idxs:
        v = vectors[i]
        got = index.query(vector=v.tolist(), top_k=3, namespace=ns, include_values=False)
        matches = got.get("matches", [])
        if not matches:
            raise RuntimeError(f"query {int(i)}: pinecone returned no matches")
        top_id = matches[0]["id"]
        sims = vectors @ v
        order = np.argsort(-sims)
        best_sim = float(sims[order[0]])
        tol = 1e-4
        ties = [corpus.ids[j] for j in np.flatnonzero(np.abs(sims - best_sim) < 1e-6)]
        top1_ok = top_id in ties
        sim_ok = abs(float(matches[0].get("score", 0.0)) - best_sim) <= tol
        ok = ok and top1_ok and sim_ok
        print(
            f"query {int(i)}: pinecone top {top_id} | exact top {corpus.ids[order[0]]} | "
            f"top1_in_ties={top1_ok} sim {float(matches[0].get('score', 0.0)):.6f} vs exact {best_sim:.6f}"
        )
    if not ok:
        raise RuntimeError("pinecone sampled search disagrees with exact search over saved vectors")


def status() -> None:
    corpus = load_corpus()
    digest = _digest()
    pid = compute_protocol_id(corpus.sha256, digest)
    ranges = {s: e for s, e in batch_ranges(len(corpus.ids))}
    for key in MODELS:
        d = checkpoint_dir(key, corpus.sha256, digest)
        done = completed_starts(d, ranges, key, corpus.sha256, pid, embed_dim=EMBED_DIM)
        print(f"{key}: {len(done)}/{len(ranges)} batches done ({sum(e - s for s, e in done.items())} passages)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["plan", "precompute", "upload", "status"])
    p.add_argument("--model", choices=list(MODELS), default="base")
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.cmd == "status":
        status()
        return
    if args.cmd == "plan":
        ranges = batch_ranges(66_575)
        print(f"{len(ranges)} chunks of <= {CHUNK}")
        sample = load_corpus().texts[:200]
        print(f"sample passage max len {max(len(x) for x in sample)} chars")
        return
    if args.cmd == "precompute":
        encode_with_resume(args.model, args.sample, args.dry_run)
        return
    if args.cmd == "upload":
        upload(args.model, args.sample)
        return


if __name__ == "__main__":
    main()
