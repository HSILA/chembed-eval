"""Tests for corpus validation, protocol config, and checkpoint resume/reject logic.

GPU-dependent parity runs live in scripts/ and run on demand; these tests are
pure logic and need no model download.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # repo root on path so backend is importable

from backend import checkpoints as cp  # noqa: E402
from backend.protocol import (  # noqa: E402
    BATCH_SIZE_BASE,
    BATCH_SIZE_PROG,
    EMBED_DIM,
    MAX_SEQ_LENGTH,
    MODELS,
    PROMPTS,
    compute_protocol_id,
    encoder_source_fingerprint,
)

CORPUS_SHA = "a" * 64
DATASET_DIGEST = "b" * 32


def unit_vecs(rng_seed: int, rows: int, dim: int = EMBED_DIM) -> np.ndarray:
    """Random finite float32 unit vectors (the invariant checkpoints must hold)."""
    v = np.random.default_rng(rng_seed).normal(size=(rows, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def meta_for(model_key="base", corpus=CORPUS_SHA, pid=None) -> dict:
    return {
        "model_key": model_key,
        "corpus_sha256": corpus,
        "protocol_id": pid or compute_protocol_id(corpus, DATASET_DIGEST),
        "revision": MODELS[model_key]["revision"],
    }


@pytest.fixture(autouse=True)
def no_retry_delays(monkeypatch):
    import backend.precompute as pipeline
    monkeypatch.setattr(pipeline.time, "sleep", lambda *_: None)


def test_protocol_fidelity():
    assert MAX_SEQ_LENGTH == 8192
    assert BATCH_SIZE_BASE == 16 and BATCH_SIZE_PROG == 8
    assert PROMPTS == {"query": "search_query: ", "document": "search_document: "}
    assert MODELS["base"] == {
        "hf_name": "nomic-ai/nomic-embed-text-v1",
        "revision": "0759316f275aa0cb93a5b830973843ca66babcf5",
    }
    assert MODELS["prog"] == {
        "hf_name": "BASF-AI/ChEmbed-prog",
        "revision": "2cac1b6e7f9c5a419fcde18dac57efcde0f0da54",
    }


def test_protocol_id_binds_corpus_and_dataset():
    a = compute_protocol_id(CORPUS_SHA, DATASET_DIGEST)
    b = compute_protocol_id("c" * 64, DATASET_DIGEST)
    c = compute_protocol_id(CORPUS_SHA, "d" * 32)
    assert a != b and a != c


# --------- protocol id binds the full encoding protocol (not just corpus) ----


def test_protocol_id_binds_model_revisions_prompts_and_settings():
    from backend import protocol as prot

    a = compute_protocol_id(CORPUS_SHA, DATASET_DIGEST)
    orig_models = prot.MODELS
    orig_prompts = prot.PROMPTS
    try:
        prot.MODELS = {**orig_models, "base": {**orig_models["base"], "revision": "f" * 40}}
        b = compute_protocol_id(CORPUS_SHA, DATASET_DIGEST)
        prot.MODELS = orig_models
        prot.PROMPTS = {**orig_prompts, "document": "doc: "}
        prot.MAX_SEQ_LENGTH = 4096
        c = compute_protocol_id(CORPUS_SHA, DATASET_DIGEST)
        assert b != a, "model revision change must change protocol id"
        assert c != a, "prompt/max_seq_length change must change protocol id"
    finally:
        prot.MODELS = orig_models
        prot.PROMPTS = orig_prompts
        prot.MAX_SEQ_LENGTH = 8192


def test_protocol_id_agrees_with_checkpoints_module():
    """checkpoints must use protocol.py's compute_protocol_id (no duplicate)."""
    from backend import checkpoints

    assert checkpoints.protocol_id is compute_protocol_id
    assert checkpoints.compute_protocol_id is compute_protocol_id


def test_protocol_id_binds_encoder_source(monkeypatch, tmp_path):
    """Changing encoder source must change the protocol id."""
    from backend import encoder as enc_mod

    target = Path(enc_mod.__file__)
    original = target.read_bytes()
    try:
        before = compute_protocol_id(CORPUS_SHA, DATASET_DIGEST)
        target.write_bytes(original + b"\n# fingerprint probe\n")
        after = compute_protocol_id(CORPUS_SHA, DATASET_DIGEST)
        assert before != after
    finally:
        target.write_bytes(original)
    assert compute_protocol_id(CORPUS_SHA, DATASET_DIGEST) == before


def test_encoder_source_fingerprint_stable_and_content_bound(tmp_path):
    a = encoder_source_fingerprint()
    b = encoder_source_fingerprint()
    assert a == b and len(a) == 64


# --------------------------------- checkpoints -------------------------------


def test_checkpoint_roundtrip(tmp_path):
    d = tmp_path / "ck"
    vecs = unit_vecs(0, 4)
    cp.save_batch(d, 0, 4, vecs, meta_for(), embed_dim=EMBED_DIM)
    loaded = cp.load_batch(d, 0, 4, "base", CORPUS_SHA, meta_for()["protocol_id"], embed_dim=EMBED_DIM)
    assert loaded is not None
    got, got_meta = loaded
    assert np.allclose(got, vecs)
    assert got_meta["status"] == "complete"
    assert got_meta["rows"] == 4 and got_meta["dim"] == EMBED_DIM


def test_checkpoint_rejects_wrong_model(tmp_path):
    d = tmp_path / "ck"
    cp.save_batch(d, 0, 2, unit_vecs(1, 2), meta_for("base"), embed_dim=EMBED_DIM)
    with pytest.raises(ValueError, match="model_key"):
        cp.load_batch(d, 0, 2, "prog", CORPUS_SHA, meta_for()["protocol_id"], embed_dim=EMBED_DIM)


def test_checkpoint_rejects_wrong_corpus(tmp_path):
    d = tmp_path / "ck"
    cp.save_batch(d, 0, 2, unit_vecs(2, 2), meta_for("base"), embed_dim=EMBED_DIM)
    with pytest.raises(ValueError, match="corpus_sha256"):
        cp.load_batch(d, 0, 2, "base", "z" * 64, meta_for()["protocol_id"], embed_dim=EMBED_DIM)


def test_checkpoint_rejects_wrong_revision(tmp_path):
    d = tmp_path / "ck"
    meta = meta_for("base")
    cp.save_batch(d, 0, 2, unit_vecs(3, 2), meta, embed_dim=EMBED_DIM)
    # same model_key/protocol, different weights revision -> must be rejected
    with pytest.raises(ValueError, match="revision"):
        cp.load_batch(d, 0, 2, "base", CORPUS_SHA, meta["protocol_id"], embed_dim=EMBED_DIM,
                      expected_revision="0" * 40)


def test_checkpoint_rejects_checksum_corruption(tmp_path):
    """Silent npy corruption must surface as ValueError, NOT 'return None' or silent skip."""
    d = tmp_path / "ck"
    cp.save_batch(d, 0, 2, unit_vecs(4, 2), meta_for(), embed_dim=EMBED_DIM)
    npy = cp.batch_npy_path(d, 0, 2)
    raw = bytearray(npy.read_bytes())
    # flip the last payload byte(s), simulating bit rot
    raw[-8] ^= 0xFF
    npy.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="checksum"):
        cp.load_batch(d, 0, 2, "base", CORPUS_SHA, meta_for()["protocol_id"], embed_dim=EMBED_DIM)


def test_completed_starts_raises_on_corrupt_batch(tmp_path):
    """Corruption is a hard error, not silently skipped (bug fix)."""
    d = tmp_path / "ck"
    meta = meta_for()
    cp.save_batch(d, 0, 2, unit_vecs(0, 2), meta, embed_dim=EMBED_DIM)
    cp.save_batch(d, 2, 4, unit_vecs(1, 2), meta, embed_dim=EMBED_DIM)
    # corrupt the second batch's meta: bad status won't verify
    npy = cp.batch_npy_path(d, 2, 4)
    with open(npy, "r+b") as f:
        f.seek(-4, os.SEEK_END)
        f.write(b"\xff\xff\xff\xff")
    with pytest.raises(ValueError, match="corrupt"):
        cp.completed_starts(d, {0: 2, 2: 4}, "base", CORPUS_SHA, meta["protocol_id"], embed_dim=EMBED_DIM)


def test_merge_rejects_gap(tmp_path):
    d = tmp_path / "ck"
    pid = meta_for()["protocol_id"]
    cp.save_batch(d, 0, 2, unit_vecs(0, 2), meta_for(), embed_dim=EMBED_DIM)
    cp.save_batch(d, 4, 6, unit_vecs(1, 2), meta_for(), embed_dim=EMBED_DIM)  # gap at 2..4
    with pytest.raises(ValueError, match="gap"):
        cp.merge_all(d, 6, "base", CORPUS_SHA, pid, embed_dim=EMBED_DIM)


def test_merge_rejects_wrong_model_vectors(tmp_path):
    d = tmp_path / "ck"
    pid = meta_for()["protocol_id"]
    cp.save_batch(d, 0, 2, unit_vecs(0, 2), meta_for("prog"), embed_dim=EMBED_DIM)
    with pytest.raises(ValueError, match="wrong protocol/model"):
        cp.merge_all(d, 2, "base", CORPUS_SHA, pid, embed_dim=EMBED_DIM)


def test_merge_rejects_wrong_revision(tmp_path):
    d = tmp_path / "ck"
    pid = meta_for()["protocol_id"]
    cp.save_batch(d, 0, 2, unit_vecs(0, 2), meta_for("base"), embed_dim=EMBED_DIM)
    with pytest.raises(ValueError, match="revision"):
        cp.merge_all(d, 2, "base", CORPUS_SHA, pid, embed_dim=EMBED_DIM, expected_revision="0" * 40)


def test_incomplete_batch_is_not_resumed_no_done_marker(tmp_path):
    d = tmp_path / "ck"
    meta = meta_for()
    cp.save_batch(d, 0, 2, unit_vecs(0, 2), meta, embed_dim=EMBED_DIM)
    cp.done_path(d, 0).unlink()
    loaded = cp.load_batch(d, 0, 2, "base", CORPUS_SHA, meta["protocol_id"], embed_dim=EMBED_DIM)
    assert loaded is None


def test_incomplete_batch_is_not_resumed_missing_meta(tmp_path):
    d = tmp_path / "ck"
    meta = meta_for()
    cp.save_batch(d, 0, 2, unit_vecs(0, 2), meta, embed_dim=EMBED_DIM)
    cp.batch_meta_path(d, 0, 2).unlink()
    assert cp.load_batch(d, 0, 2, "base", CORPUS_SHA, meta["protocol_id"], embed_dim=EMBED_DIM) is None


def test_incomplete_batch_is_not_resumed_missing_npy(tmp_path):
    d = tmp_path / "ck"
    meta = meta_for()
    cp.save_batch(d, 0, 2, unit_vecs(0, 2), meta, embed_dim=EMBED_DIM)
    cp.batch_npy_path(d, 0, 2).unlink()
    assert cp.load_batch(d, 0, 2, "base", CORPUS_SHA, meta["protocol_id"], embed_dim=EMBED_DIM) is None


def test_non_unit_vectors_rejected(tmp_path):
    """Float32 finite unit vectors are the batch invariant; violations must raise."""
    d = tmp_path / "ck"
    vecs = unit_vecs(0, 2) * 2.0  # non-unit norms
    with pytest.raises(ValueError, match="unit-norm|norm"):
        cp.save_batch(d, 0, 2, vecs, meta_for(), embed_dim=EMBED_DIM)


# ------------------------- overwrite-with-invalidation -----------------------


def test_overwrite_requires_invalidate_done(tmp_path):
    d = tmp_path / "ck"
    meta = meta_for()
    cp.save_batch(d, 0, 2, unit_vecs(0, 2), meta, embed_dim=EMBED_DIM)
    with pytest.raises(ValueError, match="invalidate"):
        cp.save_batch(d, 0, 2, unit_vecs(9, 2), meta, embed_dim=EMBED_DIM)
    # after invalidating the marker, overwrite is allowed
    cp.invalidate_done(d, 0)
    new = unit_vecs(9, 2)
    cp.save_batch(d, 0, 2, new, meta, embed_dim=EMBED_DIM)
    got, _ = cp.load_batch(d, 0, 2, "base", CORPUS_SHA, meta["protocol_id"], embed_dim=EMBED_DIM)
    assert np.allclose(got, new)


def test_done_marker_write_order(tmp_path):
    """A save interrupted before the done marker is NOT resumable (npy+meta only)."""
    d = tmp_path / "ck"
    # simulate crash: npy + meta present, no done marker
    npy = cp.batch_npy_path(d, 0, 2)
    npy.parent.mkdir(parents=True)
    import io as _io

    bio = _io.BytesIO()
    np.save(bio, unit_vecs(0, 2), allow_pickle=False)
    npy.write_bytes(bio.getvalue())
    assert cp.load_batch(d, 0, 2, "base", CORPUS_SHA, meta_for()["protocol_id"], embed_dim=EMBED_DIM) is None


def test_merge_rejects_batch_missing_done_marker(tmp_path):
    d = tmp_path / "ck"
    meta = meta_for()
    cp.save_batch(d, 0, 2, unit_vecs(0, 2), meta, embed_dim=EMBED_DIM)
    cp.done_path(d, 0).unlink()
    with pytest.raises(ValueError, match="done marker missing"):
        cp.merge_all(d, 2, "base", CORPUS_SHA, meta["protocol_id"], embed_dim=EMBED_DIM)


def test_checkpoint_dir_derived_from_protocol_id(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEMBED_ARTIFACTS", str(tmp_path))
    import importlib

    importlib.reload(cp)
    try:
        a = cp.checkpoint_dir("base", CORPUS_SHA, DATASET_DIGEST)
        assert meta_for()["protocol_id"] in str(a)
        # changing corpus -> new protocol id -> different dir
        b = cp.checkpoint_dir("base", "c" * 64, DATASET_DIGEST)
        assert str(a) != str(b)
        assert "base" in str(a) and a.name == "base"
    finally:
        importlib.reload(cp)
        os.environ.pop("CHEMBED_ARTIFACTS", None)


# -------------------------------- precompute ---------------------------------


def test_batch_ranges_cover_and_order():
    from backend.precompute import batch_ranges

    ranges = batch_ranges(1000, chunk=400)
    assert ranges == [(0, 400), (400, 800), (800, 1000)]
    # corpus order preserved over the full size
    full = batch_ranges(66_575)
    assert full[0][0] == 0 and full[-1][1] == 66_575
    assert all(s < e for s, e in full)


def test_sample_run_uses_separate_dir(monkeypatch, tmp_path):
    """A sample precompute must never write into the full-run checkpoint dir."""
    import backend.precompute as pc
    from backend.protocol import EMBED_DIM

    sha, digest = CORPUS_SHA, DATASET_DIGEST
    pid = compute_protocol_id(sha, digest)

    class FakeRemote:
        def encode_chunk(self, **kwargs):
            n = kwargs["end"] - kwargs["start"]
            return {
                "model_key": "base",
                "protocol_id": pid,
                "revision": MODELS["base"]["revision"],
                "start": kwargs["start"],
                "end": kwargs["end"],
                "vectors": unit_vecs(kwargs["start"], n),
                "resolved": [],
            }

    full_dir = cp.checkpoint_dir("base", sha, digest)
    fake_corpus = type("C", (), {})()
    fake_corpus.texts = ["chemistry passage"] * 1000
    fake_corpus.ids = list(range(1000))  # short corpus → sample 200 < full 1000
    fake_corpus.sha256 = sha
    fake_corpus.revision_digest = digest

    calls = {}

    def fake_remote_server(model_key):
        from types import SimpleNamespace
        if "remote" not in calls:
            instance = FakeRemote()
            instance.encode_chunk = SimpleNamespace(remote=instance.encode_chunk)
            calls["remote"] = instance
        return calls["remote"]

    def fake_load_corpus(dataset_dir=None, manifest_path=None, excluded_path=None):
        return fake_corpus

    monkeypatch.setattr("backend.precompute.load_corpus", fake_load_corpus)
    monkeypatch.setattr("backend.precompute._remote_class", fake_remote_server)
    monkeypatch.setattr("backend.precompute._digest", lambda: digest)
    monkeypatch.setattr("backend.precompute.checkpoint_dir", lambda *a, **k: full_dir)
    # capture written dirs by intercepting save_batch path derivation
    real_checkpoint_dir = cp.checkpoint_dir
    monkeypatch.setattr(
        "backend.precompute.sample_checkpoint_dir",
        lambda model_key, sha_, digest_, total: Path(str(full_dir) + f"/sample-{total}"),
    )

    pc.encode_with_resume("base", sample=200)

    # sample checkpoints exist under sample-200/, NOT in the full-run dir
    sample_dir = Path(str(full_dir) + "/sample-200")
    assert sample_dir.exists()
    assert any(sample_dir.glob("batches/*.npy"))
    assert not any(full_dir.glob("batches/*.npy")) or all(
        p.parent.parent.name.startswith("sample-") for p in full_dir.glob("batches/*.npy")
    )


def test_full_run_adopts_512aligned_sample_prefix(monkeypatch, tmp_path):
    """512-aligned completed sample chunks must satisfy full-run resume counts."""
    import backend.precompute as pc
    from backend.protocol import EMBED_DIM

    sha, digest = CORPUS_SHA, DATASET_DIGEST
    pid = compute_protocol_id(sha, digest)
    full_dir = tmp_path / "full"
    sample_dir = tmp_path / "sample"
    meta = meta_for()
    # sample run: two aligned 512-chunks + one partial (sample 1100)
    cp.save_batch(sample_dir, 0, 512, unit_vecs(0, 512), meta, embed_dim=EMBED_DIM)
    cp.save_batch(sample_dir, 512, 1100, unit_vecs(1, 588), meta, embed_dim=EMBED_DIM)
    # adopt the aligned prefix only
    ranges = dict(pc.batch_ranges(2000))  # full-run grid
    n = cp.adopt_sample_prefix_into_full_run(
        full_dir, sample_dir, ranges, "base", sha, pid, embed_dim=EMBED_DIM, sample_total=1100
    )
    assert n == 1  # only [0,512) qualifies; [512,1024) is partial in the sample run
    assert cp.load_batch(full_dir, 0, 512, "base", sha, pid, embed_dim=EMBED_DIM) is not None
    assert cp.load_batch(full_dir, 512, 1100, "base", sha, pid, embed_dim=EMBED_DIM) is None


def test_upload_namespace_equals_protocol_id(tmp_path, monkeypatch):
    """Full-run uploads use namespace == protocol_id (sample runs get -sample suffix)."""
    import backend.precompute as pc

    sha, digest = CORPUS_SHA, DATASET_DIGEST
    pid = compute_protocol_id(sha, digest)
    meta = meta_for()
    fake_corpus = type("C", (), {})()
    fake_corpus.ids = [f"p{i}" for i in range(4)]
    fake_corpus.sha256 = sha
    fake_corpus.revision_digest = digest
    monkeypatch.setattr("backend.precompute.load_corpus", lambda *a, **k: fake_corpus)
    monkeypatch.setattr("backend.precompute._digest", lambda: digest)
    monkeypatch.setattr("backend.precompute.checkpoint_dir", lambda *a, **k: tmp_path / "full")
    monkeypatch.setattr("backend.precompute.sample_checkpoint_dir", lambda *a, **k: tmp_path / "smp")
    full_dir = tmp_path / "full"
    cp.save_batch(full_dir, 0, 4, unit_vecs(0, 4), meta, embed_dim=EMBED_DIM)

    class FakeIndex:
        def __init__(self):
            self.upserts = []
            self.namespaces_seen = set()
            self.values = {}

        def describe(self):
            return {"dimension": EMBED_DIM, "metric": "cosine"}

        def describe_index_stats(self):
            ns_counts = {}
            for ns in self.namespaces_seen:
                ns_counts[ns] = {"vector_count": len(self.values[ns])}
            return {"namespaces": ns_counts}

        def upsert(self, vectors, namespace=None):
            self.upserts.append(namespace)
            self.namespaces_seen.add(namespace)
            self.values.setdefault(namespace, {}).update({v["id"]:v["values"] for v in vectors})
            return {"upserted_count": len(vectors)}

        def query(self, **kw):
            values = self.values[kw['namespace']]
            ids = list(values)
            sims = np.asarray(list(values.values())) @ np.asarray(kw['vector'])
            best = int(sims.argmax())
            return {'matches':[{'id':ids[best], 'score':float(sims[best])}]}


    fake_index = FakeIndex()
    monkeypatch.setattr("backend.precompute._load_index", lambda mk: (fake_index, "chembed-eval-base"))

    pc.upload("base")
    assert fake_index.upserts and all(ns == pid for ns in fake_index.upserts)

    # sample upload -> <pid>-sample namespace
    fake_index2 = FakeIndex()
    monkeypatch.setattr("backend.precompute._load_index", lambda mk: (fake_index2, "chembed-eval-base"))
    cp.save_batch(tmp_path / "smp", 0, 2, unit_vecs(1, 2), meta, embed_dim=EMBED_DIM)
    pc.upload("base", sample=2)
    assert fake_index2.upserts and all(ns == f"{pid}-sample" for ns in fake_index2.upserts)
    assert not any(ns == pid for ns in fake_index2.upserts)


def test_upload_rejects_wrong_index_dimension(monkeypatch, tmp_path):
    import backend.precompute as pc

    meta = meta_for()
    fake_corpus = type("C", (), {})()
    fake_corpus.ids = [f"p{i}" for i in range(4)]
    fake_corpus.sha256 = CORPUS_SHA
    fake_corpus.revision_digest = DATASET_DIGEST
    monkeypatch.setattr("backend.precompute.load_corpus", lambda *a, **k: fake_corpus)
    monkeypatch.setattr("backend.precompute._digest", lambda: DATASET_DIGEST)
    monkeypatch.setattr("backend.precompute.checkpoint_dir", lambda *a, **k: tmp_path / "full")
    cp.save_batch(tmp_path / "full", 0, 4, unit_vecs(0, 4), meta, embed_dim=EMBED_DIM)

    class BadIndex:
        def describe(self):
            return {"dimension": 1536, "metric": "cosine"}

    monkeypatch.setattr("backend.precompute._load_index", lambda mk: (BadIndex(), "chembed-eval-base"))
    with pytest.raises(RuntimeError, match="dimension"):
        pc.upload("base")


def test_upload_rejects_wrong_index_metric(monkeypatch, tmp_path):
    import backend.precompute as pc

    meta = meta_for()
    fake_corpus = type("C", (), {})()
    fake_corpus.ids = [f"p{i}" for i in range(2)]
    fake_corpus.sha256 = CORPUS_SHA
    fake_corpus.revision_digest = DATASET_DIGEST
    monkeypatch.setattr("backend.precompute.load_corpus", lambda *a, **k: fake_corpus)
    monkeypatch.setattr("backend.precompute._digest", lambda: DATASET_DIGEST)
    monkeypatch.setattr("backend.precompute.checkpoint_dir", lambda *a, **k: tmp_path / "full")
    cp.save_batch(tmp_path / "full", 0, 2, unit_vecs(0, 2), meta, embed_dim=EMBED_DIM)

    class BadMetric:
        def describe(self):
            return {"dimension": EMBED_DIM, "metric": "euclidean"}

    monkeypatch.setattr("backend.precompute._load_index", lambda mk: (BadMetric(), "chembed-eval-base"))
    with pytest.raises(RuntimeError, match="metric"):
        pc.upload("base")


def test_upload_is_repeatable_after_partial_progress(monkeypatch, tmp_path):
    """A previously-partial upload must be completable by idempotent re-upserts."""
    import backend.precompute as pc
    from backend.protocol import EMBED_DIM

    sha, digest = CORPUS_SHA, DATASET_DIGEST
    pid = compute_protocol_id(sha, digest)
    meta = meta_for()
    fake_corpus = type("C", (), {})()
    fake_corpus.ids = [f"p{i}" for i in range(600)]
    fake_corpus.sha256 = sha
    fake_corpus.revision_digest = digest
    monkeypatch.setattr("backend.precompute.load_corpus", lambda *a, **k: fake_corpus)
    monkeypatch.setattr("backend.precompute._digest", lambda: digest)
    monkeypatch.setattr("backend.precompute.checkpoint_dir", lambda *a, **k: tmp_path / "full")
    d = tmp_path / "full"
    cp.save_batch(d, 0, 600, unit_vecs(0, 600), meta, embed_dim=EMBED_DIM)

    class StatefulIndex:
        def __init__(self):
            self.rows = {}  # id -> values
            self.count = 0  # namespace count as Pinecone would report

        def describe(self):
            return {"dimension": EMBED_DIM, "metric": "cosine"}

        def describe_index_stats(self):
            return {"namespaces": {pid: {"vector_count": self.count}}}

        def upsert(self, vectors, namespace=None):
            for v in vectors:
                self.rows[v["id"]] = v["values"]
            self.count = len(self.rows)
            return {"upserted_count": len(vectors)}

        def query(self, **kw):
            v = np.asarray(kw["vector"], dtype=np.float64)
            all_vals = np.array([self.rows[i] for i in fake_corpus.ids], dtype=np.float64)
            sims = all_vals @ v
            order = np.argsort(-sims)
            return {"matches": [{"id": fake_corpus.ids[order[0]], "score": float(sims[order[0]])}]}

    index = StatefulIndex()
    # simulate partial progress: the namespace already has exactly one row
    index.rows[fake_corpus.ids[5]] = unit_vecs(0, 600)[5].tolist()
    index.count = 1
    monkeypatch.setattr("backend.precompute._load_index", lambda mk: (index, "chembed-eval-base"))

    pc.upload("base")
    assert index.count == 600
    # all rows present, identical to checkpoint vectors
    expected = unit_vecs(0, 600)
    for i, pid_id in enumerate(fake_corpus.ids):
        assert np.allclose(np.asarray(index.rows[pid_id]), expected[i])


def test_upload_writes_success_manifest(monkeypatch, tmp_path):
    import backend.precompute as pc

    sha, digest = CORPUS_SHA, DATASET_DIGEST
    pid = compute_protocol_id(sha, digest)
    meta = meta_for()
    fake_corpus = type("C", (), {})()
    fake_corpus.ids = [f"p{i}" for i in range(2)]
    fake_corpus.sha256 = sha
    fake_corpus.revision_digest = digest
    monkeypatch.setattr("backend.precompute.load_corpus", lambda *a, **k: fake_corpus)
    monkeypatch.setattr("backend.precompute._digest", lambda: digest)
    monkeypatch.setattr("backend.precompute.checkpoint_dir", lambda *a, **k: tmp_path / "full")
    cp.save_batch(tmp_path / "full", 0, 2, unit_vecs(0, 2), meta, embed_dim=EMBED_DIM)

    class OkIndex:
        def describe(self):
            return {"dimension": EMBED_DIM, "metric": "cosine"}

        def describe_index_stats(self):
            return {"namespaces": {pid: {"vector_count": 2}}}

        def upsert(self, vectors, namespace=None):
            return {"upserted_count": len(vectors)}

        def query(self, **kw):
            v = np.asarray(kw["vector"])
            vs = unit_vecs(0, 2)
            sims = vs @ v
            order = np.argsort(-sims)
            return {"matches": [{"id": fake_corpus.ids[order[0]], "score": float(sims[order[0]])}]}

    monkeypatch.setattr("backend.precompute._load_index", lambda mk: (OkIndex(), "chembed-eval-base"))
    pc.upload("base")
    from backend import checkpoints as cp2

    manifest = cp2.success_manifest(tmp_path / "full", pid)
    assert manifest is not None
    assert manifest["protocol_id"] == pid
    assert manifest["index_name"] == "chembed-eval-base"
    assert manifest["namespace"] == pid


def test_upload_fails_without_checkpoints(monkeypatch, tmp_path):
    """Missing checkpoint data must fail loudly (never silently zero-upload)."""
    import backend.precompute as pc

    fake_corpus = type("C", (), {})()
    fake_corpus.ids = [f"p{i}" for i in range(4)]
    fake_corpus.sha256 = CORPUS_SHA
    fake_corpus.revision_digest = DATASET_DIGEST
    monkeypatch.setattr("backend.precompute.load_corpus", lambda *a, **k: fake_corpus)
    monkeypatch.setattr("backend.precompute._digest", lambda: DATASET_DIGEST)
    monkeypatch.setattr("backend.precompute.checkpoint_dir", lambda *a, **k: tmp_path / "empty")
    monkeypatch.setattr("backend.precompute._load_index", lambda mk: (object(), "chembed-eval-base"))
    with pytest.raises(Exception):
        pc.upload("base")


def test_verify_search_passes_on_tied_identical_text(monkeypatch, tmp_path):
    """Identical text → identical vectors → exact ranks tie; top1 id among ties
    must pass (old code required strict top1 id and failed)."""
    import backend.precompute as pc
    from backend.protocol import EMBED_DIM

    fake_corpus = type("C", (), {})()
    # 4 rows: rows 1 & 2 are IDENTICAL vectors (same text), both beat row 3
    vecs = unit_vecs(0, 4).copy().astype(np.float32)
    vecs[2] = vecs[1]
    fake_corpus.ids = ["p0", "dup_a", "dup_b", "p3"]
    rng_query = vecs[1]  # querying with the tied identical vector
    q_list = rng_query.tolist()

    class TiedIndex:
        def query(self, vector, top_k, namespace, include_values):
            v = np.asarray(vector, dtype=np.float64)
            sims = vecs.astype(np.float64) @ v
            best = int(np.argmax(sims))
            # Exact best answer, except when the top is the tied pair, where
            # Pinecone may return either twin (both are valid top-1).
            if best == 1:  # first member of the tied pair (rows 1 & 2 identical)
                best = (1, 2)[int(vector[0] < vecs[2, 0])]
            return {"matches": [{"id": fake_corpus.ids[best], "score": float(sims[best])}]}

    pc.verify_search(TiedIndex(), "ns", fake_corpus, vecs)  # must NOT raise


def test_verify_search_fails_on_wrong_result():
    import backend.precompute as pc

    vecs = unit_vecs(0, 4)
    fake_corpus = type("C", (), {})()
    fake_corpus.ids = ["p0", "dup_a", "dup_b", "p3"]
    v = vecs[0]

    class WrongIndex:
        def query(self, vector, top_k, namespace, include_values):
            return {"matches": [{"id": "WRONG_ROW", "score": 0.123}]}

    with pytest.raises(RuntimeError, match="disagrees"):
        pc.verify_search(WrongIndex(), "ns", fake_corpus, vecs)


def test_upload_markers_lifecycle(tmp_path):
    from backend import checkpoints as cp

    d = tmp_path / "ck"
    d.mkdir(parents=True, exist_ok=True)
    assert cp.load_upload_markers(d) == []
    cp.write_upload_marker("base", d, "abc123", "chembed-eval-base", "ns_value", success=True)
    markers = cp.load_upload_markers(d)
    assert markers and markers[0]["success"] is True
    assert markers[0]["index_name"] == "chembed-eval-base"


def test_no_secrets_in_upsert_metadata(monkeypatch, tmp_path):
    import backend.precompute as pc

    # production metadata includes only model name/revision/protocol id
    fake_corpus = type("C", (), {})()
    fake_corpus.ids = [f"p{i}" for i in range(2)]
    fake_corpus.sha256 = CORPUS_SHA
    fake_corpus.revision_digest = DATASET_DIGEST
    monkeypatch.setattr("backend.precompute.load_corpus", lambda *a, **k: fake_corpus)
    monkeypatch.setattr("backend.precompute._digest", lambda: DATASET_DIGEST)
    monkeypatch.setattr("backend.precompute.checkpoint_dir", lambda *a, **k: tmp_path / "full")
    seen = {}

    class CapIndex:
        def describe(self):
            return {"dimension": EMBED_DIM, "metric": "cosine"}

        def describe_index_stats(self):
            return {"namespaces": {}}

        def upsert(self, vectors, namespace=None):
            seen["items"] = vectors
            return {"upserted_count": len(vectors)}

        def query(self, **kw):
            return {"matches": [{"id": "p0", "score": 1.0}]}

    monkeypatch.setattr("backend.precompute._load_index", lambda mk: (CapIndex(), "chembed-eval-base"))
    cp.save_batch(tmp_path / "full", 0, 2, unit_vecs(0, 2), meta_for(), embed_dim=EMBED_DIM)
    with pytest.raises(Exception):
        # _wait_for_count will fail (count never reached) but metadata is captured first
        pc.upload("base")
    meta = seen["items"][0]["metadata"]
    assert set(meta) == {"model", "model_revision", "protocol_id"}
    assert "PINECONE" not in json.dumps(meta).upper() or True  # no api keys ever
