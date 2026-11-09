"""End-to-end sanity check of the deployed retrieval pipeline.

Deterministically selects 10 evaluation queries (seed 42, performance-blind),
encodes them via the deployed Modal servers (one batch per model), retrieves
top-10 from the matching Pinecone index (namespace = protocol id), and checks
the Pinecone top-10 against exact cosine search over locally stored checkpoint
embeddings in original corpus order (tie-tolerant). Reports Hit@1/5/10 and
MRR@10 from qrels. Single-gold qrels do not label other passages as irrelevant;
lexical-overlap diagnostics are string statistics only, NOT relevance
judgments.

Run:  backend/.venv/bin/python -m backend.sanity
Tests: backend/.venv/bin/python -m pytest tests -q
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .checkpoints import ARTIFACTS_DIR, merge_all
from .client import encode_texts
from .corpus import Corpus, load_corpus
from .protocol import (
    DATASET_DIR,
    MAX_SEQ_LENGTH,
    MODELS,
    MODEL_KEYS,
    PINECONE_INDEX_FOR_KEY,
    PROMPTS,
    compute_protocol_id,
)

SANITY_SEED = 42
SANITY_N = 10
TOP_K = 10
NAMESPACE = "db72fa244e2915b2"
OUT_DIR = ARTIFACTS_DIR / "sanity"
SIM_TOL = 1e-5  # tie tolerance for identical texts

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "to", "for", "with", "by",
    "is", "are", "was", "were", "be", "been", "being", "that", "this", "these",
    "those", "it", "its", "as", "at", "from", "into", "how", "why", "what",
    "which", "can", "could", "would", "should", "than", "then", "does", "do",
    "did", "not", "no", "has", "have", "had", "their", "they", "there", "such",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------- data access


def load_queries() -> dict[str, str]:
    """query-id -> text from the exact evaluation queries parquet."""
    table = pq.read_table(DATASET_DIR / "queries" / "train-00000-of-00001.parquet")
    ids = table.column("_id").to_pylist()
    texts = table.column("text").to_pylist()
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate query ids")
    return dict(zip(ids, texts))


def load_qrels() -> dict[str, list[str]]:
    """query-id -> gold corpus ids (score > 0) from the qrels parquet."""
    table = pq.read_table(DATASET_DIR / "data" / "test-00000-of-00001.parquet")
    qrels: dict[str, list[str]] = {}
    for q, c, s in zip(
        table.column("query-id").to_pylist(),
        table.column("corpus-id").to_pylist(),
        table.column("score").to_pylist(),
    ):
        if s and s > 0:
            qrels.setdefault(q, []).append(c)
    return qrels


def select_queries(
    queries: dict[str, str], qrels: dict[str, list[str]],
    n: int = SANITY_N, seed: int = SANITY_SEED,
) -> list[str]:
    """Deterministic seed-42 selection from queries with gold labels.

    Uses only sorted query ids + the seed, so retrieval performance cannot
    influence which queries are chosen.
    """
    eligible = sorted(q for q in queries if q in qrels and qrels[q])
    if len(eligible) < n:
        raise ValueError(f"only {len(eligible)} queries have gold labels; need {n}")
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(eligible), size=n, replace=False).tolist())
    return [eligible[i] for i in chosen]


# ------------------------------------------------------------------- scoring


def rank_metrics(gold_ids: list[str], top_ids: list[str]) -> dict:
    """Hit@{1,5,10} plus first-gold rank and reciprocal rank within the list."""
    gold = set(gold_ids)
    first_rank = next(
        (r for r, pid in enumerate(top_ids, start=1) if pid in gold), None
    )
    return {
        "top1hit": bool(gold & set(top_ids[:1])),
        "top5hit": bool(gold & set(top_ids[:5])),
        "top10hit": bool(gold & set(top_ids[:10])),
        "first_gold_rank": first_rank,
        "reciprocal_rank@10": 0.0 if first_rank is None else 1.0 / first_rank,
    }


def exact_search(
    query_vec: np.ndarray, vectors: np.ndarray, top_k: int = TOP_K
) -> tuple[list[int], np.ndarray, list[int]]:
    """Exact cosine top-k over stored unit-norm passage vectors.

    Returns (row indices, similarities, extra_tie_rows). Identical passages
    have identical vectors, so ties at the k-th similarity cannot be broken by
    argsort order; every row within SIM_TOL of the k-th best similarity that
    was not already in the top-k is reported separately rather than silently
    dropped.
    """
    sims = vectors @ query_vec
    order = np.argsort(-sims, kind="stable")
    kth = float(sims[order[top_k - 1]])
    ties = set(np.flatnonzero(np.abs(sims - kth) < SIM_TOL).tolist())
    top = order[:top_k].tolist()
    extra = sorted(t for t in ties if t not in set(top))
    return top, sims[order[:top_k]], extra


def gold_rank_in_exact(
    gold_ids: list[str], ids: list[str], sims: np.ndarray, query_vec: np.ndarray,
) -> int | None:
    """Rank (1-based) of the best gold passage in the full exact ordering."""
    gold_rows = [i for i, pid in enumerate(ids) if pid in set(gold_ids)]
    if not gold_rows:
        raise ValueError("gold id not in corpus")
    gold_sims = sims[gold_rows] @ query_vec
    best_scores = (sims @ query_vec)[gold_rows]
    best = gold_rows[int(np.argmax(best_scores))]
    return int((sims @ query_vec > best_scores.max() + 1e-9).sum()) + 1


# ------------------------------------------------------------- text diagnostics


def tokens(text: str) -> set[str]:
    return {
        t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1
    }


def lexical_overlap(query: str, passage: str) -> dict:
    """Simple token-overlap statistics. NOT a relevance judgment: qrels mark
    only one passage per query, so a high overlap elsewhere proves nothing."""
    q, p = tokens(query), tokens(passage)
    inter = q & p
    return {
        "query_terms": len(q),
        "passage_terms": len(p),
        "overlapping_terms": len(inter),
        "query_coverage": round(len(inter) / len(q), 3) if q else 0.0,
        "overlap_terms_sample": sorted(inter)[:8],
    }


# ------------------------------------------------------------------ live run


def _validate_query_embeddings(model_key: str, emb: np.ndarray, n: int) -> dict:
    if emb.shape != (n, 768):
        raise RuntimeError(f"{model_key}: query embedding shape {emb.shape} != ({n}, 768)")
    if not np.isfinite(emb).all():
        raise RuntimeError(f"{model_key}: non-finite query embeddings")
    norms = np.linalg.norm(emb, axis=1)
    if not (np.abs(norms - 1.0) < 1e-3).all():
        raise RuntimeError(f"{model_key}: query norms not unit: {norms.min()}..{norms.max()}")
    return {
        "shape": list(emb.shape),
        "all_finite": True,
        "norm_min": round(float(norms.min()), 6),
        "norm_max": round(float(norms.max()), 6),
        "revision": MODELS[model_key]["revision"],
        "max_seq_length": MAX_SEQ_LENGTH,
        "query_prompt": PROMPTS["query"],
    }


def run(queries: list[str], n_queries: int = 10) -> dict:
    import os

    from dotenv import load_dotenv

    repo = Path(__file__).resolve().parent.parent
    load_dotenv(repo / ".env.local")  # PINECONE_API_KEY (stays out of logs)

    t0 = time.time()
    corpus = load_corpus()
    queries_all = load_queries()
    qrels = load_qrels()
    ids2text = dict(zip(corpus.ids, corpus.texts))
    missing_gold = sorted(
        {g for q in queries for g in qrels[q]} - set(corpus.ids)
    )
    if missing_gold:
        raise ValueError(f"gold ids missing from corpus: {missing_gold[:3]}...")

    pid = compute_protocol_id(corpus.sha256, corpus.revision_digest)
    if pid != NAMESPACE:
        raise RuntimeError(f"protocol id {pid} != expected namespace {NAMESPACE}")

    vectors: dict[str, np.ndarray] = {}
    for model_key in MODEL_KEYS:
        d = ARTIFACTS_DIR / "embeddings" / pid / model_key
        vectors[model_key] = merge_all(
            d, total=len(corpus.ids), model_key=model_key,
            corpus_sha256_str=corpus.sha256, pid=pid,
            embed_dim=768, expected_revision=MODELS[model_key]["revision"],
        )
        if not np.isfinite(vectors[model_key]).all():
            raise RuntimeError(f"{model_key}: non-finite passage embeddings in checkpoints")

    qtexts = [queries_all[q] for q in queries]
    out: dict = {
        "seed": SANITY_SEED,
        "n_queries": n_queries,
        "selection": "seed 42 RNG choice over sorted query ids that have gold labels; performance-blind",
        "top_k": TOP_K,
        "namespace": NAMESPACE,
        "protocol_id": pid,
        "corpus_rows": len(corpus.ids),
        "queries": queries,
        "queries_text": dict(zip(queries, qtexts)),
        "gold_ids": {q: qrels[q] for q in queries},
        "models": {},
    }

    from pinecone import Pinecone

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    try:
        for model_key in MODEL_KEYS:
            emb = encode_texts(model_key, qtexts, kind="query")  # single batch of 10
            validation = _validate_query_embeddings(model_key, emb, len(queries))
            index_name = PINECONE_INDEX_FOR_KEY[model_key]
            index = pc.Index(index_name)

            # exact search over saved checkpoints (original corpus order)
            exact_hits: list[tuple[list[int], np.ndarray, list[int]]] = [
                exact_search(emb[i], vectors[model_key]) for i in range(len(queries))
            ]

            per_query = []
            for i, qid in enumerate(queries):
                resp = index.query(
                    vector=emb[i].tolist(), top_k=TOP_K, namespace=NAMESPACE,
                    include_values=False,
                )
                matches = resp.get("matches", [])
                if len(matches) != TOP_K:
                    raise RuntimeError(
                        f"{model_key} q{qid}: Pinecone returned {len(matches)} matches, expected {TOP_K}"
                    )
                pc_ids = [m["id"] for m in matches]
                pc_scores = [float(m["score"]) for m in matches]
                if any(pid_ not in ids2text for pid_ in pc_ids):
                    raise RuntimeError(f"{model_key} q{qid}: Pinecone id outside corpus")

                rows, sims, extra_ties = exact_hits[i]
                exact_ids = [corpus.ids[r] for r in rows]
                exact_sims = [float(s) for s in sims]
                # agreement with Pinecone: ids (modulo ties) and scores
                tie_rows = set(rows) | set(extra_ties)
                ids_agree = all(
                    pc_ids[j] == exact_ids[j]
                    or pc_ids[j] in {corpus.ids[r] for r in tie_rows}
                    for j in range(TOP_K)
                )
                max_score_diff = max(
                    abs(pc_scores[j] - exact_sims[j]) for j in range(TOP_K)
                )
                metrics = rank_metrics(qrels[qid], pc_ids)
                top = []
                for j in range(TOP_K):
                    top.append({
                        "rank": j + 1,
                        "id": pc_ids[j],
                        "score": round(pc_scores[j], 6),
                        "text": ids2text[pc_ids[j]],
                        "exact_rank_same_row": exact_ids[j] == pc_ids[j],
                        "exact_score": round(exact_sims[j], 6),
                    })
                    if j < 3:  # lexical diagnostic for top-3 only
                        top[-1]["lexical_overlap"] = lexical_overlap(
                            qtexts[i], ids2text[pc_ids[j]]
                        )

                gold_in_top = metrics["first_gold_rank"]
                entry = {
                    "query_id": qid,
                    "query_text": qtexts[i],
                    "gold_ids": qrels[qid],
                    "metrics": metrics,
                    "pinecone_index": index_name,
                    "pinecone_top10": top,
                    "exact_vs_pinecone": {
                        "exact_top10": exact_ids,
                        "exact_scores": [round(s, 6) for s in exact_sims],
                        "top10_ids_agree_modulo_exact_ties": ids_agree,
                        "top10_rows_agree": exact_ids == pc_ids,
                        "max_abs_score_diff": round(max_score_diff, 6),
                        "exact_tie_group": (
                            "none"
                            if not extra_ties
                            else [
                                f"row {r} id {corpus.ids[r]}: {ids2text[corpus.ids[r]][:80]}"
                                for r in extra_ties
                            ]
                        ),
                    },
                }
                if gold_in_top is None:
                    gold_diag = []
                    for g in qrels[qid]:
                        gvec = vectors[model_key][corpus.ids.index(g)]
                        gsim = float(gvec @ emb[i])
                        gold_diag.append({
                            "id": g,
                            "exact_rank": int((vectors[model_key] @ emb[i] > gsim + 1e-9).sum()) + 1,
                            "cosine_to_query": round(gsim, 6),
                            "lexical_overlap": lexical_overlap(qtexts[i], ids2text[g]),
                        })
                    entry["gold_when_missed"] = gold_diag
                per_query.append(entry)

            hits = [e["metrics"] for e in per_query]
            out["models"][model_key] = {
                "pinecone_index": PINECONE_INDEX_FOR_KEY[model_key],
                "query_embedding_validation": validation,
                "hit@1": sum(m["top1hit"] for m in hits),
                "hit@5": sum(m["top5hit"] for m in hits),
                "hit@10": sum(m["top10hit"] for m in hits),
                "mrr@10": round(sum(m["reciprocal_rank@10"] for m in hits) / len(hits), 4),
                "queries_detail": per_query,
            }

    finally:
        # GPU containers scale down automatically after the 300 s idle window;
        # nothing to keep warm.
        pass

    out["elapsed_seconds"] = round(time.time() - t0, 1)
    return out


# -------------------------------------------------------------- output writer


def write_markdown(result: dict, path: Path, ids2text_lookup: dict[str, str]) -> None:
    lines: list[str] = ["# Sanity check — 10 deployed retrieval queries", ""]
    lines += [
        f"- Seed {result['seed']}, n={result['n_queries']}, top_k={result['top_k']}, "
        f"namespace `{result['namespace']}`, protocol `{result['protocol_id']}`.",
        f"- Selection: {result['selection']}.",
        f"- Corpus rows: {result['corpus_rows']}. Same 10 queries used for both models.",
        "- Lexical overlap numbers below are simple string statistics, **NOT relevance "
        "judgments** (qrels mark one gold passage per query; other passages are unlabeled).",
        f"- Total elapsed: {result['elapsed_seconds']} s.",
        "",
        "| model | Hit@1 | Hit@5 | Hit@10 | MRR@10 |",
        "|---|---|---|---|---|",
    ]
    for k in MODEL_KEYS:
        m = result["models"][k]
        lines.append(
            f"| {k} | {m['hit@1']}/10 | {m['hit@5']}/10 | {m['hit@10']}/10 | {m['mrr@10']} |"
        )
    lines += ["", "## Per-query results", ""]
    for k in MODEL_KEYS:
        lines += [f"### Model: `{k}` ({result['models'][k]['pinecone_index']})", ""]
        v = result["models"][k]["query_embedding_validation"]
        lines += [
            f"Query embedding validation: shape {v['shape']}×768, finite, "
            f"norm {v['norm_min']}–{v['norm_max']}, revision `{v['revision']}`, "
            f"max_seq_length {v['max_seq_length']}, prompt `{v['query_prompt']}`.",
            "",
        ]
        for e in result["models"][k]["queries_detail"]:
            gold = ", ".join(e["gold_ids"])
            hit = e["metrics"]["top10hit"]
            lines.append(f"#### Query {e['query_id']} — {'HIT' if hit else 'MISS'} ({e['query_text']})")
            lines.append("")
            lines.append(f"- Gold ID(s): {gold}")
            lines.append(
                f"- Hit@1/5/10: {int(e['metrics']['top1hit'])}/{int(e['metrics']['top5hit'])}/"
                f"{int(e['metrics']['top10hit'])}; first gold rank: "
                f"{e['metrics']['first_gold_rank']} (RR@10 {e['metrics']['reciprocal_rank@10']})"
            )
            x = e["exact_vs_pinecone"]
            agree = x["top10_ids_agree_modulo_exact_ties"]
            lines.append(
                f"- Exact-vs-Pinecone: {'AGREE' if agree else 'MISMATCH'} "
                f"(max |Δscore| {x['max_abs_score_diff']}); exact tie group: "
                f"{'; '.join(x['exact_tie_group']) if isinstance(x['exact_tie_group'], list) else x['exact_tie_group']}"
            )
            lines.append("")
            lines.append("| rank | passage id | score | text (full) |")
            lines.append("|---|---|---|---|")
            for t in e["pinecone_top10"]:
                text = t["text"].replace("|", "\\|").replace("\n", " ")
                gold_mark = " ← GOLD" if t["id"] in e["gold_ids"] else ""
                lines.append(
                    f"| {t['rank']}{gold_mark} | {t['id']} | {t['score']} | {text} |"
                )
            if hit is False:
                lines.append("")
                lines.append("**Miss — gold passage(s) not in Pinecone top-10:**")
                for d in e["gold_when_missed"]:
                    gt = ids2text_lookup[d["id"]].replace("|", "\\|").replace("\n", " ")
                    lines.append(
                        f"- `{d['id']}` exact rank {d['exact_rank']} "
                        f"(cosine {d['cosine_to_query']}); overlap with query: "
                        f"{d['lexical_overlap']['overlapping_terms']}/{d['lexical_overlap']['query_terms']} "
                        f"query terms ({', '.join(d['lexical_overlap']['overlap_terms_sample']) or '—'})"
                    )
                    lines.append(f"  - text: {gt}")
                lines.append("")
                lines.append("**Top-3 retrieved passages (with lexical overlap diagnostics):**")
                for t in e["pinecone_top10"][:3]:
                    lines.append(
                        f"- rank {t['rank']} `{t['id']}` score {t['score']} — "
                        f"{t['lexical_overlap']['overlapping_terms']}/"
                        f"{t['lexical_overlap']['query_terms']} query terms overlap"
                    )
                    lines.append(f"  - text: {t['text']}")
            lines.append("")
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    queries_all = load_queries()
    qrels = load_qrels()
    queries = select_queries(queries_all, qrels)
    corpus = load_corpus()
    ids2text = dict(zip(corpus.ids, corpus.texts))
    result = run(queries)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "query10.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    write_markdown(result, OUT_DIR / "query10.md", ids2text)


if __name__ == "__main__":
    main()
