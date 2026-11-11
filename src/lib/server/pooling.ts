import { MODEL_KEYS, RRF_K, TOP_K, type ModelKey } from "./search-config";
export { RRF_K } from "./search-config";
import type { PineconeHit } from "./pinecone";

export type PooledPassage = {
  originalPassageId: string;
  rankFor: Partial<Record<ModelKey, number>>;
  scoreFor: Partial<Record<ModelKey, number>>;
};

export type PoolingResult =
  | { ok: true; pool: PooledPassage[] }
  | { ok: false; reason: string };

/**
 * Merge per-model hit lists by original passage ID. A merged list passes only
 * when every model returned exactly TOP_K hits and every hit has a rank and
 * score; anything else is a failed/short retrieval, never silently padded.
 */
export function poolHits(
  hits: Record<ModelKey, PineconeHit[] | undefined>,
): PoolingResult {
  for (const modelKey of MODEL_KEYS) {
    const list = hits[modelKey];
    if (!list || list.length !== TOP_K) {
      return {
        ok: false,
        reason: `${modelKey} returned ${list?.length ?? 0} hits, expected ${TOP_K}`,
      };
    }
  }
  const byId = new Map<string, PooledPassage>();
  for (const modelKey of MODEL_KEYS) {
    const seen = new Set<string>();
    for (const hit of hits[modelKey]!) {
      if (seen.has(hit.originalPassageId)) {
        return { ok: false, reason: `${modelKey} returned a duplicate passage id` };
      }
      seen.add(hit.originalPassageId);
      const rank = seen.size;
      let entry = byId.get(hit.originalPassageId);
      if (!entry) {
        entry = { originalPassageId: hit.originalPassageId, rankFor: {}, scoreFor: {} };
        byId.set(hit.originalPassageId, entry);
      }
      entry.rankFor[modelKey] = rank;
      entry.scoreFor[modelKey] = hit.score;
    }
  }
  return { ok: true, pool: [...byId.values()] };
}

/** Fisher–Yates shuffle with an injectable RNG for saved tie breaking. */
export function shuffle<T>(items: T[], rng: () => number = Math.random): T[] {
  const out = [...items];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

/**
 * Reciprocal rank fusion across the base and PROG original ranks. A model that
 * did not retrieve the passage contributes 0. This is a display-level fusion
 * of the two existing ranking lists only — not a BM25 or score hybrid, and it
 * does not change how either model is queried or scored.
 */
export function rrfScore(
  rankFor: Partial<Record<ModelKey, number>>,
  k: number = RRF_K,
): number {
  let score = 0;
  for (const modelKey of MODEL_KEYS) {
    const rank = rankFor[modelKey];
    if (rank !== undefined) score += 1 / (k + rank);
  }
  return score;
}

/**
 * Display order: deduplicated pool sorted by RRF score, descending. Exact ties
 * keep the order of the pool after one input shuffle, so the tie break is
 * reproducible with an injectable rng. Stable sort preserves that shuffle for
 * equal scores.
 */
export function rankByRrf<
  P extends { originalPassageId: string; rankFor?: Partial<Record<ModelKey, number>> },
>(pool: P[], rng: () => number = Math.random): P[] {
  return shuffle(pool, rng).sort(
    (a, b) => rrfScore(b.rankFor ?? {}) - rrfScore(a.rankFor ?? {}),
  );
}

export type DisplayOrderMethod = "first_encounter" | "reciprocal_rank_fusion" | "shuffle";

/** Respect the recorded method when an older, unfinished search is retried. */
export function displayOrderMethod(settings: Record<string, unknown>): DisplayOrderMethod {
  const order = settings.display_order;
  if (order && typeof order === "object" && "method" in order) {
    if (order.method === "first_encounter" || order.method === "reciprocal_rank_fusion") {
      return order.method;
    }
  }
  return "shuffle";
}

/** A shared passage appears at its earliest original rank, with random ties. */
export function rankByFirstEncounter<
  T extends { originalPassageId: string; rankFor?: Partial<Record<ModelKey, number>> },
>(pool: T[], rng: () => number = Math.random): T[] {
  const firstRank = (passage: T) => Math.min(
    passage.rankFor?.base ?? Infinity,
    passage.rankFor?.prog ?? Infinity,
  );
  return shuffle(pool, rng).sort((a, b) => firstRank(a) - firstRank(b));
}

/** Persist one display order. Saved result positions are read back unchanged. */
export function assignDisplayPositions<
  T extends { originalPassageId: string; rankFor?: Partial<Record<ModelKey, number>> },
>(
  pool: T[],
  rng: () => number = Math.random,
  method: DisplayOrderMethod = "first_encounter",
): Map<string, number> {
  const positions = new Map<string, number>();
  const ordered = method === "first_encounter"
    ? rankByFirstEncounter(pool, rng)
    : method === "reciprocal_rank_fusion"
      ? rankByRrf(pool, rng)
      : shuffle(pool, rng);
  ordered.forEach((p, i) => positions.set(p.originalPassageId, i + 1));
  return positions;
}
