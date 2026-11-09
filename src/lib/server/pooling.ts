import { TOP_K, type ModelKey } from "./search-config";
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
  for (const modelKey of Object.keys(hits) as ModelKey[]) {
    const list = hits[modelKey];
    if (!list || list.length !== TOP_K) {
      return {
        ok: false,
        reason: `${modelKey} returned ${list?.length ?? 0} hits, expected ${TOP_K}`,
      };
    }
  }
  const byId = new Map<string, PooledPassage>();
  for (const modelKey of Object.keys(hits) as ModelKey[]) {
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

/** Fisher–Yates shuffle with an injectable RNG for deterministic tests. */
export function shuffle<T>(items: T[], rng: () => number = Math.random): T[] {
  const out = [...items];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

/**
 * Assign display positions 1..N to the shuffled pool. Callers persist
 * display_position alongside the private rank/score pairs; only the text at
 * each position is ever shown.
 */
export function assignDisplayPositions<T extends { originalPassageId: string }>(
  pool: T[],
  rng: () => number = Math.random,
): Map<string, number> {
  const positions = new Map<string, number>();
  shuffle(pool, rng).forEach((p, i) => positions.set(p.originalPassageId, i + 1));
  return positions;
}
