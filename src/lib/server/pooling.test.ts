import { describe, expect, it } from "vitest";

import { assignDisplayPositions, poolHits, rrfScore, rankByRrf, rankByFirstEncounter, displayOrderMethod, shuffle } from "@/lib/server/pooling";
import type { PineconeHit } from "@/lib/server/pinecone";
import { settingsSnapshot, RRF_K } from "@/lib/server/search-config";

const hit = (id: string, score = 0.9): PineconeHit => ({ originalPassageId: id, score });

describe("poolHits", () => {
  it("merges overlapping hits by passage id, keeping each model's rank", () => {
    const pooled = poolHits({
      base: [hit("a"), hit("b"), hit("c"), hit("d"), hit("e")],
      prog: [hit("d"), hit("e"), hit("f"), hit("g"), hit("h")],
    });
    expect(pooled.ok).toBe(true);
    if (!pooled.ok) return;
    expect(pooled.pool).toHaveLength(8);
    const d = pooled.pool.find((p) => p.originalPassageId === "d")!;
    expect(d.rankFor.base).toBe(4);
    expect(d.rankFor.prog).toBe(1);
    expect(d.scoreFor.base).toBeCloseTo(0.9);
    const f = pooled.pool.find((p) => p.originalPassageId === "f")!;
    expect(f.rankFor.base).toBeUndefined();
    expect(f.rankFor.prog).toBe(3);
  });

  it("rejects a model returning fewer than 5 hits instead of padding the pool", () => {
    expect(poolHits({ base: [hit("a"), hit("b")], prog: [hit("a"), hit("b"), hit("c"), hit("d"), hit("e")] })).toEqual({
      ok: false,
      reason: expect.stringContaining("returned 2 hits"),
    });
    expect(poolHits({ base: undefined, prog: [] })).toMatchObject({ ok: false });
  });

  it("rejects a duplicate passage id within one model's hits", () => {
    expect(
      poolHits({ base: [hit("a"), hit("a"), hit("b"), hit("c"), hit("d")], prog: [hit("e"), hit("f"), hit("g"), hit("h"), hit("i")] }),
    ).toMatchObject({ ok: false });
  });
});

describe("display positions", () => {
  it("assigns 1..N bijectively", () => {
    const pool = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"].map((id) => ({
      originalPassageId: id,
    }));
    const positions = assignDisplayPositions(pool, () => 0.42);
    expect(positions.size).toBe(pool.length);
    expect(new Set(positions.values())).toEqual(new Set(pool.map((_, i) => i + 1)));
  });

  it("shuffles deterministically with a fixed rng and never drops items", () => {
    const items = [1, 2, 3, 4, 5];
    const first = shuffle(items, () => 0.1);
    const again = shuffle(items, () => 0.1);
    expect(first).toEqual(again);
    expect([...first].sort()).toEqual(items);
  });
});

describe("rankByRrf", () => {
  const pool = (entries: [string, Record<string, number> | undefined][]) =>
    entries.map(([id, rankFor]) => ({ originalPassageId: id, ...(rankFor ? { rankFor } : {}) }));

  it("sorts by summed reciprocal-rank contribution with k=60, absent rank contributing 0", () => {
    expect(RRF_K).toBe(60);
    expect(settingsSnapshot().display_order).toMatchObject({
      method: "first_encounter",
      primary: "minimum_original_rank",
      tie_break: "random within each rank, saved once",
    });
    const p1 = { originalPassageId: "both", rankFor: { base: 1, prog: 5 } };
    const p2 = { originalPassageId: "base-only", rankFor: { base: 4 } };
    const p3 = { originalPassageId: "late-overlap", rankFor: { base: 9, prog: 9 } };
    const ordered = rankByRrf([p3, p2, p1], () => 0);
    // 1/61+1/65 > 2/69 > 1/64
    expect(rrfScore({ base: 1, prog: 5 })).toBeCloseTo(1 / 61 + 1 / 65);
    expect(rrfScore({ base: 4 })).toBeCloseTo(1 / 64);
    expect(rrfScore({ base: 9, prog: 9 })).toBeCloseTo(2 / 69);
    expect(ordered.map((p) => p.originalPassageId)).toEqual(["both", "late-overlap", "base-only"]);
  });

  it("breaks exact score ties with one input shuffle followed by a stable score sort, reproducibly", () => {
    const entries = pool([
      ["a", { base: 1 }],
      ["b", { prog: 1 }],
      ["c", { base: 2, prog: 9 }],
    ]);
    expect(rrfScore({ base: 1 })).toBe(rrfScore({ prog: 1 }));
    const first = rankByRrf(entries, () => 0.05);
    const again = rankByRrf(entries, () => 0.05);
    expect(first).toEqual(again); // reproducible
    const ids = first.map((p) => p.originalPassageId).filter((id) => id === "a" || id === "b");
    expect(new Set(ids)).toEqual(new Set(["a", "b"]));
    const tied = entries.slice(0, 2);
    expect(rankByRrf(tied, () => 0).map(p => p.originalPassageId)).toEqual(["b", "a"]);
    expect(rankByRrf(tied, () => 0.99).map(p => p.originalPassageId)).toEqual(["a", "b"]);
  });

  it("assigns display positions over the RRF-ordered pool and preserves dedup", () => {
    const deduped = poolHits({
      base: [hit("a"), hit("b"), hit("c"), hit("d"), hit("e")],
      prog: [hit("d"), hit("e"), hit("f"), hit("g"), hit("h")],
    });
    expect(deduped.ok).toBe(true);
    if (!deduped.ok) return;
    const positions = assignDisplayPositions(deduped.pool, () => 0.1);
    expect(positions.size).toBe(8); // deduped size, not 10
    expect(new Set(positions.values())).toEqual(new Set(Array.from({ length: 8 }, (_, i) => i + 1)));
  });
});


describe("first encounter display order", () => {
  it("shows a rank-3/rank-5 shared passage in the rank-3 group only once", () => {
    const merged = poolHits({
      base: [hit("a"), hit("b"), hit("shared"), hit("d"), hit("e")],
      prog: [hit("f"), hit("g"), hit("h"), hit("i"), hit("shared")],
    });
    if (!merged.ok) throw new Error(merged.reason);
    const before = JSON.stringify(merged.pool);
    for (const rng of [() => 0, () => 0.99]) {
      const ordered = rankByFirstEncounter(merged.pool, rng);
      const ids = ordered.map(p => p.originalPassageId);
      expect(ids).toHaveLength(9);
      expect(ids.filter(id => id === "shared")).toHaveLength(1);
      expect(new Set(ids.slice(0, 2))).toEqual(new Set(["a", "f"]));
      expect(new Set(ids.slice(2, 4))).toEqual(new Set(["b", "g"]));
      expect(new Set(ids.slice(4, 6))).toEqual(new Set(["shared", "h"]));
      expect(new Set(ids.slice(6, 8))).toEqual(new Set(["d", "i"]));
      expect(ids[8]).toBe("e");
    }
    expect(JSON.stringify(merged.pool)).toBe(before);
    expect(merged.pool.find(p => p.originalPassageId === "shared")?.rankFor).toEqual({ base: 3, prog: 5 });
  });

  it("breaks equal-rank ties randomly without boosting later shared passages", () => {
    const pool = [
      { originalPassageId: "base-first", rankFor: { base: 1 } },
      { originalPassageId: "prog-first", rankFor: { prog: 1 } },
      { originalPassageId: "shared-last", rankFor: { base: 5, prog: 5 } },
    ];
    const a = rankByFirstEncounter(pool, () => 0).map(p => p.originalPassageId);
    const b = rankByFirstEncounter(pool, () => 0.99).map(p => p.originalPassageId);
    expect(a).toEqual(["prog-first", "base-first", "shared-last"]);
    expect(b).toEqual(["base-first", "prog-first", "shared-last"]);
    expect([...assignDisplayPositions(pool, () => 0).keys()]).toEqual(a);
  });

  it("uses the saved method for unfinished searches created before this change", () => {
    expect(displayOrderMethod(settingsSnapshot())).toBe("first_encounter");
    expect(displayOrderMethod({ display_order: { method: "reciprocal_rank_fusion" } })).toBe("reciprocal_rank_fusion");
    expect(displayOrderMethod({})).toBe("shuffle");
    const pool = [
      { originalPassageId: "single", rankFor: { base: 1 } },
      { originalPassageId: "shared", rankFor: { base: 5, prog: 5 } },
    ];
    expect([...assignDisplayPositions(pool, () => 0, "reciprocal_rank_fusion").keys()]).toEqual(["shared", "single"]);
  });
});
