import { describe, expect, it } from "vitest";

import { assignDisplayPositions, poolHits, shuffle } from "@/lib/server/pooling";
import type { PineconeHit } from "@/lib/server/pinecone";

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
