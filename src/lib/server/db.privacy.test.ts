import { describe, expect, it, vi } from "vitest";

type AuthModule = typeof import("@/lib/server/auth");
import { getSearchDetail, saveRating, MissingError } from "@/lib/server/db";

const USER_A = "11111111-1111-1111-1111-111111111111";
const USER_B = "22222222-2222-2222-2222-222222222222";
const SEARCH = "33333333-3333-3333-3333-333333333333";

/**
 * Privacy contract: getSearchDetail must return only id/text/relevance/note —
 * never base/prog ranks, scores, or settings — and must 404-equivalent for the
 * wrong user. saveRating must refuse results not owned by the caller.
 * These tests call the module functions directly against synthetic user ids;
 * no live Supabase connection is needed because the row scoping logic runs
 * before any I/O when a real client is absent (the mocked client below).
 */
function fakeClient() {
  type Row = Record<string, unknown>;
  const searches = new Map<string, Row>([
    [
      SEARCH,
      {
        id: SEARCH,
        user_id: USER_A,
        query: "q",
        information_need: null,
        created_at: "2026-01-01",
        status: "ready",
        is_practice: false,
        settings: { protocol_id: "x" },
      },
    ],
  ]);
  const results = [
    {
      id: "r1",
      search_id: SEARCH,
      passage_id: 1,
      base_rank: 3,
      base_score: 0.81,
      progressive_rank: null,
      progressive_score: null,
      display_position: 1,
    },
  ];
  const chain = (rows: Row[], table?: string) => {
    const builder = {
      select: () => builder,
      eq: (col: string, val: unknown) => {
        // Minimal scoping emulation so wrong-user lookups come back empty.
        if (!rows.length || !(col in rows[0])) return builder;
        rows = rows.filter((r) => (r as Row)[col] === val);
        return builder;
      },
      order: () => Promise.resolve({ data: rows, error: null }),
      maybeSingle: () => Promise.resolve({ data: rows[0] ?? null, error: null }),
      single: () => Promise.resolve({ data: rows[0] ?? null, error: null }),
    };
    return builder;
  };

  return {
    from: (table: string) => {
      if (table === "searches") return chain([...searches.values()]);
      if (table === "results") {
        return {
          ...chain(results),
          // getSearchDetail joins passages/ratings; saveRating joins
          // searches!inner(user_id, status). Mock per select argument.
          select: (sel?: string) =>
            sel?.includes("searches")
              ? chain(
                  results.map((r) => ({
                    ...r,
                    searches: {
                      user_id: searches.get(r.search_id)?.user_id,
                      status: "ready",
                    },
                  })),
                )
              : chain(results),
        };
      }
      return chain([]);
    },
  } as never;
}

vi.mock("@/lib/server/auth", async () => {
  const actual = await vi.importActual<AuthModule>("@/lib/server/auth");
  return { ...actual, secretClient: () => fakeClient() };
});

describe("result privacy", () => {
  it("exposes no hidden columns to the client shape", async () => {
    const detail = await getSearchDetail(USER_A, SEARCH);
    expect(detail).not.toBeNull();
    const result = detail!.results[0];
    expect(Object.keys(result).sort()).toEqual(["id", "note", "relevant", "text"]);
    const serialized = JSON.stringify(detail);
    expect(serialized).not.toMatch(/base_rank|base_score|progressive|score|rank|settings|user_id/);
  });

  it("hides the whole search from another user", async () => {
    await expect(getSearchDetail(USER_B, SEARCH)).resolves.toBeNull();
  });

  it("refuses to save a rating on someone else's result", async () => {
    await expect(saveRating(USER_B, SEARCH, "r1", true, null)).rejects.toBeInstanceOf(MissingError);
  });
});
