import "server-only";

import type { HistoryPageSize, ListSearchesResponse } from "@/lib/contract";
import { secretClient } from "./auth";

/** Row of public.searches. */
export type SearchRow = {
  id: string;
  user_id: string;
  query: string;
  information_need: string | null;
  created_at: string;
  status: "pending" | "ready" | "failed";
  is_practice: boolean;
  settings: Record<string, unknown>;
};

/** Row of public.results with private ranks/scores never sent to the client. */
type ResultRow = {
  id: string;
  search_id: string;
  passage_id: number;
  base_rank: number | null;
  base_score: number | null;
  progressive_rank: number | null;
  progressive_score: number | null;
  display_position: number;
};

/** Row of public.ratings. */
type RatingRow = { result_id: string; relevant: boolean | null; note: string | null };

const db = () => secretClient();

function assertOwner(userId: string, row: { user_id?: string } | null) {
  if (!row || row.user_id !== userId) {
    // Same generic error for not-found and not-yours: never confirm existence.
    throw new MissingError();
  }
}

export class MissingError extends Error {}

type HistoryRow = Pick<SearchRow, "id" | "query" | "information_need" | "created_at" | "status"> & {
  results: { ratings: Pick<RatingRow, "relevant"> | null }[];
};

/** One page of the user's searches, without loading passage text or private ranks. */
export async function listSearches(
  userId: string,
  requestedPage: number,
  pageSize: HistoryPageSize,
): Promise<ListSearchesResponse> {
  const client = db();
  const { count, error: countError } = await client
    .from("searches")
    .select("id", { count: "exact", head: true })
    .eq("user_id", userId);
  if (countError) throw countError;

  const total = count ?? 0;
  const page = Math.min(requestedPage, Math.max(1, Math.ceil(total / pageSize)));
  if (total === 0) return { searches: [], page, pageSize, total };

  const start = (page - 1) * pageSize;
  const { data, error } = await client
    .from("searches")
    .select("id, query, information_need, created_at, status, results(ratings(relevant))")
    .eq("user_id", userId)
    .order("created_at", { ascending: false })
    .order("id", { ascending: false })
    .range(start, start + pageSize - 1);
  if (error?.code === "PGRST103" && page > 1) return listSearches(userId, 1, pageSize);
  if (error) throw error;

  const searches = ((data ?? []) as unknown as HistoryRow[]).map((row) => {
    const results = row.status === "ready" ? row.results : [];
    return {
      id: row.id,
      query: row.query,
      informationNeed: row.information_need,
      createdAt: row.created_at,
      status: row.status,
      ratedCount: results.filter((result) => result.ratings?.relevant != null).length,
      resultCount: results.length,
    };
  });
  return { searches, page, pageSize, total };
}

/** Idempotent create: caller-generated UUID reserved exactly once. */
export type CreateSearchClaim =
  | { kind: "created"; row: SearchRow } // this caller inserted the row
  | { kind: "inflight"; row: SearchRow } // same owner, same input, still pending
  | { kind: "ready"; row: SearchRow } // same owner, same input, already ready
  | { kind: "conflict" }; // taken by another user, or same owner with different query

export async function createSearch(
  userId: string,
  id: string,
  query: string,
  informationNeed: string | null,
  settings: Record<string, unknown>,
): Promise<CreateSearchClaim> {
  // Reserving the id must never resurrect a completed search: the upsert is
  // ignoreDuplicates, so a ready/failed row is untouched.
  const client = db();

  // Try to reserve the id. ignoreDuplicates: a concurrent identical POST or a
  // retry of this one hits the primary key and this call becomes a no-op.
  const { data: inserted, error: insErr } = await client
    .from("searches")
    .upsert(
      { id, user_id: userId, query, information_need: informationNeed, status: "pending", settings },
      { onConflict: "id", ignoreDuplicates: true },
    )
    .select("*")
    .maybeSingle();

  if (insErr) throw insErr;
  if (inserted) return { kind: "created", row: inserted as SearchRow };

  // Insert lost a race (or is a retry). Only the same owner with the exact
  // same query/need may piggyback; anything else is not-found-equivalent.
  const { data: existing, error: selErr } = await client
    .from("searches")
    .select("*")
    .eq("id", id)
    .maybeSingle();
  if (selErr) throw selErr;
  if (!existing) return { kind: "conflict" };
  const row = existing as SearchRow;
  if (row.user_id !== userId || row.query !== query) return { kind: "conflict" };
  if ((existing as SearchRow).information_need !== informationNeed) {
    return { kind: "conflict" };
  }
  if (row.status === "ready") return { kind: "ready", row };
  return { kind: "inflight", row };
}

export async function setSearchStatus(
  userId: string,
  id: string,
  status: "pending" | "ready" | "failed",
): Promise<void> {
  const { error } = await db()
    .from("searches")
    .update({ status })
    .eq("id", id)
    .eq("user_id", userId);
  if (error) throw error;
}

/**
 * Conditional status flip: succeeds (true) only if the row was in
 * `fromStatus`. Used to atomically claim a failed search for a retry.
 */
export async function setSearchStatusIfStatus(
  userId: string,
  id: string,
  fromStatus: "pending" | "ready" | "failed",
  toStatus: "pending" | "ready" | "failed",
): Promise<boolean> {
  const { data, error } = await db()
    .from("searches")
    .update({ status: toStatus })
    .eq("id", id)
    .eq("user_id", userId)
    .eq("status", fromStatus)
    .select("id");
  if (error) throw error;
  return (data ?? []).length > 0;
}

/**
 * A result batch is inserted atomically. Its presence lets a retry finish
 * marking the search ready without replacing the original display order.
 */
export async function hasSavedResults(searchId: string): Promise<boolean> {
  const { data, error } = await db()
    .from("results")
    .select("id")
    .eq("search_id", searchId)
    .limit(1);
  if (error) throw error;
  return Boolean(data?.length);
}

/** Store private ranks, scores, and display order in one batch. */
export async function saveResults(
  searchId: string,
  pool: {
    originalPassageId: string;
    rankFor: { base?: number; prog?: number };
    scoreFor: { base?: number; prog?: number };
    position: number;
  }[],
): Promise<void> {
  const ids = pool.map((p) => p.originalPassageId);
  const { data: passages, error: pErr } = await db()
    .from("passages")
    .select("id, original_passage_id")
    .in("original_passage_id", ids);
  if (pErr) throw pErr;
  const byOriginalId = new Map(
    (passages ?? []).map((p) => [p.original_passage_id, p.id as number]),
  );
  if (byOriginalId.size !== new Set(ids).size) {
    const missing = ids.filter((id) => !byOriginalId.has(id));
    throw new Error(`passages not found: ${missing.join(", ")}`);
  }

  const rows = pool.map((p) => ({
    search_id: searchId,
    passage_id: byOriginalId.get(p.originalPassageId)!,
    base_rank: p.rankFor.base ?? null,
    base_score: p.scoreFor.base ?? null,
    progressive_rank: p.rankFor.prog ?? null,
    progressive_score: p.scoreFor.prog ?? null,
    display_position: p.position,
  }));

  const { error } = await db().from("results").insert(rows);
  if (error) throw error;
}

/**
 * A search with its displayed results and ratings. Ownership is enforced by
 * user_id on the search row itself; results/ratings hang off it, so nothing
 * leaks without that row.
 */
export async function getSearchDetail(userId: string, searchId: string): Promise<{
  search: SearchRow;
  results: { id: string; text: string; relevant: boolean | null; note: string | null }[];
} | null> {
  const { data: search, error: sErr } = await db()
    .from("searches")
    .select("*")
    .eq("id", searchId)
    .eq("user_id", userId)
    .maybeSingle();
  if (sErr) throw sErr;
  if (!search) return null;
  const row = search as SearchRow;
  // Results only exist for ready searches; never expose partial/pending pools.
  if (row.status !== "ready") {
    return {
      search: {
        id: row.id,
        query: row.query,
        information_need: row.information_need,
        created_at: row.created_at,
        status: row.status,
      } as SearchRow,
      results: [],
    };
  }

  const { data, error } = await db()
    .from("results")
    .select("id, display_position, passages(text), ratings(relevant, note)")
    .eq("search_id", searchId)
    .order("display_position", { ascending: true });
  if (error) throw error;
  const rows = (data ?? []) as unknown as {
    id: string;
    display_position: number;
    passages: { text: string } | null; // to-one join → object, not array
    ratings: RatingRow | null;
  }[];
  return {
    // Public projection: internal user_id/is_practice/settings never leave db.ts.
    search: {
      id: search.id,
      query: search.query,
      information_need: search.information_need,
      created_at: search.created_at,
      status: search.status,
    } as SearchRow,
    results: rows.map((r) => {
      const rating = r.ratings ?? null;
      return {
        id: r.id,
        text: r.passages?.text ?? "",
        relevant: rating?.relevant ?? null,
        note: rating?.note ?? null,
      };
    }),
  };
}

/**
 * Upsert one rating after confirming the result belongs to the caller's
 * given search and that the search is ready. `updated_at` is bumped on every
 * write; `created_at` keeps its original value on conflict.
 */
export async function saveRating(
  userId: string,
  searchId: string,
  resultId: string,
  relevant: boolean | null,
  note: string | null,
): Promise<void> {
  const { data, error } = await db()
    .from("results")
    .select("id, searches!inner(user_id, status)")
    .eq("id", resultId)
    .eq("search_id", searchId)
    .maybeSingle();
  if (error) throw error;
  if (!data) throw new MissingError();
  const joined = data as unknown as { searches: { user_id: string; status: string } };
  assertOwner(userId, { user_id: joined.searches.user_id });
  if (joined.searches.status !== "ready") throw new MissingError();

  const { error: rErr } = await db()
    .from("ratings")
    .upsert(
      { result_id: resultId, relevant, note, updated_at: new Date().toISOString() },
      { onConflict: "result_id", ignoreDuplicates: false },
    );
  if (rErr) throw rErr;
}
