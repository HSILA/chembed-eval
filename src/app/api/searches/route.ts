import { randomUUID } from "node:crypto";

import { NextResponse, type NextRequest } from "next/server";

import { jsonError, requireUser, toErrorResponse } from "@/lib/server/api";
import { createSearch, getSearchDetail, listSearches, type CreateSearchClaim } from "@/lib/server/db";
import { settingsSnapshot } from "@/lib/server/search-config";
import { runSearchPipeline } from "@/lib/server/pipeline";
import type {
  CreateSearchResponse,
  ListSearchesResponse,
  SearchDetailResponse,
} from "@/lib/contract";

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const MAX_QUERY_LEN = 500;
const MAX_NEED_LEN = 500;

/** Rated-count/result-count rollups for a user's searches. */
async function summaryRows(userId: string) {
  const rows = await listSearches(userId);
  return Promise.all(
    rows.map(async (row) => {
      const detail = await getSearchDetail(userId, row.id);
      const ratedCount = detail
        ? detail.results.filter((r) => r.relevant !== null).length
        : 0;
      const resultCount = detail ? detail.results.length : 0;
      return { row, ratedCount, resultCount };
    }),
  );
}

/** GET /api/searches — the user's searches, newest first. */
export async function GET() {
  const auth = await requireUser();
  if ("response" in auth) return auth.response;
  try {
    const summaries = await summaryRows(auth.user.id);
    const body: ListSearchesResponse = {
      searches: summaries.map(({ row, ratedCount, resultCount }) => ({
        id: row.id,
        query: row.query,
        informationNeed: row.information_need,
        createdAt: row.created_at,
        status: row.status,
        ratedCount,
        resultCount,
      })),
    };
    return NextResponse.json(body);
  } catch (err) {
    return toErrorResponse(err);
  }
}

/**
 * POST /api/searches — {id?, query, informationNeed?} → {id}. The id is
 * caller-supplied when resuming; generated otherwise. The id is reserved
 * exactly once via an ignoreDuplicates upsert; only the reserving request (or
 * a claim of a failed run) executes the pipeline. An in-flight or already
 * ready reservation answers with the same id without rerunning anything.
 * maxDuration 120 covers cold model containers plus both Pinecone queries.
 */
export const maxDuration = 120;

export async function POST(request: NextRequest) {
  const auth = await requireUser();
  if ("response" in auth) return auth.response;

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return jsonError("invalid JSON body");
  }
  if (body === null || typeof body !== "object") {
    return jsonError("body must be a JSON object");
  }
  const { id: rawId, query: rawQuery, informationNeed: rawNeed } = body as {
    id?: unknown;
    query?: unknown;
    informationNeed?: unknown;
  };
  if (typeof rawQuery !== "string" || rawQuery.trim().length === 0) {
    return jsonError("query is required");
  }
  if (rawQuery.length > MAX_QUERY_LEN) return jsonError("query too long");
  if (
    rawNeed !== undefined &&
    rawNeed !== null &&
    (typeof rawNeed !== "string" || rawNeed.length > MAX_NEED_LEN)
  ) {
    return jsonError("informationNeed must be a string (max 500 chars) or null");
  }
  const id = typeof rawId === "string" && rawId.length > 0 ? rawId : randomUUID();
  if (!UUID_RE.test(id)) return jsonError("id must be a UUID");
  const query = rawQuery;
  const informationNeed =
    typeof rawNeed === "string" && rawNeed.length > 0 ? rawNeed : null;

  try {
    const claim = await createSearch(
      auth.user.id,
      id,
      query,
      informationNeed,
      settingsSnapshot(),
    );
    if (claim.kind === "conflict") {
      return jsonError("id already exists for a different owner or query", 409);
    }
    if (claim.kind === "inflight" && claim.row.status === "pending") {
      // An existing pending search already has a request processing it.
      const out: CreateSearchResponse = { id };
      return NextResponse.json(out, { status: 201 });
    }
    await runSearchPipeline(auth.user.id, claim.row);
    const out: CreateSearchResponse = { id };
    return NextResponse.json(out, { status: 201 });
  } catch (err) {
    return toErrorResponse(err);
  }
}
