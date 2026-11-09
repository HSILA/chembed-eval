import { NextResponse, type NextRequest } from "next/server";

import { jsonError, requireUser, toErrorResponse } from "@/lib/server/api";
import { getSearchDetail, getSearchSummary } from "@/lib/server/db";
import type { SearchDetailResponse } from "@/lib/contract";

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/** GET /api/searches/[id] — search summary plus displayed results. */
export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const auth = await requireUser();
  if ("response" in auth) return auth.response;
  const { id } = await params;
  if (!UUID_RE.test(id)) return jsonError("not found", 404);

  try {
    const summary = await getSearchSummary(auth.user.id, id);
    if (!summary) return jsonError("not found", 404);
    const detail = await getSearchDetail(auth.user.id, id);
    if (!detail) return jsonError("not found", 404);
    const body: SearchDetailResponse = {
      search: {
        id: summary.id,
        query: summary.query,
        informationNeed: summary.information_need,
        createdAt: summary.created_at,
        status: summary.status,
        ratedCount: summary.ratedCount,
        resultCount: summary.resultCount,
      },
      results: detail.results,
    };
    return NextResponse.json(body);
  } catch (err) {
    return toErrorResponse(err);
  }
}
