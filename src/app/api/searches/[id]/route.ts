import { NextResponse, type NextRequest } from "next/server";

import { jsonError, requireUser, toErrorResponse } from "@/lib/server/api";
import { getSearchDetail } from "@/lib/server/db";
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
    const detail = await getSearchDetail(auth.user.id, id);
    if (!detail) return jsonError("not found", 404);
    const { search, results } = detail;
    const body: SearchDetailResponse = {
      search: {
        id: search.id,
        query: search.query,
        informationNeed: search.information_need,
        createdAt: search.created_at,
        status: search.status,
        ratedCount: results.filter((result) => result.relevant !== null).length,
        resultCount: results.length,
      },
      results,
    };
    return NextResponse.json(body);
  } catch (err) {
    return toErrorResponse(err);
  }
}
