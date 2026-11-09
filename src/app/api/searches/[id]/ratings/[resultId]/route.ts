import { NextResponse, type NextRequest } from "next/server";

import { jsonError, requireUser, toErrorResponse } from "@/lib/server/api";
import { saveRating } from "@/lib/server/db";
import type { SimpleOkResponse } from "@/lib/contract";

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/** PATCH /api/searches/[id]/ratings/[resultId] — {relevant, note} → {ok:true}. */
export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ id: string; resultId: string }> },
) {
  const auth = await requireUser();
  if ("response" in auth) return auth.response;
  const { id, resultId } = await params;
  if (!UUID_RE.test(id) || !UUID_RE.test(resultId)) return jsonError("not found", 404);

  let body: { relevant?: unknown; note?: unknown };
  try {
    body = await request.json();
  } catch {
    return jsonError("invalid JSON body");
  }
  if (typeof body.relevant !== "boolean" && body.relevant !== null) {
    return jsonError("relevant must be a boolean or null");
  }
  if (body.note !== undefined && body.note !== null && typeof body.note !== "string") {
    return jsonError("note must be a string or null");
  }
  if (typeof body.note === "string" && body.note.length > 2000) {
    return jsonError("note too long");
  }

  try {
    await saveRating(auth.user.id, id, resultId, body.relevant, body.note ?? null);
    const out: SimpleOkResponse = { ok: true };
    return NextResponse.json(out);
  } catch (err) {
    return toErrorResponse(err);
  }
}
