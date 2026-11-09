import { NextResponse, type NextRequest } from "next/server";

import { jsonError } from "@/lib/server/api";
import { cookieClient } from "@/lib/server/supabase-clients";

/**
 * POST /api/auth/sign-in — {email, password} → {ok:true}. The only public
 * route. Supabase Auth sets the session cookies via @supabase/ssr.
 */
export async function POST(request: NextRequest) {
  let body: { email?: unknown; password?: unknown };
  try {
    body = await request.json();
  } catch {
    return jsonError("invalid JSON body");
  }
  if (typeof body.email !== "string" || typeof body.password !== "string") {
    return jsonError("email and password are required");
  }
  const { error } = await (await cookieClient()).auth.signInWithPassword({
    email: body.email,
    password: body.password,
  });
  if (error) return jsonError("sign-in failed", 401);
  return NextResponse.json({ ok: true });
}
