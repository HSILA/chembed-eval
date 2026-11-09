import { NextResponse } from "next/server";

import { getUser } from "@/lib/server/auth";
import { cookieClient } from "@/lib/server/supabase-clients";

/** POST /api/auth/sign-out — clears the session; always answers {ok:true}. */
export async function POST() {
  const user = await getUser();
  if (user) await (await cookieClient()).auth.signOut();
  return NextResponse.json({ ok: true });
}
