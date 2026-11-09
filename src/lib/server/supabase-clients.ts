import "server-only";

import { cookies } from "next/headers";
import { createServerClient } from "@supabase/ssr";
import type { SupabaseClient } from "@supabase/supabase-js";

const url = () => {
  const u = process.env.NEXT_PUBLIC_SUPABASE_URL;
  if (!u) throw new Error("NEXT_PUBLIC_SUPABASE_URL is not set");
  return u;
};

/**
 * Cookie-bound client for auth flows (publishable key, reads request cookies).
 * The cookie store is captured once and written synchronously, so session
 * refreshes performed inside route handlers are visible to the same handler
 * (and to Supabase) on the very next auth call.
 */
export async function cookieClient(): Promise<SupabaseClient> {
  const cookieStore = await cookies();
  const key = process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY;
  if (!key) throw new Error("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY is not set");
  return createServerClient(url(), key, {
    cookieOptions: {
      httpOnly: true,
      sameSite: "lax",
      secure: process.env.NODE_ENV === "production",
    },
    cookies: {
      getAll: () => cookieStore.getAll(),
      setAll(cookieList) {
        try {
          for (const { name, value, options } of cookieList) {
            cookieStore.set(name, value, options);
          }
        } catch {
          // Writes can throw during a Server Component render; that render
          // never completes anyway and the next request re-authenticates.
        }
      },
    },
  });
}
