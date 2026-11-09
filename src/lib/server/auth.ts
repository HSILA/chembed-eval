import "server-only";

import { createClient, type SupabaseClient, type User } from "@supabase/supabase-js";

import { cookieClient } from "./supabase-clients";

export type { User } from "@supabase/supabase-js";

/**
 * Secret-key client. Bypasses RLS by design — every call site must scope rows
 * by the authenticated user's id before reading or writing. Import only from
 * server-only modules.
 */
export function secretClient(): SupabaseClient {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.SUPABASE_SECRET_KEY;
  if (!url || !key) throw new Error("SUPABASE_SECRET_KEY is not set");
  return createClient(url, key, { auth: { persistSession: false } });
}

/**
 * The authenticated user from the request cookies, or null. Uses getUser()
 * only — never getSession(), whose cookie payload is unverified.
 */
export async function getUser(): Promise<User | null> {
  // getUser() validates the JWT against the Auth server; never trust
  // getSession(), which only decodes cookies without verification.
  const { data, error } = await (await cookieClient()).auth.getUser();
  if (error) return null;
  return data.user;
}
