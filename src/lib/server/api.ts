import { NextResponse } from "next/server";
import { getUser, type User } from "@/lib/server/auth";
import { MissingError } from "@/lib/server/db";

export type AuthResult = { user: User } | { response: NextResponse };

/** JSON body helper: all API errors are {error: string}. */
export function jsonError(message: string, status = 400) {
  return NextResponse.json({ error: message }, { status });
}

/**
 * Require an authenticated user. Sign-in is the only public route; call this
 * first in every other handler.
 */
export async function requireUser(): Promise<AuthResult> {
  const user = await getUser();
  if (!user) return { response: jsonError("not signed in", 401) };
  return { user };
}

/** Convert thrown errors to the uniform {error} response; 404 for MissingError. */
export function toErrorResponse(err: unknown): NextResponse {
  if (err instanceof MissingError) return jsonError("not found", 404);
  return jsonError("Something went wrong. Please try again.", 500);
}
