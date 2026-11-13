// A search normally finishes within the server's 120-second limit. Allow
// extra time before suggesting a new search, without changing saved data.
export function mayBeInterrupted(status: string, createdAt: string, now = Date.now()): boolean {
  const started = Date.parse(createdAt);
  return status === "pending" && Number.isFinite(started) && now - started > 10 * 60 * 1000;
}
