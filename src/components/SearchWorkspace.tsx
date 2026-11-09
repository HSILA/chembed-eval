"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import SearchForm from "@/components/SearchForm";
import ResultsList from "@/components/ResultsList";
import type { SearchDetailResponse, ResultView } from "@/lib/contract";

export type RunStatus = "idle" | "pending" | "ready" | "failed";

/** Valid random v4 UUID, with a spec-compliant fallback for browsers
 * lacking crypto.randomUUID (insecure contexts, older Safari). */
function makeUUID(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
  bytes[8] = (bytes[8] & 0x3f) | 0x80; // RFC 4122 variant
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0"));
  return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex
    .slice(6, 8)
    .join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10, 16).join("")}`;
}

/**
 * Current search run. A new submit generates a fresh UUID; a retry of the
 * same query reuses the last POSTed id so the backend
 * treats it as the same search instead of creating a duplicate.
 */
export default function SearchWorkspace() {
  const [status, setStatus] = useState<RunStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<ResultView[] | null>(null);
  const [runSearchId, setRunSearchId] = useState<string | null>(null);
  const [savedQuery, setSavedQuery] = useState<string | null>(null);

  const runIdRef = useRef<string | null>(null);
  const lastPostedQueryRef = useRef<string | null>(null);
  const pollRef = useRef<number | null>(null);
  /** Increments per submit; poll/after-POST responses only apply state when
   * their token still matches, so stale responses can't overwrite a newer run. */
  const tokenRef = useRef(0);

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => stopPolling, [stopPolling]);

  /** GET one search. Applies state only if its token is still current.
   * Returns the terminal/pending status, or null on failure. Never starts
   * or leaves an interval running on a non-pending outcome. */
  const loadResults = useCallback(
    async (id: string, runToken: number): Promise<"pending" | "ready" | "failed" | null> => {
      let parsed: unknown;
      try {
        const res = await fetch(`/api/searches/${id}`, { cache: "no-store" });
        parsed = await res.json().catch(() => null);
      } catch {
        parsed = null;
      }
      if (tokenRef.current !== runToken) return null; // superseded run; no state updates

      // Boundary: parse to the shared contract type; fall back to a failure
      // state if the shape doesn't match.
      let data: SearchDetailResponse | null = null;
      if (typeof parsed === "object" && parsed !== null && "search" in parsed) {
        const candidate = parsed as SearchDetailResponse;
        if (
          candidate.search &&
          typeof candidate.search === "object" &&
          typeof candidate.search.id === "string" &&
          typeof candidate.search.status === "string"
        ) {
          data = candidate;
        }
      }
      if (data === null) {
        const apiError =
          typeof parsed === "object" &&
          parsed !== null &&
          "error" in parsed &&
          typeof parsed.error === "string"
            ? parsed.error
            : null;
        stopPolling();
        setError(apiError ?? "Could not load results.");
        setStatus("failed");
        return "failed";
      }
      const search = data.search;
      setSavedQuery(search.query);
      setRunSearchId(search.id);
      if (search.status === "failed") {
        stopPolling();
        setError("The search failed on the server. You can retry it.");
        setStatus("failed");
        return "failed";
      }
      if (search.status === "ready") {
        stopPolling();
        // API returns results already ordered by display position.
        setResults(data.results ?? []);
        setStatus("ready");
        return "ready";
      }
      return "pending";
    },
    [stopPolling]
  );

  const startPolling = useCallback(
    (id: string) => {
      stopPolling();
      const runToken = tokenRef.current;
      pollRef.current = window.setInterval(() => {
        void loadResults(id, runToken);
      }, 2000);
    },
    [loadResults, stopPolling]
  );

  const submit = useCallback(
    async (query: string, retry = false) => {
      stopPolling();
      const runToken = ++tokenRef.current;

      // Only an explicit retry reuses the previous run.
      const isRetry =
        retry &&
        runIdRef.current !== null &&
        lastPostedQueryRef.current === query;
      if (!isRetry || !runIdRef.current) {
        runIdRef.current = makeUUID();
      }
      // Remember the input before POSTing so the retry button has it
      // even if the POST itself fails.
      lastPostedQueryRef.current = query;
      const id = runIdRef.current;

      setError(null);
      setResults(null);
      setStatus("pending");
      setSavedQuery(query);

      let postedId: string | null;
      try {
        const res = await fetch("/api/searches", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id, query }),
        });
        const parsed: unknown = await res.json().catch(() => null);
        if (tokenRef.current !== runToken) return; // superseded
        const postedObj: Record<string, unknown> | null =
          typeof parsed === "object" && parsed !== null
            ? (parsed as Record<string, unknown>)
            : null;
        const rawPostedId: unknown = postedObj !== null ? postedObj.id : null;
        const postedIdValue = typeof rawPostedId === "string" ? rawPostedId : null;
        if (postedIdValue === null) {
          const rawError: unknown = postedObj !== null ? postedObj.error : null;
          const apiError = typeof rawError === "string" ? rawError : null;
          setError(apiError ?? "Could not start the search.");
          setStatus("failed");
          return;
        }
        postedId = postedIdValue;
      } catch {
        if (tokenRef.current !== runToken) return; // superseded
        setError("Network error while starting the search.");
        setStatus("failed");
        return;
      }

      // Poll only while the API reports the search as pending.
      const outcome = await loadResults(postedId, runToken);
      if (outcome === "pending" && tokenRef.current === runToken && pollRef.current === null) {
        startPolling(postedId);
      }
    },
    [loadResults, startPolling, stopPolling]
  );

  return (
    <section aria-label="Search">
      <h1 className="page-title">Search passages</h1>

      <SearchForm
        onSubmit={submit}
        submitting={status === "pending"}
        retainedQuery={savedQuery}
      />

      {status === "pending" && (
        <div className="status-bar pending" role="status" aria-live="polite">
          <span className="spinner" aria-hidden="true" />
          Retrieving passages…
        </div>
      )}

      {status === "failed" && error && (
        <div className="status-bar error" role="alert">
          <span>{error}</span>
          <button
            className="retry-button"
            type="button"
            disabled={!savedQuery}
            onClick={() => savedQuery && submit(savedQuery, true)}
          >
            Retry search
          </button>
        </div>
      )}

      {results && status === "ready" && runSearchId && (
        <ResultsList searchId={runSearchId} results={results} />
      )}
    </section>
  );
}
