"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import ResultsList from "@/components/ResultsList";
import type { SearchDetailResponse } from "@/lib/contract";
import { mayBeInterrupted } from "@/lib/search-status";

export default function HistoryDetail({
  id,
  backQuery,
}: {
  id: string;
  backQuery: string;
}) {
  const [detail, setDetail] = useState<SearchDetailResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`/api/searches/${encodeURIComponent(id)}`, {
      cache: "no-store",
      signal: controller.signal,
    })
      .then(async (res) => {
        const data = await res.json().catch(() => null);
        if (controller.signal.aborted) return;
        // Generic message for not-owned/invalid/not-found — no existence oracle.
        if (!res.ok || !data?.search) {
          setError("Could not load that search.");
          return;
        }
        setDetail({ search: data.search, results: data.results ?? [] });
      })
      .catch(() => {
        if (!controller.signal.aborted) setError("Could not load that search.");
      });
    return () => controller.abort();
  }, [id, attempt]);

  function reload() {
    setDetail(null);
    setError(null);
    setAttempt((count) => count + 1);
  }

  return (
    <section aria-label="Saved search">
      <Link href={`/history?${backQuery}`} className="back-link">
        Back to history
      </Link>

      <h1 className="page-title">
        {detail ? detail.search.query || "(untitled query)" : "Saved search"}
      </h1>

      {!detail && !error && (
        <div className="status-bar pending" role="status">
          <span className="spinner" aria-hidden="true" /> Loading results…
        </div>
      )}
      {error && (
        <div className="status-bar error" role="alert">
          <span>{error}</span>
          <button type="button" className="retry-button" onClick={reload}>Retry</button>
        </div>
      )}
      {detail && detail.search.status === "failed" && (
        <div className="status-bar error" role="alert">
          <span>This search failed on the server; it has no results.</span>
        </div>
      )}
      {detail?.search.status === "pending" && mayBeInterrupted(detail.search.status, detail.search.createdAt) && (
        <div className="status-bar" role="status">
          <span>This search may have stopped before finishing. Please start a new search with the same question.</span>
          <Link className="retry-button" href="/">New search</Link>
        </div>
      )}
      {detail?.search.status === "pending" && !mayBeInterrupted(detail.search.status, detail.search.createdAt) && (
        <div className="status-bar" role="status">
          <span>This search is still running.</span>
          <button type="button" className="retry-button" onClick={reload}>Check again</button>
        </div>
      )}
      {detail?.search.status === "ready" && (
        <ResultsList searchId={detail.search.id} results={detail.results} />
      )}
    </section>
  );
}
