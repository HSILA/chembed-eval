"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import type { HistoryPageSize, ListSearchesResponse } from "@/lib/contract";
import { historyQuery } from "@/lib/history-pagination";

export default function HistoryWorkspace({
  page,
  pageSize,
}: {
  page: number;
  pageSize: HistoryPageSize;
}) {
  const router = useRouter();
  const [data, setData] = useState<ListSearchesResponse | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        const response = await fetch(`/api/searches?${historyQuery(page, pageSize)}`, {
          cache: "no-store",
          signal: controller.signal,
        });
        const body = await response.json().catch(() => null);
        if (controller.signal.aborted) return;
        if (!response.ok || !Array.isArray(body?.searches)) {
          setListError(body?.error ?? "Could not load your searches.");
          return;
        }
        setData(body as ListSearchesResponse);
      } catch {
        if (!controller.signal.aborted) setListError("Could not load your searches.");
      }
    }
    void load();
    return () => controller.abort();
  }, [page, pageSize, attempt]);

  const total = data?.total ?? 0;
  const currentPage = data?.page ?? page;
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const offset = (currentPage - 1) * pageSize;

  return (
    <section aria-label="History">
      <h1 className="page-title">Search history</h1>
      <p className="page-intro">
        Open a past search to review or update its ratings.
      </p>

      <div className="history-page-size">
        <label htmlFor="history-page-size-select">Searches per page</label>
        <select
          id="history-page-size-select"
          value={pageSize}
          onChange={(event) => {
            const size = event.target.value === "20" ? 20 : 10;
            router.push(`/history?${historyQuery(1, size)}`);
          }}
        >
          <option value={10}>10</option>
          <option value={20}>20</option>
        </select>
      </div>

      {!data && !listError && (
        <div className="status-bar pending" role="status">
          <span className="spinner" aria-hidden="true" /> Loading searches…
        </div>
      )}

      {data && data.searches.length === 0 && (
        <div className="status-bar">
          <span>No searches yet. Start one on the Search page.</span>
        </div>
      )}

      {listError && (
        <div className="status-bar error" role="alert">
          <span>{listError}</span>
          <button
            className="retry-button"
            type="button"
            onClick={() => {
              setListError(null);
              setAttempt((n) => n + 1);
            }}
          >
            Retry
          </button>
        </div>
      )}

      {data && data.searches.length > 0 && (
        <>
          <ul className="history-list" role="list">
            {data.searches.map((s) => (
              <li key={s.id}>
                <Link
                  href={`/history/${s.id}?${historyQuery(currentPage, pageSize)}`}
                  className="history-item"
                >
                  <span className="q">{s.query || "(untitled query)"}</span>
                  <span className="history-meta">
                    <span>
                      {s.status === "ready"
                        ? `${s.ratedCount} of ${s.resultCount} rated`
                        : s.status === "failed" ? "Search failed" : "Searching…"}
                    </span>
                    <time dateTime={s.createdAt}>
                      {new Date(s.createdAt).toLocaleString(undefined, {
                        dateStyle: "medium",
                        timeStyle: "short",
                      })}
                    </time>
                  </span>
                </Link>
              </li>
            ))}
          </ul>

          <nav className="history-pager" aria-label="History pagination">
            {currentPage > 1 ? (
              <Link
                className="pager-link"
                href={`/history?${historyQuery(currentPage - 1, pageSize)}`}
                aria-label={`Previous page, page ${currentPage - 1} of ${totalPages}`}
              >
                Previous
              </Link>
            ) : (
              <span className="pager-link pager-disabled" aria-hidden="true">
                Previous
              </span>
            )}
            <span className="pager-status" role="status">
              {offset + 1} to {offset + data.searches.length} of {total}
            </span>
            {currentPage < totalPages ? (
              <Link
                className="pager-link"
                href={`/history?${historyQuery(currentPage + 1, pageSize)}`}
                aria-label={`Next page, page ${currentPage + 1} of ${totalPages}`}
              >
                Next
              </Link>
            ) : (
              <span className="pager-link pager-disabled" aria-hidden="true">
                Next
              </span>
            )}
          </nav>

        </>
      )}
    </section>
  );
}
