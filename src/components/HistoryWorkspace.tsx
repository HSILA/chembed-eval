"use client";

import { useCallback, useEffect, useState } from "react";
import ResultsList from "@/components/ResultsList";
import type { SearchSummary, SearchDetailResponse } from "@/lib/contract";

export default function HistoryWorkspace() {
  const [searches, setSearches] = useState<SearchSummary[] | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<SearchDetailResponse | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  const loadList = useCallback(async () => {
    setListError(null);
    try {
      const res = await fetch("/api/searches", { cache: "no-store" });
      const data = await res.json().catch(() => null);
      if (!res.ok) {
        setListError(data?.error ?? "Could not load your searches.");
        setSearches([]);
        return;
      }
      setSearches(data.searches ?? []);
    } catch {
      setListError("Network error while loading your searches.");
      setSearches([]);
    }
  }, []);

  useEffect(() => {
    void loadList();
  }, [loadList]);

  const open = useCallback(async (id: string) => {
    setSelectedId(id);
    setDetailLoading(true);
    setDetailError(null);
    setDetail(null);
    try {
      const res = await fetch(`/api/searches/${id}`, { cache: "no-store" });
      const data = await res.json().catch(() => null);
      if (!res.ok || !data?.search) {
        setDetailError(data?.error ?? "Could not load that search.");
        return;
      }
      setDetail({
        search: data.search,
        results: data.results ?? [],
      });
    } catch {
      setDetailError("Network error while loading that search.");
    } finally {
      setDetailLoading(false);
    }
  }, []);

  return (
    <section aria-label="History">
      <h1 className="page-title">Search history</h1>
      <p className="page-intro">
        Open a past search to review or update its ratings.
      </p>

      {searches === null && (
        <div className="status-bar pending" role="status">
          <span className="spinner" aria-hidden="true" /> Loading searches…
        </div>
      )}

      {searches !== null && searches.length === 0 && (
        <div className="status-bar">
          <span>No searches yet. Start one on the Search page.</span>
        </div>
      )}

      {listError && (
        <div className="status-bar error" role="alert">
          <span>{listError}</span>
          <button className="retry-button" type="button" onClick={() => void loadList()}>
            Retry
          </button>
        </div>
      )}

      {searches !== null && searches.length > 0 && (
        <ul className="history-list" role="list">
          {searches.map((s) => (
            <li key={s.id}>
              <button
                type="button"
                className="history-item"
                aria-current={selectedId === s.id}
                onClick={() => void open(s.id)}
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
              </button>
            </li>
          ))}
        </ul>
      )}

      {detailLoading && (
        <div className="status-bar pending" role="status">
          <span className="spinner" aria-hidden="true" /> Loading results…
        </div>
      )}
      {detailError && (
        <div className="status-bar error" role="alert">
          <span>{detailError}</span>
          <button
            className="retry-button"
            type="button"
            onClick={() => selectedId && void open(selectedId)}
          >
            Retry
          </button>
        </div>
      )}
      {detail && detail.search.status === "failed" && (
        <div className="status-bar error" role="alert">
          <span>This search failed on the server; it has no results.</span>
        </div>
      )}
      {detail && detail.search.status !== "failed" && (
        <ResultsList searchId={detail.search.id} results={detail.results} />
      )}
    </section>
  );
}
