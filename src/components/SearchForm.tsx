"use client";

import { useEffect, useId, useRef, useState } from "react";

type Submit = (query: string) => void | Promise<void>;

const exampleQuery =
  "What explanations have been proposed for the photo-induced absorption bands observed in different quantum dot materials?";

export default function SearchForm({
  onSubmit,
  submitting,
  retainedQuery,
}: {
  onSubmit: Submit;
  submitting: boolean;
  retainedQuery?: string | null;
}) {
  const formId = useId();
  const [query, setQuery] = useState("");
  const [clientError, setClientError] = useState<string | null>(null);
  const inFlight = useRef(false);

  // If a search is running or just finished, its query is authoritative.
  useEffect(() => {
    if (retainedQuery != null) setQuery(retainedQuery);
  }, [retainedQuery]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (submitting || inFlight.current) return; // no duplicate submits
    if (!query.trim()) {
      setClientError("Enter a query before searching.");
      return;
    }
    setClientError(null);
    inFlight.current = true;
    try {
      await onSubmit(query.trim());
    } finally {
      inFlight.current = false;
    }
  }

  return (
    <form onSubmit={handleSubmit} noValidate>
      <label className="field-label" htmlFor={`${formId}-query`}>
        Chemistry query
      </label>
      <div className="query-box">
        <textarea
          id={`${formId}-query`}
          className="text-input"
          rows={2}
          aria-invalid={clientError ? true : undefined}
          aria-describedby={clientError ? `${formId}-err ${formId}-example` : `${formId}-example`}
          placeholder={exampleQuery}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              e.currentTarget.form?.requestSubmit();
            }
          }}
          disabled={submitting}
          maxLength={2000}
        />
        <button
          className="search-submit"
          type="submit"
          disabled={submitting}
        >
          {submitting ? (
            <span className="spinner" style={{ borderTopColor: "var(--paper)" }} aria-hidden="true" />
          ) : (
            <svg
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="round"
              aria-hidden="true"
            >
              <circle cx="11" cy="11" r="7" />
              <line x1="16.5" y1="16.5" x2="21" y2="21" />
            </svg>
          )}
          {submitting ? "Searching" : "Search"}
        </button>
      </div>
      {clientError && (
        <p className="error-text" id={`${formId}-err`} role="alert">
          {clientError}
        </p>
      )}

      <p className="hint query-example" id={`${formId}-example`}>
        For example, “{exampleQuery}”
      </p>
      <p className="hint">
        You’ll see 5 to 10 shuffled passages. Rate each one against your query.
      </p>
    </form>
  );
}
