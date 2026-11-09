"use client";

import { useEffect, useId, useRef, useState } from "react";
import type { ResultView } from "@/lib/contract";

type Relevant = boolean | null;

type SaveKind = "rating" | "note";

interface LocalItem extends ResultView {
  saving: SaveKind | null;
  saveError: string | null;
}

/**
 * Shared result list. Used by the search page right after retrieval and by
 * the history page when a past search is opened: a result is one passage
 * with one judgment, independently of how the user arrived.
 */
export default function ResultsList({
  searchId,
  results,
}: {
  searchId: string;
  results: ResultView[];
}) {
  const [items, setItems] = useState<LocalItem[]>(() =>
    results.map((r) => ({ ...r, saving: null, saveError: null }))
  );

  useEffect(() => {
    setItems(results.map((r) => ({ ...r, saving: null, saveError: null })));
  }, [results]);

  const rated = items.filter((i) => i.relevant !== null).length;

  const saves = useRef(new Map<string, Promise<void>>());

  const patch = (
    resultId: string,
    relevant: Relevant,
    note: string | null,
    kind: SaveKind,
  ) => {
    const previous = saves.current.get(resultId) ?? Promise.resolve();
    setItems((items) => items.map((item) =>
      item.id === resultId ? { ...item, saving: kind, saveError: null } : item
    ));

    // Clicking a rating can blur the note input first. Save in that order
    // so a note request cannot restore a note after Clear.
    const request: Promise<void> = previous.then(async () => {
      let error: string | null = null;
      try {
        const response = await fetch(`/api/searches/${searchId}/ratings/${resultId}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ relevant, note }),
        });
        const data = await response.json().catch(() => null);
        if (!response.ok || !data?.ok) error = data?.error ?? "Save failed.";
      } catch {
        error = "Network error. Your change was not saved.";
      }

      const finished = saves.current.get(resultId) === request;
      if (finished) saves.current.delete(resultId);
      setItems((items) => items.map((item) =>
        item.id === resultId ? {
          ...item,
          ...(error === null ? { relevant, note } : {}),
          ...(finished ? { saving: null, saveError: error } : {}),
        } : item
      ));
    });
    saves.current.set(resultId, request);
  };

  const setJudgment = (item: LocalItem, relevant: Relevant) => {
    patch(item.id, relevant, relevant === null ? null : item.note, "rating");
  };

  const setNote = (item: LocalItem, note: string) => {
    setItems((items) => items.map((row) =>
      row.id === item.id ? { ...row, note } : row
    ));
  };

  const commitNote = (item: LocalItem) => {
    if (item.saving === "rating") return;
    patch(item.id, item.relevant, item.note?.trim() || null, "note");
  };

  return (
    <div className="results" aria-live="polite">
      <p className="progress-line">
        {rated} of {items.length} passages rated
      </p>
      <ol className="passage-list" role="list">
        {items.map((item, idx) => (
          <ResultRow
            key={item.id}
            item={item}
            index={idx}
            onJudge={setJudgment}
            onNoteChange={setNote}
            onNoteCommit={commitNote}
          />
        ))}
      </ol>
    </div>
  );
}

function ResultRow({
  item,
  index,
  onJudge,
  onNoteChange,
  onNoteCommit,
}: {
  item: LocalItem;
  index: number;
  onJudge: (item: LocalItem, relevant: Relevant) => void;
  onNoteChange: (item: LocalItem, note: string) => void;
  onNoteCommit: (item: LocalItem) => void;
}) {
  const id = useId();
  const [noteOpen, setNoteOpen] = useState(Boolean(item.note));

  const stateLine = item.saving
    ? "Saving…"
    : item.saveError
      ? `Not saved: ${item.saveError}`
      : "";

  return (
    <li className="result">
      <p className="passage-number">Passage {index + 1}</p>
      <p className="result-text">{item.text}</p>
      <div className="rating-controls" role="group" aria-label={`Rating for passage ${index + 1}`}>
        <button
          type="button"
          className="rating-button yes"
          aria-pressed={item.relevant === true}
          onClick={() => onJudge(item, true)}
          disabled={item.saving === "rating"}
        >
          Relevant
        </button>
        <button
          type="button"
          className="rating-button no"
          aria-pressed={item.relevant === false}
          onClick={() => onJudge(item, false)}
          disabled={item.saving === "rating"}
        >
          Not relevant
        </button>
        <button
          type="button"
          className="rating-button"
          aria-pressed={item.relevant === null}
          onClick={() => onJudge(item, null)}
          disabled={item.saving === "rating"}
          title="Clear the rating and note"
        >
          Clear
        </button>
        <span
          className={`rating-state ${item.saveError ? "error" : ""}`}
          aria-live="polite"
        >
          {stateLine}
        </span>
      </div>

      <div className="result-note">
        {noteOpen ? (
          <div>
            <label className="field-label" htmlFor={`${id}-note`}>
              Note <span style={{ textTransform: "none" }}>(optional)</span>
            </label>
            <input
              id={`${id}-note`}
              className="text-input"
              type="text"
              value={item.note ?? ""}
              onChange={(e) => onNoteChange(item, e.target.value)}
              onBlur={() => onNoteCommit(item)}
              maxLength={1000}
              disabled={item.saving !== null}
              placeholder="e.g. useful abstract, wrong compound class, uncertain"
            />
          </div>
        ) : (
          <button
            type="button"
            className="ghost-button"
            onClick={() => setNoteOpen(true)}
          >
            {item.note ? "Edit note" : "Add note"}
          </button>
        )}
      </div>
    </li>
  );
}
