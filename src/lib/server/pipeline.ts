import "server-only";

import { poolHits, assignDisplayPositions } from "./pooling";
import { encodeQueryBoth } from "./modal-encode";
import { queryAllIndexes } from "./pinecone";
import { hasSavedResults, saveResults, setSearchStatus, setSearchStatusIfStatus, type SearchRow } from "./db";

/**
 * Run a newly reserved search or claim a failed search for retry.
 * Only one request may save the result batch.
 */
export async function runSearchPipeline(
  userId: string,
  search: SearchRow,
): Promise<void> {
  if (search.status === "ready") return;

  if (search.status === "failed") {
    // Conditional update: SET status = pending WHERE id AND status = failed.
    // A racing retry leaves the row pending and this update matches 0 rows.
    if (!(await setSearchStatusIfStatus(userId, search.id, "failed", "pending"))) {
      return; // someone else is running it; this post just returns the id
    }
  }

  try {
    if (search.status === "failed" && await hasSavedResults(search.id)) {
      await setSearchStatus(userId, search.id, "ready");
      return;
    }
    const vectors = await encodeQueryBoth(search.query);
    const hits = await queryAllIndexes(vectors);
    const pooled = poolHits(hits);
    if (!pooled.ok) throw new Error(pooled.reason);
    const positions = assignDisplayPositions(pooled.pool);
    const pool = pooled.pool.map((p) => ({
      originalPassageId: p.originalPassageId,
      rankFor: p.rankFor,
      scoreFor: p.scoreFor,
      position: positions.get(p.originalPassageId)!,
    }));
    await saveResults(search.id, pool);
    await setSearchStatus(userId, search.id, "ready");
  } catch (err) {
    try {
      await setSearchStatus(userId, search.id, "failed");
    } catch {
      // Status write failed but the originating error matters more.
    }
    throw err;
  }
}
