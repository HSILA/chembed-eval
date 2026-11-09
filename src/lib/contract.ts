/** Shared API contract: request bodies and response shapes. Client code may
 * import these types only — email/password and result metadata never carry
 * hidden fields (ranks, scores, model identity, protocol/passage internals). */

export type SearchStatus = "pending" | "ready" | "failed";

export type SearchSummary = {
  id: string;
  query: string;
  informationNeed: string | null;
  createdAt: string;
  status: SearchStatus;
  ratedCount: number;
  resultCount: number;
};

export type HistoryPageSize = 10 | 20;

export type ListSearchesResponse = {
  searches: SearchSummary[];
  page: number;
  pageSize: HistoryPageSize;
  total: number;
};

export type CreateSearchRequest = {
  id: string;
  query: string;
  informationNeed: string | null;
};

export type CreateSearchResponse = { id: string };

export type ResultView = {
  id: string;
  text: string;
  relevant: boolean | null;
  note: string | null;
};

export type SearchDetailResponse = {
  search: SearchSummary;
  results: ResultView[];
};

export type RatingRequest = { relevant: boolean | null; note: string | null };

export type SimpleOkResponse = { ok: true };

export type ApiErrorResponse = { error: string };
