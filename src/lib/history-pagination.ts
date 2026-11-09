import type { HistoryPageSize } from "./contract";

type SearchParam = string | string[] | null | undefined;

export function parseHistoryPaging(
  page: SearchParam,
  pageSize: SearchParam,
): { page: number; pageSize: HistoryPageSize } {
  const number = typeof page === "string" ? Number(page) : NaN;
  return {
    page: Number.isSafeInteger(number) && number > 0 ? number : 1,
    pageSize: pageSize === "20" ? 20 : 10,
  };
}

export function historyQuery(page: number, pageSize: HistoryPageSize): string {
  return `page=${page}&pageSize=${pageSize}`;
}
