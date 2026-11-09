import HistoryDetail from "@/components/HistoryDetail";
import { historyQuery, parseHistoryPaging } from "@/lib/history-pagination";

export default async function SearchDetailPage({
  params,
  searchParams,
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ page?: string | string[]; pageSize?: string | string[] }>;
}) {
  const [{ id }, query] = await Promise.all([params, searchParams]);
  const { page, pageSize } = parseHistoryPaging(query.page, query.pageSize);
  return <HistoryDetail key={id} id={id} backQuery={historyQuery(page, pageSize)} />;
}
