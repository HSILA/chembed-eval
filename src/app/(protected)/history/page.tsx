import HistoryWorkspace from "@/components/HistoryWorkspace";
import { parseHistoryPaging } from "@/lib/history-pagination";

export default async function HistoryPage({
  searchParams,
}: {
  searchParams: Promise<{ page?: string | string[]; pageSize?: string | string[] }>;
}) {
  const params = await searchParams;
  const { page, pageSize } = parseHistoryPaging(params.page, params.pageSize);
  return <HistoryWorkspace key={`${page}:${pageSize}`} page={page} pageSize={pageSize} />;
}
