import { redirect } from "next/navigation";
import { getUser } from "@/lib/server/auth";
import AppShell from "@/components/AppShell";

export const dynamic = "force-dynamic";

export default async function AuthenticatedLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const user = await getUser();
  if (!user) redirect("/sign-in");
  return <AppShell email={user.email ?? ""}>{children}</AppShell>;
}
