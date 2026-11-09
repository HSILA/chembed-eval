"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useState } from "react";

export default function AppShell({
  email,
  children,
}: {
  email: string;
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const router = useRouter();
  const [signingOut, setSigningOut] = useState(false);

  const isSearch = pathname === "/";
  const isHistory = pathname === "/history" || pathname.startsWith("/history/");

  async function signOut() {
    if (signingOut) return;
    setSigningOut(true);
    try {
      await fetch("/api/auth/sign-out", { method: "POST" });
    } finally {
      router.replace("/sign-in");
      router.refresh();
    }
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <Link href="/" className="brand">
          ChEmbed RAG Evaluation
        </Link>
        <nav className="nav" aria-label="Main">
          <Link href="/" aria-current={isSearch ? "page" : undefined}>
            Search
          </Link>
          <Link href="/history" aria-current={isHistory ? "page" : undefined}>
            History
          </Link>
        </nav>
        <div className="sidebar-footer">
          <p className="user-line">{email}</p>
          <button className="ghost-button" onClick={signOut} disabled={signingOut}>
            {signingOut ? "Signing out…" : "Sign out"}
          </button>
        </div>
      </aside>
      <main className="main">{children}</main>
    </div>
  );
}
