import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ChEmbed RAG Evaluation",
  description: "Relevance judgments for chemistry retrieval results",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
