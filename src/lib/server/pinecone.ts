import "server-only";

import { Pinecone, type Index, type RecordMetadata } from "@pinecone-database/pinecone";

import {
  EMBED_DIM,
  MODEL_KEYS,
  PINECONE_INDEX_FOR_KEY,
  PINECONE_NAMESPACE,
  TOP_K,
  type ModelKey,
} from "./search-config";

export type PineconeHit = { originalPassageId: string; score: number };
type QueryVector = number[];

const indexes = new Map<ModelKey, Index<RecordMetadata>>();

function indexFor(modelKey: ModelKey): Index<RecordMetadata> {
  let idx = indexes.get(modelKey);
  if (!idx) {
    const apiKey = process.env.PINECONE_API_KEY;
    if (!apiKey) throw new Error("PINECONE_API_KEY is not set");
    idx = new Pinecone({ apiKey }).index<RecordMetadata>(PINECONE_INDEX_FOR_KEY[modelKey]);
    indexes.set(modelKey, idx);
  }
  return idx;
}

function dimCheck(vec: QueryVector, modelKey: ModelKey) {
  if (vec.length !== EMBED_DIM) {
    throw new Error(`${modelKey}: query vector has ${vec.length} dims, expected ${EMBED_DIM}`);
  }
}

/** Top-K single-positive query in the fixed protocol namespace. */
export async function queryIndex(
  modelKey: ModelKey,
  vector: QueryVector,
): Promise<PineconeHit[]> {
  dimCheck(vector, modelKey);
  const namespace = indexFor(modelKey).namespace(PINECONE_NAMESPACE);
  const res = await namespace.query({ vector, topK: TOP_K, includeMetadata: false });
  const hits: PineconeHit[] = [];
  for (const match of res.matches ?? []) {
    if (typeof match.id === "string" && typeof match.score === "number") {
      hits.push({ originalPassageId: match.id, score: match.score });
    }
  }
  return hits;
}

/** Query every index in parallel. */
export async function queryAllIndexes(
  vectors: Record<ModelKey, QueryVector>,
): Promise<Record<ModelKey, PineconeHit[]>> {
  const [base, prog] = await Promise.all(MODEL_KEYS.map((k) => queryIndex(k, vectors[k])));
  return { base, prog };
}
