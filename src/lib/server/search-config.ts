/**
 * Shared server constants: models, Pinecone indexes, namespace, and the
 * settings snapshot persisted with every search. Mirrors backend/protocol.py
 * (protocol id db72fa244e2915b2) — change both together, never live values
 * alone.
 */
export const MODEL_KEYS = ["base", "prog"] as const;
export type ModelKey = (typeof MODEL_KEYS)[number];

export const MODELS: Record<ModelKey, { hfName: string; revision: string }> = {
  base: {
    hfName: "nomic-ai/nomic-embed-text-v1",
    revision: "0759316f275aa0cb93a5b830973843ca66babcf5",
  },
  prog: {
    hfName: "BASF-AI/ChEmbed-prog",
    revision: "2cac1b6e7f9c5a419fcde18dac57efcde0f0da54",
  },
};

export const PROMPT_QUERY = "search_query: "; // trailing space matters
export const MAX_SEQ_LENGTH = 8192;
export const EMBED_DIM = 768;
export const TOP_K = 5; // hits per model
export const RRF_K = 60;
export const PROTOCOL_ID = "db72fa244e2915b2";
export const PINECONE_NAMESPACE = PROTOCOL_ID;

export const PINECONE_INDEX_FOR_KEY: Record<ModelKey, string> = {
  base: process.env.PINECONE_INDEX_BASE || "chembed-eval-base",
  prog: process.env.PINECONE_INDEX_PROGRESSIVE || "chembed-eval-progressive",
};

export const MODAL_APP_NAME = process.env.MODAL_APP_NAME || "chembed-eval-models";
export const MODAL_CLASS_FOR_KEY: Record<ModelKey, string> = {
  base: process.env.MODAL_CLASS_BASE || "ModelServerBase",
  prog: process.env.MODAL_CLASS_PROGRESSIVE || "ModelServerProg",
};

export function settingsSnapshot() {
  return {
    models: MODELS,
    prompt_query: PROMPT_QUERY,
    prompt_document: "search_document: ",
    max_seq_length: MAX_SEQ_LENGTH,
    embed_dim: EMBED_DIM,
    pooling: "mean",
    normalize: true,
    dtype: "float32",
    top_k: TOP_K,
    display_order: {
      method: "first_encounter",
      primary: "minimum_original_rank",
      tie_break: "random within each rank, saved once",
      deduplication: "passage_id",
    },
    pinecone_namespace: PINECONE_NAMESPACE,
    protocol_id: PROTOCOL_ID,
  };
}
