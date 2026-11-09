import "server-only";

import { ModalClient } from "modal";

import {
  EMBED_DIM,
  MODAL_APP_NAME,
  MODAL_CLASS_FOR_KEY,
  MODELS,
  PROMPT_QUERY,
  type ModelKey,
} from "./search-config";

let sharedClient: ModalClient | null = null;

/** Modal client is constructed lazily: no credentials are read at build time. */
function getClient(): ModalClient {
  if (!sharedClient) {
    sharedClient = new ModalClient({
      tokenId: process.env.MODAL_TOKEN_ID,
      tokenSecret: process.env.MODAL_TOKEN_SECRET,
    });
  }
  return sharedClient;
}

/**
 * Encode one raw query with one deployed Modal model class. The encoder adds
 * the `search_query: ` prompt itself on the server, so the query must be
 * passed unprefixed. encode(texts, kind) returns [model_revision, vectors];
 * the revision is asserted against the pinned value so a drifted deployment
 * fails loudly.
 */
async function encodeQuery(modelKey: ModelKey, query: string): Promise<number[]> {
  const client = getClient();
  const cls = await client.cls.fromName(MODAL_APP_NAME, MODAL_CLASS_FOR_KEY[modelKey]);
  const instance = await cls.instance();
  const [revision, vectors] = (await instance
    .method("encode")
    .remote([[query], "query"])) as [string, number[][]];
  const expected = MODELS[modelKey].revision;
  if (revision !== expected) {
    throw new Error(
      `${modelKey}: deployed revision ${revision} does not match pinned ${expected}`,
    );
  }
  const vec = vectors?.[0];
  if (
    !Array.isArray(vec) ||
    vec.length !== EMBED_DIM ||
    !vec.every((v) => Number.isFinite(v))
  ) {
    throw new Error(`${modelKey}: invalid ${EMBED_DIM}-dim embedding returned`);
  }
  return vec;
}

/** Both models, in parallel, on the unprefixed query. Throws if either fails. */
export async function encodeQueryBoth(query: string): Promise<Record<ModelKey, number[]>> {
  const [base, prog] = await Promise.all([
    encodeQuery("base", query),
    encodeQuery("prog", query),
  ]);
  return { base, prog };
}

export { PROMPT_QUERY };
