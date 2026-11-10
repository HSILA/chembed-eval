# chembed-eval

A private human evaluation app for comparing chemistry retrieval models.

## Web app

Use Node.js 22 or newer. Set the web app values from [.env.example](.env.example)
in your local `.env.local` file, then run:

```sh
npm ci
npm run dev
```

Open http://localhost:3000 and sign in with an account created in Supabase Auth.
Use **Search** to rate passages and **History** to reopen a search and continue.
The server calls both private Modal models, retrieves five passages per model
from Pinecone, and saves one deduplicated list ordered by reciprocal rank fusion of
the two models' original ranks, with the model details kept private.

Run `npm test`, `npm run typecheck`, and `npm run build` to check changes.
For Vercel, use the same project settings. Store the Supabase secret key,
Pinecone API key, and Modal credentials as server-only environment variables.

## Embedding backend

The first milestone prepares the 66,575 filtered ChemRxiv passages with two deployed models:

- Nomic v1 → `chembed-eval-base` in Pinecone.
- ChEmbed progressive → `chembed-eval-progressive` in Pinecone.

Both use an 8192-token limit and fixed model revisions. Query and passage encoding share the same deployed encoder. The encoder follows the paper's dedicated ChemRxiv evaluation runner. The 2048-token overrides in other benchmark runners are not used here.

Two private Modal classes run on separate T4 GPUs, with no always-running instances and a five-minute maximum idle window. GPU startup and waiting time can incur charges.

### Setup

From this repository's root:

```sh
uv sync --project backend --locked
cp -n .env.example .env.local
uvx --from modal modal token new
```

Fill in the local Pinecone key and Hugging Face token. The token must have access to the progressive checkpoint. Save the token in Modal's `chembed-hf` secret as `HF_TOKEN` too. Do not put credentials in command arguments, logs, or Git.

The default data source is `../ChEmbed-overlap-audit/clean-eval`, containing the filtered parquet file, manifest, and exclusions. The pipeline validates its hash and IDs before encoding.

### Deploy and check

```sh
backend/.venv/bin/modal deploy -m backend.modal_app
backend/.venv/bin/python tests/test_remote_parity.py --local-only
backend/.venv/bin/python tests/test_remote_parity.py
backend/.venv/bin/python -m pytest tests/test_backend_core.py -q
```

The integration check compares the original evaluation wrappers with the deployed services, for queries and passages. It also checks that both tokenizers truncate at 8192 tokens. The reference wrapper is read from the original evaluation directory.

### Prepare and upload

```sh
backend/.venv/bin/python -m backend.precompute plan
backend/.venv/bin/python -m backend.precompute precompute --model base --sample 512
backend/.venv/bin/python -m backend.precompute precompute --model prog --sample 512
# Check sample timings and estimated compute cost before the full job.
backend/.venv/bin/python -m backend.precompute precompute --model base
backend/.venv/bin/python -m backend.precompute precompute --model prog
backend/.venv/bin/python -m backend.precompute upload --model base
backend/.venv/bin/python -m backend.precompute upload --model prog
backend/.venv/bin/python -m backend.precompute status
```

Vectors and checkpoints go under the ignored `artifacts/` directory. Keep a backup of that directory. A failed batch can be resumed without repeating completed work. The upload must validate counts and sampled searches before it is considered complete.

Model revisions and encoding settings are in `backend/protocol.py`. Model files, secrets, corpus data, and embeddings are not committed. The UI and Vercel integration are later milestones.

## Supabase database

The linked `chembed-rag` project has four server-only tables: passages, searches, results, and ratings. Supabase Auth handles accounts, with public signup disabled. The schema lives in `supabase/migrations/`.

To load the verified corpus into a new instance, set the Supabase URL and service-role key locally, then run `backend/.venv/bin/python -m backend.load_passages`. The import preserves original passage IDs and skips existing IDs on retry.
