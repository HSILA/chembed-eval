"""chembed-eval embedding backend.

Layers:
- protocol.py   exact eval settings transcribed from clean-eval (models,
                revisions, prompts, batch sizes, max_seq_length).
- corpus.py     loads + validates the filtered 66,575-passage corpus by sha256.
- encoder.py    EvalEncoder: exact encode path (MTEB wrapper chain), used for
                both live queries and batch passages.
- clean_eval_shim.py  cache-file pinning for gated ChEmbed revisions.
- checkpoints.py atomic, resumable batch artifacts under artifacts/.
- modal_app.py  two on-demand T4 model servers (one per model).
- client.py     driver-side hydration + decode of the deployed classes.
- precompute.py resumable full-corpus precompute + Pinecone upload + checks.
"""
