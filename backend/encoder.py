"""Shared encoder for deployed queries and passage batches.

Matches the paper's ChemRxiv runner: base through MTEB's Nomic wrapper;
progressive through SentenceTransformer with the evaluation wrapper's exact
prompt strings and pinned weight-resolution shim. Stored vectors are L2
normalized float32, equivalent to the evaluation's cosine normalization.
The dedicated ChemRxiv runner uses the models' 8192-token limit.
"""

from pathlib import Path
from typing import Sequence

import numpy as np

from .protocol import MODELS, PROMPTS, BATCH_SIZE_BASE, BATCH_SIZE_PROG

# clean-eval folder with the original wrapper + shim source
CLEAN_EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "ChEmbed-overlap-audit" / "clean-eval"



class EvalEncoder:
    """Encode queries and passages exactly like the original MTEB evaluation run."""

    def __init__(self, model_key: str, device: str):
        self.model_key = model_key
        if model_key not in MODELS:
            raise ValueError(f"unknown model key {model_key!r}; expected one of {sorted(MODELS)}")
        model_name = MODELS[model_key]["hf_name"]
        revision = MODELS[model_key]["revision"]

        self._device = device
        self._load(model_name, revision, model_key)

    def _load(self, model_name: str, revision: str, key: str) -> None:
        from dotenv import load_dotenv

        load_dotenv(CLEAN_EVAL_DIR / ".env")
        load_dotenv(CLEAN_EVAL_DIR.parent / ".env")

        import os

        if key == "base":
            # mteb.get_model returns the same NomicWrapper(SentenceTransformerEncoderWrapper)
            # used by the original eval; we keep its underlying SentenceTransformer
            # and drive encoding through the prompt-kwarg path (verified identical
            # to the wrapper's `search_document: `+text path; see tests/local_parity).
            import mteb

            wrapper = mteb.get_model(
                model_name,
                device=self._device,
                revision=revision,
                trust_remote_code=True,
                model_kwargs={"trust_remote_code": True},
            )
            model = wrapper.model
        else:
            token = os.environ.get("HF_TOKEN")
            if not token:
                raise RuntimeError("HF_TOKEN missing from environment for gated repos.")
            # The remote modeling code of the gated ChEmbed repos resolves its
            # own weights through hub.cached_file without revision or token;
            # pin both, then load through SentenceTransformer with the same
            # revision, then assert every resolved weight file came from the
            # pinned snapshot (no main-branch fallback).
            import backend.clean_eval_shim as clean_eval_shim_module

            clean_eval_shim_module.pin_chembed_cached_file(model_name, revision, token)
            from sentence_transformers import SentenceTransformer

            st = SentenceTransformer(
                model_name,
                device=self._device,
                revision=revision,
                trust_remote_code=True,
                model_kwargs={"trust_remote_code": True},
            )
            clean_eval_shim_module.assert_provenance(model_name, revision)
            self.resolved_weights = list(clean_eval_shim_module.RESOLVED_PATHS[model_name])
            model = st

        self.st_model = model
        self.revision = revision
        if self.st_model.max_seq_length != 8192:
            raise RuntimeError(f"Expected 8192-token context, got {self.st_model.max_seq_length}")
        # prompts active for this model, from the wrapper used by the original
        # eval (NomicWrapper for base via mteb.get_model; ChEmbedWrapper's
        # model_prompts copied for prog)
        self.model_prompts = PROMPTS

    # ---- encoding ---------------------------------------------------------

    def _encode(self, prompt: str, texts: Sequence[str], batch_size: int, normalize: bool) -> np.ndarray:
        emb = self.st_model.encode(
            texts,
            prompt=prompt,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        emb = np.asarray(emb, dtype=np.float32)
        if emb.shape != (len(texts), 768) or not np.isfinite(emb).all():
            raise RuntimeError(f"unexpected embedding shape {emb.shape}")
        if normalize:
            norms = np.linalg.norm(emb, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                raise RuntimeError(f"normalized norms off: {norms.min()}..{norms.max()}")
        return emb

    def encode_queries(self, texts: Sequence[str], batch_size: int | None = None) -> np.ndarray:
        bs = batch_size or (BATCH_SIZE_BASE if self.model_key == "base" else BATCH_SIZE_PROG)
        return self._encode(PROMPTS["query"], list(texts), bs, normalize=True)

    def encode_passages(self, texts: Sequence[str], batch_size: int | None = None) -> np.ndarray:
        bs = batch_size or (BATCH_SIZE_BASE if self.model_key == "base" else BATCH_SIZE_PROG)
        return self._encode(PROMPTS["document"], list(texts), bs, normalize=True)

    @staticmethod
    def format_passage(text: str) -> str:
        """Document text exactly as mteb feeds it: strip() only (no title column)."""
        return text.strip()
