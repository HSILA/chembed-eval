"""Modal app: two model servers on T4 GPUs, on-demand (min containers 0, one
max container each, 300 s idle window via scaledown_window).

Private invocation only — no web endpoints: run_batch_chunk is called with
.modal.Cls / Function objects hydrated by a local driver (backend/client.py);
live query encoding uses the same private encode method. A protected web
endpoint can be added when the Vercel app is built.
"""

import os

import modal

from modal import App


def _lazy_encoder():
    """Encoder imports are heavy; keep them inside image (they run on Modal
    where deps exist) while allowing local `modal deploy` without numpy."""
    from backend.encoder import EvalEncoder
    format_passage = EvalEncoder.format_passage  # noqa: F401

    return EvalEncoder, format_passage


app = modal.App("chembed-eval-models")

APP_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.9.0",
        "transformers==4.57.6",
        "sentence-transformers==5.2.0",
        "mteb==2.7.12",
        "einops==0.8.1",
        "numpy<2",
        "python-dotenv==1.2.1",
        "huggingface-hub==0.36.0",
        "pyarrow==23.0.0",
        "pydantic",
    )
    .add_local_python_source("backend", ignore=lambda p: p.suffix != ".py" or ".venv" in p.parts or "__pycache__" in p.parts)
)

GPU = "T4"
IDLE_TIMEOUT_S = 300
HF_CACHE_DIR = "/cache/hf"


def _common_volume():
    return modal.Volume.from_name("chembed-model-cache", create_if_missing=True)

# prompt/protocol constants duplicated as literals to keep module import free of
# numpy/torch; values asserted on the remote side (see tests/test_protocol_remote)
BATCH_SIZE_BASE = 16
BATCH_SIZE_PROG = 8


def _protocol_imports():
    from backend.protocol import MODELS, compute_protocol_id

    return MODELS, compute_protocol_id


class ModelServerBase:
    """Modal class serving nomic-ai/nomic-embed-text-v1 @ fixed revision."""

    @modal.enter()
    def load(self):
        EvalEncoder, _format = _lazy_encoder()
        import os

        os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
        self.model_key = "base"
        self.encoder = EvalEncoder("base", device="cuda")
        _common_volume().commit()

    @modal.method()
    def health(self) -> dict:
        import torch

        return {
            "model_key": self.model_key,
            "revision": self.encoder.revision,
            "max_seq_length": self.encoder.st_model.max_seq_length,
            "gpu": torch.cuda.get_device_name(0),
            "device_ok": torch.cuda.is_available(),
        }

    @modal.method()
    def encode(self, texts: list[str], kind: str = "passage") -> tuple[str, list[list[float]]]:
        _, format_passage = _lazy_encoder()
        if kind not in ("query", "passage"):
            raise ValueError(f"bad kind {kind!r}")
        bs = BATCH_SIZE_BASE
        if kind == "query":
            emb = self.encoder.encode_queries(texts, batch_size=bs)
        else:
            emb = self.encoder.encode_passages([format_passage(t) for t in texts], batch_size=bs)
        return (self.encoder.revision, emb.tolist())

    @modal.method()
    def encode_chunk(
        self,
        texts: list[str],
        start: int,
        end: int,
        corpus_sha256: str,
        dataset_digest: str,
    ) -> dict:
        from backend import clean_eval_shim

        MODELS, compute_protocol_id = _protocol_imports()
        _, format_passage = _lazy_encoder()

        fid = compute_protocol_id(corpus_sha256, dataset_digest)
        emb = self.encoder.encode_passages([format_passage(t) for t in texts], batch_size=BATCH_SIZE_BASE)
        return {
            "model_key": self.model_key,
            "protocol_id": fid,
            "revision": self.encoder.revision,
            "start": start,
            "end": end,
            "vectors": emb.tolist(),
            "resolved": sorted(clean_eval_shim.RESOLVED_PATHS.get(MODELS[self.model_key]["hf_name"], []))[:4],
        }


class ModelServerProg:
    """Modal class serving BASF-AI/ChEmbed-prog @ fixed revision (pinned shim)."""

    @modal.enter()
    def load(self):
        EvalEncoder, _format = _lazy_encoder()
        import os

        os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
        self.model_key = "prog"
        self.encoder = EvalEncoder("prog", device="cuda")
        _common_volume().commit()

    @modal.method()
    def health(self) -> dict:
        import torch

        return {
            "model_key": self.model_key,
            "revision": self.encoder.revision,
            "max_seq_length": self.encoder.st_model.max_seq_length,
            "gpu": torch.cuda.get_device_name(0),
            "device_ok": torch.cuda.is_available(),
        }

    @modal.method()
    def encode(self, texts: list[str], kind: str = "passage") -> tuple[str, list[list[float]]]:
        _, format_passage = _lazy_encoder()
        if kind not in ("query", "passage"):
            raise ValueError(f"bad kind {kind!r}")
        emb = self.encoder.encode_queries(texts, batch_size=BATCH_SIZE_PROG) if kind == "query" else self.encoder.encode_passages(
            [format_passage(t) for t in texts], batch_size=BATCH_SIZE_PROG
        )
        return (self.encoder.revision, emb.tolist())

    @modal.method()
    def encode_chunk(
        self,
        texts: list[str],
        start: int,
        end: int,
        corpus_sha256: str,
        dataset_digest: str,
    ) -> dict:
        from backend import clean_eval_shim

        MODELS, compute_protocol_id = _protocol_imports()
        _, format_passage = _lazy_encoder()

        fid = compute_protocol_id(corpus_sha256, dataset_digest)
        emb = self.encoder.encode_passages([format_passage(t) for t in texts], batch_size=BATCH_SIZE_PROG)
        return {
            "model_key": self.model_key,
            "protocol_id": fid,
            "revision": self.encoder.revision,
            "start": start,
            "end": end,
            "vectors": emb.tolist(),
            "resolved": sorted(clean_eval_shim.RESOLVED_PATHS.get(MODELS[self.model_key]["hf_name"], []))[:4],
        }


_common_kwargs = dict(
    image=APP_IMAGE,
    gpu=GPU,
    secrets=[modal.Secret.from_name("chembed-hf")],
    scaledown_window=IDLE_TIMEOUT_S,
    max_containers=1,
    timeout=600,
    startup_timeout=300,
    min_containers=0,
    volumes={"/cache": _common_volume()},
)

ModelServerBase = app.cls(**_common_kwargs)(ModelServerBase)
ModelServerProg = app.cls(**_common_kwargs)(ModelServerProg)
