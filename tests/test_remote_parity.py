"""Integration check against the actual evaluation wrappers.

Run from repo root: backend/.venv/bin/python tests/test_remote_parity.py
Add --local-only to omit Modal calls. Uses CPU for the local reference.
"""
import argparse
import importlib.util
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backend.encoder import EvalEncoder
from backend.client import health, encode_texts
from backend.corpus import load_corpus
from backend.protocol import MODELS, MAX_SEQ_LENGTH


def main():
    import torch
    from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
    from mteb.tasks.retrieval.eng.chemrxiv import ChemRxivRetrieval
    from mteb.types import PromptType
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    reference_path = ROOT.parent / 'ChEmbed-overlap-audit/clean-eval/ChEmbedWrapper.py'
    spec = importlib.util.spec_from_file_location('reference_chembed_wrapper', reference_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    corpus = load_corpus()
    docs = [t.strip() for t in [corpus.texts[i] for i in (7,1234,65432)]]
    queries = ['does thioamide perturb hydrogen bonds?', 'benzene ring substituents']
    for key in MODELS:
        local = EvalEncoder(key, 'cpu')
        wrapper_class = SentenceTransformerEncoderWrapper if key == 'base' else module.ChEmbedWrapper
        # Constructor performs the real wrapper's prompt setup as in evaluation.
        reference = wrapper_class(local.st_model, model_prompts={'query':'search_query: ', 'document':'search_document: '})
        tokenized = local.st_model.tokenize(['carbon ' * 10000])
        assert tokenized['input_ids'].shape[1] == MAX_SEQ_LENGTH
        if not args.local_only:
            h = health(key)
            assert h['max_seq_length'] == MAX_SEQ_LENGTH and h['device_ok']
            assert h['revision'] == MODELS[key]['revision']
            print(key, 'deployed on', h['gpu'], flush=True)
        for kind, texts in [('query', queries), ('document', docs)]:
            expected = reference.encode([{'text':texts}], task_metadata=ChemRxivRetrieval.metadata,
                hf_split='test', hf_subset='default', prompt_type=PromptType(kind),
                batch_size=2, show_progress_bar=False)
            expected = np.asarray(expected)
            expected /= np.linalg.norm(expected, axis=1, keepdims=True)
            got = local.encode_queries(texts,2) if kind == 'query' else local.encode_passages(texts,2)
            np.testing.assert_allclose(expected, got, atol=1e-5, rtol=1e-4)
            print(key, kind, 'evaluation-wrapper parity passed', flush=True)
            if not args.local_only:
                remote = encode_texts(key, texts, 'query' if kind=='query' else 'passage')
                np.testing.assert_allclose(expected, remote, atol=2e-3, rtol=1e-2)
                print(key, kind, 'deployed parity passed', flush=True)
        del local, reference
    print('Encoding checks passed.', flush=True)

if __name__ == '__main__':
    main()
