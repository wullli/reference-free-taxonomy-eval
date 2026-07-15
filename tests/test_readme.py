"""
Executes the code examples embedded in README.md so they're verified in CI.

The reference-free metrics example downloads real models (sentence-transformer,
spaCy, bart-large-mnli) when run as documented. To keep CI fast, offline, and
deterministic, the ML backends (SentenceTransformer, spaCy, transformers'
pipeline) are mocked; only the taxonomy_metrics code itself is exercised for
real.
"""
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

README_PATH = Path(__file__).parent.parent / "README.md"


def _code_blocks() -> list[str]:
    text = README_PATH.read_text()
    return re.findall(r"```python\n(.*?)```", text, re.DOTALL)


def _run_block_capturing_prints(block: str, namespace: dict) -> list:
    results: list = []
    namespace["_results"] = results
    # every print() in the README examples takes a single positional argument
    modified = block.replace("print(", "_results.append(")
    exec(modified, namespace)
    return results


def test_readme_quickstart() -> None:
    blocks = _code_blocks()
    namespace: dict = {}
    results = _run_block_capturing_prints(blocks[0], namespace)

    assert results == [
        ["apple", "pear", "carrot"],
        3,
    ]


class _FakeSentenceTransformer:
    """Stands in for sentence_transformers.SentenceTransformer: returns
    deterministic random embeddings instead of running a real model."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._rng = np.random.RandomState(0)
        self.device = torch.device("cpu")

    def eval(self) -> "_FakeSentenceTransformer":
        return self

    def encode(self, sentences, show_progress_bar: bool = False, batch_size: int = 32) -> np.ndarray:
        return self._rng.rand(len(sentences), 8).astype(np.float32)


def _fake_pipeline(task: str, model: Any = None, batch_size: int = 128, **kwargs: Any):
    """Stands in for transformers.pipeline: returns a fixed entailment-leaning
    score for every query instead of running a real NLI model."""

    def _call(dataset, top_k: Any = None):
        for _ in dataset:
            yield [
                {"label": "entailment", "score": 0.9},
                {"label": "neutral", "score": 0.05},
                {"label": "contradiction", "score": 0.05},
            ]

    return _call


def test_readme_reference_free_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    import sentence_transformers

    from taxonomy_metrics.evaluation.reference_free import adequacy

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeSentenceTransformer)
    monkeypatch.setattr(adequacy, "pipeline", _fake_pipeline)
    monkeypatch.setattr(adequacy.spacy, "load", lambda *_a, **_kw: object())

    blocks = _code_blocks()
    namespace: dict = {}
    csc_result, sp_result, nli_result = _run_block_capturing_prints(blocks[1], namespace)

    assert set(csc_result) == {"ConceptSimilarityCorrelation"}
    assert isinstance(csc_result["ConceptSimilarityCorrelation"], float)

    assert set(sp_result) == {"SemanticProximity"}
    assert isinstance(sp_result["SemanticProximity"], (float, np.floating))

    assert set(nli_result) == {"NLIVerification-Strong", "NLIVerification-Weak"}
    assert isinstance(nli_result["NLIVerification-Strong"], float)
    assert isinstance(nli_result["NLIVerification-Weak"], float)
