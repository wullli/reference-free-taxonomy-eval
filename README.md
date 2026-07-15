# Reference-Free Evaluation of Taxonomies

[![PyPI](https://img.shields.io/pypi/v/reference_free_taxonomy_eval)](https://pypi.org/project/reference-free-taxonomy-eval/)
[![Tests](https://img.shields.io/github/actions/workflow/status/wullli/reference-free-taxonomy-eval/ci.yml?branch=main&job=test&label=tests)](https://github.com/wullli/reference-free-taxonomy-eval/actions/workflows/ci.yml)
[![Lint](https://img.shields.io/github/actions/workflow/status/wullli/reference-free-taxonomy-eval/ci.yml?branch=main&job=lint&label=lint)](https://github.com/wullli/reference-free-taxonomy-eval/actions/workflows/ci.yml)
[![Type Check](https://img.shields.io/github/actions/workflow/status/wullli/reference-free-taxonomy-eval/ci.yml?branch=main&job=typecheck&label=type%20check)](https://github.com/wullli/reference-free-taxonomy-eval/actions/workflows/ci.yml)

Reference-free (and gold-standard) metrics for evaluating taxonomies — quantify how coherent, semantically adequate, and structurally sound a taxonomy is, with or without a ground-truth reference.

The metrics implemented here are described in our paper, [_Reference-Free Evaluation of Taxonomies_](https://aclanthology.org/2026.findings-acl.1273/) (Findings of ACL 2026) — see [Citation](#citation) below.

## Installation

Requires **Python >=3.10, <=3.11**. Install from [PyPI](https://pypi.org/project/reference-free-taxonomy-eval/):

```bash
pip install reference_free_taxonomy_eval
```

Or install from source:

```bash
git clone git@github.com:wullli/reference-free-taxonomy-eval.git
cd reference-free-taxonomy-eval
pip install -r requirements.txt
pip install -e .
```

Optional, if you have an NVIDIA GPU and want the accelerated `networkx` backend used by some metrics:

```bash
pip install -r requirements.gpu.txt
```

The NLI-based metric (`NLIVerificationMetric`, see below) needs a spaCy model:

```bash
python -m spacy download en_core_web_sm
```

## Quick start

Everything starts with a `Taxonomy`: a thin wrapper around a `networkx.DiGraph` built from `(parent, child)` relations plus a mapping from node ids to display names.

```python
from taxonomy_metrics.graph.taxonomy import Taxonomy

relations = [
    ("food", "fruit"),
    ("food", "vegetable"),
    ("fruit", "apple"),
    ("fruit", "pear"),
    ("vegetable", "carrot"),
]
names = {n: n for n in {p for p, _ in relations} | {c for _, c in relations}}

tax = Taxonomy(relations, id_to_name=names)

print(tax.leaves())               # ['apple', 'pear', 'carrot']
print(tax.depth())                # 3
```

## Running the reference-free metrics

Reference-free metrics score a taxonomy without needing a gold-standard taxonomy to compare against. They live in `taxonomy_metrics.evaluation.reference_free` and all share the same interface: `metric.calculate(taxonomy, node_subset=None)`.

```python
from sentence_transformers import SentenceTransformer

from taxonomy_metrics.graph.taxonomy import Taxonomy
from taxonomy_metrics.evaluation.reference_free.robustness import ConceptSimilarityCorrelationMetric, SemanticProximityMetric
from taxonomy_metrics.evaluation.reference_free.adequacy import NLIVerificationMetric

relations = [
    ("food", "fruit"),
    ("fruit", "apple"),
    ("fruit", "pear"),
    ("fruit", "stone fruit"),
    ("stone fruit", "peach"),
    ("stone fruit", "apricot"),
    ("food", "spice"),
    ("spice", "cinnamon"),
    ("spice", "pepper"),
    ("food", "vegetable"),
    ("vegetable", "carrot"),
    ("vegetable", "broccoli"),
]
names = {n: n for n in {p for p, _ in relations} | {c for _, c in relations}}
tax = Taxonomy(relations, id_to_name=names)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

csc = ConceptSimilarityCorrelationMetric(sentence_transformer=model)
sp = SemanticProximityMetric(sentence_transformer=model)
nli = NLIVerificationMetric()  # defaults to facebook/bart-large-mnli + en_core_web_sm

print(csc.calculate(tax))  # {'ConceptSimilarityCorrelation': ...}
print(sp.calculate(tax))   # {'SemanticProximity': 0.83...}
print(nli.calculate(tax))  # {'NLIVerification-Strong': ..., 'NLIVerification-Weak': ...}
```

## Citation

If you use this package, please cite:

> Pascal Wullschleger, Majid Zarharan, Donnacha Daly, Marc Pouly, and Jennifer Foster. 2026. Reference-Free Evaluation of Taxonomies. In Findings of the Association for Computational Linguistics: ACL 2026, pages 25489–25507, San Diego, California, United States. Association for Computational Linguistics.

```bibtex
@inproceedings{wullschleger-etal-2026-reference,
    title = "Reference-Free Evaluation of Taxonomies",
    author = "Wullschleger, Pascal  and
      Zarharan, Majid  and
      Daly, Donnacha  and
      Pouly, Marc  and
      Foster, Jennifer",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-acl.1273/",
    doi = "10.18653/v1/2026.findings-acl.1273",
    pages = "25489--25507",
    ISBN = "979-8-89176-395-1",
}
```

## License

MIT — see [LICENSE](LICENSE).
