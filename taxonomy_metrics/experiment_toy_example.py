import argparse
import itertools
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import pipeline

from taxonomy_metrics.evaluation.reference_free.adequacy import NLIVerificationMetric
from taxonomy_metrics.evaluation.reference_free.robustness import (
    ConceptSimilarityCorrelationMetric, SemanticProximityMetric)
from taxonomy_metrics.experiments.common import torch_device
from taxonomy_metrics.graph.mutations import MoveSubtreeNodeTaxonomyMutator
from taxonomy_metrics.graph.taxonomy import Taxonomy


def get_node2name(tax: list) -> tuple[dict[str, str], list[str], list[str]]:
    names = list(set(itertools.chain(*tax)))
    node2name = dict(zip(names, names))
    return node2name, names, deepcopy(names)


def get_dummy_taxonomy(rels: list) -> Taxonomy:
    node2name, names, ids = get_node2name(rels)
    name2node = dict(zip(names, ids))
    return Taxonomy([(name2node[p], name2node[c]) for p, c in rels], node2name)


out_path = Path(__file__).parent.parent.parent / "output"

toy_rels = [
    ("food", "fruit"),
    ("fruit", "apple"),
    ("fruit", "pear"),
    ("fruit", "stone fruit"),
    ("stone fruit", "peach"),
    ("stone fruit", "apricot"),
    ("food", "spice"),
    ("spice", "cinnamon"),
    ("spice", "pepper"),
    ("pepper", "paprika"),
    ("pepper", "chili powder"),
    ("food", "vegetable"),
    ("vegetable", "sweet pepper"),
    ("vegetable", "carrot"),
    ("vegetable", "broccoli"),
]


def main(n_mutations: int) -> None:
    toy_tax = get_dummy_taxonomy(rels=toy_rels)
    res = defaultdict(list)
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device=torch_device()
    )

    nli = pipeline(
        "text-classification", model="facebook/bart-large-mnli", batch_size=128
    )

    csc_metric = ConceptSimilarityCorrelationMetric(
        sentence_transformer=model, progress=False, use_descriptions=False
    )
    metrics = {
        "CSC": csc_metric,
        "SP": SemanticProximityMetric(sentence_transformer=model, progress=False),
        "NLI-Strict": NLIVerificationMetric(model=nli, strict=True, probability=True),
        "NLI-Weak": NLIVerificationMetric(model=nli, strict=False, probability=True),
    }
    mutator = MoveSubtreeNodeTaxonomyMutator()

    emb_dict = csc_metric.embeddings(toy_tax.id_to_name, toy_tax.id_to_name)
    csc_metric.similarity_map(emb_dict)
    taxo_copy = deepcopy(toy_tax)

    for j in tqdm(range(n_mutations)):
        res["n_mutations"].append(j)
        res["taxonomy"].append(json.dumps(list(taxo_copy.g.edges())))
        for mname, m in metrics.items():
            score = m.calculate(taxo_copy)
            res[mname].append(score)
        taxo_copy = deepcopy(taxo_copy)
        taxo_copy = mutator(taxo_copy)

    variations = pd.DataFrame(res)
    variations.to_csv(out_path / "toy_variations.csv", index=False)


if __name__ == "__main__":
    spacy.cli.download("en_core_web_sm")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_mutations", type=int, default=10)
    args = parser.parse_args()
    main(n_mutations=args.n_mutations)
