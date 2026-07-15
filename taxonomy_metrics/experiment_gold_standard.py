import argparse
import random
import timeit
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import spacy
from dotenv import load_dotenv
from networkx.utils.backends import backend_info
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import pipeline

from taxonomy_metrics.evaluation import ParentGoldStandardMetric
from taxonomy_metrics.evaluation.gold.position_metric import PositionGoldStandardMetric
from taxonomy_metrics.evaluation.reference_free.adequacy import (NLIVerificationMetric,
                                                                 RaTEMetric)
from taxonomy_metrics.evaluation.reference_free.robustness import (
    ConceptSimilarityCorrelationMetric, SemanticProximityMetric)
from taxonomy_metrics.experiments.common import positions, record_sample, torch_device
from taxonomy_metrics.graph.mutations import MoveSubtreeNodeTaxonomyMutator, TaxonomyMutator, \
    WordNetTaxonomyMutator, WPSWeightedTaxonomyMutator
from taxonomy_metrics.graph.taxonomy import Taxonomy
from taxonomy_metrics.data.loader import load_taxonomy

nx.config.warnings_to_ignore.add("cache")
print("AVAILABLE BACKENDS:", list(backend_info.keys()))
if "cugraph" in backend_info:
    nx.config.backend_priority = ["cugraph"]

api_key_path = Path(__file__).parent.parent / ".env"
if Path(api_key_path).exists():
    load_dotenv(dotenv_path=str(api_key_path))

out_path = Path(__file__).parent.parent.parent / "output"
data_path = Path(__file__).parent.parent.parent / "data"


def generate_sample(
    true_tax: Taxonomy,
    dataset: str,
    metrics: list,
    mutator: TaxonomyMutator,
) -> dict:
    data: dict[str, list[object]] = defaultdict(list)
    true_positions = positions(true_tax)
    query_weights = true_tax.num_descendants()

    taxo_copy = deepcopy(true_tax)

    times_mutated = 0
    for m in range(1, 6):
        n_nodes = len(taxo_copy.g.nodes())
        n_mutations = int(((2 ** m) / 100) * n_nodes)  # 1%, 2%, ..., 32% of nodes mutated
        n_mutations_to_go = n_mutations - times_mutated
        for _ in range(n_mutations_to_go):
            taxo_copy = mutator(taxo_copy)
            times_mutated += 1

        record_sample(
            taxonomy=taxo_copy,
            original_taxonomy=true_tax,
            true_positions=true_positions,
            n_mutations=times_mutated,
            data=data,
            dataset=dataset,
            metrics=metrics,
            taxonomy_id=str(hash(frozenset(taxo_copy.g.edges()))),
            query_weights=query_weights
        )
    return data


def main(
        n_samples: int,
        dataset: str = "semeval_food",
        seed: int = 42,
        strategy: str = "move",
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    print(f"Load {dataset} taxonomy ...", flush=True)
    start_time = timeit.default_timer()
    path = data_path / f"{dataset}"
    print(f"Path: {path}", flush=True)
    terms, taxo = load_taxonomy(str(path))
    id_to_name = {d["node_id"]: d["node_name"] for d in terms.to_dict(orient="records")}
    id_to_desc = {d["node_id"]: d["desc"] for d in terms.to_dict(orient="records")}
    print(
        f"Load taxonomy took seconds {timeit.default_timer() - start_time}", flush=True
    )

    print("Build graph ...", flush=True)
    start_time = timeit.default_timer()
    true_tax = Taxonomy(
        [tuple(t) for t in taxo[["hypernym", "hyponym"]].values.tolist()],
        id_to_name=id_to_name,
        id_to_desc=id_to_desc
    )
    cycles = nx.simple_cycles(true_tax.g)
    if cycles is not None:
        for c in cycles:
            print(f"Cycle: {c}")
    true_tax.connect()
    cycles = nx.simple_cycles(true_tax.g)
    if cycles is not None:
        for c in cycles:
            print(f"Cycle: {c}")
    print(f"Build graph took seconds {timeit.default_timer() - start_time}", flush=True)

    print("Load models ...", flush=True)
    start_time = timeit.default_timer()
    sim_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device=torch_device()
    )
    nli_model = pipeline(
        "text-classification", model="facebook/bart-large-mnli", batch_size=128, top_k=None,
    )
    print(f"Load models took seconds {timeit.default_timer() - start_time}", flush=True)

    print("Precompute embeddings and similarities ...")
    start_time = timeit.default_timer()
    print(
        f"Precomputing took seconds {timeit.default_timer() - start_time}", flush=True
    )

    mutated_taxonomies = defaultdict(list)

    print("Generate samples ...", flush=True)
    start_time = timeit.default_timer()

    csc = ConceptSimilarityCorrelationMetric(
        sentence_transformer=sim_model, progress=True, use_descriptions=True, no_cache=False
    )
    embeddings = csc.embeddings(true_tax.id_to_name, true_tax.id_to_desc)
    sim_map = csc.similarity_map(embeddings)
    sp = SemanticProximityMetric(
        sentence_transformer=sim_model,
        batch_size=128,
        progress=False,
        use_descriptions=True,
        no_cache=False,
    )
    sp.similarities = sim_map
    csc.similarities = sim_map

    mutators = {
        "move": MoveSubtreeNodeTaxonomyMutator(inner_only=False),
        "inner": MoveSubtreeNodeTaxonomyMutator(inner_only=True),
        "wordnet": WordNetTaxonomyMutator(),
        "wps": WPSWeightedTaxonomyMutator(inner_only=False, init_taxonomy=true_tax),
    }

    metrics = [
        csc,
        RaTEMetric(
            pattern_path=Path(__file__).parent.parent
                         / "resources"
                         / "rate_queries.txt",
            top_k=10,
        ),
        NLIVerificationMetric(nli_model, progress=False, probability=True),
        PositionGoldStandardMetric(),
        ParentGoldStandardMetric(),
        sp
    ]
    with tqdm(
            total=n_samples, desc="Generating Samples", mininterval=2
    ) as pbar:
        for i in range(n_samples):
            res = generate_sample(true_tax, dataset, metrics, mutators[strategy])
            pbar.set_postfix(n_samples=i + 1, refresh=True)
            for k, v in res.items():
                mutated_taxonomies[k].extend(v)
            df = pd.DataFrame(mutated_taxonomies)
            out = out_path / "ground_truth_correlation"
            out.mkdir(exist_ok=True, parents=True)
            df.to_csv(
                out
                / f"validation_{dataset}_{n_samples}samples_seed{seed}_{strategy}.csv",
                index=False,
            )
    print(
        f"Generating samples took seconds {timeit.default_timer() - start_time}",
        flush=True,
    )


if __name__ == "__main__":
    spacy.cli.download("en_core_web_sm")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="wikitax")
    parser.add_argument("--seed", type=int, default=467)
    parser.add_argument(
        "--strategy", type=str, choices=["inner", "move", "wordnet", "wps"], default="inner"
    )
    args = parser.parse_args()
    main(
        n_samples=args.n_samples,
        dataset=args.dataset,
        seed=args.seed,
        strategy=args.strategy,
    )
