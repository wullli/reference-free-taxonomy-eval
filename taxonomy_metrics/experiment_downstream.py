import argparse
import random
import timeit
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from networkx.utils.backends import backend_info
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import pipeline

from taxonomy_metrics.evaluation import PositionGoldStandardMetric, ParentGoldStandardMetric
from taxonomy_metrics.evaluation.metric import GoldStandardMetric, ReferenceFreeMetric
from taxonomy_metrics.evaluation.reference_free.adequacy import NLIVerificationMetric, RaTEMetric
from taxonomy_metrics.evaluation.reference_free.robustness import (
    ConceptSimilarityCorrelationMetric, SemanticProximityMetric)
from taxonomy_metrics.experiments.common import record_sample
from taxonomy_metrics.graph.mutations import MoveSubtreeNodeTaxonomyMutator
from taxonomy_metrics.task.dbpedia_task import DBPediaTask
from taxonomy_metrics.task.mn_ds_task import MNDSTask
from taxonomy_metrics.task.taxonomy_downstream_task import TaxonomyDownstreamTask
from taxonomy_metrics.task.web_of_science_task import WebOfScienceTask
from taxonomy_metrics.experiment_gold_standard import positions

nx.config.warnings_to_ignore.add("cache")
print("AVAILABLE BACKENDS:", list(backend_info.keys()))
if "cugraph" in backend_info:
    nx.config.backend_priority = ["cugraph"]
else:
    nx.config.backend_priority = []

TASKS: dict[str, type[TaxonomyDownstreamTask]] = {
    "web_of_science": WebOfScienceTask,
    "dbpedia": DBPediaTask,
    "mn-ds-news": MNDSTask,
}

_OUT_PATH = Path(__file__).parent.parent.parent / "output"



def save_results(data: dict[str, list[object]],
                 dataset: str,
                 n_samples: int,
                 seed: int,
                 strategy: str) -> None:
    df = pd.DataFrame(data)
    path = _OUT_PATH / "downstream_evaluation"
    path.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path
        / f"downstream_eval_{dataset}_{n_samples}samples_seed{seed}_{strategy}.csv",
        index=False,
    )


def main(dataset: str,
         strategy: str,
         n_samples: int,
         seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    start_time = timeit.default_timer()

    if dataset not in TASKS:
        raise ValueError(f"Dataset {dataset} is not supported. Choose from {list(TASKS.keys())}.")

    nli = pipeline(
        "text-classification", model="facebook/bart-large-mnli", batch_size=128
    )

    task = TASKS[dataset]()
    if task is None:
        raise NotImplementedError(f"Data loader for {dataset} is not implemented yet.")

    mutator = MoveSubtreeNodeTaxonomyMutator(inner_only=strategy == "inner")
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    st.max_seq_length = 1024

    data_split, taxonomy = task.load_data()
    taxonomy.connect()

    data: dict[str, list[object]] = defaultdict(list)
    csc = ConceptSimilarityCorrelationMetric(sentence_transformer=st, progress=True, use_descriptions=True)
    embeddings = csc.embeddings(taxonomy.id_to_name, taxonomy.id_to_desc)
    sim_map = csc.similarity_map(embeddings)
    sp = SemanticProximityMetric(
        sentence_transformer=st,
        batch_size=128,
        progress=False,
        use_descriptions=True,
    )
    sp.similarities = sim_map
    csc.similarities = sim_map

    metrics: list[ReferenceFreeMetric | GoldStandardMetric] = [
        csc,
        RaTEMetric(
            pattern_path=Path(__file__).parent.parent
                         / "resources"
                         / "rate_queries.txt",
            top_k=10,
        ),
        NLIVerificationMetric(nli, progress=False, probability=True),
        sp,
        PositionGoldStandardMetric(),
        ParentGoldStandardMetric(),
    ]

    true_positions = positions(taxonomy)
    query_weights = taxonomy.num_descendants()

    record_sample(
        taxonomy=taxonomy,
        original_taxonomy=taxonomy,
        true_positions=true_positions,
        task=task,
        n_mutations=0,
        data_split=data_split,
        data=data,
        dataset=dataset,
        metrics=metrics,
        taxonomy_id=str(hash(frozenset(taxonomy.g.edges()))),
        query_weights=query_weights
    )


    save_results(data, dataset, n_samples, seed, strategy)

    with tqdm(
            total=n_samples, desc="Generating Samples", mininterval=2
    ) as pbar:
        for i in range(n_samples):
            times_mutated = 0
            taxo_copy = deepcopy(taxonomy)
            for m in range(1, 6):
                n_nodes = len(taxonomy.g.nodes())
                n_mutations = int(((2 ** m) / 100) * n_nodes)  # 1%, 2%, ..., 32% of nodes mutated
                n_mutations_to_go = n_mutations - times_mutated
                for _ in range(n_mutations_to_go):
                    taxo_copy = mutator(taxo_copy)
                    times_mutated += 1
                record_sample(
                    taxonomy=taxo_copy,
                    original_taxonomy=taxonomy,
                    true_positions=true_positions,
                    task=task,
                    n_mutations=times_mutated,
                    data_split=data_split,
                    data=data,
                    dataset=dataset,
                    metrics=metrics,
                    taxonomy_id=str(hash(frozenset(taxo_copy.g.edges()))),
                    query_weights=query_weights
                )
                pbar.update(1)
            pbar.set_postfix(n_samples=i + 1, refresh=True)
            save_results(data, dataset, n_samples, seed, strategy)

    print(
        f"Generating samples took seconds {timeit.default_timer() - start_time}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mn-ds-news",
                        choices=list(TASKS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument(
        "--strategy", type=str, choices=["inner", "move"], default="move"
    )
    args = parser.parse_args()
    main(**vars(args))
