import itertools
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from taxonomy_metrics.graph.taxonomy import Taxonomy
from taxonomy_metrics.task.taxonomy_downstream_task import TaxonomyDownstreamTask, DataSplit

DBPEDIA_PATH = Path(__file__).parents[3] / "data" / "dbpedia"


class DBPediaTask(TaxonomyDownstreamTask):

    @staticmethod
    def _get_edges(ancestry: tuple[str]) -> list[tuple[str, str]]:
        edges = []
        for i in range(len(ancestry) - 1):
            edges.append((ancestry[i], ancestry[i + 1]))
        return edges

    @classmethod
    def load_data(cls,
                  path: Path = DBPEDIA_PATH,
                  sentence_transformer: SentenceTransformer | None = None
                  ) -> tuple[DataSplit, Taxonomy]:
        dataset = "DeveloperOats/DBPedia_Classes"
        train = pd.DataFrame(load_dataset(dataset, split="train"))
        test = pd.DataFrame(load_dataset(dataset, split="test"))

        if not cls.is_cached() and sentence_transformer is not None:
            train_content = train.text.values
            test_content = test.text.values
            train_content_vectors = sentence_transformer.encode(
                train_content, show_progress_bar=True,
                batch_size=16
            )
            test_content_vectors = sentence_transformer.encode(
                test_content, show_progress_bar=True,
                batch_size=16
            )
            np.save(path / "train_content_vectors.npy", train_content_vectors)
            np.save(path / "test_content_vectors.npy", test_content_vectors)
        else:
            train_content_vectors = np.load(path / "train_content_vectors.npy")
            test_content_vectors = np.load(path / "test_content_vectors.npy")

        split = DataSplit(train_content_vectors, test_content_vectors, train.l3.values, test.l3.values)
        train_edges = list(itertools.chain(*[cls._get_edges(a) for a in train[["l1", "l2", "l3"]].values]))
        test_edges = list(itertools.chain(*[cls._get_edges(a) for a in test[["l1", "l2", "l3"]].values]))
        concepts = set(train.l1.unique()).union(set(test.l2.unique())).union(set(test.l3.unique()))
        edges = list(train_edges) + list(test_edges)
        unique_edges = (tuple(e) for e in np.unique(edges, axis=0))
        taxonomy = Taxonomy(relations=unique_edges, id_to_name=dict(zip(concepts, concepts)))
        return split, taxonomy

    @classmethod
    def is_cached(cls, path: Path = DBPEDIA_PATH) -> bool:
        return (path / "train_content_vectors.npy").exists()


if __name__ == "__main__":
    if not DBPediaTask.is_cached():
        st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        DBPediaTask.load_data(sentence_transformer=st)
    start = timeit.default_timer()
    print("Baseline Score: ", DBPediaTask.baseline())
    end = timeit.default_timer()
    print(f"Fit and eval took {end - start:.2f} seconds")
