import timeit
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

from taxonomy_metrics.graph.taxonomy import Taxonomy
from taxonomy_metrics.task.taxonomy_downstream_task import TaxonomyDownstreamTask, DataSplit

MNDS_PATH = Path(__file__).parents[3] / "data" / "mn-ds-news"

class MNDSTask(TaxonomyDownstreamTask):
    @classmethod
    def load_data(cls,
                  path: Path = MNDS_PATH,
                  sentence_transformer: SentenceTransformer | None = None
                  ) -> tuple[
        DataSplit, Taxonomy]:

        if not cls.is_cached() and sentence_transformer is not None:
            df = pd.read_csv('https://zenodo.org/record/7394851/files/MN-DS-news-classification.csv?download=1')
            df["title_content"] = df.title + ". " + df.content
            content_vectors = sentence_transformer.encode(
                df.title_content.values, show_progress_bar=True,
                batch_size=16
            )
            np.save(path / "content_vectors.npy", content_vectors)
            df.to_csv(path / "mn-ds.csv", index=False)
        else:
            df = pd.read_csv(path / "mn-ds.csv")
            content_vectors = np.load(path / "content_vectors.npy")

        label_l1 = np.array(df.category_level_1.values, dtype=str).reshape(-1, 1)
        label_l2 = np.array(df.category_level_2.values, dtype=str).reshape(-1, 1)
        edges = np.hstack((label_l1, label_l2))

        x_train, x_test, y_train, y_test = train_test_split(
            content_vectors, df.category_level_2.values, test_size=0.2, random_state=42
        )

        concepts = set(label_l2.ravel()).union(label_l1.ravel())
        id_to_name = dict(zip(concepts, concepts))
        split = DataSplit(x_train, x_test, y_train, y_test)
        unique_edges = (tuple(e) for e in np.unique(edges, axis=0))
        taxonomy = Taxonomy(relations=unique_edges, id_to_name=id_to_name)
        return split, taxonomy

    @classmethod
    def is_cached(cls, path: Path = MNDS_PATH) -> bool:
        return (path / "mn-ds.csv").exists() and (path / "content_vectors.npy").exists()

if __name__ == "__main__":
    if not MNDSTask.is_cached():
        st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        MNDSTask.load_data(sentence_transformer=st)
    start = timeit.default_timer()
    print("Baseline Score: ", MNDSTask.baseline())
    end = timeit.default_timer()
    print(f"Fit and eval took {end - start:.2f} seconds")
