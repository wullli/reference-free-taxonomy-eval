import abc
import dataclasses
from pathlib import Path

import numpy as np
from hiclass import LocalClassifierPerParentNode
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from taxonomy_metrics.graph.taxonomy import Taxonomy


@dataclasses.dataclass
class DataSplit:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


class TaxonomyDownstreamTask(abc.ABC):

    @classmethod
    @abc.abstractmethod
    def load_data(cls,
                  path: Path = Path("./"),
                  sentence_transformer: SentenceTransformer | None = None,
                  ) -> tuple[DataSplit, Taxonomy]: ...

    @classmethod
    @abc.abstractmethod
    def is_cached(cls, path: Path = Path("./")) -> bool: ...

    @classmethod
    def score(cls, taxonomy: Taxonomy, data_split: DataSplit) -> float:
        lcpn = LocalClassifierPerParentNode(local_classifier=LogisticRegression(max_iter=500))
        ancestries = taxonomy.ancestries()
        ancestor_map = {a[-1]: a for a in ancestries}
        y_train_mutated = [ancestor_map[n] for n in data_split.y_train.ravel()]
        y_test_mutated = [ancestor_map[n] for n in data_split.y_test.ravel()]

        lcpn.fit(data_split.x_train, np.array(y_train_mutated, dtype=object))
        y_pred = lcpn.predict(data_split.x_test)
        y_pred_leaves = [[y for y in yp if y != ""][-1] for yp in y_pred]
        y_test_leaves = [yt[-1] for yt in y_test_mutated]

        f1 = f1_score(y_test_leaves, y_pred_leaves, average="macro")
        return f1

    @classmethod
    def baseline(cls) -> float:
        split, taxonomy = cls.load_data()
        return cls.score(data_split=split, taxonomy=taxonomy)
