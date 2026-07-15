from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Mapping, Set, Tuple

import numpy as np

from taxonomy_metrics.graph.taxonomy import Taxonomy


@dataclass
class ScoreAccumulator:
    tp: float = 0
    fp: float = 0
    fn: float = 0
    tn: float = 0

    @property
    def total(self) -> float:
        t = self.tp + self.fp + self.fn + self.tn
        if t == 0:
            raise ValueError("Total count is zero.")
        return t

    def precision(self, nan: bool = False) -> float:
        return (
            self.tp / (self.tp + self.fp)
            if (self.tp + self.fp) > 0.0
            else (0.0 if not nan else np.nan)
        )

    def recall(self, nan: bool = False) -> float:
        return (
            self.tp / (self.tp + self.fn)
            if (self.tp + self.fn) > 0.0
            else (0.0 if not nan else np.nan)
        )

    def f1(self, nan: bool = False) -> float:
        p = self.precision(nan=nan)
        r = self.recall(nan=nan)
        denom = p + r
        return (
            2 * p * r / denom
            if (denom > 0.0 and not np.isnan(denom))
            else (0.0 if not nan else np.nan)
        )

    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total

    def to_dict(self, prefix: str = "") -> Dict[str, float]:
        return {
            f"{prefix}F1": self.f1(),
            f"{prefix}Precision": self.precision(),
            f"{prefix}Recall": self.recall(),
        }


class GoldStandardMetric:
    @classmethod
    @abstractmethod
    def calculate(
        cls,
        pred_positions: Dict[str, List[Tuple[str, str]]],
        true_positions: Dict[str, List[Tuple[str, str]]],
        node2name: Dict[str, str],
        weights: Mapping[str, float] | None = None,
        leaves: Set[str] | None = None,
        verbose: bool = False,
        first_only: bool = True,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        raise NotImplementedError("Metric is not implemented!")


class ReferenceFreeMetric:
    name: ClassVar[str]

    def _scores(self, **values: float) -> Dict[str, float]:
        if list(values.keys()) == ["score"]:
            return {self.name: values["score"]}
        return {f"{self.name}-{k}": v for k, v in values.items()}

    @abstractmethod
    def calculate(
        self,
        taxonomy: Taxonomy,
        node_subset: Set[str] | None = None,
    ) -> dict[str, float]:
        raise NotImplementedError("Metric is not implemented!")

