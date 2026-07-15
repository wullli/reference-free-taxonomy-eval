from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import numpy as np

from taxonomy_metrics.evaluation.metric import GoldStandardMetric, ScoreAccumulator


class PositionGoldStandardMetric(GoldStandardMetric):

    @classmethod
    def calculate(
        cls,
        pred_positions: Dict[str, List[Tuple[str, str]]],
        true_positions: Dict[str, List[Tuple[str, str]]],
        node2name: Dict[str, str],
        weights: Mapping[str, float] | None = None,
        leaves: Optional[Set[str]] = None,
        verbose: bool = False,
        first_only: bool = True,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Calculate the position-f1 metric as described in the paper
        :param pred_positions: Predicted relations (parent, child) added to the taxonomy
        :param true_positions: Gold standard relations (parent, child) in the test data
        :param node2name: Mapping from node id to node name for the true taxonomy
        :param weights: Optional weights for each query
        :param leaves: The set of leaf node ids
        :param verbose: Print verbose output
        :return:
        """
        if leaves is None:
            leaves = set()
        queries = set(true_positions.keys())
        acc = ScoreAccumulator()
        leaf_acc = ScoreAccumulator()
        nonleaf_acc = ScoreAccumulator()
        scores = []
        nonleaf_scores = []
        leaf_scores = []
        for query in queries:
            truth = true_positions.get(query, [])
            if len(truth) == 0:
                print(f"Query {query} has no truth!")
                continue
            pred = pred_positions.get(query, [])

            if first_only:
                pred = pred[:1]

            tp = len(set(truth).intersection(set(pred)))
            fp = len(set(pred).difference(set(truth)))
            fn = len(set(truth).difference(set(pred)))

            scores.append([tp, fp, fn])

            query_weight = weights[query] if weights and query in weights else 1.0
            weighted_tp = tp * query_weight
            weighted_fp = fp * query_weight
            weighted_fn = fn * query_weight
            acc.tp += weighted_tp
            acc.fp += weighted_fp
            acc.fn += weighted_fn

            if query in leaves:
                leaf_acc.tp += weighted_tp
                leaf_acc.fp += weighted_fp
                leaf_acc.fn += weighted_fn
                leaf_scores.append([tp, fp, fn])
            else:
                nonleaf_acc.tp += weighted_tp
                nonleaf_acc.fp += weighted_fp
                nonleaf_acc.fn += weighted_fn
                nonleaf_scores.append([tp, fp, fn])

        return (
            {**acc.to_dict(), "scores": np.array(scores)},
            {**nonleaf_acc.to_dict("Non-leaf "), "Non-leaf scores": np.array(nonleaf_scores)},
            {**leaf_acc.to_dict("Leaf "), "Leaf scores": np.array(leaf_scores)},
        )
