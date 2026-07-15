from typing import Dict, List, Mapping, Optional, Set, Tuple


from taxonomy_metrics.evaluation.metric import GoldStandardMetric, ScoreAccumulator


class ParentGoldStandardMetric(GoldStandardMetric):

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
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Calculate the ancestor-f1 metric as described in the paper
        :param pred_positions: Predicted relations (parent, child) added to the taxonomy
        :param true_positions: Gold standard relations (parent, child) in the test data
        :param node2name: Mapping from node id to node name for the true taxonomy
        :param weights: Weights for each query
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
        for query in queries:
            truth = [p[0] for p in true_positions.get(query, [])]
            if len(truth) == 0:
                print(f"Query {query} has no truth!")
                continue

            pred = [p[0] for p in pred_positions.get(query, [])]

            if first_only:
                pred = pred[:1]

            tp = len(set(truth).intersection(set(pred)))
            fp = len(set(pred).difference(set(truth)))
            fn = len(set(truth).difference(set(pred)))

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
            else:
                nonleaf_acc.tp += weighted_tp
                nonleaf_acc.fp += weighted_fp
                nonleaf_acc.fn += weighted_fn

        return (
            acc.to_dict(),
            nonleaf_acc.to_dict("Non-leaf "),
            leaf_acc.to_dict("Leaf "),
        )
