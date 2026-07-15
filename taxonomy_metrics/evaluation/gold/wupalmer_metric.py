from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from taxonomy_metrics.evaluation.metric import GoldStandardMetric
from taxonomy_metrics.evaluation.wu_palmer import wu_palmer_similarity
from taxonomy_metrics.graph.taxonomy import Taxonomy


class WuPSimilarityMetric(GoldStandardMetric):

    @staticmethod
    def get_ancestries(
        tax: Taxonomy,
        leaf_set: Optional[Set[str]] = None,
    ) -> Dict[str, Tuple[str, ...]]:
        return {a[-1]: tuple(a) for a in tax.ancestries(subset=leaf_set)}

    @classmethod
    def calculate(  # type: ignore[override]
        cls,
        pred_positions: Dict[str, List[Tuple[str, str]]],
        true_positions: Dict[str, List[Tuple[str, str]]],
        node2name: Dict[str, str],
        seed_taxonomy: List[Tuple[str, str]],
        leaves: Optional[Set[str]] = None,
        verbose: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Calculate the wu-palmer similarity metric
        :param pred_positions: Predicted positions (parent, child) added to the taxonomy
        :param true_positions: Gold standard relations (parent, child) in the test data
        :param seed_taxonomy: The seed taxonomy
        :param node2name: Mapping from node id to node name for the true taxonomy
        :param leaves: The set of leaf node ids
        :param verbose: Print the relations
        :return:
        """
        if leaves is None:
            leaves = set()
        queries = set(true_positions.keys())
        name2node = dict(zip(node2name.values(), node2name.keys()))
        query_ids = set([name2node[n] for n in queries])

        pred_triplets = []
        true_triplets = []

        for q, positions in true_positions.items():
            for p, c in positions:
                true_triplets.append((p, q, c))

        for q, positions in pred_positions.items():
            for p, c in positions:
                pred_triplets.append((p, q, c))

        pred_tax = Taxonomy(seed_taxonomy, id_to_name=node2name)
        pred_tax.insert(pred_triplets)

        true_tax = Taxonomy(seed_taxonomy, id_to_name=node2name)
        true_tax.insert(true_triplets)

        # If we only use the new relations, we will have a disconnected graph, thus we need to add the seed relations
        # And then we need to remove the seed ancestor relations from the new relations
        pred_ancestries = cls.get_ancestries(pred_tax, leaf_set=query_ids)
        true_ancestries = cls.get_ancestries(true_tax, leaf_set=query_ids)

        scores = []
        leaf_scores = []
        nonleaf_scores = []
        for q in queries:
            true = true_ancestries.get(q, tuple())
            pred = pred_ancestries.get(q, tuple())

            score = wu_palmer_similarity(true, pred)
            scores.append(score)
            if q in leaves:
                leaf_scores.append(score)
            else:
                nonleaf_scores.append(score)

        return (
            {"WuPalmerSimilarity": np.nanmean(scores), "scores": np.array(scores)},
            {"WuPalmerSimilarity": np.nanmean(nonleaf_scores), "scores": np.array(nonleaf_scores)},
            {"WuPalmerSimilarity": np.nanmean(leaf_scores), "scores": np.array(leaf_scores)},
        )
