import pytest

from taxonomy_metrics.evaluation.metric import ScoreAccumulator
from taxonomy_metrics.evaluation.gold.parent_metric import ParentGoldStandardMetric
from taxonomy_metrics.evaluation.gold.position_metric import PositionGoldStandardMetric
from taxonomy_metrics.graph.taxonomy import Taxonomy


def test_score_accumulator_accuracy_with_zero_fp_fn() -> None:
    """Perfect classifier (FP=0, FN=0) must not raise."""
    acc = ScoreAccumulator(tp=10, fp=0, fn=0, tn=5)
    assert acc.accuracy() == pytest.approx(1.0)


def test_score_accumulator_total_raises_on_empty() -> None:
    acc = ScoreAccumulator(tp=0, fp=0, fn=0, tn=0)
    with pytest.raises(ValueError):
        _ = acc.total


def test_parent_metric_includes_recall() -> None:
    """add_gold_standard_scores must not drop Recall for ParentGoldStandardMetric."""
    from taxonomy_metrics.experiments.common import add_gold_standard_scores
    from taxonomy_metrics.graph.taxonomy import Taxonomy

    ids = {n: n for n in "ABCD"}
    tax = Taxonomy([("A", "B"), ("A", "C"), ("B", "D")], id_to_name=ids)
    true_pos = {"D": [("B", None)]}
    pred_pos = {"D": [("B", None)]}

    scores = add_gold_standard_scores(
        metric=ParentGoldStandardMetric(),
        mutated_positions=pred_pos,
        true_positions=true_pos,
        true_tax=tax,
    )
    keys = list(scores.keys())
    assert any("Recall" in k for k in keys), f"Recall missing from keys: {keys}"


def test_position_metric_excludes_raw_scores_array() -> None:
    """add_gold_standard_scores must strip the raw numpy 'scores' array."""
    import numpy as np
    from taxonomy_metrics.experiments.common import add_gold_standard_scores

    ids = {n: n for n in "ABCD"}
    tax = Taxonomy([("A", "B"), ("A", "C"), ("B", "D")], id_to_name=ids)
    true_pos = {"D": [("B", None)]}
    pred_pos = {"D": [("B", None)]}

    scores = add_gold_standard_scores(
        metric=PositionGoldStandardMetric(),
        mutated_positions=pred_pos,
        true_positions=true_pos,
        true_tax=tax,
    )
    for v in scores.values():
        assert not isinstance(v, np.ndarray), "Raw numpy array leaked into scores"
