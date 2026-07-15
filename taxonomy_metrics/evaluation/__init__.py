from taxonomy_metrics.evaluation.gold.parent_metric import ParentGoldStandardMetric
from taxonomy_metrics.evaluation.gold.position_metric import PositionGoldStandardMetric
from taxonomy_metrics.evaluation.gold.wupalmer_metric import WuPSimilarityMetric
from taxonomy_metrics.evaluation.metric import GoldStandardMetric as GoldStandardMetric

METRIC_REGISTRY = {
    "wup": WuPSimilarityMetric,
    "position": PositionGoldStandardMetric,
    "parent": ParentGoldStandardMetric,
}
