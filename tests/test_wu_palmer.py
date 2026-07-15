import pytest

from taxonomy_metrics.evaluation.wu_palmer import wu_palmer_similarity
from taxonomy_metrics.graph.taxonomy import Taxonomy


def test_similarity_identical_paths() -> None:
    path = ["A", "B", "C"]
    assert wu_palmer_similarity(path, path) == pytest.approx(1.0)


def test_similarity_root_lca() -> None:
    path1 = ["A", "B"]
    path2 = ["A", "C"]
    assert wu_palmer_similarity(path1, path2) == pytest.approx(0.5)


def test_similarity_one_is_ancestor() -> None:
    path1 = ["A", "B"]
    path2 = ["A", "B", "C"]
    assert wu_palmer_similarity(path1, path2) == pytest.approx(0.8)


def test_similarity_from_taxonomy(dummy_taxonomy: Taxonomy) -> None:
    ancestries = {a[-1]: a for a in dummy_taxonomy.ancestries(subset={"D", "E"})}
    score = wu_palmer_similarity(ancestries["D"], ancestries["E"])
    assert score == pytest.approx(1 / 3)
