import pytest
from taxonomy_metrics.graph.taxonomy import Taxonomy


@pytest.fixture()
def dummy_taxonomy() -> Taxonomy:
    relations = [
        ("A", "B"),
        ("A", "C"),
        ("B", "D"),
        ("C", "E"),
    ]
    ids = {n: n for n in "ABCDE"}
    return Taxonomy(relations, id_to_name=ids, id_to_desc=ids)
