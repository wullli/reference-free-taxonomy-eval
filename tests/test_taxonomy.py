import pytest

from taxonomy_metrics.graph.taxonomy import Taxonomy


def test_ancestries_subset(dummy_taxonomy: Taxonomy) -> None:
    leaf_paths = set(map(tuple, dummy_taxonomy.ancestries(subset={"D", "E"})))
    assert leaf_paths == {("A", "B", "D"), ("A", "C", "E")}


@pytest.fixture()
def diamond_taxonomy() -> Taxonomy:
    """DAG where D has two parents: B and C (diamond: A->B->D, A->C->D)."""
    relations = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
    ids = {n: n for n in "ABCD"}
    return Taxonomy(relations, id_to_name=ids, id_to_desc=ids)


def test_node_descendants_tree(dummy_taxonomy: Taxonomy) -> None:
    desc = dummy_taxonomy.node_descendants("A", only_leaves=True)
    assert desc == {"D", "E"}


def test_node_descendants_dag(diamond_taxonomy: Taxonomy) -> None:
    """Both B and C must include D in their descendants; A must include D too."""
    desc_b = diamond_taxonomy.node_descendants("B", only_leaves=True)
    assert "D" in desc_b

    desc_c = diamond_taxonomy.node_descendants("C", only_leaves=True)
    assert "D" in desc_c

    desc_a = diamond_taxonomy.node_descendants("A", only_leaves=True)
    assert desc_a == {"D"}


def test_node_descendants_non_leaves(dummy_taxonomy: Taxonomy) -> None:
    desc = dummy_taxonomy.node_descendants("A", only_leaves=False)
    assert desc == {"B", "C", "D", "E"}


def test_insert_returns_changes() -> None:
    ids = {n: n for n in "ABCD"}
    tax = Taxonomy([("A", "B"), ("A", "C")], id_to_name=ids)
    new_rel, removed_rel, inserted = tax.insert([(None, "D", "B")])
    assert ("D", "B") in new_rel
    assert removed_rel == []

    new_rel2, removed_rel2, inserted2 = tax.insert([("A", "X", "B")])
    ids["X"] = "X"
    assert ("A", "B") in removed_rel2
    assert ("A", "X") in new_rel2
    assert ("X", "B") in new_rel2
    assert ("A", "X", "B") in inserted2
