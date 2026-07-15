import logging
from itertools import product
from typing import Any, Dict, Set, Tuple, Iterable

import networkx as nx


class Taxonomy:
    """
    Wrapper for networkx with methods that are used repeatedly to build a Taxonomy
    I know this is ugly, don't hate me for it.
    """

    def __init__(self,
                 relations: Iterable[Tuple[str, str]],
                 id_to_name: dict,
                 id_to_desc: dict | None = None,
                 pseudo_root: str = "pseudo root") -> None:
        """
        Initialize the Taxonomy with relations, id_to_name, and id_to_desc.
        :param relations: The relations (i.e. edges) of the taxonomy as an iterator of tuples (parent, child)
        :param id_to_name: Mapping from node IDs to names
        :param id_to_desc: Mapping from node IDs to node descriptions.
        """
        self.g = nx.DiGraph(relations)
        self.pseudo_root = pseudo_root
        nodes = set(self.g.nodes())
        for nid in id_to_name:
            if nid not in nodes:
                self.g.add_node(nid)
        self.id_to_name = id_to_name
        self.original_roots = self.roots()
        self.id_to_desc = id_to_desc if id_to_desc is not None else id_to_name

    def insert(
        self, triplets: Iterable[tuple[str | None, str, str | None]]
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str, str]]]:
        """
        Insert new concepts into the taxonomy using triplets as placements (parent, query, child)
        :param triplets: A set of triplets with parent-child relations
        """
        new_relations: list[tuple[str, str]] = []
        removed_relations: list[tuple[str, str]] = []
        inserted_triplets: list[tuple[str, str, str]] = []
        for p, q, c in triplets:
            if p is not None and c is not None and (p, c) in self.g.edges():
                self.g.remove_edge(p, c)
                self.g.add_edge(p, q)
                self.g.add_edge(q, c)
                removed_relations.append((p, c))
                new_relations.append((p, q))
                new_relations.append((q, c))
                inserted_triplets.append((p, q, c))
            if c is not None and (q, c) not in self.g.edges():
                self.g.add_edge(q, c)
                new_relations.append((q, c))
            if p is not None and (p, q) not in self.g.edges():
                self.g.add_edge(p, q)
                new_relations.append((p, q))
        return new_relations, removed_relations, inserted_triplets

    def connect(self) -> None:
        roots = self.roots()
        if len(roots) > 1:
            if self.pseudo_root not in self.g.nodes():
                self.g.add_node(self.pseudo_root)
                self.id_to_desc[self.pseudo_root] = self.pseudo_root
                self.id_to_name[self.pseudo_root] = self.pseudo_root
            incoming = list(self.g.predecessors(self.pseudo_root))
            assert len(incoming) == 0, f"Custom root has incoming edges: {incoming}"
            placements = [
                (None, self.pseudo_root, r) for r in roots if r != self.pseudo_root
            ]
            self.insert(triplets=placements)
            incoming = list(self.g.predecessors(self.pseudo_root))
            assert len(incoming) == 0, f"Custom root has incoming edges: {incoming}"

    def roots(self) -> list[str]:
        return [n for n in self.g.nodes() if self.g.in_degree(n) == 0]

    def node_depth(self, node: Any) -> int:
        def _get_depth(n: Any, accumulator: int = 0) -> Iterable[int]:
            yield accumulator + 1
            edges = self.g.out_edges(n)
            for e in edges:
                yield from _get_depth(e[1], accumulator=accumulator + 1)

        depths = list(_get_depth(node))
        return max(depths)

    def depth(self) -> int:
        roots = [n for n in self.g.nodes() if self.g.in_degree(n) == 0]

        def _get_depth(node: Any, accumulator: int = 0) -> Iterable[int]:
            yield accumulator + 1
            edges = self.g.out_edges(node)
            for e in edges:
                yield from _get_depth(e[1], accumulator=accumulator + 1)

        depths = []
        try:
            for r in roots:
                depths.extend(list(_get_depth(r)))
        except RecursionError as re:
            raise re

        return max(depths)

    def node_children(self, node: Any) -> tuple:
        out_edges = self.g.out_edges(node)
        return tuple(map(lambda x: x[1], out_edges))

    def descendants(self, only_leaves: bool = True) -> Dict[str, Set[str]]:
        acc: Dict[str, Set[str]] = {}
        roots = [n for n in self.g.nodes() if self.g.in_degree(n) == 0]
        for r in roots:
            self.node_descendants(r, acc, only_leaves=only_leaves)
        return acc

    def node_descendants(
        self, node: Any, acc: dict | None = None, only_leaves: bool = True
    ) -> Set[str]:
        if acc is None:
            acc = {}

        if node in acc:
            return acc[node]
        else:
            acc[node] = set()

        for n in self.node_children(node):
            if (only_leaves and self.is_leaf(n)) or not only_leaves:
                acc[node].add(n)
            if not self.is_leaf(n):
                if n not in acc:
                    self.node_descendants(n, acc, only_leaves)
                acc[node] = acc[node].union(acc[n])
        return acc[node]

    def children(self) -> list[tuple[str, tuple[str]]]:
        children = []
        for n in self.g.nodes():
            node_children = self.node_children(n)
            children.append((n, node_children))
        return children

    def inner_nodes(self) -> list[str]:
        inner_nodes = [n for n in self.g.nodes() if self.g.out_degree(n) > 0]
        return inner_nodes

    def leaves(self) -> list[str]:
        leaves = [n for n in self.g.nodes() if self.g.out_degree(n) == 0]
        return leaves

    def is_leaf(self, node: Any) -> bool:
        return self.g.out_degree(node) == 0

    def ancestries(self, subset: Set[str] | None = None) -> list[list[str]]:
        """
        Get all ancestries for the taxonomy
        :param subset: Filtered nodes for ancestries
        """

        roots = [n for n in self.g.nodes() if self.g.in_degree(n) == 0]

        def _get_ancestries(node: Any, accumulator: tuple = ()) -> Iterable[list[str]]:
            yield list(accumulator + (node,))
            edges = self.g.out_edges(node)
            for e in edges:
                if e[1] in self.original_roots and e[1] in accumulator:
                    continue
                if e[1] in accumulator:
                    logging.warning(f"Cycle detected for node {e[1]}")
                    continue
                yield from _get_ancestries(e[1], accumulator=accumulator + (node,))

        a = []
        try:
            for r in roots:
                a.extend(list(_get_ancestries(r)))
        except RecursionError as re:
            raise re

        if subset is not None:
            a = [path for path in a if path[-1] in subset]

        return a

    def node_triplets(self, node: Any, existing: bool = False) -> Iterable[tuple[str, str, str]]:
        parents = list(self.g.predecessors(node))
        children = list(self.g.successors(node))
        if len(parents) == 0 or not existing:
            parents += [None]
        if len(children) == 0 or not existing:
            children += [None]
        for p, c in product(parents, children):
            if (p is None) and (c is None):
                continue
            yield p, node, c

    def triplets(self, existing: bool = False) -> Iterable[tuple[str, str, str]]:
        """
        Get all triplets
        :param existing: Whether to only return triplets that already exist or all possible triplets
        """
        for n in self.g.nodes():
            yield from self.node_triplets(n, existing=existing)

    def num_descendants(self) -> dict[str, int]:
        roots = self.roots()
        single_root = len(roots) == 1

        def _not_single_root(node: str) -> bool:
            return node != self.pseudo_root or (single_root and node in roots)

        return {n: len(d) for n, d in self.descendants(only_leaves=False).items() if _not_single_root(n)}
