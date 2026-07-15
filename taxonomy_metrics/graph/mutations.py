import abc
import enum
import itertools
import random
from collections import defaultdict
from typing import Hashable

import networkx as nx
import nltk
import numpy as np

from taxonomy_metrics.evaluation.wu_palmer import wu_palmer_similarity
from taxonomy_metrics.graph.taxonomy import Taxonomy
from nltk.corpus import wordnet as wn


class TaxonomyMutator(abc.ABC):

    @abc.abstractmethod
    def __call__(self, taxonomy: Taxonomy) -> Taxonomy:
        ...


class MoveSubtreeNodeTaxonomyMutator(TaxonomyMutator):

    def __init__(self, inner_only: bool = False) -> None:
        self.inner_only = inner_only
        self._distance_map: dict = {}

    def __call__(self, taxonomy: Taxonomy) -> Taxonomy:
        roots = taxonomy.roots()
        nodes = set(taxonomy.g.nodes())
        parent_candidates: set[Hashable] | list[Hashable] | np.ndarray = set()
        current_parents = set()
        node_to_move = None
        prev_leaves = taxonomy.leaves()
        while len(parent_candidates) < 1:
            candidate_nodes = [
                n for n in nodes if n not in roots and n != taxonomy.pseudo_root
            ]
            if self.inner_only:
                candidate_nodes = [n for n in candidate_nodes if not taxonomy.is_leaf(n)]
            node_to_move = random.choice(candidate_nodes)
            if self.inner_only:
                current_parents = set(
                    [
                        n
                        for n in taxonomy.g.predecessors(node_to_move)
                        if taxonomy.g.out_degree(n) > 1
                    ]
                )
                if len(current_parents) == 0:
                    continue
            else:
                current_parents = set(taxonomy.g.predecessors(node_to_move))
            descendants = nx.descendants(taxonomy.g, node_to_move) | {node_to_move}
            parent_candidates = [
                n
                for n in candidate_nodes
                if n not in descendants and n not in current_parents
            ]

        weights = None
        if len(self._distance_map) > 0:
            weights = np.array([1 - self._distance_map.get(frozenset({node_to_move, p}), 0.5)
                                for p in parent_candidates])
            top_100 = np.argsort(weights)[::-1][:100]
            parent_candidates = np.array(parent_candidates)[top_100]
            weights = weights[top_100]
            weights = weights / weights.sum()
        parent = np.random.choice(np.array(parent_candidates), p=weights)

        for current_parent in current_parents:
            taxonomy.g.remove_edge(current_parent, node_to_move)
        taxonomy.g.add_edge(parent, node_to_move)
        if self.inner_only:
            assert len(prev_leaves) == len(taxonomy.leaves()), (
                f"Leaves changed, this should not happen: "
                f"{len(prev_leaves)} -> {len(taxonomy.leaves())}"
            )
        return taxonomy


class WPSWeightedTaxonomyMutator(MoveSubtreeNodeTaxonomyMutator):

    def __init__(self, init_taxonomy: Taxonomy, inner_only: bool = False) -> None:
        super().__init__(inner_only=inner_only)

        self._distance_map: dict = {}
        ancestries = defaultdict(list)
        for a in init_taxonomy.ancestries():
            ancestries[a[-1]].append(a)

        for k1, a1 in ancestries.items():
            for k2, a2 in ancestries.items():
                if k1 == k2 or k1 == init_taxonomy.pseudo_root or k2 == init_taxonomy.pseudo_root:
                    continue
                for path1 in a1:
                    for path2 in a2:
                        dist = 1 - wu_palmer_similarity(path1, path2)
                        if dist < self._distance_map.get(frozenset({k1, k2}), 1.0):
                            self._distance_map[frozenset({k1, k2})] = dist


class WordNetTaxonomyMutator(TaxonomyMutator):
    class WordnetMutationMode(enum.Enum):
        MOST_COMMON = "most_common"
        HYPERNYM = "hypernym"
        ALL = "all"

    def __init__(self, most_common_meaning: bool = True) -> None:
        nltk.download('wordnet')
        self.most_common_meaning = most_common_meaning

    def __call__(self, taxonomy: Taxonomy, mutations_frac: float | None = None) -> Taxonomy:
        name_to_id = {v: k for k, v in taxonomy.id_to_name.items()}
        wornet_words = set(
            [n.replace(" ", "_") for n in taxonomy.id_to_name.values()
             if wn.synsets(n.replace(" ", "_"), lang="eng")]
        )

        if len(wornet_words) == 0:
            raise ValueError("No WordNet words found in the taxonomy!")

        n_mutations = int(mutations_frac * len(wornet_words)) if mutations_frac is not None else None
        limit = len(wornet_words) if n_mutations is None else min(n_mutations, len(wornet_words))
        shuffled_words = list(wornet_words)
        random.shuffle(shuffled_words)

        for w in shuffled_words[:limit]:
            synsets = wn.synsets(w, lang="eng", pos=wn.NOUN)
            if len(synsets) == 0:
                continue
            if self.most_common_meaning:
                lemmas = synsets[0].lemma_names(lang="eng")
            else:
                lemmas = list(itertools.chain(*(synset.lemma_names(lang="eng") for synset in synsets)))
            replacement = random.choice(lemmas)
            replacement = replacement.replace("_", " ")
            node_id = name_to_id[w.replace("_", " ")]
            taxonomy.id_to_name[node_id] = replacement
        return taxonomy
