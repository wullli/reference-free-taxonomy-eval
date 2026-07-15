import abc
from copy import deepcopy
from itertools import product
from typing import Any, Dict, Hashable, Set

import numpy as np
from scipy.stats import kendalltau

try:
    import cupy as cp

    print("Using CuPy backend.")
except ImportError:
    cp = np
    print("CuPy not installed.")

from collections import defaultdict

import torch
from scipy.spatial.distance import cosine
from sentence_transformers import util
from tqdm.auto import tqdm

from taxonomy_metrics.evaluation.metric import ReferenceFreeMetric
from taxonomy_metrics.graph.taxonomy import Taxonomy


def cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
    return cosine(vec_a, vec_b)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
    return 1 - cosine_distance(vec_a, vec_b)


class SentenceTransformerUnsupervisedMetric(ReferenceFreeMetric, abc.ABC):
    def __init__(
            self,
            use_descriptions: bool = False,
            sentence_transformer: Any = None,
            progress: bool = False,
            batch_size: int = 128,
            no_cache: bool = False
    ) -> None:
        self.progress: bool = progress
        self.batch_size: int = batch_size
        self.use_descriptions: bool = use_descriptions
        if isinstance(sentence_transformer, str):
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(sentence_transformer).eval()
        else:
            self.model = sentence_transformer
            if self.model is not None:
                self.model.eval()
        self.similarities: dict[frozenset[Hashable], float] | None = None
        self.path_dists: dict[frozenset[str], float] = {}
        self.emb_dists: dict[frozenset[str], float] = {}
        self.no_cache: bool = no_cache

    def embeddings(self, node2name: dict, descriptions: dict | None = None) -> dict[Hashable, np.ndarray]:
        with torch.no_grad():
            if self.use_descriptions:
                desc = descriptions if descriptions is not None else node2name
                sentence_map = {
                    nid: (
                        f"{nn}: {desc.get(nid, nn)}"
                        if isinstance(desc.get(nid, nn), str)
                        else nn
                    )
                    for nid, nn in node2name.items()
                }
            else:
                sentence_map = {nid: nn for nid, nn in node2name.items()}

        sentences = list(sentence_map.values())
        embeddings = self.model.encode(sentences, show_progress_bar=True, batch_size=32)
        embeddings = dict(zip(sentence_map.keys(), embeddings))
        return embeddings

    def similarity_map(self,
                       emb: dict,
                       subset: set[str] | None = None) -> dict[frozenset[Hashable], float]:
        with torch.no_grad():
            if subset is not None:
                centroids = {k: emb[k] for k in emb if k in subset}
            else:
                centroids = emb

            keys = np.array(list(centroids.keys()))
            emb_tensor = torch.tensor(np.array(list(centroids.values()))).to(
                self.model.device
            )
            similarity_map = {}

            def _batch_cosine_similarity(
                    emb_batch: torch.Tensor, key_batch: np.ndarray
            ) -> None:
                if len(emb_batch) == 0:
                    return
                sim_mat = util.cos_sim(emb_batch, emb_tensor).cpu().numpy()
                for i, c1 in enumerate(key_batch):
                    for j, c2 in enumerate(centroids.keys()):
                        similarity_map[frozenset((c1, c2))] = sim_mat[i, j]

            n = len(centroids)
            for i in tqdm(
                    range(0, n, self.batch_size), desc="Precomputing Similarity"
            ):
                emb_batch = emb_tensor[i:i + self.batch_size]
                key_batch = keys[i:i + self.batch_size]
                _batch_cosine_similarity(emb_batch, key_batch)
        self.similarities = similarity_map
        return self.similarities


class ConceptSimilarityCorrelationMetric(SentenceTransformerUnsupervisedMetric):
    name = "ConceptSimilarityCorrelation"

    def __init__(
            self,
            use_descriptions: bool = False,
            sentence_transformer: Any = None,
            progress: bool = False,
            no_cache: bool = False,
    ) -> None:
        super().__init__(
            use_descriptions=use_descriptions,
            sentence_transformer=sentence_transformer,
            progress=progress,
            no_cache=no_cache
        )
        self.tax: Taxonomy | None = None

    @classmethod
    def wu_palmer_similarity(
            cls, concept1: str, concept2: str, ancestries: Dict[str, list]
    ) -> float:
        from taxonomy_metrics.evaluation.wu_palmer import wu_palmer_similarity

        max_score = float("-inf")

        for a1 in ancestries.get(concept1, [()]):
            for a2 in ancestries.get(concept2, [()]):
                max_score = max(max_score, wu_palmer_similarity(a1, a2))

        return max_score

    def _set_dists(self, tax: Taxonomy) -> None:
        assert self.similarities is not None
        ancestries = defaultdict(list)
        for a in tax.ancestries():
            ancestries[a[-1]].append(a)

        nodes = tax.g.nodes()
        it = product(nodes, nodes)
        if self.progress:
            it = tqdm(it, total=len(nodes) ** 2, desc="Calculating CSC", mininterval=1)
        for c1, c2 in it:
            if c1 == c2 or c1 == tax.pseudo_root or c2 == tax.pseudo_root:
                continue
            self.path_dists[frozenset({c1, c2})] = (self.wu_palmer_similarity(c1, c2, ancestries))
            self.emb_dists[frozenset({c1, c2})] = self.similarities[frozenset({c1, c2})]

    def calculate(
            self,
            taxonomy: Taxonomy,
            node_subset: Set[str] | None = None,
    ) -> dict[str, float]:
        self.tax = deepcopy(taxonomy)
        self.tax.connect()

        if self.similarities is None or self.no_cache:
            embeddings = self.embeddings(taxonomy.id_to_name, taxonomy.id_to_desc)
            self.similarities = self.similarity_map(embeddings, node_subset)

        self._set_dists(self.tax)
        return self._scores(
            score=kendalltau(list(self.path_dists.values()), list(self.emb_dists.values())).statistic
        )


class SemanticProximityMetric(SentenceTransformerUnsupervisedMetric):
    name = "SemanticProximity"

    def calculate(
            self,
            taxonomy: Taxonomy,
            node_subset: Set[str] | None = None,
    ) -> dict[str, float]:
        tax = deepcopy(taxonomy)
        tax.connect()

        children = tax.children()
        leaves = set(tax.leaves())

        if self.similarities is None or self.no_cache:
            embeddings = self.embeddings(taxonomy.id_to_name, taxonomy.id_to_desc)
            self.similarities = self.similarity_map(embeddings)
        assert self.similarities is not None

        scores = []

        it = children
        if self.progress:
            it = tqdm(it, desc="Calculating Semantic Proximity")
        for _, cs in it:
            leaf_children = [c for c in cs if c in leaves]
            if len(leaf_children) > 1:
                min_sim = min(
                    self.similarities[frozenset((c1, c2))]
                    for c1, c2 in product(*[leaf_children, leaf_children])
                    if c1 != c2
                )
                outsiders = leaves.difference(leaf_children)
                outside_sims = np.array(
                    [
                        self.similarities[frozenset((c1, c2))]
                        for c1, c2 in product(*[leaf_children, outsiders])
                        if c1 != c2
                    ]
                )
                sorted_outside_sims = np.sort(outside_sims)[::-1]
                intruders = np.sum(sorted_outside_sims >= min_sim)
                s = 1 - (intruders / len(sorted_outside_sims))
                scores.append(s)
        return self._scores(score=float(np.mean(scores)))
