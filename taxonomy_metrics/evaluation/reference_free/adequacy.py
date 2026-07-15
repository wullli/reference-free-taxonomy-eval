from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Set, Tuple, Union, Generator, Any

import inflect
import numpy as np
import pandas as pd
import spacy
from datasets import Dataset
from nltk import WordNetLemmatizer
from tqdm.auto import tqdm
from transformers import Pipeline, pipeline
from transformers.pipelines.base import KeyDataset

from taxonomy_metrics.evaluation.metric import ReferenceFreeMetric
from taxonomy_metrics.evaluation.reference_free.hearst_pattern_builder import HearstPatternBuilder
from taxonomy_metrics.experiments.common import torch_device
from taxonomy_metrics.graph.taxonomy import Taxonomy

m = inflect.engine()


class NLIVerificationMetric(ReferenceFreeMetric):
    name = "NLIVerification"

    def __init__(
            self,
            model: Union[str, Pipeline] | None = "facebook/bart-large-mnli",
            strict: bool = False,
            probability: bool = True,
            progress: bool = False,
            propagate: bool = True,
            lemmatize: bool = False,
            hearst_pattern_builder: HearstPatternBuilder | None = None,
    ) -> None:
        self.nlp = spacy.load("en_core_web_sm")
        if isinstance(model, str):
            self.nli: Pipeline | None = pipeline("text-classification", model=model, batch_size=128)
        else:
            self.nli = model
        self.model = model
        self.lemmatize: bool = lemmatize
        self.progress: bool = progress
        self.strict: bool = strict
        self.probability: bool = probability
        self.cache: dict = {}
        self.propagate: bool = propagate
        self._prev_scores: dict = {}
        self.hearst_pattern_builder: HearstPatternBuilder = hearst_pattern_builder or HearstPatternBuilder()
        self.tax: Taxonomy | None = None
        self.scores: dict = {}
        self.n_labels: int = 3  # entailment, contradiction, neutral

    @staticmethod
    def get_pronoun(word: str) -> str:
        return "an" if word[0] in ["a", "e", "i", "o", "u"] else "a"

    @staticmethod
    def iterate_edges(path: list) -> Generator[Tuple[str, str], None, None]:
        """
        Iterate over edges from a given path of nodes.
        Yields edges as tuples in the format (second, first).
        """
        if len(path) < 2:
            return
        for i in range(len(path) - 1):
            yield path[i], path[i + 1]

    def strong_score(self, r: list[dict]) -> float:
        probs = []
        for pattern_res in r:
            pattern_prob = [d for d in pattern_res
                            if d["label"].lower() == "entailment"][0]["score"]
            probs.append(pattern_prob)

        prob = np.mean(probs)
        if self.probability:
            return prob
        return int(prob > 0.5)

    def weak_score(self, r: list[dict]) -> float:
        probs = []
        for pattern_res in r:
            pattern_prob = 1 - [d for d in pattern_res
                                if d["label"].lower() == "contradiction"][0]["score"]
            probs.append(pattern_prob)

        prob = np.mean(probs)
        if self.probability:
            return prob
        return int(prob > 0.5)

    def predict_edge_scores(self, tax: Taxonomy) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
        assert self.nli is not None
        queries: list[dict[str, str]] = []
        n_patterns = self.hearst_pattern_builder.number_of_patterns
        cached_mask = []
        used_edges = []
        for i, (p, c) in enumerate(tax.g.edges()):
            if p == tax.pseudo_root or c == tax.pseudo_root:
                continue
            pn, cn = tax.id_to_name[p], tax.id_to_name[c]
            child_desc = tax.id_to_desc.get(c, cn)
            if self.lemmatize:
                child_name, parent_name = str(self.nlp(cn)[0].lemma_), str(self.nlp(pn)[0].lemma_)
            else:
                child_name, parent_name = cn, pn

            pattern_instances = self.hearst_pattern_builder.build(
                hypernym=parent_name, hyponym=child_name,
                hypernym_plural=m.plural(parent_name),
                hyponym_article=self.get_pronoun(child_name),
                hypernym_article=self.get_pronoun(parent_name)
            )
            current_queries = [f"{child_desc}. {pattern_instance}" for pattern_instance in pattern_instances]
            queries.extend([{"text": q} for q in current_queries])
            cached_mask.append((c, p) in self.cache)
            used_edges.append((c, p))

        cached_mask_array = np.array(cached_mask)
        query_array = np.array(queries).reshape(-1, n_patterns)
        assert query_array.shape == (cached_mask_array.shape[0], n_patterns)

        query_ds = KeyDataset(
            Dataset.from_list(query_array[~cached_mask_array, :].ravel().tolist()), key="text"
        )

        outputs = []
        nli = self.nli(query_ds, top_k=None)
        if self.progress:
            nli = tqdm(nli, total=len(query_ds), desc="Running Queries")

        for out in nli:
            outputs.append(out)

        output_array = np.reshape(np.array(outputs), (-1, n_patterns, self.n_labels))
        return output_array, cached_mask_array, used_edges

    def _collect_scores(self, output_array: np.ndarray, used_edges: list, cached_mask: np.ndarray) -> tuple[
        dict, dict, list]:
        results = []
        running_strong_sum = 0.0
        running_weak_sum = 0.0
        total = len(used_edges)
        scores = defaultdict(list)
        for i in range(total):
            is_cached = cached_mask[i]
            if not is_cached:
                out = output_array[np.cumsum(~cached_mask)[i] - 1]
                results.append(out)
                self.cache[tuple(used_edges[i])] = out
            else:
                results.append(self.cache[tuple(used_edges[i])])
            strong_score = self.strong_score(results[-1])
            weak_score = self.weak_score(results[-1])
            scores["Strong"].append(strong_score)
            scores["Weak"].append(weak_score)
            running_strong_sum += strong_score
            running_weak_sum += weak_score

        res = {
            "Strong": running_strong_sum / total,
            "Weak": running_weak_sum / total,
        }
        return res, scores, results

    def _propagate_scores(self, tax: Taxonomy, scores: dict) -> dict:
        out = {}
        for score_type, score_values in scores.items():
            running_sum = 0
            score_map = dict(zip(tax.g.edges(), score_values))
            ancestries = tax.ancestries()
            if self.progress:
                ancestries = tqdm(ancestries, desc="Propagating Scores")

            n_invalid_ancestries = 0  # roots
            for a in ancestries:
                a = [n for n in a if n != tax.pseudo_root]
                if len(a) < 2:
                    n_invalid_ancestries += 1
                    continue
                propagated_score = 1
                n_edges = 0
                for edge in self.iterate_edges(a):
                    if tax.pseudo_root in edge:
                        continue
                    n_edges += 1
                    propagated_score *= score_map[edge]
                # nth root of the product of scores, geometric mean
                propagated_score = propagated_score ** (1 / n_edges)
                running_sum += propagated_score
            divisor = len(ancestries) - n_invalid_ancestries
            out[score_type] = running_sum / divisor if divisor > 0 else 0
        return out

    def calculate(
            self,
            taxonomy: Taxonomy,
            node_subset: Set[str] | None = None,
    ) -> dict[str, float]:
        if self.tax is not None and self.tax.id_to_name != taxonomy.id_to_name:
            # in case the id mapping has changed, we need to reset the cache
            self.cache = {}
        self.tax = deepcopy(taxonomy)

        if self.nli is None:
            used_edges = list(self.tax.g.edges())
            output_array = np.array([])
            cached_mask = np.full(len(used_edges), fill_value=True, dtype=bool)
        else:
            output_array, cached_mask, used_edges = self.predict_edge_scores(self.tax)
        res, scores, results = self._collect_scores(output_array, used_edges, cached_mask)

        if self.propagate:
            res = self._propagate_scores(self.tax, scores)
        self.scores = scores
        return self._scores(**res)


class RaTEMetric(ReferenceFreeMetric):
    """
    Based on https://github.com/CestLucas/RaTE
    With added caching implemented in the calculate method
    """
    name = "RaTE"

    def __init__(
            self,
            pattern_path: str | Path,
            top_k: int,
            model: str = "bert-large-uncased-whole-word-masking",
            batch_size: int = 128,
    ) -> None:
        self.pattern_path = pattern_path
        self.model = model
        self.top_k = top_k

        self.query_templates: list[str] = []
        self.column_names: list[str] = []
        self.cached_queries: dict = {}
        with open(self.pattern_path, "r") as fin:
            for line in fin.readlines():
                if not line.startswith("#"):
                    query, name = line.split(",")
                    self.query_templates.append(query.strip())
                    self.column_names.append(name.strip())

        device = torch_device()
        self.unmasker = pipeline(
            "fill-mask", model=self.model, device=device, batch_size=batch_size
        )
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize_a_word(self, word: str) -> str:
        split_word = word.split()
        ngram = len(split_word)
        if ngram == 1:
            return self.lemmatizer.lemmatize(word, pos="n")
        else:
            last_word = split_word[-1]
            lem = self.lemmatizer.lemmatize(last_word, pos="n")
            new_word = " ".join(split_word[:-1] + [lem])
            return new_word

    def generate_queries_from_template(self, token: str) -> list[str]:
        test_queries = []

        for template in self.query_templates:
            test_queries.append(template.replace("{token}", token) + " .")

        return test_queries

    def calculate(
            self,
            taxonomy: Taxonomy,
            node_subset: Set[str] | None = None,
    ) -> dict[str, float]:
        predictions: dict[str, dict[str, Any]] = {}

        eval_parents = []
        eval_children = []

        for parent, child in taxonomy.g.edges():
            eval_parents.append(taxonomy.id_to_name[parent])
            eval_children.append(taxonomy.id_to_name[child])

        n_templates = len(self.generate_queries_from_template("dummy"))
        queries = []
        cached_mask = []
        for child in eval_children:
            child_queries = self.generate_queries_from_template(child)
            queries.append(child_queries)
            cached_mask.append(child in self.cached_queries)

        cached_mask_array = np.array(cached_mask)
        queries_unmasked = np.array(
            self.unmasker(
                np.array(queries)[~cached_mask_array].ravel().tolist(), top_k=self.top_k
            )
        ).reshape(-1, n_templates, self.top_k)

        for i, child in enumerate(eval_children):
            if not cached_mask_array[i]:
                unmasked_idx = np.cumsum(~cached_mask_array)[i] - 1
                self.cached_queries[child] = queries_unmasked[unmasked_idx]
            unmasked = self.cached_queries[child]

            for j, column in enumerate(self.column_names):
                if child not in predictions:
                    predictions[child] = {}
                predictions[child][column] = unmasked[j]

        df_result = pd.DataFrame()
        df_result["hypernym"] = eval_parents
        df_result["hyponym"] = eval_children

        for column_name in self.column_names:
            query_predictions = []

            for parent, child in zip(eval_parents, eval_children):
                preds = predictions[child][column_name]

                parent_in_preds = False

                for pred_list in preds:
                    if parent == pred_list["token_str"]:
                        parent_in_preds = True
                        break
                query_predictions.append(parent_in_preds * 1)

            df_result[column_name] = query_predictions

        df_result["sum"] = df_result[self.column_names].sum(axis=1)
        return self._scores(score=len(df_result.loc[df_result["sum"] > 0]) / len(df_result))

