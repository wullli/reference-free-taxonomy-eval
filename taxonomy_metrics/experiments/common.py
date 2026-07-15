import json
from re import sub
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from taxonomy_metrics.evaluation import GoldStandardMetric
from taxonomy_metrics.evaluation.metric import ReferenceFreeMetric
from taxonomy_metrics.graph.taxonomy import Taxonomy
from taxonomy_metrics.task.taxonomy_downstream_task import DataSplit, TaxonomyDownstreamTask


def positions(tax: Taxonomy) -> dict[str, list[tuple[str, str]]]:
    """
    Get all unique (parent, query, child) triplets in the taxonomy and group them by query.
    :param tax: The taxonomy to extract positions from
    :return: A dictionary mapping each query to a list of (parent, child) tuples representing its positions.
    """
    unique_triplets = set()
    for t in tax.triplets(existing=True):
        unique_triplets.add(t)
    df = pd.DataFrame(list(unique_triplets), columns=["parent", "query", "child"])
    df = df.groupby("query").agg({"parent": list, "child": list})
    df["positions"] = df.apply(lambda x: list(zip(x.parent, x.child)), axis=1)
    df = df.drop(columns=["parent", "child"])
    df = df.reset_index(drop=False)
    pos = {row.query: row.positions for _, row in df.iterrows()}
    return pos


def add_gold_standard_scores(metric: GoldStandardMetric,
                             mutated_positions: dict[str, list[tuple[str, str]]],
                             true_positions: dict[str, list[tuple[str, str]]],
                             true_tax: Taxonomy,
                             query_weights: Mapping[str, float] | None = None) -> dict[str, float]:
    score, score_nleaf, score_leaf = metric.calculate(
        pred_positions=mutated_positions,
        true_positions=true_positions,
        node2name=true_tax.id_to_name,
        weights=query_weights,
        leaves=set(true_tax.leaves()),
        first_only=False,
    )
    all_scores = [
        (k, v)
        for d in (score, score_nleaf, score_leaf)
        for k, v in d.items()
        if not isinstance(v, np.ndarray)
    ]
    concat_scores = {}
    weight_mode = "weighted" if query_weights is not None else "unweighted"

    for k, v in all_scores:
        prefix = type(metric).__name__
        concat_scores[f"{weight_mode} {prefix} {k}"] = v

    return concat_scores


def record_sample(*,
                  taxonomy: Taxonomy,
                  original_taxonomy: Taxonomy,
                  true_positions: dict[str, list[tuple[str, str]]],
                  n_mutations: int,
                  data: dict[str, list[object]],
                  dataset: str,
                  metrics: list[ReferenceFreeMetric | GoldStandardMetric]
                           | Mapping[str, ReferenceFreeMetric | GoldStandardMetric],
                  query_weights: Mapping[str, float] | None = None,
                  taxonomy_id: str | None = None,
                  data_split: DataSplit | None = None,
                  task: TaxonomyDownstreamTask | None = None,
                  **kwargs: Any,
                  ) -> dict[str, list[object]]:
    if taxonomy_id is None:
        taxonomy_id = str(hash(frozenset(taxonomy.g.edges())))
    #data["mutated_taxonomy"].append(base64.b64encode(pickle.dumps(taxonomy)).decode("utf-8"))
    #data["original_taxonomy"].append(base64.b64encode(pickle.dumps(original_taxonomy)).decode("utf-8"))
    data["dataset"].append(dataset)
    data["n_mutations"].append(n_mutations)
    data["downstream_f1"].append(task.score(taxonomy, data_split) if task is not None
                                                                     and data_split is not None else None)
    data["taxonomy_id"].append(taxonomy_id)
    for k, v in kwargs.items():
        data[k].append(v)
    mutated_positions = positions(taxonomy)

    def _calculate_metric(data: dict, metric_name: str,
                          metric_instance: ReferenceFreeMetric | GoldStandardMetric) -> None:
        if isinstance(metric_instance, ReferenceFreeMetric):
            score = metric_instance.calculate(taxonomy)
            for k, v in score.items():
                data[f"{metric_name} {k}"].append(v)
        elif isinstance(metric_instance, GoldStandardMetric):
            all_scores = add_gold_standard_scores(
                metric=metric_instance,
                mutated_positions=mutated_positions,
                true_positions=true_positions,
                true_tax=original_taxonomy,
                query_weights=query_weights,
            )
            for k, v in all_scores.items():
                data[f"{metric_name} {k}"].append(v)

            unweighted_scores = add_gold_standard_scores(
                metric=metric_instance,
                mutated_positions=mutated_positions,
                true_positions=true_positions,
                true_tax=original_taxonomy,
                query_weights=None,
            )
            for k, v in unweighted_scores.items():
                data[f"{metric_name} {k}"].append(v)

    if isinstance(metrics, list):
        for m in metrics:
            _calculate_metric(data, type(m).__name__, m)
    else:
        for m_name, m in metrics.items():
            _calculate_metric(data, m_name, m)

    return data


def torch_device() -> torch.device:
    """
    Select the best available PyTorch device in the order: CUDA -> MPS -> CPU.
    Returns:
        torch.device: The best available device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_json(json_path: str) -> dict:
    """
    Load a json file
    :param json_path: path to the json file
    :return: the taxonomy as a dictionary
    """
    with open(json_path) as f:
        taxonomy = json.load(f)
    return taxonomy


def snake_case(s: str) -> str:
    """
    Replace hyphens with spaces, then apply regular expression substitutions for title case conversion
    and add an underscore between words, finally convert the result to lowercase
    :param s: The string to convert
    :return:
    """
    return "_".join(
        sub(
            "([A-Z][a-z]+)", r" \1", sub("([A-Z]+)", r" \1", s.replace("-", " "))
        ).split()
    ).lower()
