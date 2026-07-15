import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from dotenv import load_dotenv
from sklearn.metrics import classification_report

from taxonomy_metrics.evaluation.gold.wupalmer_metric import WuPSimilarityMetric
from taxonomy_metrics.evaluation.reference_free.adequacy import NLIVerificationMetric
from taxonomy_metrics.evaluation.wu_palmer import wu_palmer_similarity
from taxonomy_metrics.graph.taxonomy import Taxonomy
from taxonomy_metrics.data.loader import load_taxonomy

api_key_path = Path(__file__).parent.parent / ".env"
if Path(api_key_path).exists():
    load_dotenv(dotenv_path=str(api_key_path))

spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
out_path = Path(__file__).parent.parent.parent / "output"


def main(tax_dict: dict, nli_models: dict, mode: str = "Strong", node_descriptions: bool = True) -> None:
    nliv_results = defaultdict(list)

    for dataset, data in tax_dict.items():
        print(f"Calculating NLI scores for dataset: {dataset} ...")
        taxo = data["taxo"]
        id_to_name = data["id_to_name"]
        id_to_desc = data["id_to_desc"]
        true_tax = data["true_tax"]

        true_tax.connect()
        ancestries = true_tax.ancestries()
        positives_set = set()
        for a in ancestries:
            for i in range(len(a) - 1):
                for j in range(i + 1, len(a)):
                    positives_set.add((a[i], a[j]))

        positives: list[tuple[str, str]] = list(positives_set)
        negatives: list[tuple[str, str]] = []
        rs = np.random.RandomState(42)
        while len(negatives) < len(positives):
            neg_hypo = rs.choice(taxo.hyponym)
            neg_hyper = rs.choice(taxo.hypernym)
            if (
                    neg_hypo != neg_hyper
                    and (neg_hyper, neg_hypo) not in negatives
                    and (neg_hypo, neg_hyper) not in positives
            ):
                negatives.append((neg_hyper, neg_hypo))

        for model, metric in nli_models.items():
            pos_tax = Taxonomy(
                relations=positives,
                id_to_name=id_to_name,
                id_to_desc=id_to_desc,
            )
            metric.calculate(pos_tax)
            pos_scores = metric.scores[mode]
            positives = [p for p in positives if pos_tax.pseudo_root not in p]
            assert len(pos_scores) == len(positives), (f"Positives: {len(positives)}, "
                                                       f"Scores: {len(pos_scores)} "
                                                       f"Dataset: {dataset}")

            neg_tax = Taxonomy(
                relations=negatives,
                id_to_name=id_to_name,
                id_to_desc=id_to_desc,
            )
            metric.calculate(neg_tax)
            neg_scores = metric.scores[mode]
            negatives = [p for p in negatives if pos_tax.pseudo_root not in p]
            assert len(neg_scores) == len(negatives), (f"Negatives: {len(negatives)}, "
                                                       f"Scores: {len(neg_scores)} "
                                                       f"Dataset: {dataset}")

            pred_scores = list(pos_scores) + list(neg_scores)

            true_scores = [1] * len(pos_scores) + [0] * len(neg_scores)
            ancestries = WuPSimilarityMetric.get_ancestries(true_tax)
            dists = {}

            for p, c in positives:
                ancestry_p = ancestries.get(p, ())
                ancestry_c = ancestries.get(c, ())
                dists[frozenset([p, c])] = 1 - wu_palmer_similarity(
                    ancestry_p, ancestry_c
                )

            for p, c in negatives:
                ancestry_p = ancestries.get(p, ())
                ancestry_c = ancestries.get(c, ())
                dists[frozenset([p, c])] = 1 - wu_palmer_similarity(
                    ancestry_p, ancestry_c
                )

            distances_pos = [dists[frozenset([p, c])] for p, c in positives]
            distances_neg = [dists[frozenset([p, c])] for p, c in negatives]
            ds_dists = distances_pos + distances_neg

            nliv_results["model"].extend([model] * len(true_scores))
            nliv_results["dataset"].extend([dataset] * len(true_scores))
            nliv_results["pairs"].extend([tuple(p) for p in (positives + negatives)])
            nliv_results["dists"].extend(ds_dists)
            nliv_results["true_scores"].extend(true_scores)
            nliv_results["pred_scores"].extend(pred_scores)

            pred_labels = (np.array(pred_scores) >= 0.5).astype(int)
            print(classification_report(true_scores, pred_labels, zero_division=np.nan))
    try:
        use_desc = "desc" if node_descriptions else "name"
        pd.DataFrame(nliv_results).to_csv(out_path / f"nli_results_{mode}_{use_desc}.csv", index=False)
    except ValueError:
        print([{k: len(v) for k, v in nliv_results.items()}])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nli_model_name", type=str, default="facebook/bart-large-mnli"
    )
    parser.add_argument(
        "--mode", type=str, choices=["Strong", "Weak"], default="Weak"
    )
    parser.add_argument("--node_descriptions", choices=["yes", "no"], default="yes")
    args = parser.parse_args()

    nli_models = {
        args.nli_model_name: NLIVerificationMetric(
            model=args.nli_model_name,
            progress=True,
            propagate=False,  # We assess only edge probabilities
            probability=True,
        )
    }

    trial = False
    tax_dict = {}
    use_node_descriptions = args.node_descriptions == "yes"

    print("Loading taxonomies...")
    for dataset in ["semeval_food", "mesh"]:
        data_path = Path(__file__).parent.parent.parent / "data" / f"{dataset}"
        terms, taxo = load_taxonomy(str(data_path))
        id_to_name = {
            d["node_id"]: d["node_name"] for d in terms.to_dict(orient="records")
        }
        if use_node_descriptions:
            id_to_desc = {d["node_id"]: d["desc"] for d in terms.to_dict(orient="records")}
        else:
            id_to_desc = {d["node_id"]: d["node_name"] for d in terms.to_dict(orient="records")}
        relations = [tuple(r) for r in taxo[["hypernym", "hyponym"]].values.tolist()]
        true_tax = Taxonomy(relations, id_to_name=id_to_name)
        tax_dict[dataset] = {
            "taxo": taxo,
            "id_to_name": id_to_name,
            "id_to_desc": id_to_desc,
            "true_tax": true_tax,
        }
        del terms

    main(tax_dict, nli_models, mode=args.mode, node_descriptions=use_node_descriptions)
