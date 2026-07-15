import glob
from pathlib import Path

import numpy as np
import pandas as pd


def _clean_semeval_verb(node_name: str | None) -> str | None:
    if node_name is not None and node_name != "":
        return node_name.split("||")[0].split(".")[0].replace("_", " ")
    return node_name


def _clean_mesh(node_name: str | None) -> str | None:
    if node_name is not None and node_name != "":
        return " ".join(node_name.replace(",", "").split())
    return node_name


def _remove_mesh_cycles(taxo: pd.DataFrame) -> pd.DataFrame:
    taxo = taxo[
        taxo.apply(
            lambda r: not (
                (r.hyponym == "proteins")
                and (r.hypernym in ["glycoproteins", "bloodproteins"])
            ),
            axis=1,
        )
    ]
    return taxo


def load_taxonomy(
    taxonomy_dir: str, with_embeddings: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    terms_file = glob.glob(str(Path(taxonomy_dir) / "*.terms"))[0]
    taxo_file = glob.glob(str(Path(taxonomy_dir) / "*.taxo"))[0]
    desc_file = glob.glob(str(Path(taxonomy_dir) / "*.desc"))[0]

    with open(terms_file, "r") as f:
        term_lines = f.readlines()

    with open(taxo_file, "r") as f:
        taxo_lines = f.readlines()

    with open(desc_file, "r") as f:
        desc_lines = f.readlines()

    terms = np.array([t.strip().split("\t") for t in term_lines], dtype=str).T
    taxos = np.array([t.strip().split("\t") for t in taxo_lines], dtype=str)
    descs = np.array([t.strip().split("\t") for t in desc_lines], dtype=str).T

    terms = pd.DataFrame({"node_id": terms[0], "node_name": terms[1]})

    descs = pd.DataFrame({"node_name": descs[0], "desc": descs[1]})
    terms = pd.merge(
        terms, descs, left_on="node_name", right_on="node_name", how="left"
    )
    terms.desc = terms.desc.fillna(terms.node_name)
    terms.node_name = terms.node_name.apply(str.lower)

    terms["unique_node_name"] = terms["node_name"]
    taxos = pd.DataFrame(taxos, columns=["hypernym", "hyponym"])

    if "semeval_verb" in taxonomy_dir:
        terms["node_name"] = terms.node_name.apply(_clean_semeval_verb)

    if "mesh" in taxonomy_dir:
        terms["node_name"] = terms.node_name.apply(_clean_mesh)
        taxos = _remove_mesh_cycles(taxos)

    if with_embeddings:
        embed_file = glob.glob(str(Path(taxonomy_dir) / "*terms.embed"))[0]
        with open(embed_file, "r") as f:
            emb_lines = f.readlines()
        embs = np.stack([t.strip().split(" ") for t in emb_lines[1:]], dtype=str)
        print(len(embs[:, 0]))
        print(len(list(embs[:, 1:])))
        embs = pd.DataFrame({"node_id": embs[:, 0], "embedding": embs[:, 1:].tolist()})

        terms = pd.merge(terms, embs, left_on="node_id", right_on="node_id")

    terms.drop_duplicates(subset=["node_id"], keep="first", inplace=True)
    return terms, taxos
