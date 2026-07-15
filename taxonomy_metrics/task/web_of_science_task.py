import re
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

from taxonomy_metrics.graph.taxonomy import Taxonomy
from taxonomy_metrics.task.taxonomy_downstream_task import TaxonomyDownstreamTask, DataSplit

WOS_PATH = Path(__file__).parents[3] / "data" / "web_of_science" / "Meta-data"


class WebOfScienceTask(TaxonomyDownstreamTask):
    """
    A class to load a dataset from the Web of Science.
    """

    @staticmethod
    def _clean_str(string: str) -> str:
        """
        https://github.com/kk7nc/HDLTex/blob/master/HDLTex/Data_helper.py
        Tokenization/string cleaning for dataset
        Every dataset is lower cased except
        """
        string = re.sub(r"\\", "", string)
        string = re.sub(r"\'", "", string)
        string = re.sub(r"\"", "", string)
        return string.strip().lower()

    @staticmethod
    def _text_cleaner(text: str) -> str:
        """
        https://github.com/kk7nc/HDLTex/blob/master/HDLTex/Data_helper.py
        cleaning spaces, html tags, etc
        parameters: (string) text input to clean
        return: (string) clean_text
        """
        text = text.replace(".", "")
        text = text.replace("[", " ")
        text = text.replace(",", " ")
        text = text.replace("]", " ")
        text = text.replace("(", " ")
        text = text.replace(")", " ")
        text = text.replace("\"", "")
        text = text.replace("-", "")
        text = text.replace("=", "")
        rules = [
            {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
            {r'\s+': u' '},  # replace consecutive spaces
            {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
            {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
            {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
            {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
            {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
            {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
            {r'^\s+': u''}  # remove spaces at the beginning
        ]
        for rule in rules:
            for (k, v) in rule.items():
                regex = re.compile(k)
                text = regex.sub(v, text)
            text = text.rstrip()
            text = text.strip()
        clean_text = text.lower()
        return clean_text

    @classmethod
    def load_data(cls, path: Path = WOS_PATH,
                  sentence_transformer: SentenceTransformer | None = None,
                  ) -> \
            tuple[DataSplit, Taxonomy]:
        meta_file = path / "Data.xlsx"
        df = pd.read_excel(path / meta_file, sheet_name=None)["abstracts"]

        df.Abstract = df.Abstract.apply(cls._text_cleaner).apply(str.strip)
        df.area = df.area.apply(str.strip)
        df.Domain = df.Domain.apply(str.strip)

        max_id = df.Y.max()
        df.Y1 = df.Y1.apply(lambda x: x + max_id)
        df.Y1 = df.Y1.apply(str)
        df.Y2 = df.Y2.apply(str)

        l2_id_to_name = {y_idx: y_name for y_idx, y_name in dict(zip(df.Y2, df.area)).items()}
        l1_id_to_name = {y_idx: y_name
                         for y_idx, y_name in dict(zip(df.Y1, df.Domain)).items()}
        id_to_name = {**l1_id_to_name, **l2_id_to_name}

        label_l1 = np.array(df.Y1.values, dtype=str).reshape(-1, 1)
        label_l2 = np.array(df.Y2.values, dtype=str).reshape(-1, 1)
        edges = np.hstack((label_l1, label_l2))

        if not cls.is_cached() and sentence_transformer is not None:
            content = df.Abstract.values
            content_vectors = sentence_transformer.encode(
                content, show_progress_bar=True, batch_size=16
            )
            np.save(path / "content_vectors.npy", content_vectors)
        else:
            content_vectors = np.load(path / "content_vectors.npy")

        x_train, x_test, y_train, y_test = train_test_split(
            content_vectors, label_l2, test_size=0.2, random_state=42
        )
        split = DataSplit(x_train, x_test, y_train, y_test)
        unique_edges = (tuple(e) for e in np.unique(edges, axis=0))
        taxonomy = Taxonomy(relations=unique_edges, id_to_name=id_to_name)
        return split, taxonomy

    @classmethod
    def is_cached(cls, path: Path = WOS_PATH) -> bool:
        return path is not None and (path / "content_vectors.npy").exists()


if __name__ == "__main__":
    if not WebOfScienceTask.is_cached():
        st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        WebOfScienceTask.load_data(sentence_transformer=st)
    start = timeit.default_timer()
    print("Baseline Score: ", WebOfScienceTask.baseline())
    end = timeit.default_timer()
    print(f"Fit and eval took {end - start:.2f} seconds")
