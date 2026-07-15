from pathlib import Path


class HearstPatternBuilder:

    def __init__(self) -> None:
        self.hearst_patterns: list[str] = []
        pattern_path = Path(__file__).parents[3] / "resources" / "hearst_patterns.txt"
        with open(pattern_path, "r") as fr:
            for line in fr:
                template_string = line.strip()
                self.hearst_patterns.append(template_string)

        self.number_of_patterns: int = len(self.hearst_patterns)

    def build(self, hypernym: str, hyponym: str,
              hyponym_article: str, hypernym_article: str, hypernym_plural: str) -> list[str]:
        return [pattern.format(hypernym=hypernym,
                               hyponym=hyponym,
                               hyponym_article=hyponym_article,
                               hypernym_article=hypernym_article, hypernym_plural=hypernym_plural)
                for pattern in self.hearst_patterns]
