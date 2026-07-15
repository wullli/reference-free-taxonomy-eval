
from setuptools import setup, find_packages

# load requirements from requirements.txt
reqs = open("requirements.txt").read().splitlines()

setup(
    name="reference_free_taxonomy_eval",
    version="0.1.2",
    description="Reference free metrics for taxonomy evaluation",
    author="Pascal Wullschleger",
    url="https://github.com/wullli/reference-free-taxonomy-eval",
    project_urls={
        "Source": "https://github.com/wullli/reference-free-taxonomy-eval",
        "Bug Tracker": "https://github.com/wullli/reference-free-taxonomy-eval/issues",
    },
    packages=find_packages(),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    install_requires=reqs,
)
