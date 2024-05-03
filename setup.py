"""Install configuration for the routing package"""

from setuptools import setup

setup(
    name="routing",
    version="0.1.0",
    where="src",
    include="routing",
    setup_requires=[
        # "graph-tool", # Hidden dependency, requires custom install
        "thefuzz",
        "python-levenshtein",
        "geopy",
        "dash",
        "plotly",
        "tqdm",
        "bng-latlon",
        "pyarrow",
        "pandas",
        "networkx",
    ],
)
