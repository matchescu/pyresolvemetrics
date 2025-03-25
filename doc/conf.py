# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
from pathlib import Path

src_dir = Path(__file__).parent / "src"

sys.path.insert(0, str(src_dir.absolute()))

project = "PyResolveMetrics"
copyright = "2024, Andrei Olar"
author = "Andrei Olar"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
]

templates_path = []
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "haiku"
html_theme_options = {
    "sidebar_width": "240px",  # Adjust sidebar width
    "page_width": "1024px",
}

# html_static_path = ["_static"]
