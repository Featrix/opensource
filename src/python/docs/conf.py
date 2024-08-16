# Configuration file for the Sphinx documentation builder.
import os
import sys
from pathlib import Path

print("sys.path...", sys.path)
print("os.cwd = ", os.getcwd())


doc_dir = Path(__file__).parent
src_dir = doc_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# -- Project information
project = "Featrix"
copyright = "2024, Featrix, Inc"
author = "Featrix, Inc"

version = (src_dir / "VERSION").read_text().strip()
release = "0.1"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]


# Napolean settings
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
# napoleon_include_private_with_doc = False
# napoleon_include_special_with_doc = True
#
# other stuff documented here - https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

autodoc_default_options = {
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": None,
    "exclude-members": "__weakref__,model_config,model_fields,model_post_init,model_computed_fields",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]
templates_path = ["_templates"]

# -- Options for HTML output
html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"
