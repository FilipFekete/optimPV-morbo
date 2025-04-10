# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
# import optimpv from two folders up
sys.path.insert(0, os.path.abspath('../../'))
import optimpv
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'optimpv'
copyright = '2025, Vincent M. Le Corre'
author = 'Vincent M. Le Corre'
release = '1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser',
                "sphinx.ext.viewcode",
                "sphinx.ext.autodoc",
                "sphinx.ext.napoleon",
                "sphinx.ext.todo",
                "sphinx.ext.intersphinx",
                "nbsphinx",
                "sphinx_gallery.load_style",
                "IPython.sphinxext.ipython_console_highlighting",
                "sphinx.ext.mathjax",  # Maths visualization
                "sphinx.ext.graphviz",  # Dependency diagrams
                "sphinx_copybutton",
                "notfound.extension",
                'sphinx_search.extension',
                'sphinx_gallery.gen_gallery',
                "sphinx_gallery.load_style",
            ]

templates_path = ['_templates']
exclude_patterns = []
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for sphinx_gallery ----------------------------------------------
from plotly.io._sg_scraper import plotly_sg_scraper
image_scrapers = ('matplotlib', plotly_sg_scraper,)

sphinx_gallery_conf = {
     'doc_module': ('plotly',),
     'examples_dirs': 'examples',   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
     'filename_pattern': r"\.ipynb", # Notebooks to include
     'image_scrapers': image_scrapers,
    #  'ignore_pattern': r"__init__\.py", # Ignore this file
    #  'subsection_order': ExplicitOrder([
    #                 '../../examples/DD_Fit_fake_OPV.ipynb',
    #                 '../../examples/DD_Fit_real_OPV.ipynb',]),


}

nbsphinx_execute = 'never' # so we don't run the notebooks

# -- Options for HTML output -------------------------------------------------
html_context = {
    "display_github": True, # Integrate GitHub
    "github_user": "openPV-lab", # Update to the correct username
    "github_repo": "optimPV", # Update to the correct repo name
    "github_version": "main", # Version
    "conf_py_path": "/docs/source/", # Path in the checkout to the docs root
}