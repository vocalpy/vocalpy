# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import pathlib
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'VocalPy'
copyright = '2023, David Nicholson'
author = 'David Nicholson'
release = ''

# This song and dance enables builds from outside the docs directory
# https://github.com/librosa/librosa/blob/96da0eb228f591740ecbaa369cbc03fc92ea8185/docs/conf.py#L29
srcpath = os.path.abspath(pathlib.Path(os.path.dirname(__file__)) / "..")
sys.path.insert(0, srcpath)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_nb',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx_copybutton',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx_design',
    'sphinxext.opengraph',
    'sphinx_tabs.tabs',
]

templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = 'index'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None

myst_enable_extensions = [
    "dollarmath",
    # "amsmath",
    # "deflist",
    # "html_admonition",
    # "html_image",
    "colon_fence",
    # "smartquotes",
    # "replacements",
    # "linkify",
    # "substitution",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'

html_logo = "_static/vocalpy-primary-logo.png"

html_theme_options = {
    "logo_only": True,
    "show_toc_level": 1,
}

html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------
autosummary_generate = True

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}

# -- Options for nitpicky mode

# ensure that all references in the docs resolve.
nitpicky = True
nitpick_ignore = []

for line in open('nitpick-ignore.txt'):
    if line.strip() == "" or line.startswith("#"):
        continue
    dtype, target = line.split(None, 1)
    target = target.strip()
    nitpick_ignore.append((dtype, target))
