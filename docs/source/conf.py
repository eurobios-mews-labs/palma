# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('.'))

project = 'PALMA : project for automated learning machine'
copyright = '2024, Eurobios-Mews-Labs'
author = 'Eurobios-Mews-Labs'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'myst_parser', "sphinx.ext.coverage",
              'sphinx.ext.todo', 'sphinx_copybutton', 'sphinx_favicon',
              'autoapi.extension', 'versionwarning.extension', "sphinx-prompt"
              ]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'press'
html_static_path = ['_static']
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}
favicons = [
   {
      "sizes": "16x16",
      "href": "https://secure.example.com/favicon/favicon-16x16.png",
   },
   {
      "sizes": "32x32",
      "href": "https://secure.example.com/favicon/favicon-32x32.png",
   },
   {
      "rel": "apple-touch-icon",
      "sizes": "180x180",
      "href": "apple-touch-icon-180x180.png",  # use a local file in _static
   },
]
autoapi_dirs = [
    '../../palma',
]