# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PySEMTools'
copyright = '2024, Neko authors'
author = 'Neko authors'
release = 'develop'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['numpydoc', 'sphinx.ext.autodoc', 'sphinx.ext.autosummary', 'sphinx.ext.githubpages', 'sphinx_rtd_theme', 'nbsphinx']

templates_path = ['_templates']
exclude_patterns = []

# Extension configuration
numpydoc_class_members_toctree = False
numpydoc_validation_checks = {"all", "SA01", "EX01"}



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
