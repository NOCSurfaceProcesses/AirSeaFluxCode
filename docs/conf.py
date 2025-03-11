# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../'))
# sys.path.append('.')
# sys.path.append('../../')

# -- Project information -----------------------------------------------------

project = 'AirSeaFluxCode'
copyright = '2023, Stavroula Biri'
author = 'Stavroula Biri'

# The full version, including alpha/beta/rc tags
release = '1.2.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest', 'sphinx.ext.todo',
              'sphinx.ext.mathjax', 'sphinx.ext.ifconfig',
              'sphinx.ext.viewcode', 'sphinx.ext.githubpages',
              'sphinx.ext.napoleon', 'sphinx.ext.autosummary',
              'sphinx_autodoc_typehints'
              ]  # 'autoapi.extension',

# autoapi_type = 'python'
# autoapi_dirs = ['../../']
# add_module_names = False
# autoapi_keep_files = False
# autodoc_typehints = "description"

# autoapi_options = ['members', 'undoc-members', 'private-members']
# autoapi_options = ['members', 'undoc-members', 'private-members', 'show-inheritance',
#                     'show-module-summary', 'special-members', 'imported-members']
# autoapi_ignore = ['*mymodel*', '*conf*', '*gather_stats_c99.py*']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

pygments_style = 'sphinx'
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# "sphinxawesome_theme" # 'sphinx_rtd_theme' # 'alabaster'
html_theme = 'sphinx_rtd_theme'
# html_theme_options = {
#     "rightsidebar": "true",
#     "relbarbgcolor": "black"
# }
# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    'style_nav_header_background': 'white',
    'display_version': True,
    'logo_only': False,
    'collapse_navigation': False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
html_sidebars = {
    '**': ['logo-text.html', 'globaltoc.html', 'searchbox.html']
    # '**': ['logo-text.html', 'globaltoc.html', 'localtoc.html', 'searchbox.html']
}

numfig = True
math_numfig = True
numfig_secnum_depth = 2
math_eqref_format = "Eq. {number}"
