"""Sphinx configuration for physicskit."""

import os
import sys

import matplotlib

sys.path.insert(0, os.path.abspath("../.."))


# The cosmic-web formation animation (a 120x120-particle scatter animated
# over 60 frames) embeds just over Matplotlib's default 20MB HTML5-video
# limit, silently dropping trailing frames. Sphinx-Gallery's built-in
# "matplotlib" reset_modules entry calls plt.rcdefaults() before *every*
# example script runs, which would immediately undo a plain module-level
# rcParams assignment here, so the raised limit has to be reapplied as a
# reset_modules callable of its own, ordered right after "matplotlib".
def _raise_animation_embed_limit(gallery_conf, fname):
    matplotlib.rcParams["animation.embed_limit"] = 50


project = "physicskit"
copyright = "2026, physicskit contributors"
author = "physicskit team"
release = "1.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Supports NumPy-style docstrings
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx_gallery.gen_gallery",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False
# Consistent with napoleon_use_param above: render a docstring's
# "Attributes" section as a :ivar: field list rather than individual
# ".. attribute::" directives. Without this, a dataclass whose docstring
# documents its fields (for the description text) collides with
# autodoc's own automatic per-field attribute documentation, producing
# "duplicate object description" warnings for every such field.
napoleon_use_ivar = True

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Subpackages with a sphinx-gallery-formatted examples/<name>/ directory.
_GALLERY_SUBPACKAGES = [
    "astro",
    "chaos",
    "classical",
    "condensed",
    "fields",
    "fluids",
    "optics",
    "particle",
    "plasma",
    "quantum",
    "relativity",
    "rmt",
    "semiclassical",
    "statphys",
]

sphinx_gallery_conf = {
    "examples_dirs": [f"../../examples/{name}" for name in _GALLERY_SUBPACKAGES],
    "gallery_dirs": [f"api/gallery/{name}" for name in _GALLERY_SUBPACKAGES],
    # Most scripts follow the "plot_*.py" sphinx-gallery convention; the
    # physicskit.rmt paper-replication scripts predate that convention and
    # are named "*_demo.py" instead.
    "filename_pattern": r"(/plot_|_demo\.py$)",
    "download_all_examples": False,
    "within_subsection_order": "FileNameSortKey",
    "remove_config_comments": True,
    "matplotlib_animations": True,
    "reset_modules": ("matplotlib", "seaborn", _raise_animation_embed_limit),
    # Lets ".. minigallery::" (used throughout docs/source/history/) resolve
    # fully-qualified object names in addition to the file paths/globs it
    # already handles, and is required for it to not warn about falling
    # back to file-path resolution on every single invocation.
    "backreferences_dir": "gen_modules/backreferences",
    "doc_module": ("physicskit",),
}

# Merging seven previously-independent packages into one Sphinx namespace
# means common attribute names (e.g. "dim", "ndof") are now genuinely
# ambiguous across unrelated classes; autolinking to a single target for
# them isn't meaningful, so treat those as non-fatal rather than broken links.
#
# "config.cache" is suppressed because sphinx_gallery_conf legitimately
# contains a live callable (_raise_animation_embed_limit, above, passed
# via reset_modules) that Sphinx's build cache cannot pickle; this only
# disables caching for that one config value; it doesn't affect build
# correctness.
suppress_warnings = ["ref.python", "config.cache"]

html_theme = "pydata_sphinx_theme"
html_logo = "_static/images/physicskit_logo_transparent.png"
html_theme_options = {
    "github_url": "https://github.com/physicskit/physicskit",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "navigation_depth": 2,
    "logo": {
        "alt_text": "physicskit logo",
    },
}
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
