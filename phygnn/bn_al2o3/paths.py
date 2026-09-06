# -*- coding: utf-8 -*-
"""Repository-root aware default output paths for the workflow.

The workflow scripts default their outputs into ``data/``, ``models/``, and
``outputs/`` under the repository root. The root is located by walking up from
this file until ``pyproject.toml`` is found, falling back to the current
working directory, and can be overridden with the ``PHYGNN_REPO_ROOT``
environment variable.
"""
import os
from pathlib import Path


def repo_root():
    """Resolve the repo root (the directory containing pyproject.toml)."""
    env_root = os.environ.get('PHYGNN_REPO_ROOT')
    if env_root:
        return Path(env_root).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / 'pyproject.toml').exists():
            return parent

    return Path.cwd()


def data_path(name):
    """Return a default path under the repository ``data/`` directory."""
    return repo_root() / 'data' / name


def models_path(name):
    """Return a default path under the repository ``models/`` directory."""
    return repo_root() / 'models' / name


def outputs_path(name):
    """Return a default path under the repository ``outputs/`` directory."""
    return repo_root() / 'outputs' / name
