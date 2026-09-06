# -*- coding: utf-8 -*-
# ruff: noqa: E402
"""Compatibility package entry point for the nested source tree.

This repository keeps the importable package in ``phygnn/phygnn``. When the
parent folder is used as the working directory, Python first sees the outer
``phygnn`` repository folder. This shim exposes the inner package so imports
such as ``from phygnn import PhysicsGuidedNeuralNetwork`` work from either
directory.
"""

import os
from pathlib import Path

from keras.utils import get_custom_objects

_OUTER_DIR = Path(__file__).resolve().parent
_INNER_DIR = _OUTER_DIR / 'phygnn'

if _INNER_DIR.is_dir():
    __path__.insert(0, str(_INNER_DIR))

from ._version import __version__
from .base import CustomNetwork, GradientUtils
from .layers import HiddenLayers, Layers
from .layers.custom_layers import (
    GaussianAveragePooling2D,
    get_custom_layer_objects,
)
from .model_interfaces import PhygnnModel, TfModel
from .phygnn import PhysicsGuidedNeuralNetwork
from .utilities import PreProcess, tf_isin, tf_log10

get_custom_objects().update(get_custom_layer_objects())

__author__ = """Grant Buster"""
__email__ = 'grant.buster@nlr.gov'

PHYGNNDIR = str(_INNER_DIR)
TESTDATADIR = os.path.join(str(_OUTER_DIR), 'tests', 'data')
