"""
Bloodwork AI: Multi-modal bloodwork analysis platform.

This package provides tools for analyzing bloodwork data using both
microscopy images and tabular lab data to predict various clinical conditions.
"""

__version__ = "0.1.0"
__author__ = "Bloodwork AI Team"
__email__ = "team@bloodwork-ai.com"

from . import utils
from . import schemas
from . import ingestion
from . import preprocessing
from . import features
from . import models
from . import tasks
from . import training
from . import evaluation
from . import explainability
from . import serving

__all__ = [
    "utils",
    "schemas", 
    "ingestion",
    "preprocessing",
    "features",
    "models",
    "tasks",
    "training",
    "evaluation",
    "explainability",
    "serving",
]
