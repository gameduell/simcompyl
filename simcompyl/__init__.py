"""
Composable, fast Simulation framework for python.

`simulate` can be used to create and run simulations of interacting individuals.
It's design goal is to keep the simulation code simple and composable while
being able to execute a simulation efficiently, executing hundreds of steps
of thousands of individuals a second.
"""
import warnings
import logging

logging.basicConfig(level=logging.INFO)

ignores = ["numpy.dtype size changed",
           "numpy.ufunc size changed",
           "Using or importing the ABCs from "
           "'collections' instead of from 'collections.abc'"]
for i in ignores:
    warnings.filterwarnings("ignore", message=i)

from .core.model import Model, step  # noqa: E402
from .core.alloc import (Allocation, Param, Distribution,
                         Uniform, Bernoulli,
                         Continuous, Normal, Exponential)  # noqa: E402
from .core.trace import Trace  # noqa: E402
from .core.engine import Execution  # noqa: E402
from .core import trace  # noqa: E402
from .core import engine  # noqa: E402


__all__ = ['Model', 'step', 'Allocation', 'Param', 'Distribution', 'Execution',
           'Uniform', 'Bernoulli', 'Continuous', 'Normal', 'Exponential',
           'trace', 'Trace', 'engine']
