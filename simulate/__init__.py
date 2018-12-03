"""
Composable, fast Simulation framework for python.

`simulate` can be used to create and run simulations of interacting indiduals.
It's design goal is to keep the simulation code simple and composable while
being able to execute a simulation efficiently, executing hundreds of steps
of thousands of individuals a second.
"""
import warnings

ignores = ["numpy.dtype size changed",
           "numpy.ufunc size changed",
           "Using or importing the ABCs from "
           + "'collections' instead of from 'collections.abc'"]
for i in ignores:
    warnings.filterwarnings("ignore", message=i)

from .model import Model, step  # noqa: E402
from .alloc import Allocation  # noqa: E402
from . import trace  # noqa: E402
from . import engine  # noqa: E402
from . import visual  # noqa: E402


__all__ = ['Model', 'step', 'Allocation', 'trace', 'engine', 'visual']
