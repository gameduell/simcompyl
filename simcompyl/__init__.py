"""
Composable, fast Simulation framework for python.

`simulate` can be used to create and run simulations of interacting individuals.
It's design goal is to keep the simulation code simple and composable while
being able to execute a simulation efficiently, executing hundreds of steps
of thousands of individuals a second.
"""
import logging

from .core.model import Model, step
from .core.alloc import (Allocation, Param, Distribution,
                         Uniform, Bernoulli,
                         Continuous, Normal, Exponential)
from .core.trace import Trace
from .core.engine import Execution
from .core import trace
from .core import engine

logging.basicConfig(level=logging.INFO)

__all__ = ['Model', 'step', 'Allocation', 'Param', 'Distribution', 'Execution',
           'Uniform', 'Bernoulli', 'Continuous', 'Normal', 'Exponential',
           'trace', 'Trace', 'engine']

__version__ = "0.1"
