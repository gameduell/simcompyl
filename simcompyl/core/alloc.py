"""
An allocation takes care of binding different variables to concrete values.

The allocation of the parameters and random variables is handled outside the
scope of the model, as
- we only want one location where parameters and values are defined,
  but they are used in different parts/classes of a simulation
- allocation objects can be combined, so you can split the parameters of a
  larger simulation into logical junks.
- we can run the same simulation with different (partial) parameter sets
"""

import numpy as np
import weakref
import contextlib

from itertools import chain
from types import FunctionType
from inspect import signature

__all__ = ['Allocation', 'Param', 'Distribution',
           'Uniform', 'Bernoulli',
           'Continuous', 'Normal', 'Exponential']


class Param:
    """The Param descriptor defines information about a single parameter.

    Attributes
    ----------
    name : str
        name of the parameter shown to the user
    default
        default value for the given parameter
    help : str
        longer description of the parameter
    options : tuple, list or dict thereof
        restrictions on the values, could be
        - a tuple with lower and upper bounds
        - a list of available options
        - or a dict thereof, if the values is a dict

    """

    def __init__(self, name, default, help=None, options=None):
        """Create a new param descriptor with the supplied options.

        Parameters
        ----------
        name : str
            name for the parameter as shown to the user
        default
            default value for the given parameter
        help : str (optional)
            longer description of the parameter
        options : tuple, list or dict thereof (optional)
            restrictions on the values, could be
            - a tuple with lower and upper bounds
            - a list of available options
            - or a dict thereof, if the values is a dict

        """
        self.allocs = weakref.WeakKeyDictionary()
        self.name = name
        self.default = default
        self.help = help

        self.options = options

    @property
    def arity(self):
        """Return the number of primitive values this parameter is defining."""
        if (isinstance(self.default, dict) and isinstance(
                next(iter(self.default.values())), list)):
            return sum(len(vs) for vs in self.default.values())

        if isinstance(self.default, (list, dict, tuple)):
            return len(self.default)

        return 1

    def alloc(self, obj):
        """Return the `Alloc`-instance of this Param for the given object."""
        return self.allocs.setdefault(obj, Alloc(self))

    def __get__(self, instance, owner=None):
        """Access the alloc instance for this parameter."""
        if instance is None:
            return self
        else:
            return self.alloc(instance)

    def __set__(self, instance, value):
        """Update the alloc instance for this parameter."""
        self.alloc(instance).update(value)

    def __delete__(self, instance):
        """Reset the alloc instance for this parameter."""
        self.alloc(instance).reset()

    def __str__(self):
        """Show the parameter."""
        return '{}: {!r}'.format(self.name, self.default)

    def __repr__(self):
        """Represent the parameter."""
        return 'Param({!r}, {!r})'.format(self.name, self.default)


class Alloc:
    """Instance of a parameter with a concrete value.

    Attributes
    ----------
    param : Param
        the parameter descriptor this instance is based on
    value
        the current value for this instance of the parameter

    Methods
    -------
    subscribe(callback):
        subscribe a `callback(name, old, new)` for changes on the value
    update(value) :
        assigns a new value to the parameter, notifying subscribers

    """

    __slots__ = ('param', 'value', 'subscribers')

    def __init__(self, param, value=None):
        """Create a new alloc referring to the given Param object."""
        self.param = param
        self.value = value or param.default
        self.subscribers = []

    @property
    def args(self):
        """Get arguments tuple that can be used for the `sample` function."""
        return self.param.args(self)

    @property
    def name(self):
        """Return the name of the parameter."""
        return self.param.name

    @property
    def default(self):
        """Return the default value of the parameter."""
        return self.param.default

    def update(self, value):
        """Update the current value notifying subscribers about the change."""
        # TODO validation of values
        prev, self.value = self.value, value

        if prev != value:
            for s in self.subscribers:
                # TODO logging
                s(self.name, prev, value)

    def reset(self):
        """Reset the value to the default one of the parameter."""
        self.update(self.param.default)

    def subscribe(self, callback):
        """Add a `callback(name, old, new)` to be notified about changes."""
        if not isinstance(callback, FunctionType):
            msg = "Callback object should be callable, not a {}."
            raise TypeError(msg.format(type(callback).__name__))

        try:
            signature(callback).bind('', None, None)
        except TypeError as e:
            msg = "Callback should accept 3 positional args (name, old, new)."
            raise TypeError(msg) from e

        self.subscribers.append(callback)

    def unsubscribe(self, callback):
        """Remove a `callback(name, old, new)` from being notified."""
        try:
            idx = self.subscribers.index(callback)
            return self.subscribers.pop(idx)
        except ValueError:
            return None

    def __str__(self):
        """Show the allocation."""
        return '{}: {!r}'.format(self.name, self.value)

    def __repr__(self):
        """Represent the parameter."""
        return 'Alloc({!r}, {!r})'.format(self.name, self.value)


class Allocation:
    """An allocation holds parameters values for a simulation class.

    You can combine two partial allocations with the `+` operator.
    Moreover you can use a `with alloc(name=val, ...):`-context to
    temporarily overwrite values inside the allocation.
    """

    def __init__(self, **values):
        """Create a new instance of this allocation with supplied values."""
        for name, value in values.items():
            setattr(self, name, value)

    @contextlib.contextmanager
    def __call__(self, **kws):
        """Temporarily set some parameter values inside a context."""
        vals = {}
        try:
            for name, value in kws.items():
                vals[name] = getattr(self, name).value
                setattr(self, name, value)
            yield self
        finally:
            for name, value in vals.items():
                setattr(self, name, value)

    def __add__(self, other):
        """Combine two allocation object to form a new one.

        Notes
        -----
        Note that changes to the combined allocation will also be applied to
        the original allocation.

        Returns
        -------
        combined
            A combined allocation with parameters of both allocations.

        Raises
        ------
        TypeError
            If one parameter is defined in multiple allocations.

        """
        if isinstance(other, Allocation):
            return CombinedAllocation(self, other)
        else:
            return NotImplemented

    def __contains__(self, name):
        """Check if a parameter with the given name is defined."""
        for n, a in self.items():
            if name == n or name == a.name:
                return True
        else:
            return False

    def __getitem__(self, name):
        """Get the parameter with the given name."""
        for n, a in self.items():
            if name == n or name == a.name:
                return a
        else:
            msg = "Unknown parameter {!r} for {}."
            raise IndexError(msg.format(name, self))

    def keys(self):
        """Iterate through the attribute names of all parameters."""
        for name, _ in self.items():
            yield name

    def values(self):
        """Iterate through all the allocated values of this allocation."""
        for _, value in self.items():
            yield value

    def items(self):
        """Iterate though all the name/value parameter pairs."""
        for name in dir(type(self)):
            desc = getattr(type(self), name)
            if isinstance(desc, Param):
                yield name, desc.alloc(self)

    def __str__(self):
        """Output a short string of the allocation."""
        return "{}({})".format(type(self).__name__,
                               ",".join(["{}={}".format(attr, alloc)
                                         for attr, alloc in self.items()
                                         if alloc.value != alloc.default]))

    def __repr__(self):
        """Output a rich representation of the  allocation."""
        result = "{}():".format(type(self).__name__)
        for name, alloc in self.items():
            result += "\n- {}: {}={}".format(alloc.name, name, alloc)
        return result


class CombinedAllocation(Allocation):
    """A combination of other allocation objects."""

    def __init__(self, *bases):
        super().__init__()

        conflicts = set()
        keys = set()
        for b in bases:
            conflicts.update(keys.intersection(b.keys()))
            keys.update(b.keys())

        if conflicts:
            msg = "Conflict as {} found in multiple base allocations."
            raise TypeError(msg.format(','.join(conflicts)))

        self._bases = bases

    def __iter__(self):
        return iter(self._bases)

    def items(self):
        """Iterate though all keys of all base allocations."""
        return chain(*[a.items() for a in self._bases])

    def __getattr__(self, name):
        if not name.startswith('_'):
            for alloc in self._bases:
                if hasattr(alloc, name):
                    return getattr(alloc, name)

        msg = '{!r} object has no attribute {!r}'
        raise AttributeError(msg.format(type(self).__name__, name))

    def __setattr__(self, name, value):
        if not name.startswith('_'):
            for alloc in self._bases:
                if hasattr(alloc, name):
                    return setattr(alloc, name, value)

        super().__setattr__(name, value)

    def __delattr__(self, name):
        for alloc in self._bases:
            if hasattr(alloc, name):
                return delattr(alloc, name)

        super().__delattr__(name)

    def __dir__(self):
        return (list(super().__dir__())
                + [d for alloc in self._bases for d in dir(alloc)])

    def __str__(self):
        return " + ".join(str(b) for b in self._bases)

    def __repr__(self):
        result = ""
        for alloc in self._bases:
            result += "\n" + repr(alloc)
        return result.strip("\n")


class Distribution(Param):
    """Specialized subclass of parameters for random distributions."""

    def sample(self):
        """Return a function that will create samples of the distribution.

        Returns
        -------
        impl(...)
            function taking parameters values as args returning a sample of
            the distribution

        """
        raise NotImplementedError

    def args(self, val=None):
        """Get arguments tuple that can be used for the `sample` function."""
        val = val.value if val is not None else self.default
        if isinstance(val, dict):
            return tuple(val.values())
        elif isinstance(val, (tuple, list)):
            return tuple(val)
        else:
            return val,


class Uniform(Distribution):
    """Parameters for a uniform distribution."""

    def __init__(self, name, low, high, help=None, options=None):
        """Create a new uniform distribution.

        Parameters
        ----------
        low : int
            lowest number for uniform distribution
        high : int
            highest number for uniform distribution
        help : str (optional)
            long description for this parameter
        options : tuple (optional)
            bounds for low and high
        """
        super().__init__(name, (low, high), help=help, options=options)

    def sample(self):
        """Get sampling method for the uniform distribution."""
        def impl(low, high):
            return np.random.randint(low, high)
        return impl


class Bernoulli(Distribution):
    """Parameters for a bernoulli distribution."""

    def __init__(self, name, p, **kws):
        """Create a new bernoulli distribution.

        Parameters
        ----------
        p : float
            probability of success for single trials
        low : int
            lowest number for uniform distribution
        high : int
            highst number for uniform distribution
        help : str (optional)
            long description for this parameter
        options : tuple (optional)
            bounds for low and high
        """
        super().__init__(name, p, **kws)

    def sample(self):
        """Get sampling method for the bernoulli distribution."""
        def impl(p):
            return np.random.binomial(1, p) > .5
        return impl


class Continuous(Distribution):
    """Parameters for a uniform continues distribution."""

    def __init__(self, name, low, high, help=None, options=None):
        """Create a new continuous distribution.

        Parameters
        ----------
        low : float
            lowest number for uniform distribution
        high : float
            highst number for uniform distribution
        help : str (optional)
            long description for this parameter
        options : tuple (optional)
            bounds for low and high
        """
        super().__init__(name, (low, high), help=help, options=options)

    def sample(self):
        """Get sampling method for the continuous distribution."""
        def impl(low, high):
            return np.random.uniform(low, high)
        return impl


class Normal(Distribution):
    """Parameters for a normal distribution."""

    def __init__(self, name, loc, scale, help=None, options=None):
        """Create a new continuious distribution.

        Parameters
        ----------
        loc : float
           mean of the normal distribution
        scale : float
           deviation of the normal distribution
        help : str (optional)
            long description for this parameter
        options : tuple (optional)
            bounds for loc and scale of the normal distribution
        """
        super().__init__(name, {'loc': loc, 'scale': scale},
                         help=help, options=options)

    def sample(self):
        """Get sampling method for the normal distribution."""
        def impl(loc, scale):
            return np.random.normal(loc, scale)
        return impl


class Exponential(Distribution):
    """Parameters for a exponential distribution."""

    def __init__(self, name, scale, **kws):
        """Create a new normal distribution.

        Parameters
        ----------
        scale : float
            scale of the exponential distribution
        help : str (optional)
            long description for this parameter
        options : tuple (optional)
            bounds for scale of the distribution
        """
        super().__init__(name, scale, **kws)

    def sample(self):
        """Get sampling method for the exponential distribution."""
        def impl(scale):
            return np.random.exponential(scale)
        return impl
