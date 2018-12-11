"""
An allocation takes care of binding different vairables to concrete values.

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

__all__ = ['Allocation', 'Param',
           'Uniform', 'Bernoulli',
           'Continuous', 'Normal', 'Exponential']


class Alloc:
    """Instance of a parameter with a concreate value.

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
        assigns a new value to the parameter, notifying subcribers

    """

    __slots__ = ('param', 'value', 'subscribers')

    def __init__(self, param, value=None):
        """Create a new alloc refering to the given Param object."""
        self.param = param
        self.value = value or param.default
        self.subscribers = []

    @property
    def name(self):
        """Return the name of the parameter."""
        return self.param.name

    def update(self, value):
        """Update the current value notifying subscribers about the change."""
        # TODO validation of values
        prev, self.value = self.value, value

        for s in self.subscribers:
            # TODO logging
            s(self.name, prev, value)

    def reset(self):
        self.update(self.param.default)

    def subscribe(self, callback):
        """Add a `callback(name, old, new)` to be notified about changes."""
        self.subscribers.append(callback)

    def unsubscribe(self, callback):
        """Remove a `callback(name, old, new)` from being notified."""
        self.subscribers = [s for s in self.subscribes if s != callback]

    def __str__(self):
        """Show the allocation."""
        return '{}: {!r}'.format(self.name, self.value)

    def __repr__(self):
        """Represent the parameter."""
        return 'Alloc({!r}, {!r})'.format(self.name, self.value)


class Param:
    """The Param descriptor defines infomation about a single parameter.

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
        - a list of avialable options
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
            - a list of avialable options
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
                next(iter(self.default.allocs())), list)):
            return sum(len(vs) for vs in self.default.allocs())

        if isinstance(self.default, (list, dict, tuple)):
            return len(self.default)

        return 1

    def alloc(self, object):
        """Return the `Alloc`-instance of this Param for the given object."""
        return self.allocs.setdefault(Alloc(self, object))

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


class Distribution(Param):
    """Specialized subclass of parameters for random distributions."""

    def sample(self):
        """Return a function that will create samples of the distribution.

        Returns
        -------
        impl(...)
            function taking parameters values as args returing a sample of
            the distribution

        """
        raise NotImplementedError


class Uniform(Distribution):
    """Parameters for a uniform distribution."""

    def __init__(self, name, low, high, help=None, options=None):
        """Cerate a new uniform distribution.

        Parameters
        ----------
        low : int
            lowest number for uniform distribution
        high : int
            heighst number for uniform distribution
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
        """Cerate a new bernoulli distribution.

        Parameters
        ----------
        p : float
            probability of success for single triels
        low : int
            lowest number for uniform distribution
        high : int
            heighst number for uniform distribution
        help : str (optional)
            long description for this parameter
        options : tuple (optional)
            bounds for low and high
        """
        super().__init__(name, p, **kws)

    def sample(self):
        """Get sampling method for the bernoulli distribution."""
        def impl(p):
            return np.random.binomial(1, p)
        return impl


class Continuous(Distribution):
    """Parameters for a uniform continues distribution."""

    def __init__(self, name, low, high, help=None, options=None):
        """Cerate a new continious distribution.

        Parameters
        ----------
        low : float
            lowest number for uniform distribution
        high : float
            heighst number for uniform distribution
        help : str (optional)
            long description for this parameter
        options : tuple (optional)
            bounds for low and high
        """
        super().__init__(name, (low, high), help=help, options=options)

    def sample(self):
        """Get sampling method for the continous distribution."""
        def impl(low, high):
            return np.random.uniform(low, high)
        return impl


class Normal(Distribution):
    """Parameters for a normal distribution."""

    def __init__(self, name, loc, scale, help=None, options=None):
        """Cerate a new continious distribution.

        Parameters
        ----------
        loc : float
           mean of the normal distribution
        sclae : float
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
    """Parameters for a exponentail distribution."""

    def __init__(self, name, scale, **kws):
        """Cerate a new normal distribution.

        Parameters
        ----------
        sclae : float
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


class Allocation:
    """An allocation holds values for parameters and variables of a simulation.

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
        if isinstance(other, Allocation):
            return CombinedAllocation(getattr(self, '_allocs', (self,))
                                      + getattr(other, '_allocs', (other,)))
        else:
            return NotImplemented

    def keys(self):
        for name in dir(type(self)):
            desc = getattr(type(self), name)
            if isinstance(desc, Param):
                yield name

    def values(self):
        for name in self.keys():
            yield getattr(self, name)

    def items(self):
        for name in self.keys():
            yield (name, getattr(self, name))

    def __repr__(self):
        result = "{}():".format(type(self).__name__)
        for name, value in self.items():
            result += "\n- {}: {}={}".format(value.param.text, name, value)
        return result


class CombinedAllocation(Allocation):
    """
    A combination of other allocations
    """
    def __init__(self, allocs):
        self._allocs = allocs

    def keys(self):
        for a in self._allocs:
            yield from a.keys()

        return chain(a.keys() for a in self._allocs)
    def __getattr__(self, name):
        for alloc in self._allocs:
            if hasattr(alloc, name):
                return getattr(alloc, name)

        raise AttributeError('{!r} object has no attribute {!r}'.format(type(self).__name__, name))

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)

        for alloc in self._allocs:
            if hasattr(alloc, name):
                return setattr(alloc, name, value)

        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name.startswith('_'):
            super().__delattr__(name)

        for alloc in self._allocs:
            if hasattr(alloc, name):
                return delattr(alloc, name)

        super().__delattr__(name)

    def __dir__(self):
        return (super().__dir__()
                + [d for alloc in self._allocs for d in dir(alloc)])

    def __repr__(self):
        result = ""
        for alloc in self._allocs:
            result += "\n" + repr(alloc)
        return result.strip("\n")
