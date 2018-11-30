import numpy as np
import weakref
import contextlib

from itertools import chain

__all__ = ['Allocation', 'Param', 
           'Uniform', 'Bernoulli', 
           'Continuous', 'Normal', 'Exponential']

class Allocation:
    """
    An allocation holds concrete values for parameters and variables used to run steps of a simulation.
    """
    def __init__(self, **kws):
        for name, value in kws.items():
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
            return CombinedAllocation(getattr(self, '_allocs', (self,)) + getattr(other, '_allocs', (other,)))
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
            super().__delattr__(name, value)

        for alloc in self._allocs:
            if hasattr(alloc, name):
                return delattr(alloc, name)

        super().__delattr__(name, value)

    def __dir__(self):
        return super().__dir__() + [d for alloc in self._allocs for d in dir(alloc)]

    def __repr__(self):
        result = ""
        for alloc in self._allocs:
            result += "\n" + repr(alloc)
        return result.strip("\n")


class Param:
    def __init__(self, text, default, *, lower=None, upper=None, step=None, options=None):
        self.values = weakref.WeakKeyDictionary()
        self.text = text
        self.default = default
        self.lower = lower
        self.upper = upper
        self.step = step
        self.options = options
        
    @property
    def arity(self):
        if isinstance(self.default, dict) and isinstance(next(iter(self.default.values())), list):
            return sum(len(vs) for vs in self.default.values())
        
        if isinstance(self.default, (list, dict, tuple)):
            return len(self.default)
        
        return 1

    def value(self, instance):
        return self.values.setdefault(instance, Value(self, instance))

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        else:
            return self.value(instance)

    def __set__(self, instance, value):
        self.value(instance).update(value)

    def __delete__(self, instance):
        self.value(instance).reset()

    def __repr__(self):
        return 'Param({!r}, {!r})'.format(self.text, self.default)
    
    
class Value:
    def __init__(self, param, alloc):
        self.param = param
        self.alloc = alloc
        self.value = param.default

    @property
    def default(self):
        return self.param.default

    @property
    def arity(self):
        return self.param.arity

    def update(self, value):
        self.value = value

    def reset(self):
        self.value = self.default

    def __repr__(self):
        return '{} = {!r}'.format(self.param.text, self.value)
    
    
class Distribution(Param):
    pass


class Uniform(Distribution):
    def __init__(self, text, low, high, **kws):
        super().__init__(text, (low, high), **kws)
        
    def sample(self):
        def impl(low, high):
            return np.random.randint(low, high)
        return impl
    
    
class Bernoulli(Distribution):
    def __init__(self, text, p, **kws):
        super().__init__(text, p, **kws)
        
    def sample(self):
        def impl(p):
            return np.random.binomial(1, p)
        return impl


class Continuous(Distribution):
    def __init__(self, text, low, high, **kws):
        super().__init__(text, (low, high), **kws)
        
    def sample(self):
        def impl(low, high):
            return np.random.uniform(low, high)
        return impl
    
    
class Normal(Distribution):
    def __init__(self, text, loc, scale, **kws):
        super().__init__(text, {'loc': loc, 'scale': scale}, **kws)
        
    def sample(self):
        def impl(loc, scale):
            return np.random.normal(loc, scale)
        return impl
    
    
class Exponential(Distribution):
    def __init__(self, text, scale, **kws):
        super().__init__(text, scale, **kws)
        
    def sample(self):
        def impl(scale):
            return np.random.exponential(scale)
        return impl
    
