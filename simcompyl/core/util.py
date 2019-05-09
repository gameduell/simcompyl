"""Simple utility functions."""
import functools
import weakref
from contextlib import contextmanager
import logging

LOG = logging.getLogger(__name__)

__all__ = ['lazy', 'Resolvable']


class LazyDescriptor:
    """Descriptor for @lazy attributes."""
    def __init__(self, method):
        self.method = method
        self.refs = weakref.WeakKeyDictionary()

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if instance not in self.refs:
            self.refs[instance] = self.method(instance)
        return self.refs[instance]

    def __set__(self, instance, value):
        if instance in self.refs:
            raise AttributeError("Can't update a lazy value once it is set.")
        self.refs[instance] = value

    def __delete__(self, instance):
        self.refs.pop(instance, None)


def lazy(method):
    """
    Define a lazy evaluated attribute.

    The annotated method gets evaluated once when the attribute is accessed
    for the first time, remembering the returned value.
    If the attribute was not used yet, a value could be assigned once and
    that value will be returned on further accesses.

    If you `del`ite the attribute, it will go into it's original state, either
    evaluating the method once more or being free to be assigned a new value.
    """
    return functools.wraps(method)(LazyDescriptor(method))


class Unresolved:
    """Unresolved object that can be used for lazy specs."""
    def __init__(self, *args, **kws):
        self.args = args
        self.kws = kws

    def __eq__(self, other):
        if isinstance(other, Unresolved):
            return self.args == other.args and self.kws == other.kws
        else:
            return False

    def __hash__(self):
        return hash(self.args) ^ hash(frozenset(sorted(self.kws.items())))

    def __iter__(self):
        return iter(self.args + tuple(self.kws.values()))


class Resolvable:
    """Mixin with resolving behaviour.

    This objects lets you bind a `resolver` inside a `resolving` context,
    that can be later used to `resolve` things.

    If no `resolver` is bound, it will return the result of the `unresolved`
    method, normally an instance of the `Unresolved` class.
    """
    _resolver = None

    def resolve(self, *args, **kws):
        """Use the current resolver to resolve or call to unresolved."""
        if self._resolver is None:
            return self.unresolved(*args, **kws)
        else:
            return self._resolver(*args, **kws)

    def unresolved(self, *args, **kws):
        """Returns an unresolved object."""
        return Unresolved(*args, **kws)

    @contextmanager
    def resolving(self, resolver):
        """Context in which a resolver is bound."""
        LOG.debug(f"resolving for {self} with {resolver}", self, resolver)
        self._resolver = resolver
        try:
            yield
        finally:
            del self._resolver
