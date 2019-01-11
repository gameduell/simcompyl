"""Simple utility functions."""
import functools
import weakref
from contextlib import contextmanager

__all__ = ['lazy']


class LazyDescriptor:
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
    def __init__(self, *args, **kws):
        self.args = args
        self.kws = kws

    def __iter__(self):
        return iter(self.args + tuple(self.kws.values()))


class Resolvable:
    """Mixin with resolving behaviour.

    This objects lets will bind a `resolver` inside a `resolving` context,
    that can be later used to `resolve` things.

    If no `resolver` is bound, it will return the result of the `unresolved`
    method, normaly an instance of the `Unresolved` class.
    """

    def resolve(self, *args, **kws):
        if hasattr(self, '_resolver'):
            return self._resolver(*args, **kws)
        else:
            return self.unresolved(*args, **kws)

    def unresolved(self, *args, **kws):
        return Unresolved(*args, **kws)

    @contextmanager
    def resolving(self, resolver):
        self._resolver = resolver
        yield
        del self._resolver
