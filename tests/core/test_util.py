from simcompyl.core.util import lazy, LazyDescriptor, Resolvable, Unresolved
import pytest


def test_lazy():
    class Foo:
        def __init__(self):
            self.i = 42

        @lazy
        def foo(self):
            i = self.i
            self.i = 13
            return i

    assert isinstance(Foo.foo, LazyDescriptor)

    foo = Foo()
    assert foo.i == 42
    assert foo.foo == 42
    assert foo.i == 13
    assert foo.foo == 42

    bar = Foo()
    assert bar.i == 42
    assert bar.foo == 42
    assert bar.i == 13
    assert bar.foo == 42

    baz = Foo()
    baz.foo = 23
    assert baz.i == 42
    assert baz.foo == 23
    assert baz.i == 42
    assert baz.foo == 23

    with pytest.raises(AttributeError):
        foo.foo = 0

    with pytest.raises(AttributeError):
        bar.foo = 0

    with pytest.raises(AttributeError):
        baz.foo = 0

    del baz.foo
    assert baz.foo == 42
    assert baz.i == 13
    assert baz.foo == 42

    del baz.foo
    assert baz.foo == 13


def test_resolvable():
    assert Unresolved(1, b=2) == Unresolved(1, b=2)
    assert Unresolved(1, b=2) != (1, 2)
    assert hash(Unresolved(1, b=2)) == hash(Unresolved(1, b=2))
    assert tuple(Unresolved(1, 2, c=3, b=4)) == (1, 2, 3, 4)

    res = Resolvable()
    assert res.resolve(1, 2, a=3) == Unresolved(1, 2, a=3)

    def foo(c, b, a):
        return a, b, c

    with res.binding(foo):
        assert res.resolve(1, a=3, b=2) == (3, 2, 1)
    assert res.resolve(1, 2, a=3) == Unresolved(1, 2, a=3)


