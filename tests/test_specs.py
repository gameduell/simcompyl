import pytest
from collections import Counter

from simulate.model import Specs


def test_activation():
    a = {}
    b = {}
    specs = Specs()

    with specs.activate(a):
        specs(a1=1)
        with specs.activate(b):
            specs(b1=1.)
        specs(a2=2)

    assert a == {'a1': 1, 'a2': 2}
    assert b == {'b1': 1.}

    assert dict(specs) == {'a1': 1, 'b1': 1., 'a2': 2}


def test_resolve():
    specs = Specs()

    cnt = Counter()

    def custom(name, spec):
        cnt['total'] += 1
        cnt[name] += 1
        return (cnt['total'], cnt[name], name, spec)

    a_ = specs(a=0)

    with specs.resolving(custom):
        b = specs(b=1)
        a = specs(a=...)
        c = specs(c=2)
        aa = specs(a=0)

    b_ = specs(b=1)

    assert a_ == 0
    assert b == (1, 1, 'b', 1)
    assert a == (2, 1, 'a', 0)
    assert c == (3, 1, 'c', 2)
    assert aa == (4, 2, 'a', 0)
    assert b_ == 1

    assert dict(cnt) == {'total': 4, 'a': 2, 'b': 1, 'c': 1}


def test_validation():
    specs = Specs()

    specs(i=int, s=str, fs=[float]*4)
    assert specs['i'] == int
    assert specs['s'] == str
    assert specs['fs'] == [float]*4
    specs(dct={'a': int, 'b': float})
    assert specs['dct'] == {'a': int, 'b': float}

    specs(i=..., s=..., fs=..., dct=...)
    assert specs['i'] == int
    assert specs['s'] == str
    assert specs['fs'] == [float]*4
    assert specs['dct'] == {'a': int, 'b': float}

    specs(dct={'a': ..., 'b': ..., 'c': str})
    assert specs.specs['dct'] == {'a': int, 'b': float, 'c': str}

    specs(i=int)
    specs(s=str, fs=[float]*4)
    specs(dct={'b': float, 'd': [bool]})
    assert specs.specs['dct'] == {'a': int, 'b': float, 'c': str, 'd': [bool]}

    specs(fs=[float]*2)
    assert specs['fs'] == [float]*4
    specs(fs=[float]*6)
    assert specs['fs'] == [float]*6

    with pytest.raises(TypeError):
        specs(b=...)
    assert 'b' not in specs.specs

    with pytest.raises(TypeError):
        specs(i=..., b=...)
    assert specs['i'] == int
    assert 'b' not in specs.specs

    specs(b=bool)

    with pytest.raises(TypeError):
        specs(dct={'e': ...})
    assert specs.specs['dct'] == {'a': int, 'b': float, 'c': str, 'd': [bool]}

    with pytest.raises(TypeError):
        specs(dct={'a': ..., 'e': ...})
    assert specs.specs['dct'] == {'a': int, 'b': float, 'c': str, 'd': [bool]}

    with pytest.raises(TypeError):
        specs(i=float)
    assert specs['i'] == int

    with pytest.raises(TypeError):
        specs(fs=[int])
    assert specs['fs'] == [float]*6

    with pytest.raises(TypeError):
        specs(dct={'a': ..., 'c': bool})
    assert specs.specs['dct'] == {'a': int, 'b': float, 'c': str, 'd': [bool]}

    assert set(specs.specs) == {'i', 's', 'fs', 'dct', 'b'}


def test_validate_methods():
    specs = Specs()

    class Foo:
        def foo():
            pass

        def bar():
            pass

    class Bar(Foo):
        def bar():
            pass

        def baz():
            pass

    class Baz:
        def baz():
            pass

    bar = Bar()
    bar_ = Bar()
    foo = super(Bar, bar)
    baz = Baz()

    specs(foo=foo.foo, bar=foo.bar)
    assert dict(specs) == {'foo': foo.foo, 'bar': foo.bar}

    specs(foo=...)
    assert dict(specs) == {'foo': foo.foo, 'bar': foo.bar}

    specs(bar=bar.bar)
    assert dict(specs) == {'foo': foo.foo, 'bar': bar.bar}

    specs(baz=bar.baz)
    assert dict(specs) == {'foo': foo.foo, 'bar': bar.bar, 'baz': bar.baz}

    with pytest.raises(TypeError):
        specs(foo=foo.bar)

    with pytest.raises(TypeError):
        specs(bar=bar_.bar)

    with pytest.raises(TypeError):
        specs(baz=baz.baz)

    specs(baz=...)

    assert dict(specs) == {'foo': foo.foo, 'bar': bar.bar, 'baz': bar.baz}
