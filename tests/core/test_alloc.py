import simcompyl as sim
from simcompyl.core.alloc import Param, Alloc

import pytest
from collections import Counter


def test_alloc():
    class Foo(sim.Allocation):
        s = sim.Param("string", 'a string')
        n = sim.Param("number", 42, descr="A natural number", options=(1, None))

    class Bar(sim.Allocation):
        s = sim.Param("baz", 'other string', options=['a', 'b', 'c'])

    assert isinstance(Foo.s, Param)
    assert Bar.s.default == 'other string'

    assert 'string' in str(Bar.s)
    assert 'other string' in str(Bar.s)
    assert 'string' in repr(Bar.s)
    assert 'other string' in repr(Bar.s)

    foo = Foo(n=13)
    bar = Bar()

    assert isinstance(foo.s, Alloc)
    assert foo.n.name == "number"
    assert bar.s.param == Bar.s

    assert foo.s.value == 'a string'
    assert bar.s.value == 'other string'
    assert foo.n.value == 13

    foo.n = 8
    assert foo.n.value == 8
    assert foo.n.default == 42
    assert 'number' in str(foo.n)
    assert '8' in str(foo.n)
    assert 'number' in repr(foo.n)
    assert '8' in repr(foo.n)

    bar.s.update('foo')
    assert foo.s.value == 'a string'
    assert bar.s.value == 'foo'

    del foo.n
    assert foo.n.value == 42

    del bar.s
    assert bar.s.value == 'other string'

    with foo(n=1):
        assert foo.n.value == 1
        assert foo.s.value == "a string"
        with foo(n=2):
            assert foo.n.value == 2
        assert foo.n.value == 1

    assert foo.n.value == 42

    assert 'n' in foo
    assert 'number' in foo
    assert 'baz' not in foo

    assert dict(foo) == {'n': foo.n, 's': foo.s}
    assert set(foo.values()) == {foo.n, foo.s}

    assert bar['s'] == bar.s
    assert bar['baz'] == bar.s
    with pytest.raises(IndexError):
        foo['baz']

    bar.s = 'foo'
    assert 'Bar' in str(bar)
    assert 'baz' in str(bar)
    assert 'foo' in str(bar)

    assert 'Bar' in repr(bar)
    assert 'baz' in repr(bar)
    assert 'foo' in repr(bar)


def test_subscription():
    class Foo(sim.Allocation):
        n = sim.Param('count', 0)

    traces = []

    def trace(name, old, new):
        traces.append((name, old, new))

    foo = Foo()
    foo.n.subscribe(trace)
    assert traces == []

    foo.n.update(1)
    assert traces == [("count", 0, 1)]

    foo.n.update(3)
    assert traces == [("count", 0, 1), ("count", 1, 3)]

    foo.n.reset()
    assert traces == [("count", 0, 1), ("count", 1, 3), ("count", 3, 0)]

    foo.n.unsubscribe(trace)
    foo.n.update(1)
    assert traces == [("count", 0, 1), ("count", 1, 3), ("count", 3, 0)]

    foo.n.unsubscribe(trace)
    foo.n.update(1)
    assert traces == [("count", 0, 1), ("count", 1, 3), ("count", 3, 0)]

    with pytest.raises(TypeError):
        foo.n.subscribe('foo')

    with pytest.raises(TypeError):
        foo.n.subscribe(lambda x: x)

    with pytest.raises(TypeError):
        foo.n.subscribe(lambda a, b, c, d: (a, b, c, d))

    foo.n.subscribe(lambda a, b, c, d=2: (a, b, c, d))
    foo.n.subscribe(lambda a, *b: None)
    foo.n.reset()


def test_arity():
    class Foo(sim.Allocation):
        simple = sim.Param("simple", 42)
        short = sim.Param("short", {'foo': [1, 2],
                                    'bar': ['a', 'b']})
        long = sim.Param("long", {'foo': [0] * 100,
                                  'bar': ['a'] * 100})

        list = sim.Param("list", list(range(10)))

    assert Foo.simple.arity == 1
    assert Foo.short.arity == 4
    assert Foo.long.arity == 200
    assert Foo.list.arity == 10

    foo = Foo()

    foo.short = {'foo': [1, 2, 3],
                 'bar': ['a', 'b', 'c']}
    assert Foo.short.arity == 4

    foo.long = {'foo': [1] * 80,
                'bar': ['b'] * 80}
    assert Foo.short.arity == 4

    foo.list = [0] * 100
    assert Foo.list.arity == 10


def test_combined():
    class Foo(sim.Allocation):
        s = sim.Param("string", 'a string')
        n = sim.Param("number", 42, descr="A natural number", options=(1, None))

    class Bar(sim.Allocation):
        bar = sim.Param("bar", 'other string', options=['a', 'b', 'c'])

    class Baz(sim.Allocation):
        baz = sim.Param("baz", True)

    foo, bar = foobar = Foo() + Bar()

    assert isinstance(foobar, sim.Allocation)
    assert isinstance(foo, sim.Allocation)
    assert isinstance(bar, sim.Allocation)
    assert isinstance(foobar.bar, Alloc)
    assert foobar.s.name == "string"
    assert foobar.n.value == 42
    assert dict(foobar) == {'s': foobar.s,
                            'n': foobar.n,
                            'bar': foobar.bar}
    assert "s" in foobar
    assert "number" in foobar
    assert foobar["string"] == foobar.s
    assert 'bar' in dir(foobar)

    foobar.n = 1337
    assert foobar.n.value == 1337
    assert foo.n.value == 1337

    del foobar.n
    assert foobar.n.value == 42
    assert foo.n.value == 42

    with foobar(bar=''):
        assert foobar.bar.value == ''
        assert bar.bar.value == ''

    assert foobar.bar.value == 'other string'
    assert bar.bar.value == 'other string'

    foobar.bar.value = 'bla'
    assert 'bar' in str(foobar)
    assert 'bla' in str(foobar)
    assert 'Foo' in str(foobar)

    assert 'number' in repr(foobar)
    assert '42' in repr(foobar)
    assert 'Foo' in repr(foobar)

    bar.bar = '...'
    assert foobar.bar.value == '...'

    with pytest.raises(AttributeError):
        foobar.nil

    bar._attr = None
    with pytest.raises(AttributeError):
        foobar._attr

    foobar._other = None
    with pytest.raises(AttributeError):
        bar._other

    with pytest.raises(AttributeError):
        del foobar.nil

    with pytest.raises(TypeError):
        Foo() + Foo()

    with pytest.raises(TypeError):
        Foo() + object()

    foobaz = (foo, bar), baz = foobar + Baz()

    assert isinstance(foo, Foo)
    assert isinstance(bar, Bar)
    assert isinstance(baz, Baz)

    foobar.n = 23
    foobaz.bar = 'bar'
    foobaz.baz = False

    assert foo.n.value == 23
    assert foobar.n.value == 23
    assert foobaz.n.value == 23

    assert bar.bar.value == 'bar'
    assert foobar.bar.value == 'bar'
    assert foobaz.bar.value == 'bar'

    assert not baz.baz.value
    assert not foobaz.baz.value


def test_distributions():
    with pytest.raises(NotImplementedError):
        sim.Distribution("test", None).sample()

    n = 42

    def check(dist, bounds=None, typ=None, mode=None, args=None):
        dist = dist.__get__(dist)

        if args:
            assert dist.args == args

        sample = dist.param.sample()
        args = dist.args

        cnt = Counter()
        for _ in range(n):
            s = sample(*args)

            if isinstance(mode, int):
                cnt[s] += 1
            elif isinstance(mode, tuple):
                low, high = mode
                cnt[s >= low and s <= high] += 1
            else:
                assert mode is None

            if bounds:
                low, high = bounds
                if low is not None:
                    assert s >= low
                if high is not None:
                    assert s <= high

            if typ:
                assert isinstance(s, typ)

        if cnt:
            m = max(cnt.items(), key=lambda it: it[1])[0]
            if isinstance(mode, int):
                assert m == mode
            elif isinstance(mode, tuple):
                assert m

    uni = sim.Uniform("uniform", 3, 6)
    check(uni, bounds=(3, 6), typ=int, args=(3, 6))

    ber = sim.Bernoulli("bernoulli", .1)
    check(ber, bounds=(False, True), typ=bool, mode=False)

    con = sim.Continuous("continuous", 3, 6)
    check(con, bounds=(3, 6), typ=float, args=(3, 6))

    nor = sim.Normal("normal", 4., 2)
    check(nor, typ=float, mode=(1, 7), args=(4., 2))

    exp = sim.Exponential("exponential", 1.5)
    check(exp, bounds=(0, None), typ=float, mode=(0, 2), args=(1.5,))
