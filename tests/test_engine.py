import simcompyl as sim

from collections import Counter
import pytest


class Foobar(sim.Model):
    def iterate(self):
        rnd = self.random(a=bool, b=int, c=float)

        foo, bar = self.state(foo=float,
                              bar=[bool] * 8)

        baz = self.params(baz={'a': [float] * 8,
                               'b': [float] * 8,
                               'c': [bool] * 8})

        @self.derive(bazz=int)
        def dprimitiv(bazz):
            return bazz

        @self.derive(bazz=int)
        def dtuple(bazz):
            return bazz, -bazz

        @self.derive(baz={'c': [bool] * 8,
                          'd': [int] * 8},
                     bazz=int)
        def darray(baz, bazz):
            return [d if c else bazz for c, d in zip(baz['c'], baz['d'])]

        @self.derive(baz={'a': [float] * 8,
                          'b': [float] * 8})
        def dmulti(baz):
            return [[a * b for b in baz['b']] for a in baz['a']]


        def impl(params, state):
            pass

        return impl


class FooAlloc:
    a = sim.Bernoulli("a", .2)
    b = sim.Uniform("b", 0, 10)
    c = sim.Normal("c", 2, 1)
    
    baz = sim.Param("baz", {'a': [.0, .1, .5, .9],
                            'b': [.1, .5, .9,  1],
                            'c': [False, True, True, False],
                            'd': [0, 1, 2, -1]})
    bazz = sim.Param("bazz", -4)

    n_steps = sim.Param("steps", 4)
    n_samples = sim.Param("samples", 100)


def test_resolving(engine):
    model = Foobar()
    alloc = FooAlloc()

    exec = engine(model, alloc)

    params = exec.params()
    state = exec.state()

    foo = exec.resolve_state("foo", float)
    bar = exec.resolve_state("bar", [int]*8)

    assert state[0, foo] == 0
    assert list(state[0, bar]) == [0] * 8


    baz = exec.resolve_params("baz", {'a': ..., 'b': ..., 'c': ...})

    assert list(baz.a(params)) == list(alloc.baz.value['a'])
    assert list(baz.b(params)) == list(alloc.baz.value['b'])
    assert list(baz.c(params)) == list(alloc.baz.value['c'])

    a = exec.resolve_random("a", bool)
    cnt = Counter()
    for i in range(100):
        s = a(params)
        assert s in {True, False}

        cnt[s] += 1
        if len(cnt) == 2:
            break
    assert cnt[True] >= 1
    assert cnt[False] >= 1

    b = exec.resolve_random("b", int)
    cnt = Counter()
    for i in range(100):
        s = b(params)
        assert s >= 0
        assert s < 10

        cnt[s] += 1
        if len(cnt) == 4:
            break
    assert len(cnt) == 4

    c = exec.resolve_random("c", float)
    cnt = Counter()
    for i in range(100):
        s = c(params)
        cnt[1 <= s <= 3] += 1

    assert len(cnt) == 2
    assert cnt[True] >= cnt[False]

    dprimitiv = exec.resolve_derives("dprimitiv(bazz)",
                                   model.derives['dprimitiv(bazz)'])
    assert dprimitiv(params) == -4

    dtuple = exec.resolve_derives("dtuple(bazz)",
                                     model.derives['dtuple(bazz)'])
    assert tuple(dtuple(params)) == (-4, 4)

    darray = exec.resolve_derives("darray(baz,bazz)",
                                  model.derives['darray(baz,bazz)'])
    assert list(darray(params)) == [-4, 1, 2, -4]

    dmulti = exec.resolve_derives("dmulti(baz)",
                                  model.derives['dmulti(baz)'])
    assert ([list(m) for m in dmulti(params)] ==
            [[a*b for b in alloc.baz.value['b']]
             for a in alloc.baz.value['a']])


def test_misc():
    model = sim.Model()
    alloc = FooAlloc()

    exec = sim.Execution(model, alloc)

    def foo():
        """noop."""

    with pytest.raises(AttributeError):
        exec.compile(foo)


def test_numba():
    model = sim.Model()
    alloc = FooAlloc()

    exec = sim.Execution(model, alloc)
    exec.run()
