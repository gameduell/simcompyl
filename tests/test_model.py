import simulate as sim

from simulate.core.model import Step
from types import FunctionType
import pytest


def test_basics():
    mdl = sim.Model()

    assert isinstance(mdl.init, Step)
    assert isinstance(mdl.iterate, Step)
    assert isinstance(mdl.apply, Step)
    assert isinstance(mdl.finish, Step)

    assert isinstance(mdl.init(), FunctionType)
    assert isinstance(mdl.iterate(), FunctionType)
    assert isinstance(mdl.apply(), FunctionType)
    assert isinstance(mdl.finish(), FunctionType)

    mdl.init()(None, None)
    mdl.iterate()(None, None)
    mdl.apply()(None, None)
    mdl.finish()(None, None)


def test_specs():
    mdl = sim.Model()

    assert dict(mdl.params) == {'n_steps': int, 'n_samples': int}
    assert set(mdl.steps) == {'init', 'iterate', 'apply', 'finish'}

    assert len(mdl.params(foo={'a': [int], 'b': [bool]},
                          baz=float)) == 2
    assert dict(mdl.params) == {'n_steps': int,
                                'n_samples': int,
                                'foo': {'a': [int], 'b': [bool]},
                                'baz': float}

    assert len(mdl.state(foo={'a': [int], 'b': [bool]},
                         baz=float)) == 2
    assert dict(mdl.state) == {'foo': {'a': [int], 'b': [bool]},
                               'baz': float}

    assert len(mdl.random(foo=bool, baz=float)) == 2
    assert dict(mdl.random) == {'foo': bool, 'baz': float}

    def foo(bar, baz):
        return bar['s'] + baz
    mdl.derive(bar={'s': float, 't': float}, baz=float)(foo)

    assert dict(mdl.derives) == {'foo(bar,baz)': (
        foo, {'bar': {'s': float, 't': float}, 'baz': float})}

    assert dict(mdl.params) == {'n_steps': int,
                                'n_samples': int,
                                'foo': {'a': [int], 'b': [bool]},
                                'bar': {'s': float, 't': float},
                                'baz': float}


def test_binds():
    class FooBar(sim.Allocation):
        foo = sim.Param("foo", 42)
        bar = sim.Param("bar", [True, True, False, True, False, False])

    mdl = sim.Model()
    foo, bar = mdl.params(foo=int,
                          bar=[bool] * 6)

    default = mdl.engine
    assert mdl.alloc is None
    assert default.alloc is None

    alloc = FooBar()
    assert mdl.bind(alloc) == mdl
    assert mdl.alloc == alloc

    assert default.alloc == alloc

    assert mdl.bind(engine=sim.engine.NumbaEngine(), compile=False) == mdl
    assert mdl.engine != default
    assert mdl.engine.alloc == alloc


def test_execute():
    class Test(sim.Model):
        @sim.step
        def iterate(self):
            x = self.state(x=int)
            y = self.state(y=int)

            def impl(params, state):
                state[x] += 1
                state[y] -= 1
            return impl

    mdl = Test()
    with pytest.raises(ValueError):
        mdl.execute()

    class FooBar(sim.Allocation):
        n_steps = sim.Param("Steps", 2)
        n_samples = sim.Param("Samples", 3)

    mdl.bind(FooBar())
    assert dict(mdl.params) == {'n_steps': int, 'n_samples': int}
    assert dict(mdl.state) == {'x': int, 'y': int}

    out = mdl.execute()
    assert len(out) == 3
    assert out.x[0] == 2
    assert out.y[0] == -2

    eng = sim.engine.NumbaEngine()
    mdl.execute(engine=eng)
    assert eng.alloc == mdl.alloc


def test_graph():
    class Test(sim.Model):
        @sim.step
        def iterate(self):
            _iter = super().iterate()
            _foo = self.foo()
            _bar = self.bar()

            def impl(params, state):
                _iter(params, state)
                _foo(params, state)
                _bar(params, state)
            return impl

        @sim.step
        def foo(self):
            x = self.state(x=int)
            y = self.state(y=int)

            def impl(params, state):
                state[x] += 1
                state[y] -= 1
            return impl

        @sim.step
        def bar(self):
            def impl(params, state):
                pass
            return impl

    graph = Test().graph(rankdir='TD')

    # TODO really test creation of flow-graph
    assert graph

    # TODO test complex model with calls only in super method ...
