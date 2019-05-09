import simcompyl as sim

from simcompyl.core.model import Step
from types import FunctionType

import pytest


def test_basics():
    mdl = sim.Model()

    assert isinstance(mdl.init, Step)
    assert isinstance(mdl.iterate, Step)
    assert isinstance(mdl.apply, Step)

    assert isinstance(mdl.init.impl, FunctionType)
    assert isinstance(mdl.iterate.impl, FunctionType)
    assert isinstance(mdl.apply.impl, FunctionType)

    mdl.init.impl(None, None)
    mdl.iterate.impl(None, None)
    mdl.apply.impl(None, None)


def test_specs():
    mdl = sim.Model()

    assert dict(mdl.params) == {'n_steps': int, 'n_samples': int}
    assert set(mdl.steps) == {'init', 'iterate', 'apply'}

    assert len(mdl.params(foo={'a': [int], 'b': [bool]},
                          baz=float)) == 2
    assert dict(mdl.params) == {'n_steps': int,
                                'n_samples': int,
                                'foo': {'a': [int], 'b': [bool]},
                                'baz': float}

    assert len(mdl.state(foo=bool,
                         baz=float)) == 2
    assert dict(mdl.state) == {'foo': bool,
                               'baz': float}

    assert len(mdl.random(foo=bool, baz=float)) == 2
    assert dict(mdl.random) == {'foo': bool, 'baz': float}

    def foo(bar, baz):
        return bar['s'] + baz

    mdl.derive(bar={'s': float, 't': float}, baz=float)(foo)

    assert dict(mdl.derives) == {'foo(bar,baz)': (
        foo, {'bar': {'s': float, 't': float}, 'baz': float})}

    with pytest.raises(TypeError):
        mdl.params(bar=bool)

    def deriving():
        def foo(ps):
            return ps

        mdl.derive(bar={'s': float, 't': float})(foo)
        mdl.derive(baz=float)(foo)

        return foo

    foo1 = deriving()

    assert dict(mdl.derives) == {
        'foo(bar,baz)': (foo, {'bar': {'s': float, 't': float}, 'baz': float}),
        'foo(bar)': (foo1, {'bar': {'s': float, 't': float}}),
        'foo(baz)': (foo1, {'baz': float})}

    foo2 = deriving()
    assert dict(mdl.derives) == {
        'foo(bar,baz)': (foo, {'bar': {'s': float, 't': float}, 'baz': float}),
        'foo(bar)': (foo2, {'bar': {'s': float, 't': float}}),
        'foo(baz)': (foo2, {'baz': float})}

    with pytest.raises(TypeError):
        def foo(ps):
            return ps

        mdl.derive(bar={'s': float, 't': float})(foo)

    assert dict(mdl.params) == {'n_steps': int,
                                'n_samples': int,
                                'foo': {'a': [int], 'b': [bool]},
                                'baz': float}

    traces = []

    def trace(what):
        def resolve(name, spec):
            traces.append((what, name, spec))
            return spec
        return resolve

    with mdl.resolving(steps=trace('steps'),
                       state=trace('state'),
                       params=trace('params'),
                       random=trace('random'),
                       derives=trace('derived')):
        mdl.steps(init=...)
        mdl.state(foo=...)
        mdl.params(foo=...)
        mdl.random(foo=...)

    assert traces == [('steps', 'init', mdl.steps(init=...)),
                      ('state', 'foo', bool),
                      ('params', 'foo', {'a': [int], 'b': [bool]}),
                      ('random', 'foo', bool)]


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
