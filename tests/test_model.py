import simulate as sim

from simulate.model import Step
from types import FunctionType


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
