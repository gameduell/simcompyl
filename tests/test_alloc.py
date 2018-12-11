import simulate as sim

from simulate.core.alloc import Param, Alloc


def test_alloc():
    class Foo(sim.Allocation):
        s = sim.Param("string", 'a string')
        n = sim.Param("number", 42, help="A natural number", options=(1, None))

    class Bar(sim.Allocation):
        s = sim.Param("c", 'other string', options=['a', 'b', 'c'])

    assert isinstance(Foo.s, Param)
    assert Bar.s.default == 'other string'

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

    bar.s.update('foo')
    assert foo.s.value == 'a string'
    assert bar.s.value == 'foo'
