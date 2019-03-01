import pytest

import simcompyl as sim
from simcompyl.core.model import Step, StepDescriptor, Specs, SpecsCollection


class Base:
    def __init__(self):
        self.__specs__ = SpecsCollection(steps=Specs(),
                                         other=Specs())
        self._steps, self.other = self.__specs__.values()


def test_base():
    class Foo(Base):
        @sim.step
        def foo(self):
            bar = self.bar()
            self.other(foo='Foo')

            def impl(trace):
                trace.append((Foo, self, 'foo'))
                bar(trace)

            self.Foo_foo_impl = impl
            return impl

        @sim.step
        def bar(self):
            def impl(trace):
                trace.append((Foo, self, 'bar'))

            self.Foo_bar_impl = impl
            return impl

    class Bar(Foo):
        @sim.step
        def bar(self):
            _bar = super().bar()
            self.other(bar='Bar')

            def impl(trace):
                trace.append((Bar, self, 'bar'))
                _bar(trace)

            self.Bar_bar_impl = impl
            return impl

    assert isinstance(Foo.foo, StepDescriptor)
    assert isinstance(Foo.bar, StepDescriptor)
    assert isinstance(Bar.foo, StepDescriptor)
    assert isinstance(Bar.bar, StepDescriptor)

    foo = Foo()
    assert isinstance(foo.foo, Step)
    assert isinstance(foo.bar, Step)
    assert foo.foo.__name__ == 'foo'
    assert foo.bar.__name__ == 'bar'

    impl = foo.foo()
    assert impl == foo.Foo_foo_impl
    assert foo.foo.impl == foo.Foo_foo_impl
    assert foo.foo.steps == {'bar': foo.Foo_bar_impl}

    assert foo.foo.other == {'foo': 'Foo'}
    assert dict(foo.other) == {'foo': 'Foo'}

    trace = []
    impl(trace)
    assert trace == [(Foo, foo, 'foo'),
                     (Foo, foo, 'bar')]

    bar = Bar()
    assert isinstance(foo.foo, Step)
    assert isinstance(foo.bar, Step)

    impl = bar.foo()
    assert impl == bar.Foo_foo_impl
    assert bar.foo.impl == bar.Foo_foo_impl
    assert bar.foo.steps == {'bar': bar.Bar_bar_impl}

    assert bar.foo.other == {'foo': 'Foo'}
    assert bar.bar.other == {'bar': 'Bar'}
    assert dict(bar.other) == {'foo': 'Foo', 'bar': 'Bar'}

    trace = []
    impl(trace)
    assert trace == [(Foo, bar, 'foo'),
                     (Bar, bar, 'bar'),
                     (Foo, bar, 'bar')]


def test_fails():
    class Foo(Base):
        @sim.step
        def foo(self):
            pass

    foo = Foo()

    with pytest.raises(TypeError):
        foo.foo()
