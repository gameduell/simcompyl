import simulate as sim

from simulate.model import Step, Specs, StepDescriptor


def test_base():
    class Foo:
        def __init__(self):
            self.steps = Specs()
            self.state = Specs()
            self.params = Specs()
            self.random = Specs()
            self.derives = Specs()

        @sim.step
        def foo(self):
            bar = self.bar()

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

    trace = []
    impl(trace)
    assert trace == [(Foo, bar, 'foo'),
                     (Bar, bar, 'bar'),
                     (Foo, bar, 'bar')]

def test_fails():
    pass
