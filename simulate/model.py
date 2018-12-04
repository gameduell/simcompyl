"""Base for defining efficient simulation models."""

from weakref import WeakKeyDictionary
from collections import namedtuple
from contextlib import contextmanager, ExitStack
from functools import wraps
from types import MethodType

from .engine import NumbaEngine

__all__ = ['Model', 'step']


class Specs:
    """
    Central point for specifing and accessing specific aspects of a simulation.

    Specs are callabes accepting keywords with a specification that will be
    validated against other uses of the same name.

    Internally, this object is also used to glue things up:

    First of all, to give a more detailed report about the usage of the
    simulation aspects, the specs will not only recorded inside the specs own
    dictionary, but as well inside an `active` dictionary, that will be
    `activate`d in a context of the models `@step`.

    Morover, the call will return accessors according to the spec that can be
    used during the execution of the simulation. This is achived by letting the
    execution engine register itself inside the `resolving` context and giving
    contorl to the engine to create the accessor.
    """

    def __init__(self):
        """Create a new specification asspect of the simulation."""
        self.specs = {}
        self._resolve = None
        self._activated = [{}]

    def __contains__(self, name):
        return name in self.specs

    def __iter__(self):
        return iter(self.specs)

    def __getitem__(self, name):
        return self.specs[name]

    def keys(self):
        return self.specs.keys()

    def values(self):
        return self.specs.values()

    def items(self):
        return self.specs.items()

    def resolve(self, name, spec):
        """
        Resolve the specification returning a default accessor.

        Note that this method will be replacied inside `resolving` contexts
        by the engine.

        @param name: str
            name of the spec to resolve
        @param spec: type ...
            value of the specification, should be a type or something simular
        @return
            returns an accessor object that can be used inside the simulation
        """
        return spec

    @contextmanager
    def resolving(self, resolve):
        """
        Activate a context where accessors are resolved with the given method.

        @param resolve: function
            function, that given a name and as spec will create an accessor
        """
        self.resolve = resolve
        try:
            yield
        finally:
            after = self.resolve
            del self.resolve
            assert after == resolve

    @contextmanager
    def activate(self, specs):
        """
        Activate a dict to record all the specs while the context is active.

        @param specs: dict
            dict to register the specs inside
        """
        self._activated.append(specs)
        try:
            yield
        finally:
            after = self._activated.pop()
            assert after == specs

    @property
    def active(self):
        """Return the current active spec-dict."""
        return self._activated[-1]

    def validate(self, name, spec, prev):
        if spec is ...:
            if prev is None:
                msg = "Invalid specs elipsis for {!r}, no previous definition."
                raise TypeError(msg.format(name))
            else:
                return prev

        if isinstance(spec, dict) and (prev is None or isinstance(prev, dict)):
            result = {}
            for key, item in spec.items():
                result[key] = self.validate(key, item, (prev or {}).get(key))
            return result

        if isinstance(spec, list):
            if not spec:
                msg = "List spec should not be empty for {!r}."
                raise TypeError(msg.format(name))

            fst = spec[0]
            if not all(fst == s for s in spec):
                msg = "List spec should only contain single type for {!r}."
                raise TypeError(msg.format(name))

            prv = prev[0] if prev else None
            size = max(len(spec), len(prev or []))
            return [self.validate(None, fst, prv)] * size

        if all(hasattr(prev, a) for a in ['__self__',
                                          '__call__',
                                          '__name__']):
            if spec.__name__ != prev.__name__:
                msg = "Method spec {} with different methods {} previously {}."
                raise TypeError(msg.format(name, spec.__name__, prev.__name__))

            if spec.__self__ != prev.__self__:
                msg = "Method spec {} on different objects {} previously {}."
                raise TypeError(msg.format(name, spec.__self__, prev.__self__))

            return spec

        if prev is not None and spec != prev:
            msg = "Conflicting type definition of {} with {}, previously {}."
            raise TypeError(msg.format(name, spec, prev))

        return spec

    def __call__(self, return_namedtuple=None, **specs):
        """
        Register and validate the given names with the supplied specification.

        @param return_namedtuple: boolean
            weather to return a namedtuple even when only one spec is given
        @param specs: dict
            name, spec pairs, with a ...-spec using the previous value
        @return
            acccessor(s) to the spec(s)
        """
        results = {}
        for name, spec in specs.items():
            prev = self.specs.get(name)
            result = results[name] = self.validate(name, spec, prev)

            if isinstance(result, dict):
                self.specs.setdefault(name, {}).update(result)
                self.active.setdefault(name, {}).update(result)
            else:
                self.specs[name] = result
                self.active[name] = result

        if return_namedtuple or len(results) > 1:
            Access = namedtuple('Access', list(results))
            return Access(**{name: self.resolve(name, spec)
                             for name, spec in results.items()})
        else:
            (name, spec), = results.items()
            return self.resolve(name, spec)


def step(method):
    """
    Annotate a method used as a step inside a simulation model.

    @param method: method
        annotated method should return the actual implementation of the step

    The method should first bind local variables to accesors returned by
    the models `state`, `params` or `random` specs or by calls to other or
    super steps of the models. It then can define and return the state
    transforming implementation using these local variables.
    This implementation should accept a parameter object and the state as
    its first arguments, but also can have custom arguments or return types.
    The parameter object should be past on to the params and random accessors.

    >>> @step
    ... def somestep(self):
    ...     _somestep = super().somestep()
    ...     _otherstep = self.otherstep()
    ...     v = self.state(v=int)
    ...     a, b = self.params(a=int, b=int)
    ...     r = self.random(r=bool)
    ...
    ...     def impl(params, state, arg):
    ...          arg = 0
    ...
    ...          while state[v] >= a and state[v] < b
    ...              if b(param):
    ...                  arg = _otherstep(params, state)
    ...              _somestep(params, state, arg)
    ...
    ...     return impl

    This way, the engine can create a optimized version of the implementation.
    """
    return wraps(method)(StepDescriptor(method))


class StepDescriptor:
    """The descriptor returned by the step annotation, creating `Step`s."""

    def __init__(self, method):
        self.method = method
        self.refs = WeakKeyDictionary()

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        elif instance in self.refs:
            return self.refs[instance]
        else:
            ref = Step(instance, self.method.__get__(instance, owner))
            self.refs[instance] = ref
            return ref


class Step:
    """
    The step instance returned when accessing a models step.

    A step will take care of activating itself before invoking the underlaying
    method, so all dependencies will be registered insde the step. Moreover,
    it will register the resulting implementation inside the `steps` Specs
    of the model and returns the accessor for the specs.

    @see step
    """

    def __init__(self, model, method):
        self.__self__ = model
        self.method = method

        self.steps = {}
        self.state = {}
        self.params = {}
        self.random = {}
        self.derives = {}

    @property
    def __name__(self):
        """Return the name of the step."""
        return self.method.__name__

    @property
    def impl(self):
        """Return the underlying implementation registered in the model."""
        return self.__self__.steps.specs[self.__name__]

    def __call__(self):
        """Return the specs accessor of the step."""
        with self.__self__.steps.activate(self.steps), \
                self.__self__.state.activate(self.state), \
                self.__self__.params.activate(self.params), \
                self.__self__.random.activate(self.random), \
                self.__self__.derives.activate(self.derives):
            impl = self.method()
            impl.__self__ = self.__self__
            impl.__name__ = self.__name__
            impl.__step__ = self

        return self.__self__.steps(**{self.__name__: impl})


class Impl:
    def __init__(self, impl, step):
        self.impl = impl
        self.step = step

    @property
    def __name__(self):
        return self.step.__name__


class Model:
    """
    Base for implementing a fast, compoable simulation model.

    The model consists of `steps`, that will transform a `state` using
    as set of `paramms` and `random` variables.

    * `init` will be called once at the begin of the simulation for each sample
    * `iterate` and `apply` will be called repititly according to the parameter
      `n_step`, where `iterate` works on individuals of the population, while
      `apply` works on the complete population.
    * finally, a `finish` step is invoced with the complete population
    """

    def __init__(self):
        """Create a new model instance initializing its specs."""
        self.steps = Specs()
        self.state = Specs()
        self.params = Specs()
        self.random = Specs()
        self.derives = Specs()

        self.alloc = None
        self.engine = NumbaEngine().bind(self, compile=False)
        self.traces = []

        self.__setup__()

    def __setup__(self):
        """Register the basic layout of the model."""
        self.params(n_samples=int)
        self.params(n_steps=int)

        self.init()
        self.iterate()
        self.apply()
        self.finish()

    def use(self, engine, compile=None):
        """
        Use a different engine for executing the model.

        @param engine: Engine
            the new engine to use
        @param compile: boolean
            pass on the compile option to the engine, so it will already
            precompile the model
        @return self
        """
        self.engine = engine.bind(self, self.alloc, compile=compile)
        return self

    def bind(self, alloc, compile=None):
        """
        Bind the allocation object to this sumlation.

        @param alloc: Allocation
            the allocation to be used by the simulation model
        @param compile: boolean
            pass on the compile option to the engine, so it will already
            precompile the model
        @return self
        """
        self.alloc = alloc
        self.engine.bind(self, self.alloc, compile=compile)
        return self

    def execute(self, engine=None, **params):
        """
        Execute the simulation on a engine recoding it's resulting state.

        @param engine: Engine (optional)
            the engine to be used for execution
        @param params: dict
            parameters to set on the allocation before executing the model
        @return
            the state after running the simulation
        """
        if engine is None:
            engine = self.engine
        else:
            engine.bind(self, self.alloc)

        with self.alloc(**params):
            return engine.execute(self.traces)

    @contextmanager
    def trace(self, *traces, **opts):
        """
        Activte the supplied `Trace` objects inside this context.

        @param traces: Trace
            `Trace` to activate when `execute`ing the simulation
        @param opts: dict
            options to pass on to the trace objects
        """
        if len(traces) == 1:
            trace, = traces
            if opts:
                trace = trace.options(**opts)
            trace.on(self)
            self.traces.append(trace)
            try:
                yield trace.prepare()
                trace.finalize()
            finally:
                after = self.traces.pop()
                assert after == trace

        else:
            with ExitStack() as stack:
                yield tuple([stack.enter_context(self.trace(t, **opts))
                             for t in traces])

    @contextmanager
    def resolving(self, *, steps, state, params, random, derives):
        """Use the supplied function for resoliving allocation in `Specs`."""
        with self.steps.resolving(steps), \
                self.state.resolving(state), \
                self.params.resolving(params), \
                self.random.resolving(random), \
                self.derives.resolving(derives):
            yield

    def graph(self, **attrs):
        """Create a grpahivz graph out of the steps call tree."""
        gv = __import__("graphviz")

        graph = gv.Digraph()
        graph.attr(rankdir='LR')
        graph.attr(**attrs)

        with graph.subgraph() as g:
            g.attr(rank='min')
            g.edge('init', 'iterate')
            g.edge('iterate', 'apply')
            g.edge('apply', 'iterate', constraint='false')
            g.edge('apply', 'finish')

        for name, impl in self.steps.specs.items():
            calls = [s for s in impl.__step__.steps if s != name]
            while name in impl.__step__.steps:
                step = impl.__step__.steps[name].__impl__
                calls.extend([s for s in step.steps if s != name])

            with graph.subgraph() as g:
                g.attr(rank='same')
                for a, b in zip(calls, calls[1:]):
                    g.edge(a, b, style='invis')

            for c in calls:
                graph.edge(name, c)

        return graph

    def derive(self, **deps):
        """
        Create a new dynamic prarameter from the supplied parameters.

        @param deps: dict
            parameters used to create the dynamic value
        @return
            annotating function
        """
        def annotate(fn):
            name = '{}({})'.format(fn.__name__, ','.join(deps))
            return wraps(fn)(self.derives(**{name: (fn, deps)}))
        return annotate

    @step
    def init(self):
        """
        Return an implemenetation initializing the state for each individual.

        @param params
             the parameters object that should be passed on to the `params`
             and `random` accessors inside the implementation of the step
        @param state
             line of the state that should be updated by indexing with the
             `state` accessors during the implementation of this step
        @return
             the implementation function accepting these parameters
        """
        def impl(params, state):
            pass
        return impl

    @step
    def iterate(self):
        """
        Return an implemenetation that runs one simulation step on each sample.

        @param params
             the parameters object that should be passed on to the `params`
             and `random` accessors inside the implementation of the step
        @param state
             line of the state that should be updated by indexing with the
             `state` accessors during the implementation of this step
        @return
             the implementation function accepting these parameters
        """
        def impl(params, state):
            pass
        return impl

    @step
    def apply(self):
        """
        Return an implemenetation that updates the complete population.

        @param params
             the parameters object that should be passed on to the `params`
             and `random` accessors inside the implementation of the step
        @param state
             state of all samples that should be updated by indexing with the
             `state` accessors during the implementation of this step
        @return
             the implementation function accepting these parameters
        """
        def impl(params, state):
            pass
        return impl

    @step
    def finish(self):
        """
        Return an implemenetation finializes the complete population.

        @param params
             the parameters object that should be passed on to the `params`
             and `random` accessors inside the implementation of the step
        @param state
             state of all samples that should be updated by indexing with the
             `state` accessors during the implementation of this step
        @return
             the implementation function accepting these parameters
        """
        def impl(params, state):
            pass
        return impl
