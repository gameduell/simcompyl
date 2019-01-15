"""Base for defining efficient simulation models."""

from weakref import WeakKeyDictionary
from collections import namedtuple
from contextlib import contextmanager, ExitStack
from functools import wraps
from types import FunctionType

from .util import Resolvable

__all__ = ['Model', 'step']


class Specs(Resolvable):
    """Object for specifing and accessing specific aspects of a simulation.

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

    def __init__(self, collect=None):
        """Create a new specification asspect of the simulation."""
        self.specs = {}
        self._collect = collect
        self._resolve = None
        self._activated = [{}]

    def __contains__(self, name):
        """Check weather there is a spec with the given name."""
        return name in self.specs

    def __iter__(self):
        """Iterate through all names defined in this specs."""
        return iter(self.specs)

    def __getitem__(self, name):
        """Return the spec for a specific name."""
        return self.specs[name]

    def keys(self):
        """Get all names this Specs."""
        return self.specs.keys()

    def values(self):
        """Get all the Specs values."""
        return self.specs.values()

    def items(self):
        """Get name, value paris of this Specs."""
        return self.specs.items()

    def unresolved(self, name, spec):
        """Return the spec when no resolver is bound."""
        return spec

    @contextmanager
    def activate(self, specs):
        """Activate a dict to record all the specs in the context.

        Parameters
        ----------
        specs : dict
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

    def validate(self, name, spec, prev=None):
        """Validate a spec with the given name against a previous definition.

        Parameters
        ----------
        name : str
            name for the spec
        spec : type or simular
            the new type for the spec
        prev : type or simular (optional)
            previous defined spec if available.

        """
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

        if all(hasattr(prev, a) for a in ['__self__', '__call__', '__name__']):
            if spec.__name__ != prev.__name__:
                msg = "Method spec {} with different methods {} previously {}."
                raise TypeError(msg.format(name, spec.__name__, prev.__name__))

            if spec.__self__ != prev.__self__:
                msg = "Method spec {} on different objects {} previously {}."
                raise TypeError(msg.format(name, spec.__self__, prev.__self__))

            return spec

        if isinstance(prev, tuple) and hasattr(prev[0], '__code__'):
            if spec[0].__code__ != prev[0].__code__:
                msg = ("Function spec {} with different code {}:{} "
                       "previously {}:{}.")
                raise TypeError(msg.format(name,
                                           prev[0].__code__.co_filename,
                                           prev[0].__code__.co_firstlineno,
                                           spec[0].__code__.co_filename,
                                           spec[0].__code__.co_firstlineno))

            return spec

        if prev is not None and spec != prev:
            msg = "Conflicting type definition of {} with {}, previously {}."
            raise TypeError(msg.format(name, spec, prev))

        return spec

    def __call__(self, return_namedtuple=None, **specs):
        """Register and validate the given names with the supplied specs.

        Parameters
        ----------
        return_namedtuple : boolean
            weather to return a namedtuple even when only one spec is given
        **specs : dict
            name, spec pairs, using the previous value for `name=...` spces

        Retuerns
        --------
            accessor
                Either a namedtuple of accessors or a single accessor,
                that can be used to access a value during simulation

        """
        if self._collect:
            self._collect(**specs)

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

        if return_namedtuple or len(results) != 1:
            Access = namedtuple('Access', list(results))
            return Access(**{name: self.resolve(name, spec)
                             for name, spec in results.items()})
        else:
            (name, spec), = results.items()
            return self.resolve(name, spec)


class SpecsCollection(dict):
    """Specialized dict puting together multipe specs.

    This class adds the ability to manage all contained specs togheter.
    """

    def initialize(self, obj):
        """Initialize dict-attributes on an object for later `activate` calls.

        Parameters
        ----------
            obj: object
                a python object where attributes are set to new dictionaries
                for all the specs defined inside the collection

        """
        for name, sepcs in self.items():
            setattr(obj, name, {})

    @contextmanager
    def activate(self, obj):
        """Activate dicts of the object for each spec in the collection.

        Parameters
        ----------
            obj: object
                a python object with dict-attributes for each spec in this
                collection.

        """
        with ExitStack() as stack:
            for name, specs in self.items():
                stack.enter_context(specs.activate(getattr(obj, name)))
            yield


def step(method):
    """Annotate a method used as a step inside a simulation model.

    The method should first bind local variables to accesors returned by
    the models `state`, `params` or `random` specs or by calls to other or
    super steps of the models. It then can define and return the state
    transforming implementation using these local variables.
    This implementation should accept a parameter object and the state as
    its first arguments, but also can have custom arguments or return types.
    The parameter object should be past on to the params and random accessors.

    The object where the `@step` methods are defined has to support a
    `__sepcs__` attribute, returning a SpecsCollection of all specs that should
    be recorded in each step, at least a `steps` `Specs`, which records all
    calls to `@step` annotated methods.

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

    Parameters
    ----------
    method : method
        the annotated method returning the actual implementation

    """
    return wraps(method)(StepDescriptor(method))


class StepDescriptor:
    """The descriptor returned by the step annotation, creating `Step`s."""

    def __init__(self, method):
        self.method = method
        self.refs = WeakKeyDictionary()

    def __get__(self, instance, owner=None):
        """Return the step instance for this descriptor."""
        if instance is None:
            return self
        elif instance in self.refs:
            return self.refs[instance]
        else:
            ref = Step(instance, self.method.__get__(instance, owner))
            self.refs[instance] = ref
            return ref


class Step:
    """The step instance returned when accessing a models step.

    A step will take care of activating itself before invoking the underlaying
    method, so all dependencies will be registered insde the step. Moreover,
    it will register the resulting implementation inside the `steps` Specs
    of the model and returns the accessor for the specs.
    """

    def __init__(self, model, method):
        self.__self__ = model
        self.method = method
        self.__self__.__specs__.initialize(self)

    @property
    def __name__(self):
        """Return the name of the step."""
        return self.method.__name__

    @property
    def impl(self):
        """Return the underlying implementation registered in the model."""
        return self.__self__.__specs__['steps'].specs[self.__name__]

    def __call__(self):
        """Return the specs accessor of the step."""
        with self.__self__.__specs__.activate(self):
            impl = self.method()
            if not isinstance(impl, FunctionType):
                msg = ("A @step method should return a function, "
                       "but {} returned a {} object.")
                raise TypeError(msg.format(self.__name__, type(impl).__name__))

            impl.__self__ = self.__self__
            impl.__name__ = self.__name__
            impl.__step__ = self

        return self.__self__.__specs__['steps'](**{self.__name__: impl})


class Model:
    """Base for implementing a fast, compoable simulation model.

    The model consists of `steps`, that will transform a `state` using
    as set of `paramms` and `random` variables.

    Steps
    -----
    * `init` will be called once at the begin of the simulation for each sample
    * `iterate` and `apply` will be called repititly according to the parameter
      `n_step`, where `iterate` works on individuals of the population, while
      `apply` works on the complete population.
    * finally, a `finish` step is invoced with the complete population

    """

    def __init__(self):
        """Create a new model instance initializing its specs."""
        allocs = Specs()
        self.__specs__ = SpecsCollection(steps=Specs(),
                                         state=Specs(),
                                         params=Specs(collect=allocs),
                                         random=Specs(),
                                         derives=Specs(),
                                         allocs=allocs)
        (self.steps, self.state,
         self.params, self.random,
         self.derives, self.allocs) = self.__specs__.values()

        self.__setup__()

    def __setup__(self):
        """Register the basic layout of the model."""
        self.params(n_samples=int)
        self.params(n_steps=int)

        self.init()
        self.iterate()
        self.apply()
        self.finish()

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
                impl = impl.__step__.steps[name]
                calls.extend([s for s in impl.__step__.steps if s != name])

            with graph.subgraph() as g:
                g.attr(rank='same')
                for a, b in zip(calls, calls[1:]):
                    g.edge(a, b, style='invis')

            for c in calls:
                graph.edge(name, c)

        return graph

    def derive(self, **deps):
        """Create a new dynamic prarameter from the supplied parameters.

        Parameters
        ----------
        deps : dict
            parameters used to create the dynamic value

        Returns
        -------
            accessor
                accessor to the dynamic parameter for inside the simulation

        """
        def annotate(fn):
            name = '{}({})'.format(fn.__name__, ','.join(deps))
            self.allocs(**deps)
            return self.derives(**{name: (fn, deps)})
        return annotate

    @step
    def init(self):
        """Return an implemenetation initializing the state of each individual.

        Returns
        -------
             impl(params, state)
                the implementation function accepting params and state args

        """
        def impl(params, state):
            pass
        return impl

    @step
    def iterate(self):
        """Return an implemenetation running one step on each sample.

        Returns
        -------
             impl(params, state)
                the implementation function accepting params and state args

        """
        def impl(params, state):
            pass
        return impl

    @step
    def apply(self):
        """Return an implemenetation that updates the complete population.

        Returns
        -------
             impl(params, state)
                the implementation function accepting params and state args

        """
        def impl(params, state):
            pass
        return impl

    @step
    def finish(self):
        """Return an implemenetation finializes the complete population.

        Returns
        -------
             impl(params, state)
                the implementation function accepting params and state args

        """
        def impl(params, state):
            pass
        return impl
