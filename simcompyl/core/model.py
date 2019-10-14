"""Base for defining efficient simulation models."""

from weakref import WeakKeyDictionary
from collections import namedtuple
from contextlib import contextmanager, ExitStack
from functools import wraps
from types import FunctionType
import numpy as np

import logging

from .util import Resolvable, lazy
from .trace import Trace


LOG = logging.getLogger(__name__)

__all__ = ['Model', 'step']


class Specs(Resolvable):
    """Object for specifying and accessing specific aspects of a simulation.

    Specs are callables accepting keywords with a specification that will be
    validated against other uses of the same name.

    Internally, this object is also used to glue things up:

    First of all, to give a more detailed report about the usage of the
    simulation aspects, the specs will not only recorded inside the specs own
    dictionary, but as well inside an `active` dictionary, that will be
    `activate`d in a context of the models `@step`.

    Moreover, the call will return accessors according to the spec that can be
    used during the execution of the simulation. This is archived by letting
    the execution engine register itself inside the `binding` context and
    giving control to the engine to create the accessor.
    """

    def __init__(self, name, collect=None):
        """Create a new specification aspect of the simulation."""
        self.specs = {}
        self._name = name
        self._collect = collect
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
        LOG.debug(f"spec default resolution for {self._name} of {name}")
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
        spec : type or similar
            the new type for the spec
        prev : type or similar (optional)
            previous defined spec if available.

        """
        LOG.debug(f"validation for {self._name} of {name}={spec} ? {prev}")
        if spec is ...:
            if prev is None:
                msg = "Invalid spec ellipsis for {!r}, no previous definition."
                raise TypeError(msg.format(name))
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
            return [self.validate('', fst, prv)] * size

        if all(hasattr(prev, a) for a in ['__self__', '__call__', '__name__']):
            assert hasattr(spec, '__self__')

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
            name, spec pairs, using the previous value for `name=...` specs

        Returns
        -------
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

        (name, spec), = results.items()
        return self.resolve(name, spec)


class SpecsCollection(dict):
    """Specialized dict putting together multiple specs.

    This class adds the ability to manage all contained specs together.
    """

    def initialize(self, obj):
        """Initialize dict-attributes on an object for later `activate` calls.

        Parameters
        ----------
        obj: object
            a python object where attributes are set to new dictionaries
            for all the specs defined inside the collection

        """
        for name, _ in self.items():
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
        LOG.debug(f"specs activation of {obj}")
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

    >>> # noinspection PyUnresolvedReferences
    ... @step
    ... def somestep(self):
    ...     _somestep = super().somestep()
    ...     _otherstep = self.otherstep()
    ...     v = self.state(v=int)
    ...     a, b = self.params(a=int, b=int)
    ...     r = self.random(r=bool)
    ...
    ... # noinspection PyUnresolvedReferences
    ... def impl(params, state, arg):
    ...          arg = 0
    ...
    ...          while a <= state[v] < b
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

        if instance in self.refs:
            result = self.refs[instance]
        else:
            method = self.method.__get__(instance, owner)
            self.refs[instance] = result = Step(instance, method)

        return instance.steps(**{result.__name__: result})


class Step:
    """The step instance returned when accessing a models step.

    A step will take care of activating itself before invoking the underlying
    method, so all dependencies will be registered inside the step. Moreover,
    it will register the resulting implementation inside the `steps` Specs
    of the model and returns the accessor for the specs.
    """

    def __init__(self, model, method):
        self.model = self.__self__ = model
        self.method = method
        self.model.__specs__.initialize(self)
        self.impl

    @property
    def hier(self):
        """Hierarchy of super step calls."""
        hier = []
        step = self
        while step:
            hier.append(step)
            step = step.steps.get(self.__name__)
        return hier

    @property
    def name(self):
        """Return the name of the step."""
        return self.method.__name__

    @property
    def component(self):
        return self.method.__qualname__.rsplit('.', 1)[0]

    __name__ = name

    @property
    def impl(self):
        """Return the underlying implementation registered in the model."""
        with self.__self__.__specs__.activate(self):
            impl = self.method()
            if not isinstance(impl, FunctionType):
                msg = ("A @step method should return a function, "
                       "but {} returned a {} object.")
                raise TypeError(msg.format(self.__name__, type(impl).__name__))

            impl.__self__ = self.__self__
            impl.__name__ = self.__name__
            impl.__step__ = self
        return impl

    def __call__(self):
        """We are a pseudo-function."""
        raise TypeError("Steps should not be called directly.")

    def __str__(self):
        return f'@{self.method.__qualname__}'

    def __repr__(self):
        return f'Step({self.method.__qualname__} of {self.method.__self__!r})'


class Model:
    """Base for implementing a fast, compoable simulation model.

    The model consists of `steps`, that will transform a `state` using
    as set of `paramms` and `random` variables.

    Steps
    -----
    * `init` will be called once at the begin of the simulation for each sample
    * `iterate` and `apply` will be called repeated according to the parameter
      `n_step`, where `iterate` works on individuals of the population, while
      `apply` works on the complete population.

    """

    def __init__(self):
        """Create a new model instance initializing its specs."""
        allocs = Specs('allocs')
        self.__specs__ = SpecsCollection(steps=Specs('steps'),
                                         state=Specs('state'),
                                         params=Specs('params', collect=allocs),
                                         random=Specs('random'),
                                         derives=Specs('derives'),
                                         allocs=allocs)
        (self.steps, self.state,
         self.params, self.random,
         self.derives, self.allocs) = self.__specs__.values()

        self.__setup__()

    def __str__(self):
        return f"{type(self).__name__}@{id(self):x}"

    __repr__ = __str__

    def __setup__(self):
        """Register the basic layout of the model."""
        return (self.init,
                self.iterate,
                self.apply)

    def __dir__(self):
        """List state keys as well."""
        return list(super().__dir__()) + list(self.state)

    def __getitem__(self, item):
        """Access state variables to trace them."""
        if not isinstance(item, tuple):
            item = (item,)
        missing = set(item) - set(self.state)
        if missing:
            raise KeyError("{} not found in models state."
                           .format(", ".join(missing)))
        spec = self.state(True, **{i: ... for i in item})._asdict()
        return Trace(**spec)

    def __call__(self, **assigns):
        """Return custom traces form functions based on state variables."""
        return Trace(**assigns)

    def _ipython_key_completions_(self):
        return list(self.state)

    @contextmanager
    def binding(self, *, steps, state, params, random, derives):
        """Use the supplied function for binding allocation in `Specs`."""
        LOG.debug(f"resolving by binding {steps}, ...")
        with self.steps.binding(steps), \
                self.state.binding(state), \
                self.params.binding(params), \
                self.random.binding(random), \
                self.derives.binding(derives):
            yield

    def hier(self, **attrs):
        gv = __import__("graphviz")
        graph = gv.Digraph()
        graph.attr(rankdir="BT")
        graph.attr(**attrs)
        graph.attr('node', shape='box')
        graph.attr('edge', arrowhead='empty')

        stack = [type(self)]
        visited = set()

        while stack:
            cls, *stack = stack
            bases = [base for base in cls.__bases__ if issubclass(base, Model)]
            for base in bases:
                graph.edge(cls.__qualname__, base.__qualname__)
            stack.extend([base for base in bases if base not in visited])
            visited.update(bases)
        return graph

    def graph(self, details=True, rankdir='LR', cm='Pastel2', internals=False,
              **attrs):
        """Create a grpahivz graph out of the steps call tree."""
        gv = __import__("graphviz")
        mp = __import__("matplotlib.cm")

        hier = [t for t in type(self).mro() if issubclass(t, Model)]
        colors = {t.__qualname__: getattr(mp.cm, cm)(i/len(hier), bytes=True)
                  for i, t in enumerate(hier)}

        graph = gv.Digraph()
        graph.attr(rankdir=rankdir, compound='true')
        graph.attr('edge', arrowhead='vee')
        graph.attr(**attrs)

        with graph.subgraph() as core:
            core.attr(rank='min', newrank='true')
            core.attr('node', shape='point', style='invis')
            core.attr('edge', style='invis')

            core.edge('_init', '_iterate')
            core.edge('_iterate', '_apply')
            core.edge('_apply', '_iterate', constraint='false')

        graph.edge('_init', str(self.init),
                   lhead='cluster_init', minlen='2', weight='2')
        graph.edge('_iterate', str(self.iterate),
                   lhead='cluster_iterate', minlen='2', weight='2')
        graph.edge('_apply', str(self.apply),
                   lhead='cluster_apply', minlen='2', weight='2')

        for node in self.steps.values():
            with graph.subgraph(name=f'cluster_{node.__name__}') as s:
                s.attr(label=f'<<B>@{node.__name__}</B>>', labeljust='l')
                for sub in node.hier:
                    if details:
                        c = colors[sub.component]
                        label = f'''<
                        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                        <TR><TD BGCOLOR="#{c[0]:02x}{c[1]:02x}{c[2]:02x}"><B>{sub.component}</B></TD></TR><HR/>
                        <TR><TD ALIGN="LEFT">state: {', '.join(sub.state)}</TD></TR>
                        <TR><TD ALIGN="LEFT">params: {', '.join(sub.allocs)}</TD></TR>
                        <TR><TD ALIGN="LEFT">random: {', '.join(sub.random)}</TD></TR>
                        </TABLE>
                        >'''
                        s.node(str(sub), label=label, shape='plaintext')
                    else:
                        s.node(str(sub), label=sub.component, shape='box')

                for a, b in zip(node.hier, node.hier[1:]):
                    s.edge(f'{a}', f'{b}', minlen='1', weight='20')

        for node in self.steps.values():
            calls = []
            stack = list(node.steps.values())
            while stack:
                call, *stack = stack
                if call.__name__ == node.__name__:
                    stack = list(call.steps.values()) + stack
                else:
                    calls.append(call)

            opts = {'color': 'grey'} if internals else {'style': 'invis'}
            if len(calls) > 1:
                with graph.subgraph(name=f'calls_{node.__name__}') as s:
                    #s.attr(rank='same')
                    for a, b in zip(calls, calls[1:]):
                        s.edge(str(a), str(b), minlen='1', **opts)

            for sub in node.hier:
                constraint = 'true'
                for call in sub.steps.values():
                    if call.__name__ != sub.name:
                        graph.edge(str(sub), str(call),
                                   minlen='1',
                                   constraint=constraint,
                                   lhead=f'cluster_{call.__name__}')
                        #constraint = 'false'

        return graph

    def derive(self, **deps):
        """Create a new dynamic parameter from the supplied parameters.

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

            LOG.debug(f"derive on {name} using {deps}")
            self.allocs(**deps)
            return self.derives(**{name: (fn, deps)})
        return annotate

    @step
    def init(self):
        """Return an implementation initializing the state of each individual.

        Returns
        -------
             impl(params, state)
                the implementation function accepting params and state args

        """
        _ = self.params(n_samples=int)
        def impl(_, __):
            pass
        return impl

    @step
    def iterate(self):
        """Return an implementation running one step on each sample.

        Returns
        -------
             impl(params, state)
                the implementation function accepting params and state args

        """
        _ = self.params(n_steps=int)
        def impl(_, __):
            pass
        return impl

    @step
    def apply(self):
        """Return an implementation that updates the complete population.

        Returns
        -------
             impl(params, state)
                the implementation function accepting params and state args

        """
        def impl(_, __):
            pass

        return impl
