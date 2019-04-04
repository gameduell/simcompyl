"""Engine actually executes the simulations code."""
import numba as nb
import numpy as np
import pandas as pd

from collections import namedtuple
from contextlib import contextmanager

from .util import lazy
from .trace import Trace

import logging

logger = logging.getLogger(__name__)


class BasicExecution:
    """Engine to execute a simulation defined by a model."""

    def __init__(self, model, alloc):
        self.model = model
        self.alloc = alloc
        self.traces = []
        self.trace_cache = {}

    @contextmanager
    def trace(self, *traces, target=None, **options):
        """Activate given traces."""
        logger.debug("engine activates traces %s using %s target",
                     traces, target or 'default')

        traces = [tr.to(target, **options) if isinstance(tr, Trace) else tr
                  for tr in traces]

        self.traces.extend(traces)
        try:
            yield (traces[0].prepare()
                   if len(traces) == 1
                   else [tr.prepare() for tr in traces])

        finally:
            logger.debug("engine deactivates traces %s traces")
            for tr in traces:
                tr.finalize()
            self.traces = self.traces[:-len(traces)]

    def tracing(self, trace):
        """Cache of a trace method."""
        if trace.__freeze__() in self.trace_cache:
            logger.debug("engine cached trace %s", trace)
            return self.trace_cache[trace.__freeze__()]

        ctx = TraceContext(self)

        logger.debug("engine resolving trace %s", trace)
        with self.resolving():
            cc = self.compile(ctx.resolve_function(trace.trace(ctx)),
                              vectorize=False)
        self.trace_cache[trace.__freeze__()] = cc
        return cc

    def run(self, **allocs):
        """Execute the model."""
        if allocs:
            with self.alloc(**allocs):
                return self.run()

        params = self.params()
        state = self.state()

        init = self.init
        iterate = self.iterate
        apply = self.apply
        finish = self.finish

        trs = [(tr.publish, self.tracing(tr.trace))
               for tr in self.traces]

        if trs:
            def trace(pr, st):
                for pub, tr in trs:
                    pub(*tr(pr, st))
        else:
            def trace(_, __):
                pass

        logger.info("engine initializes simulation %s", self.model)
        init(params, state)
        trace(params, state)

        logger.info("engine iterates simulation %s", self.model)
        for i in range(self.alloc.n_steps.value):
            iterate(params, state)
            apply(params, state)
            trace(params, state)

        logger.info("engine finishes simulation %s", self.model)
        finish(params, state)

        return self.frame(state)

    @contextmanager
    def resolving(self):
        """Context where this engine should resolve accessors of the model."""
        logger.debug("engine registers itself as resolver for %s", self.model)
        with self.model.resolving(steps=self.resolve_steps,
                                  state=self.resolve_state,
                                  params=self.resolve_params,
                                  random=self.resolve_random,
                                  derives=self.resolve_derives):
            yield

    @lazy
    def init(self):
        """Cache of the initialization of the model."""
        logger.debug("engine resolves %s", self.model.init)
        with self.resolving():
            return self.compile(self.model.init(), vectorize=True)

    @lazy
    def iterate(self):
        """Cache of the iteration of the model."""
        logger.debug("engine resolves %s", self.model.iterate)
        with self.resolving():
            return self.compile(self.model.iterate(), vectorize=True)

    @lazy
    def apply(self):
        """Cache of the application of the model."""
        logger.debug("engine resolves %s", self.model.apply)
        with self.resolving():
            return self.compile(self.model.apply(), vectorize=False)

    @lazy
    def finish(self):
        """Cache of the finalization of the model."""
        logger.debug("engine resolves %s", self.model.finish)
        with self.resolving():
            return self.compile(self.model.finish(), vectorize=True)

    def params(self):
        """Parameters as passed to optimized simulation methods."""
        logger.info("engine creates params arg")
        return self.alloc

    def state(self):
        """State as passed to optimized simulation methods."""
        logger.info("engine creates state arg")
        shape = (self.alloc.n_samples.value,
                 sum(len(v) if isinstance(v, list) else 1
                     for v in self.model.state.specs.values()))
        return np.zeros(shape, dtype=float)

    def frame(self, state):
        """Create a dataframe for the given state representation."""
        logger.debug("engine creates frame of given state")
        columns = sum([[n for _ in s] if isinstance(s, list) else [n]
                       for n, s in self.model.state.specs.items()], [])
        return pd.DataFrame(state, columns=columns)

    def resolve_steps(self, name, impl):
        """Create an accessor for a step implementation method."""
        logger.debug("engine resolves step %s", name)

        return impl

    def resolve_state(self, name, typ):
        """Create an accessor for parts of the state."""
        logger.debug("engine resolves state %s", name)

        idx = sum(([n for _ in s] if isinstance(s, list) else [n]
                   for n, s in self.model.state.specs.items()), []).index(name)

        if isinstance(typ, list):
            logger.debug("engine resolved state %s to range(%i, %i)",
                         name, idx, idx + len(typ))
            return list(range(idx, idx + len(typ)))
        else:
            logger.debug("engine resolved state %s to %d", name, idx)
            return idx

    def resolve_params(self, name, typ):
        """Create an accessor for a parameter value."""
        logger.debug("engine resolves param %s", name)

        if isinstance(typ, dict):
            def define(key):
                def dct_getter(params):
                    return getattr(params, name).value[key]
                return dct_getter

            Getters = namedtuple('Getters', list(typ))
            getters = {n: define(n) for n in typ}
            logger.debug("engine resolved param %s to %s", name, getters)
            return Getters(**getters)
        else:
            def std_getter(params):
                return getattr(params, name).value
            logger.debug("engine resolved param %s to %s", name, std_getter)
            return std_getter

    def resolve_random(self, name, typ):
        """Create an accessor for a random distribution."""
        logger.debug("engine resolves random %s", name)

        def getter(params):
            alloc = getattr(params, name)
            sample = alloc.param.sample()
            return sample(*alloc.args)
        return getter

    def resolve_derives(self, name, spec):
        """Create an accessor for a derivied parameter."""
        fn, deps = spec
        logger.debug("engine resolves derives %s", name)

        def getter(params):
            return fn(*[getattr(params, n).value for n in deps])

        return getter

    def resolve_function(self, function):
        """Resolve a function that is called during simulation."""
        logger.debug("engine resolves function %s", function.__name__)
        return function

    def compile(self, impl, vectorize=True):
        """Create a compiled version of a step implementation."""
        # TODO more doc
        logger.info("engine compiles %s (vectorize=%s)", impl, True)
        if vectorize:
            def vector(params, state):
                for i in range(state.shape[0]):
                    impl(params, state[i])
            return vector
        else:
            return impl


class NumbaExecution(BasicExecution):
    """Engine to create a llvm-compiled simulation using the numba package."""

    def __init__(self, model, alloc,
                 use_gufunc=True, parallel=True, fastmath=True):
        """Create a new numba engine instance.

        Parameters
        ----------
        use_gufunc : bool
            execute vectorizing simulation methods with `numba.guvectorize`
        parallel : bool
            pass parallel to or specify 'parallel' target for numba
        fastmath : bool
            use numba fastmath
        """
        super().__init__(model, alloc)

        self.use_gufunc = use_gufunc
        self.parallel = parallel
        self.fastmath = fastmath

    def njit(self, *args, **kws):
        """Call to jit in no-python mode."""
        return nb.njit(*args, **kws)

    def vjit(self, dtypes, shapes,
             fastmath=None, nopython=None, parallel=None):
        if self.use_gufunc:
            return nb.guvectorize(dtypes, shapes, 
                                  fastmath=fastmath, nopython=nopython,
                                  target='parallel' if parallel else 'cpu')
        else:
            def vectorize(impl):
                ext = [tuple(dt.dtype[:, :] if i == 1 else dt
                             for i, dt in enumerate(dts))
                       for dts in dtypes]

                @self.njit(ext,
                           fastmath=fastmath,
                           nopython=nopython,
                           parallel=parallel)
                def vect(params, state):
                    for i in nb.prange(len(state)):
                        impl(params, state[i])
                return vect
            return vectorize

    def compile(self, impl, vectorize=True):
        """Use numba to compile the specified function."""
        if not hasattr(impl, 'py_func'):
            msg = ("Function {} seems not to be a numba function, "
                   "perhaps you forgot the @sim.step decorator")
            raise AttributeError(msg.format(impl))

        logger.info("engine compiles %s (vectorize=%s)", impl, vectorize)
        if vectorize:
            return self.vjit([(nb.float64[:], nb.float64[:])], '(m),(n)',
                             fastmath=self.fastmath,
                             nopython=True,
                             parallel=self.parallel)(impl.py_func)
        else:
            return self.njit([(nb.float64[:], nb.float64[:, :])],
                             fastmath=self.fastmath,
                             parallel=self.parallel)(impl.py_func)

    @lazy
    def layout(self):
        """Cache for positioning parameters into an numpy array."""
        logger.info("engine creates layout for parameters.")
        layout = []

        for name, spec in self.model.params.specs.items():
            if isinstance(spec, dict):
                bound = len(next(iter(spec.values())))
                # XXX check if all items define same bound
                layout.append(name)
                for n, ts in spec.items():
                    assert len(ts) == bound
                    layout.extend((name, n) for _ in ts)
            else:
                layout.append(name)

        for name in self.model.random.specs.keys():
            layout.extend([name] * 6)

        for name, (fn, deps) in self.model.derives.specs.items():
            args = [{n: [t() for t in ts]
                     for n, ts in typ.items()}
                    if isinstance(typ, dict) else typ()
                    for typ in deps.values()]
            result = fn(*args)

            if isinstance(result, list):
                result = np.array(result)

            if isinstance(result, np.ndarray):
                layout.extend([name] * len(result.shape))
                layout.extend([name] * len(result.ravel()))

            elif isinstance(result, tuple):
                layout.extend([name] * len(result))

            else:
                layout.append(name)

        return layout

    def params(self):
        """Create a numpy array for the parameters of the simulation."""
        logger.info("engine creates parameters numpy array")
        params = []

        for name, spec in self.model.params.specs.items():
            val = getattr(self.alloc, name).value

            if isinstance(spec, dict):
                bound = len(next(iter(spec.values())))
                size = len(next(iter(val.values())))
                assert size <= bound

                params.append(size)
                for key in spec.keys():
                    params.extend(val[key])
                    params.extend([0] * (bound - len(val[key])))

            else:
                params.append(val)

        for name in self.model.random.specs.keys():
            val = getattr(self.alloc, name).value

            if isinstance(val, dict):
                ls = list(val.values())
            elif isinstance(val, tuple):
                ls = list(val)
            else:
                ls = [val]

            params.extend((ls + [0] * 6)[:6])

        for name, (fn, deps) in self.model.derives.specs.items():
            spec = [{n: [t() for t in ts]
                     for n, ts in typ.items()}
                    if isinstance(typ, dict) else typ()
                    for typ in deps.values()]
            virt = fn(*spec)

            args = [getattr(self.alloc, name).value for name in deps.keys()]
            real = fn(*args)

            if isinstance(virt, list):
                virt = np.array(virt)

            if isinstance(real, list):
                real = np.array(real)

            if isinstance(virt, np.ndarray):
                assert(len(virt.ravel()) >= len(real.ravel()))
                params.extend(real.shape)
                params.extend(real.ravel())
                params.extend([0] * (len(virt.ravel()) - len(real.ravel())))

            elif isinstance(virt, tuple):
                assert len(virt) == len(real)
                params.extend(real)

            else:
                params.append(real)

        assert len(params) == len(self.layout)

        return np.array(params, dtype=float)

    def resolve_steps(self, name, impl):
        """Return the binding for the given step."""
        logger.debug("engine resolves step %s", name)
        return self.njit(impl)

    def resolve_params(self, name, typ):
        """Return the binding for a given parameter."""
        logger.debug("engine resolves param %s", name)

        if isinstance(typ, dict):
            lx = self.layout.index(name)

            def define(ix, _):
                @self.njit
                def dct_getter(params):
                    size = int(params[lx])
                    return params[ix:ix + size]
                return dct_getter

            Getters = namedtuple('Getters', list(typ))
            getters = {n: define(self.layout.index((name, n)), ts)
                       for n, ts in typ.items()}
            logger.debug("engine resolved param %s to %s", name, getters)
            return Getters(**getters)

        else:
            idx = self.layout.index(name)

            @self.njit
            def std_getter(params):
                return typ(params[idx])
            logger.debug("engine resolved param %s to %s", name, std_getter)
            return std_getter

    def resolve_random(self, name, typ):
        """Return the binding for a given distribution parameter."""
        logger.debug("engine resolves random %s", name)

        dist = getattr(self.alloc, name)
        idx = self.layout.index(name)

        sample = self.njit(dist.param.sample())
        # XXX type casting for random
        if dist.param.arity == 1:
            @self.njit
            def rand(params):
                return sample(params[idx])

        elif dist.param.arity == 2:
            @self.njit
            def rand(params):
                return sample(params[idx], params[idx + 1])

        else:
            raise NotImplementedError

        return rand

    def resolve_derives(self, name, spec):
        """Return the binding for a give derived parameter."""
        logger.debug("engine resolves derives %s", name)
        idx = self.layout.index(name)

        fn, deps = spec
        args = [{n: [t() for t in ts]
                 for n, ts in typ.items()} if isinstance(typ, dict) else typ()
                for typ in deps.values()]
        result = fn(*args)

        if isinstance(result, list):
            result = np.array(result)

        if isinstance(result, np.ndarray):
            s = len(result.shape)
            typ = result.dtype

            if s == 1:
                @self.njit
                def der(params):
                    size = int(params[idx])
                    return params[idx + 1:idx + 1 + size]
            elif s == 2:
                @self.njit
                def der(params):
                    a = int(params[idx])
                    b = int(params[idx + 1])
                    return (params[(idx + 2):idx + 2 + a * b]
                            .copy().reshape((a, b)).astype(typ))
            else:
                raise NotImplementedError

        elif isinstance(result, tuple):
            s = len(result)

            @self.njit
            def der(params):
                return params[idx:idx + s]

        else:
            @self.njit
            def der(params):
                return params[idx]

        return der

    def resolve_function(self, function):
        """Return the binding for the given step."""
        logger.debug("engine resolves function %s", function.__name__)
        return self.njit(function)


class PseudoNumbaExecution(NumbaExecution):
    def __init__(self, model, alloc):
        super().__init__(model, alloc, use_gufunc=False)

    def njit(self, *args, **kws):
        def decorate(fn):
            return fn

        if len(args) == 1 and not kws:
            return decorate(args[0])

        return decorate

    def compile(self, impl, vectorize=True):
        if vectorize:
            return self.vjit([()], None)(impl)
        else:
            return self.njit(impl)


DefaultExecution = Execution = NumbaExecution


class TraceContext:
    """Context given to interact with the engine."""

    def __init__(self, engine):
        self.engine = engine

    def resolve_function(self, function):
        """Resolve a function called during simulation."""
        return self.engine.resolve_function(function)

    @property
    def state(self):
        """Provide access to the models state."""
        return self.engine.model.state

    @property
    def params(self):
        """Provide access to the models state."""
        return self.engine.model.params
