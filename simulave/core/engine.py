"""Engine actually executes the simulations code."""

import numba as nb
import numpy as np
import pandas as pd

from collections import namedtuple
from contextlib import contextmanager

from .util import lazy
from .trace import Frame


class BasicExecution:
    """Engine to execute a simulation defined by a model."""

    def __init__(self, model, alloc):
        self.model = model
        self.alloc = alloc
        self.traces = []

    def invalidate(self):
        """Clear up the cached simulation methods."""
        del self.init
        del self.iterate
        del self.apply
        del self.finish
        # TODO clear lru_cache

    @contextmanager
    def trace(self, *traces, target=Frame):
        """Activate given traces."""
        traces = [target(tr) for tr in traces]
        self.traces.extend(traces)

        yield (traces[0].prepare()
               if len(traces) == 1
               else [tr.prepare() for tr in traces])

        for tr in traces:
            tr.finalize()
        self.traces = self.traces[:-len(traces)]

    # @lru_cache()
    def tracing(self, trace):
        """Cache of a trace method."""
        ctx = TraceContext(self)

        with self.resolving():
            return self.compile(ctx.resolve_function(trace.trace(ctx)),
                                vectorize=False)

    def run(self, **allocs):
        """Execute the model."""
        # TODO more doc

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
            def trace(params, state):
                for p, t in trs:
                    p(t(params, state))
        else:
            def trace(params, state):
                pass

        init(params, state)
        trace(params, state)

        for i in range(self.alloc.n_steps.value):
            iterate(params, state)
            apply(params, state)
            trace(params, state)

        finish(params, state)

        return self.frame(state)

    @contextmanager
    def resolving(self):
        """Context where this engine should resolve accessors of the model."""
        with self.model.resolving(steps=self.resolve_steps,
                                  state=self.resolve_state,
                                  params=self.resolve_params,
                                  random=self.resolve_random,
                                  derives=self.resolve_derives):
            yield

    @lazy
    def init(self):
        """Cache of the initialization of the model."""
        with self.resolving():
            return self.compile(self.model.init(), vectorize=True)

    @lazy
    def iterate(self):
        """Cache of the iteration of the model."""
        with self.resolving():
            return self.compile(self.model.iterate(), vectorize=True)

    @lazy
    def apply(self):
        """Cache of the application of the model."""
        with self.resolving():
            return self.compile(self.model.apply(), vectorize=False)

    @lazy
    def finish(self):
        """Cache of the finalization of the model."""
        with self.resolving():
            return self.compile(self.model.finish(), vectorize=True)

    def params(self):
        """Parameters as passed to optimized simulation methods."""
        return self.alloc

    def state(self):
        """State as passed to optimized simulation methods."""
        shape = (self.alloc.n_samples.value,
                 sum(len(v) if isinstance(v, list) else 1
                     for v in self.model.state.specs.values()))
        return np.zeros(shape, dtype=float)

    def frame(self, state):
        """Create a dataframe for the given state representation."""
        columns = sum([[n for _ in s] if isinstance(s, list) else [n]
                       for n, s in self.model.state.specs.items()], [])
        return pd.DataFrame(state, columns=columns)

    def resolve_steps(self, name, impl):
        """Create an accessor for a step implementation method."""
        return impl

    def resolve_state(self, name, typ):
        """Create an accessor for parts of the state."""
        idx = sum(([n for _ in s] if isinstance(s, list) else [n]
                   for n, s in self.model.state.specs.items()), []).index(name)
        if isinstance(typ, list):
            return list(range(idx, idx + len(typ)))
        else:
            return idx

    def resolve_params(self, name, typ):
        """Create an accessor for a parameter value."""
        print(name, typ)

        def getter(params):
            return getattr(params, name)
        return getter

    def resolve_random(self, name, typ):
        """Create an accessor for a random distribution."""
        def getter(params):
            alloc = getattr(params, name)
            sample = alloc.param.sample()
            return sample(*alloc.args)
        return getter

    def resolve_derives(self, name, spec):
        """Create an accessor for a derivied parameter."""
        fn, deps = spec

        def getter(params):
            return fn(*[getattr(params, name) for name in deps])

        return getter

    def resolve_function(self, function):
        """Resolve a function that is called during simulation."""
        return self.resolve_steps(function.__name__, function)

    def compile(self, impl, vectorize=True):
        """Create a compiled version of a step implementation."""
        # TODO more doc
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

        /reParameters
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

        self.target = 'parallel' if parallel else 'cpu'

    def compile(self, impl, vectorize=True):
        """Use numba to compile the specified function."""
        if vectorize and self.use_gufunc:
            return nb.guvectorize([(nb.float64[:], nb.float64[:])], '(m),(n)',
                                  fastmath=self.fastmath,
                                  nopython=True,
                                  target=self.target)(impl.py_func)
        elif vectorize and not self.use_gufunc:
            @nb.njit([(nb.float64[:], nb.float64[:, :])],
                     fastmath=self.fastmath,
                     parallel=self.parallel)
            def vect(params, state):
                for i in nb.prange(len(state)):
                    impl(params, state[i])

            return vect
        else:
            return nb.njit([(nb.float64[:], nb.float64[:, :])],
                           fastmath=True,
                           parallel=True)(impl.py_func)

    def invalidate(self):
        """Invilidate all cached methods as well as the params layout cache."""
        super().invalidate()
        del self.layout

    @lazy
    def layout(self):
        """Cache for positioning parameters into an numpy array."""
        layout = []

        for name, spec in self.model.params.specs.items():
            if isinstance(spec, dict):
                bound = len(next(iter(spec.values())))
                # XXX check if all items define same bound
                layout.append(name)
                for n, ts in spec.items():
                    assert len(ts) == bound
                    layout.extend((name, n) for t in ts)
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
        params = []

        for name, spec in self.model.params.specs.items():
            val = getattr(self.alloc, name).value

            if isinstance(spec, dict):
                bound = len(next(iter(spec.values())))
                size = len(next(iter(val.values())))
                assert size <= bound

                params.append(size)
                for name in spec.keys():
                    params.extend(val[name])
                    params.extend([0] * (bound - len(val[name])))

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
        return nb.njit(impl)

    def resolve_params(self, name, typ):
        """Return the binding for a given parameter."""
        if isinstance(typ, dict):
            lx = self.layout.index(name)

            def define(idx, typs):
                @nb.njit
                def getter(params):
                    size = int(params[lx])
                    return params[idx:idx + size]
                return getter

            Getters = namedtuple('Getters', list(typ))
            getters = {n: define(self.layout.index((name, n)), ts)
                       for n, ts in typ.items()}
            return Getters(**getters)

        else:
            idx = self.layout.index(name)

            @nb.njit
            def getter(params):
                return typ(params[idx])
            return getter

    def resolve_random(self, name, typ):
        """Return the binding for a given distribution parameter."""
        dist = getattr(self.alloc, name)
        idx = self.layout.index(name)

        sample = nb.njit(dist.param.sample())
        # XXX type casting for random
        if dist.param.arity == 1:
            @nb.njit
            def rand(params):
                return sample(params[idx])

        elif dist.param.arity == 2:
            @nb.njit
            def rand(params):
                return sample(params[idx], params[idx + 1])

        else:
            raise NotImplementedError

        return rand

    def resolve_derives(self, name, spec):
        """Return the binding for a give derived parameter."""
        idx = self.layout.index(name)

        fn, deps = spec
        args = [{n: [t() for t in ts]
                 for n, ts in typ.items()} if isinstance(typ, dict) else typ()
                for typ in deps.values()]
        result = fn(*args)

        if isinstance(result, np.ndarray):
            s = len(result.shape)
            typ = result.dtype

            if s == 1:
                @nb.njit
                def der(params):
                    size = int(params[idx])
                    return params[idx + 1:idx + 1 + size]
            elif s == 2:
                @nb.njit
                def der(params):
                    a = int(params[idx])
                    b = int(params[idx + 1])
                    return (params[(idx + 1):idx + 1 + a * b]
                            .copy().reshape((a, b)).astype(typ))
            else:
                raise NotImplementedError

        elif isinstance(result, tuple):
            s = len(result)

            @nb.njit
            def der(params):
                return params[idx:idx + s]

        else:
            @nb.njit
            def der(params):
                return params[idx]

        return der


DefaultExecution = Execution = NumbaExecution


class TraceContext:
    """Context given to interact with the engine."""

    def __init__(self, engine):
        self.engine = engine

    def resolve_function(self, function):
        """Resolve a function called during simulation."""
        return self.engine.resolve_function(function)

    def state(self, **defs):
        """Provide access to the models state."""
        return self.engine.model.state(**defs)
