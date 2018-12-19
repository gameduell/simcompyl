"""Engine actually executes the simulations code."""

import numba as nb
import numpy as np
import pandas as pd

from functools import lru_cache
from collections import namedtuple
from contextlib import contextmanager

from .util import lazy


class Engine:
    def __init__(self):
        self.model = None
        self.alloc = None

    def invalidate(self):
        del self.init
        del self.iterate
        del self.apply
        del self.finish
        # TODO clear lru_cache

    def bind(self, model, alloc=None, compile=None):
        if model is not self.model or alloc is not self.alloc:
            self.invalidate()

        self.model = model
        self.alloc = alloc

        if compile is None and alloc is not None:
            compile = True
        if compile:
            self.init
            self.iterate
            self.apply
            self.finish

        return self

    def execute(self, traces):
        params = self.params()
        state = self.state()

        init = self.init
        iterate = self.iterate
        apply = self.apply
        finish = self.finish

        trs = [(tr.manager.publish, self.trace(tr)) for tr in traces]

        if trs:
            @nb.jit
            def trace(params, state):
                for p, t in trs:
                    p(t(params, state))
        else:
            @nb.jit
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
        with self.model.resolving(steps=self.resolve_steps,
                                  state=self.resolve_state,
                                  params=self.resolve_params,
                                  random=self.resolve_random,
                                  derives=self.resolve_derives):
            yield

    @lazy
    def init(self):
        with self.resolving():
            return self.compile(self.model.init(), vectorize=True)

    @lazy
    def iterate(self):
        with self.resolving():
            return self.compile(self.model.iterate(), vectorize=True)

    @lazy
    def apply(self):
        with self.resolving():
            return self.compile(self.model.apply(), vectorize=False)

    @lazy
    def finish(self):
        with self.resolving():
            return self.compile(self.model.finish(), vectorize=True)

    @lru_cache()
    def trace(self, trace):
        with self.model.resolving(steps=self.resolve_steps,
                                  state=self.resolve_state,
                                  params=self.resolve_params,
                                  random=self.resolve_random,
                                  derives=self.resolve_derives):
            return self.compile(nb.njit(trace.trace(self.model)), vectorize=False)

    def params(self):
        return self.alloc

    def state(self):
        shape = (self.alloc.n_samples.value, sum(len(v) if isinstance(v, list) else 1
                                                 for v in self.model.state.specs.values()))
        return np.zeros(shape, dtype=float)

    def frame(self, state):
        return pd.DataFrame(state, columns=sum([[n for _ in s] if isinstance(s, list) else [n]
                                                for n, s in self.model.state.specs.items()], []))

    def resolve_steps(self, impl, vectorize=True):
        return impl

    def resolve_state(self, name, typ):
        idx = sum(([n for _ in s] if isinstance(s, list) else [n]
                   for n, s in self.model.state.specs.items()), []).index(name)
        if isinstance(typ, list):
            return list(range(idx, idx+len(typ)))
        else:
            return idx

    def resolve_params(self, name, typ):
        print(name, typ)
        def getter(params):
            return getattr(params, name)
        return getter

    def resolve_random(self, name, typ):
        def getter(params):
            return getattr(params, name)
        return getter

    def resolve_derives(self, name, spec):
        fn, deps = spec

        def getter(params):
            return fn(*[getattr(params, name) for name in deps])

        return getter

    def compile(self, impl, vectorize=True):
        if vectorize:
            def vector(params, state):
                for i in range(state.shape[0]):
                    impl(params, state[i])
            return vector
        else:
            return impl


class NumbaEngine(Engine):
    def __init__(self, use_gufunc=True):
        super().__init__()
        self.use_gufunc = use_gufunc

    def compile(self, impl, vectorize=True):
        if vectorize and self.use_gufunc:
            return nb.guvectorize([(nb.float64[:], nb.float64[:])], '(m),(n)',
                                  cache=True,
                                  fastmath=True,
                                  nopython=True,
                                  target='parallel')(impl.py_func)
        elif vectorize and not self.use_gufunc:
            @nb.njit([(nb.float64[:], nb.float64[:, :])],
                     cache=True,
                     fastmath=True,
                     parallel=True)
            def vect(params, state):
                for i in nb.prange(len(state)):
                    impl(params, state[i])

            return vect
        else:
            return nb.njit([(nb.float64[:], nb.float64[:, :])],
                           cache=True,
                           fastmath=True,
                           parallel=True)(impl.py_func)

    def invalidate(self):
        super().invalidate()
        del self.layout

    @lazy
    def layout(self):
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
            layout.extend([name]*6)

        for name, (fn, deps) in self.model.derives.specs.items():
            args = [{n: [t() for t in ts] for n, ts in typ.items()} if isinstance(typ, dict) else typ()
                    for typ in deps.values()]
            result = fn(*args)

            if isinstance(result, np.ndarray):
                layout.extend([name] * len(result.shape))
                layout.extend([name] * len(result.ravel()))

            elif isinstance(result, tuple):
                layout.extend([name]*len(result))

            else:
                layout.append(name)

        return layout

    def params(self):
        params = []

        for name, spec in self.model.params.specs.items():
            val = getattr(self.alloc, name).value

            if isinstance(spec, dict):
                bound = len(next(iter(spec.values())))
                l = len(next(iter(val.values())))
                assert l <= bound

                params.append(l)
                for name in spec.keys():
                    params.extend(val[name])
                    params.extend([0] * (bound-len(val[name])))

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

            params.extend((ls + [0]*6)[:6])

        for name, (fn, deps) in self.model.derives.specs.items():
            spec = [{n: [t() for t in ts] for n, ts in typ.items()} if isinstance(typ, dict) else typ()
                    for typ in deps.values()]
            virt = fn(*spec)

            args = [getattr(self.alloc, name).value for name in deps.keys()]
            real = fn(*args)

            if isinstance(virt, np.ndarray):
                assert(len(virt.ravel()) >= len(real.ravel()))
                params.extend(real.shape)
                params.extend(real.ravel())
                params.extend([0]*(len(virt.ravel())-len(real.ravel())))

            elif isinstance(virt, tuple):
                assert len(virt) == len(real)
                params.extend(real)

            else:
                params.append(real)

        assert len(params) == len(self.layout)

        return np.array(params, dtype=float)

    def resolve_steps(self, name, impl):
        """Return the binding for the given StepInstance"""
        return nb.njit(impl)

    def resolve_params(self, name, typ):
        """Return the binding for a given parameter"""

        if isinstance(typ, dict):
            lx = self.layout.index(name)

            def define(idx, typs):
                @nb.njit
                def getter(params):
                    l = int(params[lx])
                    return params[idx:idx+l]
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
        """Return the binding for a given distribution parameter"""
        dist = getattr(self.alloc, name)
        idx = self.layout.index(name)

        # XXX type casting for random
        if dist.arity == 1:
            sample = nb.njit(dist.param.sample())
            @nb.njit
            def rand(params):
                return sample(params[idx])

        elif dist.arity == 2:
            sample = nb.njit(dist.param.sample())
            @nb.njit
            def rand(params):
                return sample(params[idx], params[idx+1])

        else:
            raise NotImplementedError

        return rand

    def resolve_derives(self, name, spec):
        idx = self.layout.index(name)

        fn, deps = spec
        args = [{n: [t() for t in ts] for n, ts in typ.items()} if isinstance(typ, dict) else typ()
                for typ in deps.values()]
        result = fn(*args)

        if isinstance(result, np.ndarray):
            s = len(result.shape)
            typ = result.dtype

            if s == 1:
                @nb.njit
                def der(params):
                    size = int(params[idx])
                    return params[idx+1:idx+1+size]
            elif s == 2:
                @nb.njit
                def der(params):
                    a = int(params[idx])
                    b = int(params[idx+1])
                    return params[idx+1:idx+1+a*b].copy().reshape((a, b)).astype(typ)
            else:
                raise NotImplementedError

        elif isinstance(result, tuple):
            s = len(result)

            @nb.njit
            def der(params):
                return params[idx:idx+s]

        else:
            @nb.njit
            def der(params):
                return params[idx]

        return der
