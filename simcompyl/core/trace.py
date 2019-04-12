"""A trace defines how to keep track of statistics during the simulation."""
import operator
import inspect

import time
import logging
from itertools import product, chain

import pandas as pd
import numpy as np

LOG = logging.getLogger(__name__)

__all__ = ['Trace', 'Frame', 'Holotrace']


def _defop(op, arity=None, reverse=False, reduce=False):
    """Create an implementation of an operation on a trace."""
    if arity is None:
        assert reduce is False

        def __applyop__(self, other):
            return BinaryOp(self, other, op=op)

        def __applyrop__(self, other):
            return BinaryOp(self, other, op=op, reverse=True)

    else:
        assert arity == 1

        def __applyop__(self):
            return Apply0(self, op, reduce=reduce)

    if reverse:
        assert arity is None
        return __applyop__, __applyrop__
    return __applyop__


def _freeze(*objs):
    if len(objs) == 1:
        obj, = objs

        if isinstance(obj, Trace):
            return obj.__freeze__()
        if isinstance(obj, dict):
            return tuple((k, _freeze(obj[k])) for k in obj.keys())
        if isinstance(obj, (list, tuple, np.ndarray)):
            return tuple(_freeze(val) for val in obj)
        return obj

    return _freeze(objs)


class Trace:
    """Abstract base class for trace transformations.

    A trace keeps track of some statistics of the data during the simulation.
    """

    def __new__(cls, *args, **kws):
        """Create a new trace."""
        if cls == Trace and not args:
            if all(isinstance(v, (type, list)) for v in kws.values()):
                LOG.debug("trace creates as State for {}", kws)
                return State(**kws)

            if all(hasattr(v, '__call__') and not isinstance(v, type)
                   for v in kws.values()):
                LOG.debug("trace creates as Assign for {}", kws)
                return Assign(**kws)

            msg = "Expecting a state spec or assignments as keywords."
            raise TypeError(msg)

        return super().__new__(cls)

    def __init__(self, *sources, columns=None, index=None, name=None):
        """Create transformation with an given source and output columns."""
        if sources:
            if columns is None:
                columns = sources[0].columns
            if index is None:
                index = sources[0].index
            if name is None and sources[0].name != 'value':
                name = sources[0].name

        name = name or (columns[0] if len(columns) == 1 else 'value')
        LOG.debug("trace initialized tracing {} {} from {}",
                  columns, name, sources)
        self.name = name
        self.sources = sources
        self.index = index
        self.columns = columns

    def __freeze__(self):
        """Create a hashable representation of self."""
        # XXX: we can't use hash & equal as we redefine for algebra
        return _freeze(self.sources,
                       self.columns,
                       None if self.index is None else list(self.index),
                       self.name)

    def to(self, target=None, *args, **kws):
        """Make an instance of the target class with this traces."""
        if target is None:
            target = Frame
        return target(self, *args, **kws)

    def traces(self, ctx):
        """Tracing methods of the sources."""
        return [ctx.resolve_function(tr.trace(ctx)) for tr in self.sources]

    def trace(self, ctx):
        """Return numba compatible implementation method."""
        source, = self.traces(ctx)

        def impl(params, raw):
            idx, values = source(params, raw)
            return idx, values
        return impl

    def naming(self, *names):
        """Name the columns of the trace."""
        if len(names) != len(self.columns):
            raise ValueError("Supplied names should match length of columns.")
        return Trace(self, columns=names)

    def label(self, name):
        """Give a label to the values being traced."""
        return Trace(self, name=name)

    def assign(self, **assings):
        """Assign new variables by evaluating functions on the source."""
        return Assign(self, **assings)

    def __getattr__(self, name):
        """Accessors for a single column of the trace."""
        if name in self.columns:
            return Columns(self, [name])

        raise AttributeError("{!r} object has no {!r} attribute."
                             .format(type(self).__name__, name))

    def __getitem__(self, item):
        """Sub-selection of slice or columns.

        Parameters
        ----------
        item : slice or list
            - slice: sub-selection of a slice on samples
            - list: sub-selection on the columns of the trace
        """
        if isinstance(item, str):
            return Columns(self, [item])
        if isinstance(item, slice):
            return Slice(self, item)
        if isinstance(item, list):
            return Columns(self, item)
        if isinstance(item, Trace):
            return Filter(self, item)

        msg = "Getting item with a {}".format(type(item))
        raise NotImplementedError(msg)

    __eq__ = _defop(operator.eq)
    __ne__ = _defop(operator.ne)
    __lt__ = _defop(operator.lt)
    __le__ = _defop(operator.le)
    __gt__ = _defop(operator.gt)
    __ge__ = _defop(operator.ge)

    __inv__ = _defop(operator.inv)
    __or__ = _defop(operator.or_)
    __and__ = _defop(operator.and_)
    __xor__ = _defop(operator.xor)

    __pos__ = _defop(operator.pos, 1)
    __neg__ = _defop(operator.neg, 1)
    __bool__ = _defop(operator.truth, 1)
    __abs__ = _defop(operator.abs, 1)

    __add__, __radd__ = _defop(operator.add, reverse=True)
    __sub__, __rsub__ = _defop(operator.sub, reverse=True)
    __mul__, __rmul__ = _defop(operator.mul, reverse=True)
    __mod__, __rmod__ = _defop(operator.mod, reverse=True)
    __truediv__, __rtruediv__ = _defop(operator.truediv, reverse=True)
    __floordiv__, __rfloordiv__ = _defop(operator.floordiv, reverse=True)
    __pow__, __rpow__ = _defop(operator.pow, reverse=True)
    __lshift__, __rlshift__ = _defop(operator.lshift, reverse=True)
    __rshift__, __rrshift__ = _defop(operator.rshift, reverse=True)

    def take(self, num=5):
        """Take the specified number of samples form the data."""
        return self[:num]

    sum = _defop(np.sum, arity=1, reduce=True)
    mean = _defop(np.mean, arity=1, reduce=True)
    median = _defop(np.median, arity=1, reduce=True)

    def quantile(self, quantiles=(.1, .25, .5, .75, .9)):
        """Trace quantiles over the data."""
        ps = np.array(quantiles) * 100
        return Apply1(self, np.percentile, ps,
                      reduce=len(ps),
                      index=pd.Index(quantiles, name='quantile'))

    def __str__(self):
        """Show content of the trace."""
        return "{}[{}]".format(type(self).__name__,
                               ",".join(map(str, self.columns)))

    __repr__ = __str__


class State(Trace):
    """Trace some part of the state."""

    def __init__(self, **specs):
        columns = []
        for name, spec in specs.items():
            if isinstance(spec, list):
                columns.extend(["{}.{}".format(name, i)
                                for i in range(len(spec))])
            else:
                columns.append(name)
        super().__init__(columns=columns)
        self.specs = specs

    def __freeze__(self):
        return super().__freeze__() + _freeze(self.specs)

    def trace(self, ctx):
        """Return implementation that selects columns on the state."""
        ixs = np.array([ix if isinstance(ix, list) else [ix]
                        for ix in ctx.state(True, **self.specs)]).ravel()

        def impl(_, raw):
            return np.arange(len(raw)), raw[:, ixs]
        return impl


class Param(Trace):
    """Bring some parameter into the trace algebra."""

    def __init__(self, **specs):
        columns = []
        length = None

        for name, spec in specs.items():
            if isinstance(spec, dict):
                for key, types in spec.items():
                    if length and len(types) != length:
                        msg = "Inconsistent trace length."
                        raise ValueError(msg)
                    length = len(types)
                    columns.append("{}.{}".format(name, key))
            else:
                columns.append(name)
                length = 1
        super().__init__(columns=columns)

        self.specs = specs
        self.length = length

    def __freeze__(self):
        return super().__freeze__() + _freeze(self.specs,)

    def trace(self, ctx):
        """Return implementation that selects parameters."""
        fns = []
        for param in ctx.params(True, **self.specs):
            prs = param if isinstance(param, tuple) else [param]
            for pr in prs:
                fns.append(pr)
        fns = tuple(fns)
        length = self.length

        if self.length == 1:
            def base(fn):
                @ctx.resolve_function
                def impl(params, _):
                    return np.arange(length), np.array([fn(params)])
                return impl
        else:
            def base(fn):
                @ctx.resolve_function
                def impl(params, _):
                    raw = list(fn(params))
                    return (np.arange(length),
                            np.array(raw).reshape(1, len(raw)))
                return impl

        def reduce(*fns):
            if len(fns) == 1:
                return base(*fns)

            fn, *fns = fns
            first = base(fn)
            rest = reduce(*fns)

            @ctx.resolve_function
            def impl(params, _):
                idx, hd = first(params, _)
                _, rs = rest(params, _)
                return idx, np.concatenate((hd, rs))

            return impl

        red = reduce(*fns)

        if self.length > 1:
            def impl(params, state):
                idx, res = red(params, state)
                return idx, res.T
        else:
            def impl(params, state):
                idx, res = red(params, state)
                return idx, res

        return impl


class Assign(Trace):
    """Assign a new columns with values from supplied functions."""

    def __init__(self, source=None, **assigns):
        required = set()
        outputs = list(assigns.keys())
        for assign in assigns.values():
            sig = inspect.signature(assign)
            required.update(sig.parameters)
        columns = list(required)

        if source is None:
            source = State(**{c: ... for c in columns})
            sel = len(columns)
            columns = outputs
        else:
            sel = 0
            columns = source.columns + outputs

        missing = set(source.columns) - required
        if missing:
            msg = "Input of assign is missing the following columns: {}"
            raise ValueError(msg.format(", ".join(missing)))

        super().__init__(source, columns=columns)
        self.assigns = assigns
        self.select = sel

    def __freeze__(self):
        return super().__freeze__() + _freeze(self.assigns,)

    def trace(self, ctx):
        """Add assignments to the trace."""
        def bound(src, ix, fn):
            if len(ix) == 1:
                @ctx.resolve_function
                def call(raw):
                    return fn(raw[:, ix[0]])
            elif len(ix) == 2:
                @ctx.resolve_function
                def call(raw):
                    return fn(raw[:, ix[0]], raw[:, ix[1]])
            else:
                msg = "Assign with {} args not supported currently."
                raise NotImplementedError(msg.format(len(ix)))

            @ctx.resolve_function
            def impl(params, raw):
                idx, ins = src(params, raw)

                new = np.empty((len(ins), 1))
                new[:, 0] = call(ins)
                # XXX Perhaps we should do allocation to avoid concat rep.
                return idx, np.concatenate((ins, new), axis=1)
            return impl

        source, = self.traces(ctx)
        for assign in self.assigns.values():
            params = inspect.signature(assign).parameters
            idx = np.array([self.sources[0].columns.index(p) for p in params])
            source = bound(source, idx, ctx.resolve_function(assign))

        sel = self.select

        def select(params, raw):
            idx, ins = source(params, raw)
            return idx, ins[:, sel:]

        return select


class Filter(Trace):
    """Filtering along the samples using a predicate trace."""

    def __init__(self, source, predicate):
        super().__init__(source, predicate)

    @property
    def source(self):
        """Get the source to the filter."""
        return self.sources[0]

    @property
    def predicate(self):
        """Get the predicate of the filter."""
        return self.sources[1]

    def trace(self, ctx):
        """Return implementation applying the predicate."""
        source, predicate = self.traces(ctx)

        if len(self.predicate.columns) == 1:
            def impl(params, raw):
                idx, src = source(params, raw)
                _, pred = predicate(params, raw)
                pred = pred.astype(np.bool_)
                return idx[pred[:, 0]], src[pred[:, 0]]

        elif len(self.predicate.columns) == len(self.source.columns):
            def impl(params, raw):
                idx, src = source(params, raw)
                _, pred = predicate(params, raw)
                pred = pred.astype(np.bool_)
                return idx[pred], src[pred]

        else:
            raise ValueError("Sizes of predicate not match source.")

        return impl


class BinaryOp(Trace):
    """Apply a operator on the trace."""

    def __init__(self, source, other, op, reverse=False):

        if isinstance(other, Trace):
            if (source.columns != other.columns
                    and (len(source.columns) != 1 or len(other.columns) != 1)):
                raise ValueError("Columns to binary operator don't match!")
            super().__init__(source, other)
        else:
            super().__init__(source)
        self.other = other
        self.op = op
        self.reverse = reverse

    def __freeze__(self):
        return super().__freeze__() + _freeze(self.other, self.op)

    def trace(self, ctx):
        """Return implementation applying the binary op."""
        other = self.other
        op = self.op
        dtype = op(np.array([]), np.array([])).dtype

        if self.reverse:
            @ctx.resolve_function
            def apply(fst, snd):
                return op(snd, fst)
        else:
            @ctx.resolve_function
            def apply(fst, snd):
                return op(fst, snd)

        if isinstance(other, pd.Series):
            other = other.loc[self.columns]

        if isinstance(other, (pd.Series, np.ndarray, list)):
            other = np.array(other)
            source, = self.traces(ctx)

            def impl(params, raw):
                idx, ins = source(params, raw)
                m, n = ins.shape
                result = np.empty((m, n), dtype=dtype)

                for i in range(n):
                    result[:, i] = apply(ins[:, i], other[i])
                return idx, result

        elif isinstance(other, Trace):
            fst, snd = self.traces(ctx)

            def impl(params, raw):
                idx, a = fst(params, raw)
                _, b = snd(params, raw)
                m, n = a.shape

                result = np.empty((m, n))
                for i in range(n):
                    result[:, i] = apply(a[:, i], b[:, i])
                return idx, result

        else:
            source, = self.traces(ctx)

            def impl(params, raw):
                idx, src = source(params, raw)
                return idx, apply(src, other)

        return impl


class Apply(Trace):
    """Transformation applying a given function."""

    def __init__(self, source, method, *args, reduce=False, index=None):
        if reduce is True and index is None:
            index = pd.Index([method.__name__], name='method')
        super().__init__(source, index=index)
        self.method = method
        self.args = args
        self.reduce = reduce

    def __freeze__(self):
        return super().__freeze__() + _freeze(self.method, self.args)

    def trace(self, ctx):
        """Return implementation applying a method on each column."""
        apply = ctx.resolve_function(self.apply())
        reduce = int(self.reduce)
        source, = self.traces(ctx)

        def impl(params, raw):
            idx, ins = source(params, raw)
            m, n = ins.shape
            if reduce:
                # TODO really just use a range
                idx = np.arange(reduce)
                m = reduce

            result = np.empty((m, n))
            for i in range(n):
                result[:, i] = apply(ins[:, i])
            return idx, result
        return impl


class Apply0(Apply):
    """Apply a 0-arg method on each column of the trace."""
    def apply(self):
        """Return implementation applying the 0-arg method."""
        method = self.method
        assert not self.args

        def impl(raw):
            return method(raw)
        return impl


class Apply1(Apply):
    """Apply a 1-arg method on each column of the trace."""

    def apply(self):
        """Return implementation applying the 1-arg method."""
        method = self.method
        assert len(self.args) == 1
        arg, = self.args

        def impl(raw):
            return method(raw, arg)
        return impl


class Columns(Trace):
    """Sub-selection on columns."""

    def __init__(self, source, columns):
        if not all(c in source.columns for c in columns):
            missing = set(columns) - set(source.columns)
            raise KeyError("{} missing in source {}".format(
                ", ".join(map(str, missing)),
                ", ".join(map(str, source.columns))))
        super().__init__(source, columns=columns)

    @property
    def source(self):
        """Get the source of the column operation."""
        return self.sources[0]

    def trace(self, ctx):
        """Return implementation selecting columns."""
        ixs = np.array([self.source.columns.index(c) for c in self.columns])
        source, = self.traces(ctx)

        def impl(params, raw):
            idx, src = source(params, raw)
            return idx, src[:, ixs]
        return impl


class Slice(Trace):
    """Select a slice of the data."""

    def __init__(self, source, select, index=None):
        if index is None:
            if source.index:
                index = source.index[select]
            elif (select.stop or -1) >= 0:
                index = pd.RangeIndex(select.start, select.stop, select.step,
                                      name='sample')
        super().__init__(source, index=index)
        self.select = select

    def __freeze__(self):
        return super().__freeze__() + _freeze(self.select.start,
                                              self.select.stop,
                                              self.select.step)

    def trace(self, ctx):
        """Return implementation selecting a slice."""
        start = self.select.start or 0
        stop = self.select.stop
        step = self.select.step or 1
        source, = self.traces(ctx)

        def impl(params, raw):
            idx, ins = source(params, raw)
            return idx[start:stop:step], ins[start:stop:step, :]
        return impl


class Tracer:
    """Base class for publishing data for traces."""
    buffer = None

    def __init__(self, trace, skip=1):
        self.trace = trace
        self.skip = skip
        self.traces = []
        self.idxs = []
        self.count = 0

    def frame(self, idxs=(), traces=(), offset=0):
        """Create a Dataframe out of the given traces."""
        columns = pd.Index(self.trace.columns, name='variable')

        if self.trace.index is None:
            srx = pd.RangeIndex(len(traces[0]) if traces else 1,
                                name='sample')
        else:
            srx = self.trace.index

        trx = pd.RangeIndex(offset,
                            offset + (len(traces) or 1) * self.skip,
                            self.skip,
                            name='trace')
        if self.trace.index is not None:
            idxs = [self.trace.index for _ in idxs or [0]]
        elif not idxs:
            idxs = [[0]]

        idx = pd.MultiIndex.from_tuples(chain(*(product([t], idx)
                                                for t, idx in zip(trx, idxs))),
                                        names=(trx.name, srx.name))
        if traces:
            df = pd.DataFrame(np.concatenate(traces),
                              index=idx,
                              columns=columns)
        else:
            df = pd.DataFrame(0, index=idx, columns=columns)

        LOG.debug("created {}-frame of {:d} traces for {}",
                  'x'.join(str(n) for n in df.shape), len(traces), self)

        return df

    def prepare(self):
        """Prepare the internals, called before running a new simulation."""
        self.traces = []
        self.idxs = []
        self.count = 0

    @property
    def data(self):
        """Return a dataframe containing published data."""
        return self.frame()

    def publish(self, idx, trace):
        """Handle the trace data called when it becomes available."""
        if (self.count % self.skip) == 0:
            self.traces.append(trace)
            self.idxs.append(idx)
        self.count += 1

    def finalize(self):
        """Take care to finalize all the data given to the manager."""

    def __str__(self):
        """Show class alongside the trace."""
        return "{}({})".format(type(self).__name__, self.trace)

    def __repr__(self):
        """Show class alongside the trace."""
        return "{}({!r})".format(type(self).__name__, self.trace)


class Frame(Tracer):
    """Publishes data into a dataframe"""
    def prepare(self):
        super().prepare()
        self.buffer = self.frame().iloc[:0]
        return self.buffer

    @property
    def data(self):
        return self.buffer

    def finalize(self):
        df = self.frame(self.idxs, self.traces)
        self.data[df.columns] = df
        return df


class Holotrace(Tracer):
    """Publish traces to a `holoviews.Buffer`."""

    def __init__(self, trace, skip=1, batch=1, timeout=None):
        super().__init__(trace, skip=skip)

        from holoviews import streams

        self.batch = batch
        self.timeout = timeout
        self.last = None
        self.offset = 0
        self.buffer = streams.Buffer(self.frame(),
                                     index=False,
                                     length=np.iinfo(int).max)

    def prepare(self):
        """Clear and return the holoviews trace buffer."""
        super().prepare()
        self.buffer.clear()
        self.last = time.time()
        self.offset = 0
        return self.buffer

    @property
    def data(self):
        """Access the data inside the buffer."""
        return self.buffer.data

    def push(self):
        """Push remaining traces towards the holoviews buffer."""
        LOG.debug("{} pushes {:d} traces.", self, len(self.traces))
        if self.traces:
            df = self.frame(self.idxs, self.traces, offset=self.offset)

            self.offset += len(self.traces) * self.skip
            self.traces = []

            self.buffer.send(df)
            self.last = time.time()

    def publish(self, idx, trace):
        """
        Publish traced data.

        @param trace raw numpy trace data
        """
        super().publish(idx, trace)

        if len(self.traces) >= self.batch:
            self.push()

        elif (self.timeout is not None
              and self.traces
              and time.time() - self.last > self.timeout):
            self.push()

    def finalize(self):
        """Ensure that all traces end up in the holoviews buffer."""
        self.push()

    def dataset(self, data):
        """Return the buffers data as a holoviews Dataset."""
        import holoviews as hv
        data = (data
                .stack(dropna=False)
                .swaplevel(-1, -2)
                .to_frame(name=self.trace.name))
        return hv.Dataset(data,
                          kdims=list(data.index.names),
                          vdims=[self.trace.name])

    def plot(self, obj, *args, **kws):
        """Plot the trace data using a holoview object."""
        import holoviews as hv
        if isinstance(obj, type) and issubclass(obj, hv.element.chart.Chart):
            def plotting(data):
                if data.empty:
                    data = self.frame()
                return self.dataset(data).to(obj, *args, **kws).overlay()
        else:
            def plotting(data):
                if data.empty:
                    data = self.frame()
                return obj(self.dataset(data), *args, **kws)
        return hv.DynamicMap(plotting, streams=[self.buffer])
