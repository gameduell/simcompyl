"""A trace defines how to keep track of statistics during the simulation."""

import pandas as pd
import numpy as np
import operator
import inspect

import time
import logging

from collections import namedtuple
from .util import lazy

logger = logging.getLogger(__name__)


def _defop(op, arity=None, reverse=False, reduce=False):
    """Create an implementation of an operation on a trace."""
    if arity is None:
        assert reduce is False

        def __applyop__(self, other):
            return BinaryOp(self, other, op=op)

        def __applyrop__(self, other):
            return BinaryOp(self, other, op=op, reverse=True)

    elif arity == 1:
        def __applyop__(self):
            return Apply0(self, op, reduce=reduce)

    elif arity == 2:
        def __applyop__(self):
            return Apply1(self, op, reduce=reduce)

    else:
        raise TypeError("Can't define an operator with {} args".format(arity))

    if reverse:
        assert arity is None
        return __applyop__, __applyrop__
    else:
        return __applyop__


def _freeze(*objs):
    if len(objs) == 1:
        obj, = objs

        if isinstance(obj, Trace):
            return obj.__freeze__()
        elif isinstance(obj, dict):
            return tuple((k, _freeze(obj[k])) for k in obj.keys())
        elif isinstance(obj, (list, tuple, np.ndarray)):
            return tuple(_freeze(val) for val in obj)
        else:
            return obj
    else:
        return _freeze(objs)


class Trace:
    """Abstract base class for trace transformations.

    A trace keeps track of some statistics of the data during the simulation.
    """

    def __new__(cls, *args, **kws):
        """Create a new trace."""
        if cls == Trace and not args:
            if all(isinstance(v, (type, list)) for v in kws.values()):
                logger.debug("trace creates as State for %s", kws)
                return State(**kws)

            if all(hasattr(v, '__call__') for v in kws.values()):
                logger.debug("trace creates as Assign for %s", kws)
                return Assign(**kws)

            msg = "Expecting a state spec or assignments as keywords."
            raise TypeError(msg)

        return super().__new__(cls)

    def __init__(self, *sources, columns=None, index=None, label=None):
        """Create transformation with an given source and output columns."""
        if sources:
            if columns is None:
                columns = sources[0].columns
            if index is None:
                index = sources[0].index
            if label is None and sources[0].label != 'value':
                label = sources[0].label

        label = label or (columns[0] if len(columns) == 1 else 'value')
        logger.debug("trace initialized tracing %s %s from %s",
                     columns, label, sources)
        self.label = label
        self.sources = sources
        self.index = index
        self.columns = columns

    def __freeze__(self):
        return _freeze(self.sources,
                       self.columns,
                       None if self.index is None else list(self.index),
                       self.label)


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
            return source(params, raw)
        return impl

    def naming(self, *names):
        """Name the columns of the trace."""
        if len(names) != len(self.columns):
            raise ValueError("Supplied names should match length of columns.")
        return Trace(self, columns=names)

    def values(self, label):
        return Trace(self, label=label)

    def assign(self, **assings):
        """Assign new variables by evaluating passed functions on the source."""
        return Assign(self, **assings)

    def __getattr__(self, name):
        """Accessor for a single column of the trace."""
        if name in self.columns:
            return Columns(self, [name])

        raise AttributeError("{!r} object has no {!r} attribute."
                             .format(type(self).__name__, name))

    def __getitem__(self, item):
        """Subselection of slice or columns.

        Parameters
        ----------
        item : slice or list
            - slice: sub-selection of a slice on samples
            - list: sub-selection on the columns of the trace
        """
        if isinstance(item, str):
            return Columns(self, [item])
        elif isinstance(item, slice):
            return Slice(self, item)
        elif isinstance(item, list):
            return Columns(self, item)
        elif isinstance(item, Trace):
            return Filter(self, item)
        else:
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

    def quantile(self, qs=(.1, .25, .5, .75, .9)):
        """Trace quantiles over the data."""
        ps = np.array(qs) * 100
        return Apply1(self, np.percentile, ps,
                      reduce=len(ps),
                      index=pd.Index(qs, name='quantile'))

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
            return raw[:, ixs]
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

        if self.length == 1:
            def base(fn):
                @ctx.resolve_function
                def impl(params, _):
                    return np.array([fn(params)])
                return impl
        else:
            def base(fn):
                @ctx.resolve_function
                def impl(params, _):
                    raw = list(fn(params))
                    return np.array(raw).reshape(1, len(raw))
                return impl


        def reduce(fn, *fns):
            if not fns:
                return base(fn)
            else:
                first = base(fn)
                rest = reduce(*fns)

                @ctx.resolve_function
                def impl(params, _):
                    rs = rest(params, _)
                    return np.concatenate((first(params, _), rs))

                return impl

        red = reduce(*fns)

        if self.length > 1:
            def impl(params, state):
                return red(params, state).T
        else:
            def impl(params, state):
                return red(params, state)

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
            def impl(pr, raw):
                ins = src(pr, raw)

                new = np.empty((len(ins), 1))
                new[:, 0] = call(ins)
                # XXX Perhaps we should do allocation to avoid concat rep.
                return np.concatenate((ins, new), axis=1)
            return impl

        source, = self.traces(ctx)
        for assign in self.assigns.values():
            params = inspect.signature(assign).parameters
            idx = np.array([self.sources[0].columns.index(p) for p in params])
            source = bound(source, idx, ctx.resolve_function(assign))

        sel = self.select

        def select(pr, raw):
            ins = source(pr, raw)
            return ins[:, sel:]

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
                src = source(params, raw)
                pred = predicate(params, raw).astype(np.bool_)
                return src[pred[:, 0]]

        elif len(self.predicate.columns) == len(self.source.columns):
            def impl(params, raw):
                return source(params, raw)[predicate(params, raw) != 0]

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
            def apply(a, b):
                return op(b, a)
        else:
            @ctx.resolve_function
            def apply(a, b):
                return op(a, b)

        if isinstance(other, pd.Series):
            other = other.loc[self.columns]

        if isinstance(other, (pd.Series, np.ndarray, list)):
            other = np.array(other)
            source, = self.traces(ctx)

            def impl(params, raw):
                ins = source(params, raw)
                m, n = ins.shape
                result = np.empty((m, n), dtype=dtype)

                for i in range(n):
                    result[:, i] = apply(ins[:, i], other[i])
                return result

        elif isinstance(other, Trace):
            fst, snd = self.traces(ctx)

            def impl(params, raw):
                a, b = fst(params, raw), snd(params, raw)
                m, n = a.shape

                result = np.empty((m, n))
                for i in range(n):
                    result[:, i] = apply(a[:, i], b[:, i])
                return result

        else:
            source, = self.traces(ctx)

            def impl(params, raw):
                return apply(source(params, raw), other)

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
            ins = source(params, raw)
            m, n = ins.shape
            if reduce:
                m = reduce

            result = np.empty((m, n))
            for i in range(n):
                result[:, i] = apply(ins[:, i])
            return result
        return impl


class Apply0(Apply):
    """Apply a 0-arg method on each column of the trace."""

    def apply(self):
        """Return implmentation applying the 0-arg method."""
        method = self.method

        def impl(raw):
            return method(raw)
        return impl


class Apply1(Apply):
    """Apply a 1-arg method on each column of the trace."""

    def apply(self):
        """Return implmentation applying the 1-arg method."""
        method = self.method
        arg, = self.args

        def impl(raw):
            return method(raw, arg)
        return impl


class Columns(Trace):
    """Subselection on columns."""

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
            return source(params, raw)[:, ixs]
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
            ins = source(params, raw)
            return ins[start:stop:step, :]
        return impl


class Frame:
    """Create pandas dataframes out of traces."""
    data = None

    def __init__(self, trace, skip=1):
        self.trace = trace
        self.skip = skip

    def frame(self, traces=(), offset=0):
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

        idx = pd.MultiIndex.from_product([trx, srx],
                                         names=(trx.name, srx.name))

        if not traces:
            return pd.DataFrame(0, index=idx, columns=columns)

        return pd.DataFrame(np.concatenate(traces),
                            index=idx,
                            columns=columns)

    def prepare(self):
        """Prepare the internals, called before running a new simulation."""
        self.traces = []
        self.data = self.frame().iloc[:0]
        self.count = 0
        return self.data

    def publish(self, trace):
        """Handle the trace data called when it becomes available."""
        if (self.count % self.skip) == 0:
            self.traces.append(trace)
        self.count += 1

    def finalize(self):
        """Take care to finalize all the data given to the manager."""
        df = self.frame(self.traces)
        self.data[df.columns] = df
        return df

    def __str__(self):
        """Show class alongside the trace."""
        return "{}({})".format(type(self).__name__, self.trace)

    def __repr__(self):
        """Show class alongside the trace."""
        return "{}({!r})".format(type(self).__name__, self.trace)


class Holotrace(Frame):
    """Publish traces to a `holoviews.Buffer`."""

    def __init__(self, trace, skip=1, batch=1, timeout=None):
        super().__init__(trace, skip=skip)

        import holoviews as hv
        assert hv

        self.batch = batch
        self.timeout = timeout
        self.last = None

    @lazy
    def buffer(self):
        """Get the `holoviews.Buffer` where the data will be published to."""
        from holoviews import streams
        return streams.Buffer(self.frame(),
                              index=False,
                              length=np.iinfo(int).max)

    def prepare(self):
        """Clear and return the holoviews trace buffer."""
        self.buffer.clear()
        self.traces = []
        self.count = 0
        self.last = time.time()
        self.offset = 0
        return self.buffer

    @property
    def data(self):
        """Access the data inside the buffer."""
        return self.buffer.data

    def push(self):
        """Push remaining traces towards the holoviews buffer."""
        if self.traces:
            df = self.frame(self.traces, offset=self.offset)

            self.offset += len(self.traces) * self.skip
            self.traces = []

            self.buffer.send(df)
            self.last = time.time()

    def publish(self, trace):
        """
        Publish traced data.

        @param trace raw numpy trace data
        """
        if self.count % self.skip == 0:
            self.traces.append(trace)

        if len(self.traces) >= self.batch:
            self.push()

        elif (self.timeout is not None
                and self.traces
                and time.time() - self.last > self.timeout):
            self.push()

        self.count += 1


    def finalize(self):
        """Ensure that all traces end up in the holoviews buffer."""
        self.push()

    def dataset(self, data=None):
        """Return the buffers data as a holoviews Dataset."""
        import holoviews as hv
        if data is None:
            data = self.buffer.data
        data = (data
                .stack(dropna=False)
                .swaplevel(-1, -2)
                .to_frame(name=self.trace.label))
        return hv.Dataset(data,
                          kdims=list(data.index.names),
                          vdims=[self.trace.label])

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
