"""A trace defines how to keep track of statistics during the simulation."""

import pandas as pd
import numpy as np
import operator
import inspect

import time

from .util import lazy


def _defop(op, arity=2, reverse=False):
    """Create an implementation of an operation on a trace."""
    if arity == 1:
        def __applyop__(self):
            return Apply(self, op)

    elif arity == 2:
        def __applyop__(self, other):
            return BinaryOp(self, other)

        def __applyrop__(self, other):
            return BinaryOp(self, other, op=op, reverse=True)

    else:
        raise TypeError("Can't define an operator with {} args".format(arity))

    if reverse:
        return __applyop__, __applyrop__
    else:
        return __applyop__


class Trace:
    """Abstract base class for trace transformations.

    A trace keeps track of some statistics of the data during the simulation.
    """

    def __new__(cls, *args, **kws):
        """Create a new trace."""
        input = args[0] if args else None
        columns = kws.get('columns', input)
        mods = dict(kws)
        mods.pop('skip', None)

        if cls == Trace:
            if input is None and columns is None and mods:
                return Assign(*args, **kws)

            elif (isinstance(columns, list)
                  and all(isinstance(c, str) for c in columns)
                  and not kws
                  and (input is None or input is columns)):
                return Source(*args, **kws)
        return super().__new__(cls)

    def __init__(self, *inputs, columns=None, skip=None):
        """Create transformation with an given input and output columns."""
        if columns is None and inputs:
            columns = inputs[0].columns
        if skip is None and inputs:
            skip = inputs[0].skip
        self.inputs = inputs
        self.columns = columns
        self.skipping = skip

    def traces(self, ctx):
        """Tracing methods of the inputs."""
        return [ctx.resolve_function(tr.trace(ctx)) for tr in self.inputs]

    def trace(self, ctx):
        """Return numba compatible implementation method."""
        input, = self.traces(ctx)

        def impl(params, raw):
            return input(params, raw)
        return impl

    def __frame__(self, traces):
        """Create a pandas dataframe for the given transformed traces."""
        return pd.DataFrame(traces, columns=self.columns)

    def name(self, *names):
        """Name the columns of the trace."""
        if len(names) != len(self.columns):
            raise ValueError("Supplied names should match length of columns.")
        return Trace(self, columns=names)

    def skip(self, n=10):
        """Only take every n-th step as a trace."""
        return Trace(self, skip=n)

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

    sum = _defop(np.sum, arity=1)
    mean = _defop(np.mean, arity=1)
    median = _defop(np.median, arity=1)

    def quantile(self, qs=[.1, .25, .5, .75, .9]):
        """Trace quantiles over the data."""
        ps = np.array(qs) * 100
        return Apply1(self, np.percentile, ps, reduce=len(ps))


class Source(Trace):
    """Source for trace transformations, selecting columns of the state."""

    def __init__(self, columns, skip=None):
        assert isinstance(columns, list)
        assert all(isinstance(c, str) for c in columns)
        super().__init__(columns=columns, skip=skip)

    def trace(self, ctx):
        """Return implementation that selects columns on the state."""
        ixs = np.array(ctx.state(**{c: ... for c in self.columns}))

        def impl(params, raw):
            return raw[:, ixs]
        return impl


class Assign(Trace):
    """Assign a new columns with values from supplied functions."""

    def __init__(self, input=None, skip=None, **assigns):
        required = set()
        outputs = list(assigns.keys())
        for assign in assigns.values():
            sig = inspect.signature(assign)
            required.update(sig.parameters)
        columns = list(required)

        if input is None:
            input = Source(columns, skip=skip)
            sel = len(columns)
            columns = outputs
        else:
            sel = 0
            columns = input.columns + outputs

        missing = set(input.columns) - required
        if missing:
            msg = "Input of assign is missing the following columns: {}"
            raise ValueError(msg.format(", ".join(missing)))

        super().__init__(input, columns=columns)
        self.assigns = assigns
        self.select = sel

    def trace(self, ctx):
        """Add assignments to the trace."""
        def bound(input, idx, fn):
            if len(idx) == 1:
                @ctx.resolve_function
                def call(raw):
                    return fn(raw[:, idx[0]])
            elif len(idx) == 2:
                @ctx.resolve_function
                def call(raw):
                    return fn(raw[:, idx[0]], raw[:, idx[1]])
            else:
                msg = "Assign with {} args not supported currently."
                raise NotImplementedError(msg.format(len(idx)))

            @ctx.resolve_function
            def impl(params, raw):
                ins = input(params, raw)

                new = np.empty((len(ins), 1))
                new[:, 0] = call(ins)
                # XXX Perhaps we should do allocation to avoid concat rep.
                return np.concatenate((ins, new), axis=1)
            return impl

        input, = self.traces(ctx)
        for assign in self.assigns.values():
            params = inspect.signature(assign).parameters
            idx = np.array([self.inputs[0].columns.index(p) for p in params])
            fn = ctx.resolve_function(assign)
            input = bound(input, idx, fn)

        sel = self.select

        def select(params, raw):
            ins = input(params, raw)
            return ins[:, sel:]

        return select


class Filter(Trace):
    """Filtering along the samples using a predicate trace."""

    def __init__(self, input, predicate):
        super().__init__(input, predicate)

    @property
    def input(self):
        """Get the input to the filter."""
        return self.inputs[0]

    @property
    def predicate(self):
        """Get the predicate of the filter."""
        return self.inputs[1]

    def trace(self, ctx):
        """Return implementation applying the predicate."""
        input, predicate = self.traces(ctx)

        if len(self.predicate.columns) == 1:
            def impl(params, raw):
                return input(params, raw)[predicate(params, raw), :]

        elif len(self.predicate.columns) == len(self.input.columns):
            def impl(params, raw):
                return input(params, raw)[predicate(params, raw)]

        else:
            raise ValueError("Sizes of predicate not match input.")

        return impl


class BinaryOp(Trace):
    """Apply a operator on the trace."""

    def __init__(self, input, other, op, reverse=False):
        if isinstance(other, Trace):
            super().__init__(input, other)
        else:
            super().__init__(input)
        self.other = other
        self.op = op
        self.reverse = reverse

    def trace(self, ctx):
        """Return implementation applying the binary op."""
        other = self.other
        op = self.op

        if self.reverse:
            @ctx.reslove
            def apply(a, b):
                return op(b, a)
        else:
            @ctx.reslove
            def apply(a, b):
                return op(a, b)

        if isinstance(other, pd.Series):
            other = other.loc[self.columns]

        if isinstance(other, (pd.Seres, np.array, list)):
            other = np.array(other)
            input, = self.traces(ctx)

            def impl(params, raw):
                ins = input(params, raw)
                m, n = ins.shape
                result = np.empty((m, n))

                for i in range(n):
                    result[:, i] = apply(ins[:, i], other[i])
                return result

        elif isinstance(other, Trace):
            if (other.columns != self.columns
                    and (len(self.columns) != 1 or len(other.columns) != 1)):
                raise ValueError("Columns to binary operator don't match!")

            fst, snd = self.traces(ctx)

            def impl(params, raw):
                a, b = fst(params, raw), snd(params, raw)
                m, n = a.shape

                result = np.empty((m, n))
                for i in range(n):
                    result[:, i] = apply(a[i], b[i])

        else:
            input, = self.traces(ctx)

            def impl(params, raw):
                return apply(input(params, raw), other)

        return impl


class Apply(Trace):
    """Transformation applying a given function."""

    def __init__(self, input, method, *args, reduce=False):
        super().__init__(input)
        self.method = method
        self.args = args
        self.reduce = reduce

    def trace(self, ctx):
        """Return implementation applying a method on each column."""
        apply = ctx.resolve_function(self.apply())
        reduce = int(self.reduce)
        input, = self.traces(ctx)

        def impl(params, raw):
            ins = input(params, raw)
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

    def __init__(self, input, columns):
        assert all(c in input.columns for c in columns)
        super().__init__(input, columns)

    def trace(self, ctx):
        """Return implementation selecting columns."""
        ixs = [self.input.columns.index(c) for c in self.columns]
        input, = self.traces(ctx)

        def impl(params, raw):
            return input(params, raw)[:, ixs]
        return impl


class Slice(Trace):
    """Select a slice of the data."""

    def __init__(self, input, slice):
        super().__init__(input)
        self.slice = slice

    def trace(self, ctx):
        """Return implementation selecting a slice."""
        start = self.slice.start or 0
        stop = self.slice.stop
        step = self.slice.step or 1
        input, = self.traces(ctx)

        def impl(params, raw):
            ins = input(params, raw)
            return ins[start:stop:step, :]
        return impl


class Frame:
    """Create pandas dataframes out of traces."""

    def __init__(self, trace):
        self.trace = trace

    def empty(self):
        """Create a empty dataframe according to frame definition."""
        return self.frame([])

    def frame(self, traces, offset=0):
        """Create a dataframe out of the given traces."""
        # TODO index with offset
        if not traces:
            return pd.DataFrame(columns=self.trace.columns)

        shape = traces[0].shape
        if len(shape) == 1:
            return pd.DataFrame(traces, columns=self.trace.columns)
        elif len(shape) == 2:
            return pd.DataFrame(np.concatenate(traces),
                                columns=self.trace.columns)
        else:
            msg = "A trace with {} dimensions is not supported"
            raise NotImplementedError(msg.format(len(shape)))

    def prepare(self):
        """Prepare the internals, called before running a new simulation."""
        self.traces = []
        self.data = self.empty()
        return self.data

    def publish(self, trace):
        """Handle the trace data called when it becomes available."""
        self.traces.append(trace)

    def finalize(self):
        """Take care to finalize all the data given to the manager."""
        df = self.frame(self.traces)
        self.data[df.columns] = df


class Holotrace(Frame):
    """Publish traces to a `holoviews.Buffer`."""

    def __init__(self, trace, batch=1, timeout=None):
        super().__init__(trace)

        import holoviews as hv
        hv.__version__

        self.batch = batch
        self.timeout = timeout

    @lazy
    def buffer(self):
        """Get the `holoviews.Buffer` where the data will be published to."""
        import holoviews as hv
        return hv.streams.Buffer(self.empty(),  # truncate=False),
                                 index=False,
                                 length=np.iinfo(int).max)

    def prepare(self):
        """Clear the holoviews blist of tracesuffer and rr."""
        self.buffer.clear()
        self.traces = []
        self.offset = 0
        self.last = time.time()
        return self.buffer

    @property
    def data(self):
        """Access the data inside the buffer."""
        return self.buffer.data

    def push(self):
        """Push remaining traces towards the holoviews buffer."""
        if self.traces:
            df = self.frame(self.traces)
            if isinstance(df.index, pd.MultiIndex):
                df.index.set_levels(df.index.levels[0] + self.offset,
                                    level=0, inplace=True)
            else:
                df.index += self.offset

            self.offset += len(self.traces)  # * self.trace.skipping
            self.traces = []

            self.buffer.send(df)
            self.last = time.time()

    def publish(self, trace):
        """
        Publish traced data.

        @param trace raw numpy trace data
        """
        self.traces.append(trace)

        if len(self.traces) >= self.batch:
            self.push()
        elif (self.timeout is not None
                and time.time() - self.last > self.timeout):
            self.push()

    def finalize(self):
        """Ensure that all traces end up in the holoviews buffer."""
        self.push()
