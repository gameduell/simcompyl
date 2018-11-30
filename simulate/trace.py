"""A trace defines how to keep track of statistics during the simulation."""

import numba
import pandas as pd
import numpy as np

import time

from .util import lazy


class Trace:
    """Keeps track of some statitics during simulation."""

    # fallback if trace does not define columns
    columns = []
    skipping = 1

    def __init__(self, columns=[]):
        """
        Create a trace opbject to keep track of some statistic.

        @param columns create the statistic on the given columns.
        """
        self.columns = columns
        self.model = None

    def on(self, model):
        """
        Trace the statistics using the supplied model.

        @return self
        """
        self.model = model
        return self

    def target(self, manager):
        """
        Explicitly specify the target manager to use.

        @param manager TraceManager to use for publishing tracing data.
        @see TraceManager
        """
        self.manager = manager.bind(self)
        return self
    
    def holo(self):
        return self.target(HoloViewsManager())

    @lazy
    def manager(self):
        """Get the TraceManager used by this trace."""
        return TraceManager().bind(self)

    def options(self, *, sample=0, skip=0, **opts):
        """
        Set options for this trace, may result in changes to the manager.

        @param sample restrict the number of samples to use for creating traces
        @param skip only create a trace every skip-th call to loop
        @return self
        """
        if sample:
            self = SamplingTrace(self, n=sample, model=self.model)

        if skip:
            orig = self.manager
            del self.manager
            self.target(SkippingManager(orig, skip=skip))
            self.skipping = skip

        self.manager.options(**opts)
        return self

    def __getattr__(self, name):
        """Access attributes of the manager."""
        if not name.startswith('_') and hasattr(self.manager, name):
            return getattr(self.manager, name)
        raise AttributeError("{!r} object has no attribute {!r}."
                             .format(type(self).__name__, name))

    def __columns__(self, model):
        """Return an indexing array to the columns of this trace."""
        return self.model.state(return_namedtuple=True, 
                                **{c: ... for c in self.columns})

    def empty(self, truncate=True):
        """
        Create a empty trace form an uninitialized state.

        @param truncate weather to return an empty dataframe
                        just with the structure of the frame or a filled frame.
        """
        n = self.model.alloc.n_samples.value
        engine = self.model.engine
        
        tr = engine.trace(self)
        
        params = engine.params()
        state = engine.state()
        
        return self.frame([tr(params, state)]).iloc[:0 if truncate else None]
    
    def prepare(self):
        return self.manager.prepare()
    
    def finalize(self):
        return self.manager.finalize()

    def trace(self, model):
        """Return a implementation transforming the state into a trace."""
        def impl(params, state):
            return NotImplemented
        return impl

    def frame(self, traces, index=None, columns=None):
        """
        Create a dataframe from a list of traces.

        @param traces list of raw traces
        @param index indexes inside each trace
        @param columns columns inside each trace
        """
        trx = pd.RangeIndex(len(traces), name='trace')
        trx *= self.skipping

        shape = traces[0].shape

        # columns
        if columns is None and shape[-1] == len(self.columns):
            columns = pd.Index(self.columns, name='variable')
            
        if index is None and len(shape) > 1:
            index = pd.RangeIndex(shape[0], name='individual')
            
        if index is None:
            index = trx
        else:
            index = pd.MultiIndex.from_product([trx, index], 
                                               names=(trx.name, index.name))
            
        return pd.DataFrame(np.concatenate(traces),
                            columns=columns, index=index).unstack()
            

class Columns(Trace):
    """Trace values of some variables for the complete population."""
    def trace(self, model):
        """Trace implementation just selecting the columns."""
        ix = np.array(self.__columns__(model)).ravel()

        def impl(params, state):
            return state[:, ix]
        return impl


class SamplingTrace(Trace):
    """Sub-sample the complete population before running another trace."""

    def __init__(self, orig, n, model=None):
        """
        Create a trace on a specific amount of samples.

        @param orig the original trace object
        @param n number of samples to trace
        """
        self.orig = orig
        self.n = n
        self.skipping = orig.skipping
        self.model = model

    def on(self, model):
        """Forward the bind to the original trace object."""
        self.orig.on(model)
        return super().on(model)

    def trace(self, model):
        """Use another trace, but only with a limited number of samples."""
        orig = numba.jit(self.orig.trace(model))
        n = self.n

        def impl(params, state):
            return orig(params, state[:n])
        return impl

    def frame(self, traces, **opts):
        """Create a frame using the original trace object."""
        return self.orig.frame(traces, **opts)
    
    
class Sum(Trace):
    def trace(self, model):
        ix = np.array(self.__columns__(model)).ravel()
        
        def impl(params, raw):
            result = np.empty((1, len(ix)))
            for i, x in enumerate(ix):
                result[0, i] = raw[:, x].sum()
            return result
        return impl

    def frame(self, traces, index=None, **opts):
        """Create a dataframe with the quaniles."""
        if index is None:
            index = pd.Index(['Sum'], name='mesure')
        return super().frame(traces, index=index, **opts)
    
    
class Mean(Trace):
    def trace(self, model):
        ix = np.array(self.__columns__(model)).ravel()
        
        def impl(params, raw):
            result = np.empty((1, len(ix)))
            for i, x in enumerate(ix):
                result[0, i] = raw[:, x].mean()
            return result
        return impl

    def frame(self, traces, index=None, **opts):
        """Create a dataframe with the quaniles."""
        if index is None:
            index = pd.Index(['Mean'], name='mesure')
        return super().frame(traces, index=index, **opts)


class Quantiles(Trace):
    """Trace that captures quantiles over some variables."""
    def __init__(self, columns, qs=[.1, .25, .5, .75, .9]):
        """
        Create a quantile trace over given variables.

        @param columns names of the variables
        @param qs list of quantiles to create
        """
        super().__init__(columns)
        self.qs = qs

    def trace(self, model):
        """Return a method that creates the array with the quantiles."""
        ix = np.array(self.__columns__(model)).ravel()
        percents = np.array(self.qs)*100

        def impl(params, raw):
            result = np.empty((len(percents), len(ix)))
            for i, x in enumerate(ix):
                result[:, i] = np.percentile(raw[:, x], percents)
            return result
        return impl

    def frame(self, traces, index=None, **opts):
        """Create a dataframe with the quaniles."""
        if index is None:
            index = pd.Index(self.qs, name='quantile')
        return super().frame(traces, index=index, **opts)


class TraceManager:
    """Manages the datastructures to keep track of the traces."""

    def options(self):
        """Set options on the manager if available."""
        return self

    def bind(self, trace):
        """Manage the given trace object."""
        self.trace = trace
        return self

    def prepare(self):
        """
        prepare the internals, called before running a new simulation.

        @return the resulting data object, a DataFrame in the base impl.
        """
        self.traces = []
        self.data = self.trace.empty()
        return self.data

    def publish(self, trace):
        """
        Take care of handling the trace data called when it becomes available.

        @param trace the raw numpy data of the trace
        """
        self.traces.append(trace)

    def finalize(self):
        """Take care to finalize all the data given to the manager."""
        df = self.trace.frame(self.traces)
        self.data[df.columns] = df


class SkippingManager(TraceManager):
    """Manager that only traces every `skip`-th loop to an original manager."""

    def __init__(self, orig, skip=1):
        """
        Create a new manager that will skip traces.

        @param orig original manager that will be called on every skip-th trace
        @param skip number of what traces to forward
        """
        self.orig = orig
        self.count = 0
        self.skip = skip

    def __getattr__(self, name):
        """Access attributes of the original manager."""
        return getattr(self.orig, name)

    def bind(self, trace):
        """Give the trace object to the original manager."""
        self.orig.bind(trace)
        return self

    def options(self, *, skip=0, **opts):
        """
        Update options, forwarding to the original manager.

        @param skip update the number of traces to work on
        """
        if skip:
            self.skip = skip
        self.orig.options(**opts)
        return self

    def prepare(self):
        """prepare the original manager."""
        return self.orig.prepare()

    def publish(self, trace):
        """Publish only each `self.skip`-th trace to the original manager."""
        if self.count % self.skip == 0:
            self.orig.publish(trace)
        self.count += 1

    def finalize(self):
        """Finalize the original manager."""
        return self.orig.finalize()


class HoloViewsManager(TraceManager):
    """Manager that publishes to a holoviews buffer."""

    def __init__(self):
        """Create a manager that publishes traces to a `holoviews.Buffer`."""
        # early fail
        import holoviews as hv
        hv.__version__
        self.batch = 1
        self.timeout = None

    def options(self, *, batch=1, timeout=None, **opts):
        """
        Update options of the manager.

        @param batch number of traces to send to bokeh at once
        """
        self.batch = batch
        self.timeout = timeout
        return super().options(**opts)

    @lazy
    def buffer(self):
        """Get the `holoviews.Buffer` where the data will be published to."""
        import holoviews as hv
        return hv.streams.Buffer(self.trace.empty(truncate=False),
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
            df = self.trace.frame(self.traces)
            if isinstance(df.index, pd.MultiIndex):
                df.index.set_levels(df.index.levels[0]+self.offset,
                                    level=0, inplace=True)
            else:
                df.index += self.offset

            self.offset += len(self.traces) * self.trace.skipping
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
        elif self.timeout is not None and time.time() - self.last > self.timeout:
            self.push()
        

    def finalize(self):
        """Ensure that all traces end up in the holoviews buffer."""
        self.push()
