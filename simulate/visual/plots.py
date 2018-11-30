"""Adds plotting capabilities on top of trace classes for common statistics."""

import holoviews as hv
import pandas as pd
import numba

from ..util import lazy
from .. import trace

class Plot(trace.Trace):
    """
    Base class for holoviews plotting.

    >>> plot = SomePlot().on(model)
    >>> plot.show()
    >>> # graph will show-up here
    >>> with plot:
    ...     model.execute()
    """
    names = {}
        
    def options(self, names={}, **kws):
        super().options(**kws)
        self.names = names
        return self

    @lazy
    def manager(self):
        """Use HoloviewsManager as default trace manager."""
        return trace.HoloViewsManager().bind(self)

    def plot(self, data, name='value'):
        """
        Create the holoviews plotting object with the given data.

        @param data the dataframe to plot.
        """            
        data = data.stack(dropna=False)
        if len(data.columns) == 1:
            vdims = [c for c in data.columns]
        else:
            data = data.stack(dropna=False).rename(name)
            vdims = [name]
        return hv.Dataset(data, 
                          kdims=[(n, self.names.get(n, n.replace('_', ' ').title()))
                                 for n in data.index.names],
                          vdims=[(n, self.names.get(n, n.replace('_', ' ').title()))
                                 for n in vdims])

    def show(self):
        """Create a DynamicMap showing the holoviews plot for a statistic."""
        return hv.DynamicMap(self.plot, streams=[self.buffer])
                #.options(width=276, height=210)
                #.opts(plot={'sizing_mode': 'scale_width'}))


class Mean(Plot, trace.Mean):
    """Shows the quantiles for some state variables."""

    def plot(self, data):
        """Plot the quantiles as a overlay of curves."""
        return super().plot(data).to(hv.Curve).options(tools=['hover']).overlay("variable").overlay()


class Quantiles(Plot, trace.Quantiles):
    """Shows the quantiles for some state variables."""

    def plot(self, data):
        """Plot the quantiles as a overlay of curves."""
        return super().plot(data).to(hv.Curve).options(tools=['hover']).overlay()


class Spread(Plot, trace.Quantiles):
    """Shows the spread of some state variables using an area plot."""

    def __init__(self, columns):
        """
        Create a spread plot for the given variables.

        @param columns variables to create spreads for
        """
        super().__init__(columns, qs=[.5, .25, .75])
        
    def trace(self, model):
        _impl = numba.jit(super().trace(model))
        value, lower, upper = range(3)
        
        def impl(params, state):
            qs = _impl(params, state)
            qs[lower, :] = qs[value, :] - qs[lower, :]
            qs[upper, :] = qs[upper, :] - qs[value, :]
            return qs
        return impl
    
    def frame(self, traces):
        """Create a frame with lower, median and upper values."""
        return super().frame(traces, index=pd.Index(['value', 'neg', 'pos'], name='IQR'))

    def plot(self, data):
        """Create the spread plot for the data."""
        if data.empty:
            data = data.append(pd.Series(index=data.columns, name=0))
            
        ds = hv.Dataset(data.stack(0, dropna=False), 
                        kdims=['trace', 'variable'], 
                        vdims=['value', 'neg', 'pos'])
        return (ds.to(hv.Spread).options(muted_alpha=.1).overlay() * ds.to(hv.Curve, vdims='value').options(tools=['hover'])).overlay()
