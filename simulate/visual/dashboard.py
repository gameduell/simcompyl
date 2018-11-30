import contextlib
import functools

import bokeh.models as bkm
import bokeh.layouts as bkl
import bokeh.io as bki

import holoviews as hv
import param
import pandas as pd

from simulate.util import lazy


#__all__ = ['Dashboard', 'AllocationWidgets', 'AllocationControl']


class AllocationControl:
    """
    Control for the allocation of a model, used to wire UI components to the allocation.
    """
    def __init__(self, alloc):
        """
        @param the Allocation object for a model
        """
        self.alloc = alloc
        self.subscribers = []
        
    def subscribe(self, subscriber):
        """
        Subscribe to changes of allocation.
        @param subcriber function that accpets keyword arguments for changed values.
        
        Note that only changes to allocation done through te control are captured,
        so you should always use `update` to push changes to the allocation.
        """
        self.subscribers.append(subscriber)
        
    def update(self, **values):
        """
        Update values of the allocation, notifying subscribers about the changes.
        @param values variable/value pairs that should be updated
        @return values that actually did change
        """
        for n, v in dict(values).items():
            if getattr(self.alloc, n).value == v:
                del values[n]
            else:
                setattr(self.alloc, n, v)
            
        if values:
            for s in self.subscribers:
                s(**values)
        return values
    
    
def expand(title, content, expand=True):
    fold = bkm.Panel(title="<", child=bkm.Spacer())
    main = bkm.Panel(title=title, child=bkl.row(
            bkm.Spacer(width=60), content))
    expand = bkm.Tabs(tabs=[fold, main], active=int(expand))
    return expand


def widgets(alloc, ctrl):
    def _setup_wiring(name, widget, attr="value"):
        def ui_change(attr, old, new):
            ctrl.update(**{name: new})
        widget.on_change(attr, ui_change)  
        
        def st_change(**kws):
            if name in kws:
                widget.update(**{attr: kws[name]})
        ctrl.subscribe(st_change)
        
    def _mk_widgets(name, value):
        param = value.param
        
        if isinstance(param.default, bool):
            widget = bkm.Toggle(label=param.text, active=value.value)
            _setup_wiring(name, widget, 'active')
            return [widget]
                              
        if isinstance(param.default, (int, float)):
            widget = bkm.Slider(title=param.text,
                                value=value.value,
                                start=param.lower, 
                                end=param.upper, 
                                step=param.step)
            _setup_wiring(name, widget)
            return [widget]
        
        if isinstance(param.default, tuple) and len(value.default) == 2:
            widget = bkm.RangeSlider(title=param.text,
                                    value=value.value,
                                    start=param.lower,
                                    end=param.upper,
                                    step=param.step)
            _setup_wiring(name, widget)
            return [widget]
        
        if isinstance(param.default, str):
            widget = bkm.Select(title=param.text,
                                value=value.value,
                                options=param.options)
            _setup_wiring(name, widget)
            return [widget]
            
        if isinstance(param.default, dict): 
            items = next(iter(param.default.values()))
            if isinstance(items, (list, tuple)):
                source = bkm.ColumnDataSource({k: v[:10] for k,v in param.default.items()})
                size = max(map(len, param.default.values()))+2
                
                columns = [bkm.TableColumn(field=name, title=name, width=64) for name, values in value.value.items()]
                caption = bkm.Div(text=param.text + ':')
                table = bkm.DataTable(source=source, columns=columns, editable=True, height=size*24)
                _setup_wiring(name, source, 'data')
                #table = bkm.Div(text=pd.DataFrame(value.value).to_html())
                return [caption, table]
            else:
                raise ValueError("Unhandled dict-param with {} items"
                                 .format(type(items).__name__))
        
        raise ValueError("Unhandled param type {}"
                         .format(type(param.default).__name__))
    
    return bkl.widgetbox(
        sum([_mk_widgets(name, value) 
             for name, value in alloc.__params__.items()], []))

    
class Dashboard:
    """
    Creates an interactive dashboard around a model, alloc and plot definition
    """
    def __init__(self, model, ctrl, tabs, plots):
        self.model = model
        self.ctrl = ctrl
        self.tabs = tabs
        self.plots = plots
        
        self.renderer = hv.renderer('bokeh').instance(mode='server')
    
    def run(self):
        #for e in self.expands:
        #    if e.active:
        #        e.active = 0
            
        with contextlib.ExitStack() as stack:
            for ps in self.plots:
                for p in ps:
                    stack.enter_context(p)
            self.model.execute()
    
    def render(self, doc):
        self.button = bkm.Button(label="Run Simulation")
        self.button.on_click(self.run)
        
        layout = bkl.column(self.tabs,
                            bkl.widgetbox(self.button),
                            bkl.layout([[self.renderer.get_plot(p.show()).state
                                         for p in ps] for ps in self.plots], 
                                       sizing_mode='scale_width'), 
                            sizing_mode='scale_width')
        doc.add_root(layout)
        return doc
        
    def show(self, notebook_url='localhost:8888'):
        import __main__
        if hasattr(__main__, '__file__'):
            return self.render(bki.curdoc())
        else:
            bki.output_notebook()
            return bki.show(self.render, notebook_url=notebook_url)
