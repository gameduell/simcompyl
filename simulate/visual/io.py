import bokeh as bk
import bokeh.layouts as bkl
import bokeh.models as bkm
import bokeh.io as bki

import pathlib
import json

from ..util import lazy


__all__ = ['IOInterface']


class IOInterface:
    load_title = 'Load by name:'
    save_title = 'Save by name:'
    link_text = 'Link to <a href="?params={sel}">{sel}</a> Config'
    export_text = 'To/From your Computer:'
    
    title = 'Save/Load'
    
    def __init__(self, alloc, ctrl, path='params', extension='.json'):
        self.alloc = alloc
        self.ctrl = ctrl
        self.path = path
        self.extension = extension
        
        self.select = bkm.Select(options=['Defaults'] + [p.stem for p in pathlib.Path(path).glob("*"+extension)],
                                 value='Defaults',
                                 title=self.load_title)
        self.load = bkm.Button(label='Load')
        self.link = bkm.Div(text=self.link_text.format(sel=self.select.options[0]))
                            
        self.wipe = bkm.Button(label='Wipe')
        
        self.name = bkm.TextInput(title=self.save_title)
        self.save = bkm.Button(label='Save')
        self.space = bkm.Div(text=self.export_text)
        self.exports = bkm.Button(label='Export')
        self.imports = bkm.Button(label='Import')
        
        self.select.on_change('value', self.on_select)
        self.save.on_click(self.on_save)
        self.load.on_click(self.on_load)
        self.wipe.on_click(self.on_wipe)
    
    def fn(self, stem):
        return self.path + "/" + stem + self.extension
    
    def refresh(self, select='Defaults'):
        opts = ['Defaults'] + [p.stem for p in pathlib.Path(self.path).glob("*"+self.extension)]
        self.select.value = select
        self.select.options = opts
        
    def on_select(self, attr, old, new):
        self.link.text = self.link_text.format(sel=new)

    def on_load(self):
        if self.select.value == 'Defaults':
            obj = {name: value.default 
                   for name, value in self.alloc.__params__.items()}
        else:
            with open(self.fn(self.select.value), 'r') as f:
                obj = json.load(f)
        self.ctrl.update(**obj)

    def on_wipe(self):
        pathlib.Path(self.fn(self.select.value)).unlink()
        self.refresh()

    def on_save(self):
        obj = {n: v.value for n, v in self.alloc.__params__.items()}
        with open(self.fn(self.name.value), 'w+') as f:
            json.dump(obj, f)
            
        self.refresh(self.name.value)

    def on_import(self, attr, old, new):
        pass

    def on_export(self, attr, old, new):
        pass
    
    @lazy
    def output(self):
        import bokeh.io as bki
        ctx = bki.curdoc().session_context
        #if ctx:
        #    sel = ctx.request.arguments.get('params', [None])[0]
        #    if sel is not None:
        #        self.select.value = sel.decode()
        #        self.on_load()
        
        return bkl.row([bkl.widgetbox([self.select, self.load, self.link, self.wipe]),
                        bkl.widgetbox([self.name, self.save, self.space, 
                                       self.exports, self.imports])])
