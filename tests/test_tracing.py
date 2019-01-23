import simcompyl as sim

import pytest
from pandas.testing import assert_frame_equal


class Generate(sim.Model):
    @sim.step
    def init(self):
        init, _ = self.params(init=float,
                              complex={'a': [int]*5,
                                       'b': [float]*5,
                                       'c': [float]*5})
        speed = self.random(speed=float)
        grow, const, alt = self.state(grow=float, const=float, alt=float)

        def impl(params, state):
            state[const] = state[alt] = speed(params)
            state[grow] = init(params)
        return impl

    @sim.step
    def iterate(self):
        grow, const, alt = self.state(grow=float, const=float, alt=float)

        def impl(params, state):
            v = state[const]
            state[grow] += v
            state[alt] = -state[alt]
        return impl


class Simulation(sim.Allocation):
    n_samples = sim.Param("Number of Samples", 100)
    n_steps = sim.Param("Steps to Simulate", 20)


class Amounts(sim.Allocation):
    init = sim.Param("Initial Value", 0)
    speed = sim.Normal("speed", 0, .2, help="Distribution of speeds")
    radius = sim.Normal("Radius", 1, .2, help="Distribution of radius")
    complex = sim.Param("Complex", {'a': [1, 2, 3],
                                    'b': [0, 2, 1],
                                    'c': [1, 4, 9]})


def test_tracing(engine):
    model = Generate()
    alloc = Amounts() + Simulation()
    exec = engine(model, alloc)

    assert exec.run().shape == (100, 3)

    tr = model['grow', 'const', 'alt']

    assert 'grow' in dir(model)
    assert 'const' in model._ipython_key_completions_()
    assert "State" in str(tr)
    assert all(c in repr(tr) for c in tr.columns)

    with exec.trace(tr.take(6)) as td:
        exec.run()

    assert td.shape == (21 * 6, 3)
    assert_frame_equal(td.grow.unstack().diff()[1:],
                       td.const.unstack()[1:])
    assert_frame_equal(td.alt.unstack().diff().abs()[1:],
                       2 * td.const.unstack().abs()[1:])

    with exec.trace(tr.mean()) as td:
        exec.run()
    assert td.shape == (21, 3)
    assert_frame_equal(td.grow.unstack().diff()[1:],
                       td.const.unstack()[1:])
    assert_frame_equal(td.alt.unstack().abs()[1:],
                       td.const.unstack().abs()[1:])

    with exec.trace(tr[['grow', 'const']].quantile([.1, .5, .9])) as td:
        exec.run()
    assert td.shape == (21 * 3, 2)
    assert (td.xs(.1, level=1) <= td.xs(.5, level=1)).all(None)
    assert (td.xs(.5, level=1) <= td.xs(.5, level=1)).all(None)
    assert_frame_equal(td.grow.unstack().diff()[1:],
                       td.const.unstack()[1:])

    mr = (2 * model.grow - model.alt).name('math')
    with exec.trace(mr.take(6), tr.take(6)) as (md, td):
        exec.run()
    assert md.shape == (21 * 6, 1)
    assert_frame_equal(md.math.unstack().diff().diff().abs()[2:],
                       td.const.unstack().abs()[2:] * 4)

    mr = model(math=lambda grow, alt: 2 * grow + alt)

    with exec.trace(mr.take(6), tr.take(6)) as (md, td):
        exec.run()
    assert md.shape == (21 * 6, 1)
    assert_frame_equal(md.math.unstack().diff().diff().abs()[2:],
                       td.const.unstack().abs()[2:] * 4)

    with exec.trace(mr.mean(), tr.mean()) as (md, td):
        exec.run()
    assert md.shape == (21, 1)
    assert_frame_equal(md.math.unstack().diff().diff().abs()[2:],
                       td.const.unstack().abs()[2:] * 4)

    fr = tr[model['alt'] > 0]

    with exec.trace(fr.take(6)) as fd:
        exec.run()
    assert fd.shape == (21 * 6, 3)
    assert (fd.alt > 0).all(None)

    with exec.trace(fr.mean()) as fd:
        exec.run()
    assert (fd.alt > 0).all(None)
    assert_frame_equal(fd.grow[::2].unstack().diff()[1:],
                       fd.const[::2].unstack()[1:] * 2)
    assert_frame_equal(fd.grow[1::2].unstack().diff()[1:],
                       fd.const[1::2].unstack()[1:] * 2)

    with exec.trace(mr[tr.alt > 0].take(6)) as md:
        exec.run()
    assert md.shape == (21 * 6, 1)

    assert (md.unstack()[::2].diff()[1:] > 0).all(None)
    assert (md.unstack()[1::2].diff()[1:] < 0).all(None)

    with exec.trace(((tr - tr.mean()) ** 2).mean()) as td:
        exec.run()
    assert td.shape == (21, 3)
    assert_frame_equal(td.alt.unstack(), td.const.unstack())
    vdd = td.grow.diff().diff()
    assert (vdd[2:].diff()[1:].abs() < .01).all(None)

    pr = sim.trace.Param(init=float)
    with exec.trace(pr) as ps:
        exec.run(init=6)

    assert ps.shape == (21, 1)
    assert (ps == 6).all(None)

    pr = sim.trace.Param(complex={'a': [int]*3, 'c': [float]*3})
    with exec.trace(pr) as ps:
        exec.run()

    assert ps.shape == (3 * 21, 2)
    assert list(ps.columns) == ['complex.a', 'complex.c']


def test_invalids():
    with pytest.raises(AttributeError):
        sim.Trace(a=int, b=int).c

    with pytest.raises(KeyError):
        sim.Trace(a=int, b=int)['c']

    with pytest.raises(ValueError):
        sim.Trace(a=int, b=int) / sim.Trace(c=int, d=int)

    with pytest.raises(ValueError):
        sim.Trace(a=int, b=int).name("a", "b", "c")

    with pytest.raises(NotImplementedError):
        sim.Trace(a=int, b=int)[...]

    with pytest.raises(ValueError):
        sim.Trace(a=int, b=int).assign(d=lambda a, c: a - c)

    class Ctx:
        def resolve_function(self, fn):
            return fn

        def state(self, return_namedtuple=False, **defs):
            return list(defs)

    tr = sim.Trace(long=lambda a, b, c, d, e, f: 0)
    with pytest.raises(NotImplementedError):
        tr.trace(Ctx())

    model = Generate()
    with pytest.raises(KeyError):
        model['alt', 'nil']

    with pytest.raises(AttributeError):
        model.nil


def test_holoview():
    import holoviews as hv
    hv.extension('bokeh')

    model = Generate()
    alloc = Amounts() + Simulation()
    exec = sim.engine.BasicExecution(model, alloc)

    tr = model['grow', 'const', 'alt']

    ht = tr.take(6).to(sim.trace.Holotrace)

    assert 'Holotrace' in str(ht)
    assert str(ht.trace) in str(ht)
    assert repr(ht.trace) in repr(ht)

    assert list(ht.data.columns) == ['grow', 'const', 'alt']
    assert list(ht.data.index.names) == ['trace', 'sample']

    dm = ht.plot(hv.Curve)
    hv.render(dm)  # trigger rendering, so dynamic map gets filled

    over, = dm
    gc, cc, ac, = over

    assert isinstance(dm, hv.DynamicMap)
    assert isinstance(over, hv.NdOverlay)
    assert isinstance(gc, hv.Curve)
    assert gc.kdims == [hv.Dimension('trace')]
    assert gc.vdims == [hv.Dimension('value')]
    assert isinstance(cc, hv.Curve)
    assert cc.kdims == [hv.Dimension('trace')]
    assert cc.vdims == [hv.Dimension('value')]
    assert isinstance(ac, hv.Curve)
    assert ac.kdims == [hv.Dimension('trace')]
    assert ac.vdims == [hv.Dimension('value')]

    bf = ht.buffer
    assert len(bf.data) == 1
    with exec.trace(ht):
        exec.run(n_steps=2)
    assert ht.buffer == bf
    assert len(bf.data) == 3 * 6
