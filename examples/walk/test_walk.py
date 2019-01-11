import simulhave as sim

import holoviews as hv
import numpy as np
import pytest


pytestmark = [pytest.mark.integration,
              pytest.mark.filterwarnings(
                  "ignore:numpy.dtype size changed.*:RuntimeWarning")]


def test_execute(engine):
    from . import walk

    model = walk.ComplexWalk()
    alloc = walk.Simulation() + walk.BasicDistance()

    exec = engine(model, alloc)

    out = exec.run(initial_energy=10)

    limit = alloc.n_samples.value / 2

    assert set(out.columns) == {'x', 'y', 'energy'}
    assert len(out.query('x**2 + y**2 < 10**2')) > limit
    assert len(out.query('energy <= 0')) > limit

    out = exec.run()
    assert len(out.query('energy <= 0')) < limit


def test_tracing(engine):
    from . import walk

    model = walk.ComplexWalk()
    alloc = walk.Simulation() + walk.BasicDistance()

    exec = engine(model, alloc)

    tr = sim.Trace(['x', 'y']).take(12).skip(6)
    with exec.trace(tr) as pos:
        exec.run()

    assert len(pos) == (alloc.n_steps.value + 1) * 12
    assert list(pos.columns) == ['x', 'y']

    qr = (sim.Trace(dist=lambda x, y: np.sqrt(x ** 2 + y ** 2))
          .quantile([.25, .5, .75]))

    with exec.trace(tr, qr, target=sim.trace.Holotrace) as (pos, dist):
        exec.run()

    assert isinstance(dist, hv.streams.Buffer)
    assert len(dist.data) == (alloc.n_steps.value + 1) * 3
    assert list(dist.data.columns) == ['dist']
