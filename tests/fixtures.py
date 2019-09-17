import pytest
import simcompyl as sim


@pytest.fixture(params=['DebugingExecution',
                        'PseudoNumbaExecution',
                        pytest.param('NumbaExecution', marks=pytest.mark.integration)])
def engine(request):
    """Test with different engines when executing models."""
    return getattr(sim.engine, request.param)
