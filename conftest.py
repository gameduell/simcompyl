import pytest

@pytest.fixture(params=['BasicExecution', 'NumbaExecution'])
def engine(request):
    import simulate as sim
    return getattr(sim.engine, request.param)



def pytest_addoption(parser):
    parser.addoption('--integration', action='store_true', dest="integration",
                     default=False, help="enable integration tests")

def pytest_configure(config):
    if not config.option.integration:
        setattr(config.option, 'markexpr', 'not integration')

