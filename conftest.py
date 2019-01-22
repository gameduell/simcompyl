"""Configuration for py.test."""

from tests.fixtures import *


def pytest_addoption(parser):
    """Accept an `--integration` arg for tests marked as integration."""
    parser.addoption('--integration', action='store_true', dest="integration",
                     default=False, help="enable integration tests")

                     
def pytest_configure(config):
    """Activate integration tests only when `integration` options is set."""
    if not config.option.integration:
        setattr(config.option, 'markexpr', 'not integration')
