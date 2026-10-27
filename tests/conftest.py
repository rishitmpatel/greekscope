import warnings

import pytest


@pytest.fixture(autouse=True)
def _warnings_are_errors():
    """Any warning raised by library code (RuntimeWarning, IntegrationWarning...) fails the test."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield
