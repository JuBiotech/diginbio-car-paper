import pytest

import plotting


def test_interesting_groups():
    class MockPosterior:
        def __init__(self, keys):
            self.keys = lambda: frozenset(keys)
            
    result = plotting.interesting_groups(
        MockPosterior(["X0_base", "S0"])
    )
    assert result["biomass"] == ["X0_base"]
    assert result["biotransformation"] == ["S0"]

    result = plotting.interesting_groups(
        MockPosterior(["ls_X", "scaling_X", "log_X_factor", "k_design"])
    )
    assert result["biomass"] == ["ls_X", "scaling_X", "log_X_factor"], result
    assert result["biotransformation"] == ["k_design"]
    pass
