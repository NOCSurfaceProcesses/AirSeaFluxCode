import numpy as np
import pandas as pd

from AirSeaFluxCode.util_subs import validate_kelvin


def test_validate_kelvin():
    # Generate some Celsius
    t = 30 - 20 * np.random.rand(100)
    assert np.all(validate_kelvin(t) > 273.16)

    # Check pandas immutability doesn't cause error
    s = pd.Series(t)
    assert np.all(validate_kelvin(s.to_numpy()) > 273.16)
