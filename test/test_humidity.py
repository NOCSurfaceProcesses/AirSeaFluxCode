import pytest

import numpy as np

from AirSeaFluxCode.hum_subs.humidity import (
    dew_point_from_qsat,
    dew_point_to_specific_humidity,
    sat_specific_humidity_air,
    specific_humidity_to_dew_point,
)
from AirSeaFluxCode.util_subs.utils import CtoK


@pytest.mark.parametrize(
    "qmeth",
    [
        "Hardy",
        "MagnusTetens",
        "GoffGratch",
        "Buck",
        "Buck2",
        "WMO",
        "WMO2018",
        "Wexler",
        "Sonntag",
        "Bolton",
        "HylandWexler",
        "IAPWS",
        "Preining",
        "MurphyKoop",
    ],
)
def test_dewpoint_sat_specific_humidity_transform(qmeth: str):
    # TEST: dew-point to saturation specific humidity to dew-point for different
    #       qmeth values.

    # Generate some random data
    np.random.seed(90210)
    N = 5
    P = np.repeat(1013, N)

    # Initial air temperatures
    T = (CtoK + 10) + 10 * np.random.randn(N)

    # Initial dew-point temperatures
    Td = (CtoK + 5) + 10 * np.random.randn(N)

    # Convert to specific humidity
    qair = sat_specific_humidity_air(hum=("Td", Td), T=T, P=P, qmeth=qmeth)
    # Convert back to dew-point temperatuer
    dpt = dew_point_from_qsat(q=qair, T=T, P=P, qmeth=qmeth)
    # Check they are the same!
    assert np.allclose(dpt, Td), f"Mismatch for {qmeth = }"


def test_dewpoint_specific_humidity_transform():
    # TEST: dew-point to specific humidity to dew-point

    # Generate some random data
    np.random.seed(90210)
    N = 5
    P = np.repeat(1013, N)

    # Initial dew-point temperatures
    Td = (CtoK + 5) + 10 * np.random.randn(N)

    # Convert to specific humidity
    qair = dew_point_to_specific_humidity(Td=Td, P=P)
    # Convert back to dew-point temperatuer
    dpt = specific_humidity_to_dew_point(q=qair, P=P)
    # Check they are the same!
    assert np.allclose(dpt, Td)
