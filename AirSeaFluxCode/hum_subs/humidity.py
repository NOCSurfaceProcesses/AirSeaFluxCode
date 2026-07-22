# Copyright 2023-2026, Stavroula Biri
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Humidity Sub-routines including conversions between humidity measurement types,
such as dew-point, specific humidity.
"""

from typing import Tuple
import numpy as np
import warnings

from .vapor_pressure import VaporPressure
from .qsat import qsat_air, qsat_sea
from ..util_subs import CtoK, validate_kelvin


def get_hum(
    hum: Tuple[str, np.ndarray],
    T: np.ndarray,
    sst: np.ndarray,
    P: np.ndarray,
    qmeth: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get saturation specific humidity output for both air and sea.

    Parameters
    ----------
    hum : tuple[str, float | numpy.ndarray]
        humidity input switch 2x1 [x, values] default is relative humidity
            x='rh' : relative humidity [%]
            x='q' : specific humidity [g/kg]
            x='Td' : dew point temperature [K]
    T : float | numpy.ndarray
        air temperature [K]
    sst : float | numpy.ndarray
        sea surface temperature [K]
    P : float | numpy.ndarray
        air pressure at sea level [hPa]
    qmeth : str
        method to calculate saturation specific humidity from vapor pressure.
        See `AirSeaFluxCode.hum_subs.VaporPressure` for options.

    Returns
    -------
    qair : float | numpy.ndarray
        saturation specific humidity of air [g/kg]
    qsea : float | numpy.ndarray
        saturation specific humidity over sea surface [g/kg]

    See Also
    --------
    AirSeaFluxCode.hum_subs.VaporPressure
        For options for `qmeth` parameter.
    """
    qair = sat_specific_humidity_air(hum, T, P, qmeth)  # q of air [g/kg]
    qsea = qsat_sea(sst, P, qmeth)  # surface water q [g/kg]
    return qair, qsea


def sat_specific_humidity_air(
    hum: Tuple[str, np.ndarray],
    T: np.ndarray,
    P: np.ndarray,
    qmeth: str,
) -> np.ndarray:
    """
    Get specific humidity from of the air from input humidity.

    Parameters
    ----------
    hum : tuple[str, numpy.ndarray | float]
        humidity input switch 2x1 [x, values] default is relative humidity

        - x='rh' : relative humidity [%]
        - x='q' : specific humidity [g/kg]
        - x='Td' : dew point temperature [K]

    T : float | numpy.ndarray
        air temperature [K]
    P : float | numpy.ndarray
        air pressure at sea level [hPa]
    qmeth : str
        method to calculate saturation specific humidity from vapor pressure.
        See `AirSeaFluxCode.hum_subs.VaporPressure` for options.

    Returns
    -------
    float | numpy.ndarray
        saturation specific humidity of air [g/kg]

    See Also
    --------
    AirSeaFluxCode.hum_subs.VaporPressure
        For options for `qmeth` parameter.
    """
    if (hum[0] == "rh") or (hum[0] == "no"):
        RH = hum[1]
        if np.all(RH < 1):
            warnings.warn(
                "All relative humidity values < 1. "
                + "Input relative humidity units should be %. "
                + "Continuing with calculations assuming values are correct."
            )
        return qsat_air(T, P, RH, qmeth)  # q of air [g/kg]
    elif hum[0] == "q":
        qair = hum[1]  # [g/kg]
        if np.all(qair < 1):
            warnings.warn(
                "All humidity values < 1. "
                + "Input humidity units should be g/kg. "
                + "Continuing with calculations assuming values are correct."
            )
        return qair
    elif hum[0] == "Td":
        # dew point temperature (K)
        Td = validate_kelvin(hum[1], "Dew-point temperature")
        # Note this is a simplified 'Buck' - can exclude the Pressure component
        # as divided out (esd/ed)
        esd = 611.21 * np.exp(17.502 * ((Td - CtoK) / (Td - 32.19)))
        es = 611.21 * np.exp(17.502 * ((T - CtoK) / (T - 32.19)))
        RH = 100 * esd / es
        return qsat_air(T, P, RH, qmeth)  # q of air [g/kg]
    else:
        raise ValueError(f"(specific_humidity_air) Unknown humidity input: {hum[0]}")


# -----------------------------------------------------------------------------


def relative_humidity_from_qsat(
    q: np.ndarray,
    T: np.ndarray,
    P: np.ndarray,
    qmeth: str,
) -> np.ndarray:
    """
    Compute Relative Humidity from saturation specific humidity.

    Parameters
    ----------
    q : numpy.ndarray | float
        Saturation specific humidity of air [g/kg]
    T : numpy.ndarray | float
        Air temperature [K]
    P : float
        Air pressure at sea level [hPa]
    qmeth : str
        Method to calculate specific humidity from vapor pressure - see
        `AirSeaFluxCode.hum_subs.VaporPressure` for options.

    Returns
    -------
    numpy.ndarray | float
        Relative humidity [%]

    See Also
    --------
    AirSeaFluxCode.hum_subs.VaporPressure
        For options for `qmeth` parameter.
    """
    T = np.asarray(T)
    T = validate_kelvin(T)
    es = VaporPressure(T, P, phase="liquid", meth=qmeth)
    em = q * P / (622 + 0.378 * q)
    rh = 100 * em / es
    return rh


def dew_point_from_qsat(
    q: np.ndarray,
    T: np.ndarray,
    P: np.ndarray,
    qmeth: str,
) -> np.ndarray:
    """
    Compute Dew-point Temperature from saturation specific humidity, in Kelvin.
    This is a reverse of the `get_hum` functionality which converts dew-point
    temperature into relative humidity.

    Parameters
    ----------
    q : numpy.ndarray | float
        Saturation specific humidity of air [g/kg]
    T : numpy.ndarray | float
        Air temperature [K]
    P : float
        Air pressure at sea level [hPa]
    qmeth : str
        Method to calculate specific humidity from vapor pressure - see
        `AirSeaFluxCode.hum_subs.VaporPressure` for options.

    Returns
    -------
    numpy.ndarray | float
        Dew-point temperature [K]

    See Also
    --------
    AirSeaFluxCode.hum_subs.VaporPressure
        For options for `qmeth` parameter.
    AirSeaFluxCode.hum_subs.get_hum
        The reverse operation.
    """
    T = validate_kelvin(T)
    rh = relative_humidity_from_qsat(q, T, P, qmeth=qmeth)
    es = 611.21 * np.exp(17.502 * ((T - CtoK) / (T - 32.19)))
    esd = 0.01 * rh * es
    A = np.log(esd / 611.21) / 17.502
    num = A * (CtoK - 32.18)
    denom = 1 - A
    return num / denom + CtoK


def gamma(
    opt: str,
    sst: np.ndarray,
    t: np.ndarray,
    q: np.ndarray,
    cp: np.ndarray,
) -> np.ndarray:
    """
    Compute the adiabatic lapse-rate.

    Parameters
    ----------
    opt : str
        type of adiabatic lapse rate dry or "moist"
        dry has options to be constant "dry_c", for dry air "dry", or
        for unsaturated air with water vapor "dry_v"
    sst : float | numpy.ndarray
        sea surface temperature [K]
    t : float | numpy.ndarray
        air temperature [K]
    q : float | numpy.ndarray
        specific humidity of air [g/kg]
    cp : float
        specific capacity of air at constant Pressure

    Returns
    -------
    float | numpy.ndarray
        lapse rate [K/m]
    """
    sst = validate_kelvin(sst, "sst")
    t = validate_kelvin(t, "t")
    q = np.copy(q) / 1000  # convert to [kg/kg]
    if opt == "moist":
        t = np.maximum(t, 180)
        q = np.maximum(q, 1e-6)
        w = q / (1 - q)  # mixing ratio w = q/(1-q)
        iRT = 1 / (287.05 * t)
        # latent heat of vaporization of water as a function of temperature
        lv = (2.501 - 0.00237 * (sst - CtoK)) * 1e6
        gamma = (
            9.8
            * (1 + lv * w * iRT)
            / (1005 + np.power(lv, 2) * w * (287.05 / 461.495) * iRT / t)
        )
    elif opt == "dry_c":
        gamma = 0.0098 * np.ones(t.shape)
    elif opt == "dry":
        gamma = 9.81 / cp
    elif opt == "dry_v":
        w = q / (1 - q)  # mixing ratio
        f_v = 1 - 0.85 * w  # (1+w)/(1+w*)
        gamma = f_v * 9.81 / cp
    else:
        raise ValueError('(gamma) Unknown "opt" value')
    return gamma


# -----------------------------------------------------------------------------


def dew_point_to_vapor_pressure(
    Td: np.ndarray,
) -> np.ndarray:
    """
    Convert dew point temperature to partial vapor pressure following Buck
    (1981).

    Parameters
    ----------
    Td : numpy.ndarray | float
        Dew point (or temperature) value [K]

    Returns
    -------
    numpy.ndarray | float
        Partial vapor pressure [Pa]
    """
    Td = validate_kelvin(Td)
    esd = 611.21 * np.exp(17.502 * ((Td - CtoK) / (Td - 32.19)))
    return esd


def vapor_pressure_to_dew_point(
    vapor_pressure: np.ndarray,
) -> np.ndarray:
    """
    Convert vapor pressure to dew-point temperature, inverting the equation of
    Buck (1981).

    Parameters
    ----------
    vapor_pressure : numpy.ndarray | float
        Partial vapor pressure to H2O [Pa]

    Returns
    -------
    numpy.ndarray | float
        Dew-point temperature [K]
    """
    b = (np.log(vapor_pressure) - np.log(611.21)) / 17.502
    h = 240.97 * b
    t = 1 - b
    dew_point = h / t
    return dew_point + CtoK


def vapor_pressure_to_specific_humidity(
    vapor_pressure: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    """
    Converts vapor pressure to specific humidity.

    Parameters
    ----------
    vapor_pressure : numpy.ndarray | float
        Partial vapor pressure to H2O [Pa]

    P : numpy.ndarray | float
        Air pressure [hPa]

    Returns
    -------
    numpy.ndarray | float
        Specific humidity of air [g/kg]
    """
    gas_const_frac = 0.622
    return (
        1000.0
        * gas_const_frac
        * vapor_pressure
        / (P * 100 - (1 - gas_const_frac) * vapor_pressure)
    )


def specific_humidity_to_vapor_pressure(
    q: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    """
    Inverse of `vapor_pressure_to_specific_humidity`

    Parameters
    ----------
    q: numpy.ndarray | float

    P: numpy.ndarray | float
        Air pressure [hPa]

    Returns
    -------
    numpy.ndarray | float
        Vapor pressure [Pa]
    """
    gas_const_frac = 0.622
    a = q / (1000 * gas_const_frac)
    e_if_mixing_ratio = a * (P * 100)
    return e_if_mixing_ratio / (1 + a * (1 - gas_const_frac))


def dew_point_to_specific_humidity(
    Td: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    """
    Dew point to specific humidity

    Parameters
    ----------
    Td : numpy.ndarray | float
        Dew point temperature [K]

    P : numpy.ndarray | float
        Air pressure [hPa]

    Returns
    -------
    numpy.ndarray | float
        Specific humidity [g/kg]
    """
    return vapor_pressure_to_specific_humidity(
        dew_point_to_vapor_pressure(Td),
        P,
    )


def specific_humidity_to_dew_point(
    q: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    """
    Specific humidity to dew_point

    Parameters
    ----------
    q : numpy.ndarray | float
        Specific humidity [g/kg]

    P : numpy.ndarray | float
        Air pressure [hPa]

    Returns
    -------
    dew_point : numpy.ndarray | float
        Dew point temperature [K]
    """
    return vapor_pressure_to_dew_point(specific_humidity_to_vapor_pressure(q, P))
