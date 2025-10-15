# Copyright 2023-2025, Stavroula Biri
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

"""Height Adjustment functions."""

import numpy as np

from AirSeaFluxCode.util_subs import kappa
from AirSeaFluxCode.flux_subs import psit_calc, psim_calc


def adjust_temperature(
    temp: np.ndarray | float,
    tsr: np.ndarray | float,
    h_in: np.ndarray | float,
    h_out: np.ndarray | float,
    monob: np.ndarray | float,
    tlapse: np.ndarray | float = 0.0097611,
    meth: str = "S80",
) -> np.ndarray | float:
    """
    Estimate the air temperature at a target output height above the surface from a
    known air temperature at a known input height using a Monin-Obhukov length-scale.

    Parameters
    ----------
    temp : numpy.ndarray
        The air temperature value at the input height `h_in`. The temperature can be
        provided in either Celsius or Kelvin.
    tsr : numpy.ndarray
        Star temperature [K].
    h_in : numpy.ndarray | float
        The input height(s) for the known air temperature values [m].
    h_out : numpy.ndarray | float
        The target output height(s) for the air temperature [m].
    monob : numpy.ndarray
        The Monin-Obhukov length [m]
    tlapse : numpy.ndarray | float
        The adiabatic lapse-rate [K/m]. Can be computed using
        :py:func:`AirSeaFluxCode.hum_subs.gamma`, defaults to the 0.0097611 [K/m], the
        dry adiabatic lapse rate.
    meth : str
        The flux method used to compute stability functions. One of "S80", "S88",
        "LP82", "YT96", "UA", "NCAR", "C30", "C35", "ecmwf", "Beljaars".

    Returns
    -------
    numpy.ndarray
        The estimated air temperature at the target output height. Units match
        the input air temperature value.
    """
    return (
        temp
        + tlapse * (h_in - h_out)
        - (
            tsr
            / kappa
            * (
                np.log(h_in / h_out)
                - psit_calc(h_in / monob, meth)
                + psit_calc(h_out / monob, meth)
            )
        )
    )


def adjust_wind_speed(
    spd: np.ndarray | float,
    usr: np.ndarray | float,
    h_in: np.ndarray | float,
    h_out: np.ndarray | float,
    monob: np.ndarray | float,
    meth: str = "S80",
) -> np.ndarray | float:
    """
    Estimate a wind-speed at a target output height above the surface from a
    known wind-speed at a known input height using a Monin-Obhukov length-scale.

    Parameters
    ----------
    spd : numpy.ndarray
        The wind-speed value at the input height `h_in` [m/s].
    usr : numpy.ndarray
        Star wind-speed / friction velocity [m/s].
    h_in : numpy.ndarray | float
        The input height(s) for the known wind-speed values [m].
    h_out : numpy.ndarray | float
        The target output height(s) for the wind-speed [m].
    monob : numpy.ndarray
        The Monin-Obhukov length [m]
    meth : str
        The flux method used to compute stability functions. One of "S80", "S88",
        "LP82", "YT96", "UA", "NCAR", "C30", "C35", "ecmwf", "Beljaars".

    Returns
    -------
    numpy.ndarray
        The estimated wind-speed at the target output height [m/s]
    """
    return spd - usr / kappa * (
        np.log(h_in / h_out)
        - psim_calc(h_in / monob, meth)
        + psim_calc(h_out / monob, meth)
    )


def adjust_humidity(
    qair: np.ndarray | float,
    qsr: np.ndarray | float,
    h_in: np.ndarray | float,
    h_out: np.ndarray | float,
    monob: np.ndarray | float,
    meth: str = "S80",
) -> np.ndarray | float:
    """
    Estimate a specific humidity (of air) at a target output height above the surface
    from a known specific humidity (of air) at a known input height using a
    Monin-Obhukov length-scale.

    Parameters
    ----------
    qair : numpy.ndarray
        The specific humidity of air value at the input height `h_in` [g/kg].
    qsr : numpy.ndarray
        Star specific humidity [g/kg].
    h_in : numpy.ndarray | float
        The input height(s) for the known specific humidity values [m].
    h_out : numpy.ndarray | float
        The target output height(s) for the specific humidity [m].
    monob : numpy.ndarray
        The Monin-Obhukov length [m]
    meth : str
        The flux method used to compute stability functions. One of "S80", "S88",
        "LP82", "YT96", "UA", "NCAR", "C30", "C35", "ecmwf", "Beljaars".

    Returns
    -------
    numpy.ndarray
        The estimated specific humidity at the target output height [g/kg]
    """
    return qair - qsr / kappa * (
        np.log(h_in / h_out)
        - psit_calc(h_in / monob, meth)
        + psit_calc(h_out / monob, meth)
    )
