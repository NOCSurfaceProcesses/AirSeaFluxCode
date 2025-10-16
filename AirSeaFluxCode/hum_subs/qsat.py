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

import numpy as np

from .vapor_pressure import VaporPressure


def qsat_sea(T, P, qmeth):
    r"""
    Compute surface saturation specific humidity [g/kg].

    Parameters
    ----------
    T : float
        temperature [K]
    P : float
        pressure [mb]
    qmeth : str
        method to calculate vapor pressure

    Returns
    -------
    qs : float
        surface saturation specific humidity [g/kg]
    """
    ex = VaporPressure(T, P, 'liquid', qmeth)
    es = 0.98*ex  # reduction at sea surface
    qs = 622*es/(P-0.378*es)
    return qs  # [g/kg]
# -----------------------------------------------------------------------------


def qsat_air(T, P, rh, qmeth):
    r"""
    Compute saturation specific humidity [g/kg].

    Parameters
    ----------
    T : float
        temperature [K]
    P : float
        pressure [mb]
    rh : float
       relative humidity [%]
    qmeth : str
        method to calculate vapor pressure

    Returns
    -------
    q : float
        specific humidity [g/kg]
    """
    T = np.asarray(T)
    es = VaporPressure(T, P, 'liquid', qmeth)
    em = 0.01*rh*es
    q = 622*em/(P-0.378*em)
    return q  # [g/kg]
# -----------------------------------------------------------------------------
