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

"""Coolskin and Warmlayer Subroutines"""

import numpy as np
from warnings import warn
from ..util_subs import PossibleCelsiusWarning


def delta(aw, Q, usr, grav):
    """
    Compute the thickness (m) of the viscous skin layer.

    Based on Fairall et al., 1996 and cited in IFS Documentation Cy46r1
    eq. 8.155 p. 164

    Parameters
    ----------
    aw : float
        thermal expansion coefficient of sea-water  [1/K]
    Q : float
        part of the net heat flux actually absorbed in the warm layer [W/m^2]
    usr : float
        friction velocity in the air (u*) [m/s]
    grav : float
        gravity                      [ms^-2]

    Returns
    -------
    delta : float
        the thickness (m) of the viscous skin layer

    """
    rhow, visw, tcw = 1025, 1e-6, 0.6
    # u* in the water
    usr_w = np.maximum(usr, 1e-4) * np.sqrt(1.2 / rhow)  # rhoa=1.2
    rcst_cs = 16 * grav * np.power(visw, 3) / np.power(tcw, 2)
    lm = 6 * (1 + (np.abs(Q) * aw * rcst_cs / np.power(usr_w, 4)) ** 0.75) ** (-1 / 3)
    ztmp = visw / usr_w
    delta = np.where(Q > 0, np.minimum(6 * ztmp, 0.007), np.minimum(lm * ztmp, 0.007))
    return delta


def get_dqer(dter, sst, qsea, lv):
    """
    Calculate humidity correction.

    Parameters
    ----------
    dter : float
        cool skin correction         [K]
    sst : float
        sea surface temperature      [K]
    qsea : float
        specific humidity over sea   [g/kg]
    lv : float
       latent heat of vaporization   [J/kg]

    Returns
    -------
    dqer : float
       humidity correction            [g/kg]

    """
    if np.nanmin(sst) < 200:
        warn(
            "(get_dqer) - sst assumed to be Kelvin, values < 200 detected.",
            PossibleCelsiusWarning,
        )
    wetc = 0.622 * lv * qsea / (287.1 * np.power(sst, 2))
    dqer = wetc * dter
    return dqer
