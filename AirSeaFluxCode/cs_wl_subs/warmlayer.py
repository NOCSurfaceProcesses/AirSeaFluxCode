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

"""Warmlayer Sub-routines"""

import numpy as np
from ..util_subs import CtoK, kappa, validate_kelvin


def wl_ecmwf(rho, Rs, Rnl, cp, lv, usr, tsr, qsr, sst, skt, dtc, grav):
    """
    Calculate warm layer correction following IFS Documentation cy46r1.
    and aerobulk (Brodeau et al., 2016)

    Parameters
    ----------
    rho : float
        density of air               [kg/m^3]
    Rs : float
        downward solar radiation    [Wm-2]
    Rnl : float
        net thermal radiation  [Wm-2]
    cp : float
       specific heat of air at constant pressure [J/K/kg]
    lv : float
       latent heat of vaporization   [J/kg]
    usr : float
        friction velocity           [m/s]
    tsr : float
       star temperature              [K]
    qsr : float
       star humidity                 [g/kg]
    sst : float
        bulk sst                    [K]
    skt : float
        skin sst from previous step [K]
    dtc : float
        cool skin correction        [K]
    grav : float
       gravity                      [ms^-2]

    Returns
    -------
    dtwl : float
        warm layer correction       [K]

    """
    sst = validate_kelvin(sst, "sst")
    rhow, cpw, visw, rd0 = 1025, 4190, 1e-6, 3
    Rns = 0.945 * Rs
    #  Previous value of dT / warm-layer, adapted to depth:
    # thermal expansion coefficient of sea-water (SST accurate enough#)
    aw = 2.1e-5 * np.power(np.maximum(sst - CtoK + 3.2, 0), 0.79)
    # *** Rd = Fraction of solar radiation absorbed in warm layer (-)
    a1, a2, a3 = 0.28, 0.27, 0.45
    b1, b2, b3 = -71.5, -2.8, -0.06  # [m-1]
    Rd = 1 - (a1 * np.exp(b1 * rd0) + a2 * np.exp(b2 * rd0) + a3 * np.exp(b3 * rd0))
    shf = rho * cp * usr * tsr
    lhf = rho * lv * usr * qsr * 0.001  # qsr [g/kg]
    Qnsol = shf + lhf + Rnl
    usrw = np.maximum(usr, 1e-4) * np.sqrt(1.2 / rhow)  # u* in the water
    zc3 = rd0 * kappa * grav / np.power(1.2 / rhow, 3 / 2)
    zc4 = (0.3 + 1) * kappa / rd0
    zc5 = (0.3 + 1) / (0.3 * rd0)
    for jwl in range(10):  # iteration to solve implicitly eq. for warm layer
        dsst = skt - sst - dtc
        # Buoyancy flux and stability parameter (zdl = -z/L) in water
        ZSRD = (Qnsol + Rns * Rd) / (rhow * cpw)
        ztmp = np.maximum(dsst, 0)
        zdl = np.where(
            ZSRD > 0,
            2 * (np.power(usrw, 2) * np.sqrt(ztmp / (5 * rd0 * grav * aw / visw)))
            + ZSRD,
            np.power(usrw, 2) * np.sqrt(ztmp / (5 * rd0 * grav * aw / visw)),
        )
        usr = np.maximum(usr, 1e-4)
        zdL = zc3 * aw * zdl / np.power(usr, 3)
        # Stability function Phi_t(-z/L) (zdL is -z/L) :
        zphi = np.where(
            zdL > 0,
            (
                1
                + (5 * zdL + 4 * np.power(zdL, 2))
                / (1 + 3 * zdL + 0.25 * np.power(zdL, 2))
                + 2 / np.sqrt(1 - 16 * (-np.abs(zdL)))
            ),
            1 / np.sqrt(1 - 16 * (-np.abs(zdL))),
        )
        zz = zc4 * (usrw) / zphi
        zz = np.maximum(np.abs(zz), 1e-4) * np.sign(zz)
        dtwl = np.maximum(0, (zc5 * ZSRD) / zz)
    return dtwl
