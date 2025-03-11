import numpy as np

from .stratification import psim_calc, psit_calc
from ..util_subs import kappa


def get_Rb(grav, usr, hin_u, hin_t, tv, dtv, wind, monob, meth):
    """
    Calculate bulk Richardson number.

    Parameters
    ----------
    grav : float
        acceleration due to gravity [m/s2]
    usr : float
        friction wind speed [m/s]
    hin_u : float
        u sensor height [m]
    hin_t : float
        t sensor height [m]
    tv : float
        virtual temperature [K]
    dtv : float
        virtual temperature difference, air and sea [K]
    wind : float
        wind speed [m/s]
    monob : float
        Monin-Obukhov length from previous iteration step [m]
    meth : str
        bulk parameterisation method option: "S80", "S88", "LP82", "YT96",
        "UA", "NCAR", "C30", "C35", "ecmwf", "Beljaars"

    Returns
    -------
    Rb  : float
       Richardson number

    """
    # now input dtv
    # tvs = sst*(1+0.6077*qsea) # virtual SST
    # dtv = tv - tvs          # virtual air - sea temp. diff
    # adjust wind to t measurement height
    uz = (wind-usr/kappa*(np.log(hin_u/hin_t)-psim_calc(hin_u/monob, meth) +
                          psim_calc(hin_t/monob, meth)))
    Rb = grav*dtv*hin_t/(tv*uz*uz)
    return Rb

# ---------------------------------------------------------------------


def get_LRb(Rb, hin_t, monob, zo, zot, meth):
    """
    Calculate Monin-Obukhov length following ecmwf (IFS Documentation cy46r1).

    default for methods ecmwf and Beljaars

    Parameters
    ----------
    Rb  : float
       Richardson number
    hin_t : float
        t sensor height [m]
    monob : float
        Monin-Obukhov length from previous iteration step [m]
    zo   : float
        surface roughness       [m]
    zot   : float
        temperature roughness length       [m]
    meth : str
        bulk parameterisation method option: "S80", "S88", "LP82", "YT96",
        "UA", "NCAR", "C30", "C35", "ecmwf", "Beljaars"

    Returns
    -------
    monob : float
        M-O length [m]

    """
    zol = Rb*(np.power(
        np.log((hin_t+zo)/zo)-psim_calc((hin_t+zo)/monob, meth) +
        psim_calc(zo/monob, meth), 2)/(np.log((hin_t+zo)/zot) -
                                       psit_calc((hin_t+zo)/monob, meth) +
                                       psit_calc(zot/monob, meth)))
    monob = hin_t/zol
    return monob

# ---------------------------------------------------------------------


def get_Ltsrv(tsrv, grav, tv, usr):
    """
    Calculate Monin-Obukhov length from tsrv.

    Parameters
    ----------
    tsrv : float
        virtual star temperature [K]
    grav : float
        acceleration due to gravity [m/s2]
    tv : float
        virtual temperature [K]
    usr : float
        friction wind speed [m/s]

    Returns
    -------
    monob : float
        M-O length [m]

    """
    tsrv = np.maximum(np.abs(tsrv), 1e-9)*np.sign(tsrv)
    monob = (np.power(usr, 2)*tv)/(grav*kappa*tsrv)
    return monob

# ---------------------------------------------------------------------
