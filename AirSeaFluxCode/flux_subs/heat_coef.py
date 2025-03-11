import numpy as np
from ..util_subs import (kappa, visc_air)


def ctqn_calc(corq, zol, cdn, usr, zo, Ta, meth):
    """
    Calculate neutral heat and moisture exchange coefficients.

    Parameters
    ----------
    corq : flag to select
           "ct" or "cq"
    zol  : float
        height over MO length
    cdn  : float
        neutral drag coefficient
    usr : float
        friction velocity      [m/s]
    zo   : float
        surface roughness       [m]
    Ta   : float
        air temperature         [K]
    meth : str

    Returns
    -------
    ctqn : float
        neutral heat exchange coefficient
    zotq : float
        roughness length for t or q
    """
    if meth in ["S80", "S88", "YT96"]:
        cqn = np.ones(Ta.shape)*1.20*0.001  # from S88
        ctn = np.ones(Ta.shape)*1.00*0.001
        zot = 10/(np.exp(np.power(kappa, 2) / (ctn*np.log(10/zo))))
        zoq = 10/(np.exp(np.power(kappa, 2) / (cqn*np.log(10/zo))))
    elif meth == "LP82":
        cqn = np.where((zol <= 0), 1.15*0.001, 1*0.001)
        ctn = np.where((zol <= 0), 1.13*0.001, 0.66*0.001)
        zot = 10/(np.exp(np.power(kappa, 2)/(ctn*np.log(10/zo))))
        zoq = 10/(np.exp(np.power(kappa, 2)/(cqn*np.log(10/zo))))
    elif meth == "NCAR":
        # Eq. (9),(12), (13) Large & Yeager, 2009
        cqn = np.maximum(34.6*0.001*np.sqrt(cdn), 0.1e-3)
        ctn = np.maximum(np.where(zol < 0, 32.7*1e-3*np.sqrt(cdn),
                                  18*1e-3*np.sqrt(cdn)), 0.1e-3)
        zot = 10/(np.exp(np.power(kappa, 2)/(ctn*np.log(10/zo))))
        zoq = 10/(np.exp(np.power(kappa, 2)/(cqn*np.log(10/zo))))
    elif meth == "UA":
        # Zeng et al. 1998 (25)
        rr = usr*zo/visc_air(Ta)
        zoq = zo/np.exp(2.67*np.power(rr, 1/4)-2.57)
        zot = np.copy(zoq)
        cqn = np.power(kappa, 2)/(np.log(10/zo)*np.log(10/zoq))
        ctn = np.power(kappa, 2)/(np.log(10/zo)*np.log(10/zoq))
    elif meth == "C30":
        rr = zo*usr/visc_air(Ta)
        zoq = np.minimum(5e-5/np.power(rr, 0.6), 1.15e-4)  # moisture roughness
        zot = np.copy(zoq)  # temperature roughness
        cqn = np.power(kappa, 2)/np.log(10/zo)/np.log(10/zoq)
        ctn = np.power(kappa, 2)/np.log(10/zo)/np.log(10/zot)
    elif meth == "C35":
        rr = zo*usr/visc_air(Ta)
        zoq = np.minimum(5.8e-5/np.power(rr, 0.72), 1.6e-4)  # moisture rough.
        zot = np.copy(zoq)  # temperature roughness
        cqn = np.power(kappa, 2)/np.log(10/zo)/np.log(10/zoq)
        ctn = np.power(kappa, 2)/np.log(10/zo)/np.log(10/zot)
    elif meth in ["ecmwf", "Beljaars"]:
        # eq. (3.26) p.38 over sea IFS Documentation cy46r1
        zot = 0.40*visc_air(Ta)/usr
        zoq = 0.62*visc_air(Ta)/usr
        cqn = np.power(kappa, 2)/np.log(10/zo)/np.log(10/zoq)
        ctn = np.power(kappa, 2)/np.log(10/zo)/np.log(10/zot)
    else:
        raise ValueError("Unknown method ctqn: "+meth)

    if corq == "ct":
        ctqn = ctn
        zotq = zot
    elif corq == "cq":
        ctqn = cqn
        zotq = zoq
    else:
        raise ValueError("Unknown flag - should be ct or cq: "+corq)

    return ctqn, zotq
# ---------------------------------------------------------------------


def ctq_calc(cdn, cd, ctqn, hin, hout, psitq):
    """
    Calculate heat and moisture exchange coefficients at reference height.

    Parameters
    ----------
    cdn : float
        neutral drag coefficient
    cd  : float
        drag coefficient at reference height
    ctqn : float
        neutral heat or moisture exchange coefficient
    hin : float
        original temperature or humidity sensor height [m]
    hout : float
        reference height                   [m]
    psitq : float
        heat or moisture stability function

    Returns
    -------
    ctq : float
       heat or moisture exchange coefficient
    """
    ctq = (ctqn*np.sqrt(cd/cdn) /
           (1+ctqn*((np.log(hin/hout)-psitq)/(kappa*np.sqrt(cdn)))))

    return ctq
# ---------------------------------------------------------------------
