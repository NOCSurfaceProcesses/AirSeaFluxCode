import numpy as np
import warnings

from .qsat import qsat_air, qsat_sea
from ..util_subs import CtoK


def get_hum(hum, T, sst, P, qmeth):
    """
    Get specific humidity output.

    Parameters
    ----------
    hum : array
        humidity input switch 2x1 [x, values] default is relative humidity
            x='rh' : relative humidity [%]
            x='q' : specific humidity [g/kg]
            x='Td' : dew point temperature [K]
    T : float
        air temperature [K]
    sst : float
        sea surface temperature [K]
    P : float
        air pressure at sea level [hPa]
    qmeth : str
        method to calculate specific humidity from vapor pressure

    Returns
    -------
    qair : float
        specific humidity of air [g/kg]
    qsea : float
        specific humidity over sea surface [g/kg]

    """
    if ((hum[0] == 'rh') or (hum[0] == 'no')):
        RH = hum[1]
        if np.all(RH < 1):
            warnings.warn(
                "All relative humidity values < 1. " +
                "Input relative humidity units should be %. " +
                "Continuing with calculations assuming values are correct."
            )
        qsea = qsat_sea(sst, P, qmeth)  # surface water q [g/kg]
        qair = qsat_air(T, P, RH, qmeth)  # q of air [g/kg]
    elif hum[0] == 'q':
        qair = hum[1]  # [g/kg]
        if np.all(qair < 1):
            warnings.warn(
                "All humidity values < 1. " +
                "Input humidity units should be g/kg. " +
                "Continuing with calculations assuming values are correct."
            )
        qsea = qsat_sea(sst, P, qmeth)  # surface water q [g/kg]
    elif hum[0] == 'Td':
        Td = hum[1]  # dew point temperature (K)
        Td = np.where(Td < 200, np.copy(Td)+CtoK, np.copy(Td))
        T = np.where(T < 200, np.copy(T)+CtoK, np.copy(T))
        esd = 611.21*np.exp(17.502*((Td-273.16)/(Td-32.19)))
        es = 611.21*np.exp(17.502*((T-273.16)/(T-32.19)))
        RH = 100*esd/es
        qair = qsat_air(T, P, RH, qmeth)  # q of air [g/kg]
        qsea = qsat_sea(sst, P, qmeth)    # surface water q [g/kg]
    else:
        raise ValueError('(get_hum) Unknown humidity input')
    return qair, qsea
# -----------------------------------------------------------------------------


def gamma(opt, sst, t, q, cp):
    """
    Compute the adiabatic lapse-rate.

    Parameters
    ----------
    opt : str
        type of adiabatic lapse rate dry or "moist"
        dry has options to be constant "dry_c", for dry air "dry", or
        for unsaturated air with water vapor "dry_v"
    sst : float
        sea surface temperature [K]
    t : float
        air temperature [K]
    q : float
        specific humidity of air [g/kg]
    cp : float
        specific capacity of air at constant Pressure

    Returns
    -------
    gamma : float
        lapse rate [K/m]

    """
    q = np.copy(q) / 1000  # convert to [kg/kg]
    if np.nanmin(sst) < 200:  # if sst in Celsius convert to Kelvin
        sst = sst+CtoK
    if np.nanmin(t) < 200:  # if t in Celsius convert to Kelvin
        t = t+CtoK
    if opt == "moist":
        t = np.maximum(t, 180)
        q = np.maximum(q,  1e-6)
        w = q/(1-q)  # mixing ratio w = q/(1-q)
        iRT = 1/(287.05*t)
        # latent heat of vaporization of water as a function of temperature
        lv = (2.501-0.00237*(sst-CtoK))*1e6
        gamma = 9.8*(1+lv*w*iRT)/(1005+np.power(lv, 2)*w*(287.05/461.495) *
                                  iRT/t)
    elif opt == "dry_c":
        gamma = 0.0098*np.ones(t.shape)
    elif opt == "dry":
        gamma = 9.81/cp
    elif opt == "dry_v":
        w = q/(1-q)  # mixing ratio
        f_v = 1-0.85*w  # (1+w)/(1+w*)
        gamma = f_v*9.81/cp
    else:
        raise ValueError('(gamma) Unknown "opt" value')
    return gamma
# -----------------------------------------------------------------------------
