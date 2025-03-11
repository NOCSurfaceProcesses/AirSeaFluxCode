import numpy as np

from .vapor_pressure import VaporPressure
from ..util_subs import CtoK


def qsat_sea(T, P, qmeth):
    r"""
    Compute surface saturation specific humidity [g/kg].

    Parameters
    ----------
    T : float
        temperature [$^\circ$\,C]
    P : float
        pressure [mb]
    qmeth : str
        method to calculate vapor pressure

    Returns
    -------
    qs : float
        surface saturation specific humidity [g/kg]
    """
    T = np.asarray(T)
    if np.nanmin(T) > 200:  # if Ta in Kelvin convert to Celsius
        T = T-CtoK
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
        temperature [$^\circ$\,C]
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
    if np.nanmin(T) > 200:  # if Ta in Kelvin convert to Celsius
        T = T-CtoK
    es = VaporPressure(T, P, 'liquid', qmeth)
    em = 0.01*rh*es
    q = 622*em/(P-0.378*em)
    return q  # [g/kg]
# -----------------------------------------------------------------------------
