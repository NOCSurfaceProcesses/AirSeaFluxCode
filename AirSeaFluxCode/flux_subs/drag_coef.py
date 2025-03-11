import numpy as np
from ..util_subs import (kappa, visc_air)

# ---------------------------------------------------------------------


def cdn_calc(u10n, usr, Ta, grav, meth):
    """
    Calculate neutral drag coefficient.

    Parameters
    ----------
    u10n : float
        neutral 10m wind speed [m/s]
    usr : float
        friction velocity      [m/s]
    Ta   : float
        air temperature        [K]
    grav : float
        gravity               [m/s^2]
    meth : str

    Returns
    -------
    cdn : float
    zo  : float
    """
    cdn = np.zeros(Ta.shape)*np.nan
    if meth == "S80":  # eq. 14 Smith 1980
        cdn = np.maximum((0.61+0.063*u10n)*0.001, (0.61+0.063*6)*0.001)
    elif meth == "LP82":
        #  Large & Pond 1981 u10n <11m/s & eq. 21 Large & Pond 1982
        cdn = np.where(u10n < 11, 1.2*0.001, (0.49+0.065*u10n)*0.001)
    elif meth in ["S88", "UA", "ecmwf", "C30", "C35", "Beljaars"]:
        cdn = cdn_from_roughness(u10n, usr, Ta, grav, meth)
    elif meth == "YT96":
        # convert usr in eq. 21 to cdn to expand for low wind speeds
        cdn = np.power((0.10038+u10n*2.17e-3+np.power(u10n, 2)*2.78e-3 -
                        np.power(u10n, 3)*4.4e-5)/u10n, 2)
    elif meth == "NCAR":  # eq. 11 Large and Yeager 2009
        cdn = np.where(u10n > 0.5, (0.142+2.7/u10n+u10n/13.09 -
                                    3.14807e-10*np.power(u10n, 6))*1e-3,
                       (0.142+2.7/0.5+0.5/13.09 -
                        3.14807e-10*np.power(0.5, 6))*1e-3)
        cdn = np.where(u10n > 33, 2.34e-3, np.copy(cdn))
        cdn = np.maximum(np.copy(cdn), 0.1e-3)
    else:
        raise ValueError("Unknown method cdn: "+meth)

    zo = 10/np.exp(kappa/np.sqrt(cdn))
    return cdn, zo
# ---------------------------------------------------------------------


def cdn_from_roughness(u10n, usr, Ta, grav, meth):
    """
    Calculate neutral drag coefficient from roughness length.

    Parameters
    ----------
    u10n : float
        neutral 10m wind speed [m/s]
    usr : float
        friction velocity      [m/s]
    Ta   : float
        air temperature        [K]
    grav : float                [m/s]
        gravity
    meth : str

    Returns
    -------
    cdn : float
    """
    #  cdn = (0.61+0.063*u10n)*0.001
    zo, zc, zs = np.zeros(Ta.shape), np.zeros(Ta.shape), np.zeros(Ta.shape)
    for it in range(5):
        if meth == "S88":
            # Charnock roughness length (eq. 4 in Smith 88)
            zc = 0.011*np.power(usr, 2)/grav
            #  smooth surface roughness length (eq. 6 in Smith 88)
            zs = 0.11*visc_air(Ta)/usr
            zo = zc + zs  # eq. 7 & 8 in Smith 88
        elif meth == "UA":
            # valid for 0<u<18m/s # Zeng et al. 1998 (24)
            zo = 0.013*np.power(usr, 2)/grav+0.11*visc_air(Ta)/usr
        elif meth == "C30":  # eq. 25 Fairall et al. 1996a
            a = 0.011*np.ones(Ta.shape)
            a = np.where(u10n > 10, 0.011+(u10n-10)*(0.018-0.011)/(18-10),
                         np.where(u10n > 18, 0.018, a))
            zo = a*np.power(usr, 2)/grav+0.11*visc_air(Ta)/usr
        elif meth == "C35":  # eq.6-11 Edson et al. (2013)
            zo = (0.11*visc_air(Ta)/usr +
                  np.minimum(0.0017*19-0.0050, 0.0017*u10n-0.0050) *
                  np.power(usr, 2)/grav)
        elif meth in ["ecmwf", "Beljaars"]:
            # eq. (3.26) p.38 over sea IFS Documentation cy46r1
            zo = 0.018*np.power(usr, 2)/grav+0.11*visc_air(Ta)/usr
        else:
            raise ValueError("Unknown method for cdn_from_roughness "+meth)

        cdn = np.power(kappa/np.log(10/zo), 2)
    return cdn
# ---------------------------------------------------------------------


def cd_calc(cdn, hin, hout, psim):
    """
    Calculate drag coefficient at reference height.

    Parameters
    ----------
    cdn : float
        neutral drag coefficient
    hin : float
        wind speed height       [m]
    hout : float
        reference height        [m]
    psim : float
        momentum stability function

    Returns
    -------
    cd : float
    """
    cd = (cdn/np.power(1+(np.sqrt(cdn)*(np.log(hin/hout)-psim))/kappa, 2))
    return cd
# ---------------------------------------------------------------------
