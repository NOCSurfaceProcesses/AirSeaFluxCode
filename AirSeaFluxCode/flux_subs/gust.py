import numpy as np


def get_gust(beta, zi, ustb, Ta, usr, tsrv, grav):
    """
    Compute gustiness.

    Parameters
    ----------
    beta : float
        constant
    zi : int
        scale height of the boundary layer depth [m]
    ustb : float
        gust wind in stable conditions   [m/s]
    Ta : float
        air temperature   [K]
    usr : float
        friction velocity [m/s]
    tsrv : float
        star virtual temperature of air [K]
    grav : float
        gravity

    Returns
    -------
    ug : float        [m/s]
    """
    if np.nanmax(Ta) < 200:  # convert to K if in Celsius
        Ta = Ta+273.16
    # minus sign to allow cube root
    Bf = (-grav/Ta)*usr*tsrv
    ug = np.ones(np.shape(Ta))*ustb
    ug = np.where(Bf > 0, np.maximum(beta*np.power(Bf*zi, 1/3), ustb), ustb)
    return ug
# ---------------------------------------------------------------------


def apply_GF(gust, spd, wind, step):
    """
    Apply gustiness factor according if gustiness ON.

    There are different ways to remove the effect of gustiness according to
    the user's choice.

    Parameters
    ----------
    gust : int
        option on how to apply gustiness
        0: gustiness is switched OFF
        1: gustiness is switched ON following Fairall et al.
        2: gustiness is switched ON and GF is removed from TSFs u10n, uref
        3: gustiness is switched ON and GF=1
        4: gustiness is switched ON following ECMWF 
        5: gustiness is switched ON following Zeng et al. (1998) 
        6: gustiness is switched ON following C35 matlab code
    spd : float
        wind speed                      [ms^{-1}]
    wind : float
        wind speed including gust       [ms^{-1}]
    step : str
        step during AirSeaFluxCode the GF is applied: "u", "TSF"

    Returns
    -------
    GustFact : float
        gustiness factor.

    """
    # 1. following C35 documentation, 2. use GF to TSF, u10n uzout,
    # 3. GF=1, 4. UA,  5. C35 code 6. ecmwf aerobulk)
    # ratio of gusty to horizontal wind; gustiness factor
    if step in ["u"]:
        GustFact = wind*0+1
        if gust[0] in [1, 2]:
            GustFact = np.sqrt(wind/spd)
        elif gust[0] == 6:
            # as in C35 matlab code
            GustFact = wind/spd
    elif step == "TSF":
        # remove effect of gustiness  from TSFs
        # here it is a 3xspd.shape array
        GustFact = np.ones([3, spd.shape[0]], dtype=float)
        GustFact[0, :] = wind/spd
        GustFact[1:3, :] = wind*0+1
        # following Fairall et al. (2003)
        if gust[0] == 2:
            # usr is divided by (GustFact)^0.5 (here applied to sensible and
            # latent as well as tau)
            GustFact[1:3, :] = np.sqrt(wind/spd)
        elif gust[0] == 3:
            GustFact[0, :] = wind*0+1
    return GustFact
# ---------------------------------------------------------------------
