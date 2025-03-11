import numpy as np

from ..util_subs import kappa
from .stratification import psim_calc, psit_calc


def get_strs(hin, monob, wind, zo, zot, zoq, dt, dq, cd, ct, cq, meth):
    """
    Calculate star wind speed, temperature and specific humidity.

    Parameters
    ----------
    hin : float
        sensor heights [m]
    monob : float
        M-O length     [m]
    wind : float
        wind speed     [m/s]
    zo : float
        momentum roughness length    [m]
    zot : float
        temperature roughness length [m]
    zoq : float
        moisture roughness length    [m]
    dt : float
        temperature difference       [K]
    dq : float
        specific humidity difference [g/kg]
    cd : float
       drag coefficient
    ct : float
        temperature exchange coefficient
    cq : float
        moisture exchange coefficient
    meth : str
        bulk parameterisation method option: "S80", "S88", "LP82", "YT96",
        "UA", "NCAR", "C30", "C35", "ecmwf", "Beljaars"

    Returns
    -------
    usr : float
        friction wind speed [m/s]
    tsr : float
        star temperature    [K]
    qsr : float
        star specific humidity [g/kg]

    """
    usr = wind*np.sqrt(cd)
    tsr = ct*wind*dt/usr
    qsr = cq*wind*dq/usr

    if meth == "UA":
        # Zeng et al. 1998
        # away from extremes UA follows e.g. S80

        # momentum
        hol0 = hin[0]/np.copy(monob)
        # very unstable (Zeng et al. 1998 eq 7)
        usr = np.where(
            hol0 <= -1.574, wind*kappa/(np.log(-1.574*monob/zo) -
                                        psim_calc(-1.574, meth) +
                                        psim_calc(zo/monob, meth) +
                                        1.14*(np.power(-hin[0]/monob, 1/3) -
                                              np.power(1.574, 1/3))), usr)
        # very stable (Zeng et al. 1998 eq 10)
        usr = np.where(
            hol0 > 1, wind*kappa/(np.log(monob/zo)+5-5*zo/monob +
                                  5*np.log(hin[0]/monob)+hin[0]/monob-1), usr)

        # temperature
        hol1 = hin[1]/np.copy(monob)
        # very unstable (Zeng et al. 1998 eq 11)
        tsr = np.where(
            hol1 < -0.465, kappa*dt/(np.log((-0.465*monob)/zot) -
                                     psit_calc(-0.465, meth) +
                                     0.8*(np.power(0.465, -1/3) -
                                          np.power(-hin[1]/monob, -1/3))), tsr)
        # very stable (Zeng et al. 1998 eq 14)
        tsr = np.where(
            hol1 > 1, kappa*(dt)/(np.log(monob/zot)+5-5*zot/monob +
                                  5*np.log(hin[1]/monob)+hin[1]/monob-1), tsr)

        # humidity
        hol2 = hin[2]/monob
        # very unstable (Zeng et al. 1998 eq 11)
        qsr = np.where(
            hol2 < -0.465, kappa*dq/(np.log((-0.465*monob)/zoq) -
                                     psit_calc(-0.465, meth) +
                                     psit_calc(zoq/monob, meth) +
                                     0.8*(np.power(0.465, -1/3) -
                                          np.power(-hin[2]/monob, -1/3))), qsr)
        # very stable (Zeng et al. 1998 eq 14)
        qsr = np.where(hol2 > 1, kappa*dq/(np.log(monob/zoq)+5-5*zoq/monob +
                                           5*np.log(hin[2]/monob) +
                                           hin[2]/monob-1), qsr)
    return usr, tsr, qsr
# ---------------------------------------------------------------------


def get_tsrv(tsr, qsr, Ta, qair):
    """
    Calculate virtual star temperature.

    Parameters
    ----------
    tsr : float
        star temperature [K]
    qsr : float
        star specific humidity [g/kg]
    Ta : float
        air temperature [K]
    qair : float
        air specific humidity [g/kg]

    Returns
    -------
    tsrv : float
        virtual star temperature [K]

    """
    # NOTE: 0.6077 goes with mixing ratio or [kg/kg] humidity
    # tsrv = tsr*(1+0.6077*qair)+0.6077*Ta*qsr  # q [kg/kg]
    tsrv = 0.001*(tsr*(1000+0.6077*qair)+0.6077*Ta*qsr)  # q [g/kg]
    return tsrv

# ---------------------------------------------------------------------
