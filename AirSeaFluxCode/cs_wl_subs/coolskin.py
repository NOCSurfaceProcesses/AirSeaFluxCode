import numpy as np
from ..util_subs import CtoK
from .cs_wl_subs import delta


def cs(sst, d, rho, Rs, Rnl, cp, lv, usr, tsr, qsr, grav, opt):
    """
    Compute cool skin.

    Based on COARE3.5 (Fairall et al. 1996, Edson et al. 2013)

    Parameters
    ----------
    sst : float
        sea surface temperature      [K]
    d   : float
        cool skin thickness           [m]
    rho : float
        density of air               [kg/m^3]
    Rs : float
        downward shortwave radiation [Wm-2]
    Rnl : float
        net upwelling IR radiation       [Wm-2]
    cp : float
        specific heat of air at constant pressure [J/K/kg]
    lv : float
        latent heat of vaporization   [J/kg]
    usr : float
        friction velocity             [ms^-1]
    tsr : float
        star temperature              [K]
    qsr : float
        star humidity                 [g/kg]
    grav : float
        gravity                      [ms^-2]
    opt  : str
        method to follow
    Returns
    -------
    dter : float
        cool skin correction         [K]
    delta : float
        cool skin thickness           [m]
    """
    # coded following Saunders (1967) with lambda = 6
    if (np.nanmin(sst) > 200):  # if sst in Kelvin convert to Celsius
        sst = sst-CtoK
    # ************  cool skin constants  *******
    # density of water, specific heat capacity of water, water viscosity,
    # thermal conductivity of water
    tcw = 0.6
    Rns = 0.945*Rs  # albedo correction
    shf = rho*cp*usr*tsr
    lhf = rho*lv*usr*qsr*0.001  # qsr [g/kg]
    Qnsol = shf+lhf+Rnl
    if opt == "C35":
        cpw = 4000
        aw = 2.1e-5*np.power(sst+3.2, 0.79)
        # d = delta(aw, Qnsol, usr, grav, rho, opt)
        fs = 0.065+11*d-6.6e-5/d*(1-np.exp(-d/8.0e-4))  # eq. 17 F96
        # in F96 first term in eq. 17 is 0.137 insted of 0.065
        Q = Qnsol+Rns*fs
        Qb = aw*Q+0.026*np.minimum(lhf, 0)*cpw/lv  # eq. 8 F96
        d = delta(aw, Qb, usr, grav)
    elif opt == "ecmwf":
        aw = np.maximum(1e-5, 1e-5*(sst-CtoK))
        # d = delta(aw, Qnsol, usr, grav, rho, opt)
        for jc in range(4):
            # fraction of the solar radiation absorbed in layer delta eq. 8.153
            # and Eq.(5) Zeng & Beljaars, 2005
            fs = 0.065+11*d-6.6e-5/d*(1-np.exp(-d/8e-4))  # eq. 8.153 Cy46r1
            Q = Qnsol+Rns*fs
            d = delta(aw, Q, usr, grav)
    dter = Q*d/tcw  # eq. 4 F96
    return dter, d
# ---------------------------------------------------------------------


def cs_C35(sst, rho, Rs, Rnl, cp, lv, delta, usr, tsr, qsr, grav):
    """
    Compute cool skin.

    Based on COARE3.5 (Fairall et al. 1996, Edson et al. 2013)

    Parameters
    ----------
    sst : float
        sea surface temperature      [K]
    rho : float
        density of air               [kg/m^3]
    Rs : float
        downward shortwave radiation [Wm-2]
    Rnl : float
        net upwelling IR radiation       [Wm-2]
    cp : float
        specific heat of air at constant pressure [J/K/kg]
    lv : float
        latent heat of vaporization   [J/kg]
    delta : float
        cool skin thickness           [m]
    usr : float
        friction velocity             [m/s]
    tsr : float
        star temperature              [K]
    qsr : float
        star humidity                 [g/kg]
    grav : float
        gravity                      [ms^-2]

    Returns
    -------
    dter : float
        cool skin correction         [K]
    dqer : float
        humidity corrction            [g/kg]
    delta : float
        cool skin thickness           [m]
    """
    # coded following Saunders (1967) with lambda = 6
    if np.nanmin(sst) > 200:  # if sst in Kelvin convert to Celsius
        sst = sst-CtoK
    # ************  cool skin constants  *******
    # density of water, specific heat capacity of water, water viscosity,
    # thermal conductivity of water
    rhow, cpw, visw, tcw = 1022, 4000, 1e-6, 0.6
    aw = 2.1e-5*np.power(np.maximum(sst+3.2, 0), 0.79)
    bigc = 16*grav*cpw*np.power(rhow*visw, 3)/(np.power(tcw, 2)*np.power(
        rho, 2))
    Rns = 0.945*Rs  # albedo correction
    shf = rho*cp*usr*tsr
    lhf = rho*lv*usr*qsr*0.001  # qsr [g/kg]
    Qnsol = shf+lhf+Rnl
    fs = 0.065+11*delta-6.6e-5/delta*(1-np.exp(-delta/8.0e-4))
    Q = Qnsol+Rns*fs
    Qb = aw*Q+0.026*np.minimum(lhf, 0)*cpw/lv
    xlamx = 6*np.ones(sst.shape)
    xlamx = np.where(Qb > 0, 6, 6/(1+(bigc*np.abs(Qb)/usr**4)**0.75)**0.333)
    delta = np.where(
        Qb > 0, np.minimum(xlamx*visw/(np.sqrt(rho/rhow)*usr), 0.01),
        xlamx*visw/(np.sqrt(rho/rhow)*usr))
    dter = Q*delta/tcw
    return dter, delta
# ----------------


def cs_ecmwf(rho, Rs, Rnl, cp, lv, usr, tsr, qsr, sst, grav):
    """
    cool skin adjustment based on IFS Documentation cy46r1

    Parameters
    ----------
    rho : float
        density of air               [kg/m^3]
    Rs : float
        downward solar radiation [Wm-2]
    Rnl : float
        net thermal radiation     [Wm-2]
    cp : float
       specific heat of air at constant pressure [J/K/kg]
    lv : float
       latent heat of vaporization   [J/kg]
    usr : float
       friction velocity         [m/s]
    tsr : float
       star temperature              [K]
    qsr : float
       star humidity                 [g/kg]
    sst : float
        sea surface temperature  [K]
    grav : float
       gravity                      [ms^-2]

    Returns
    -------
    dtc : float
        cool skin temperature correction [K]

    """
    if np.nanmin(sst) < 200:  # if sst in Celsius convert to Kelvin
        sst = sst+CtoK
    aw = np.maximum(1e-5, 1e-5*(sst-CtoK))
    Rns = 0.945*Rs  # (net solar radiation (albedo correction)
    shf = rho*cp*usr*tsr
    lhf = rho*lv*usr*qsr*0.001  # qsr [g/kg]
    Qnsol = shf+lhf+Rnl  # eq. 8.152
    d = delta(aw, Qnsol, usr, grav)
    for jc in range(4):  # because implicit in terms of delta...
        # # fraction of the solar radiation absorbed in layer delta eq. 8.153
        # and Eq.(5) Zeng & Beljaars, 2005
        fs = 0.065+11*d-6.6e-5/d*(1-np.exp(-d/8e-4))
        Q = Qnsol+fs*Rns
        d = delta(aw, Q, usr, grav)
    dtc = Q*d/0.6  # (rhow*cw*kw)eq. 8.151
    return dtc


def cs_Beljaars(rho, Rs, Rnl, cp, lv, usr, tsr, qsr, grav, Qs):
    """
    cool skin adjustment based on Beljaars (1997)
    air-sea interaction in the ECMWF model

    Parameters
    ----------
    rho : float
        density of air           [kg/m^3]
    Rs : float
        downward solar radiation [Wm-2]
    Rnl : float
        net thermal radiaion     [Wm-2]
    cp : float
       specific heat of air at constant pressure [J/K/kg]
    lv : float
       latent heat of vaporization   [J/kg]
    usr : float
       friction velocity         [m/s]
    tsr : float
       star temperature              [K]
    qsr : float
       star humidity                 [g/kg]
    grav : float
       gravity                      [ms^-2]
    Qs : float
      radiation balance

    Returns
    -------
    Qs : float
      radiation balance
    dtc : float
        cool skin temperature correction [K]

    """
    tcw = 0.6       # thermal conductivity of water (at 20C) [W/m/K]
    visw = 1e-6     # kinetic viscosity of water [m^2/s]
    rhow = 1025     # Density of sea-water [kg/m^3]
    cpw = 4190      # specific heat capacity of water
    aw = 3e-4       # thermal expansion coefficient [K-1]
    Rns = 0.945*Rs  # net solar radiation (albedo correction)
    shf = rho*cp*usr*tsr
    lhf = rho*lv*usr*qsr*0.001  # qsr [g/kg]
    Q = Rnl+shf+lhf+Qs
    xt = 16*Q*grav*aw*cpw*np.power(rhow*visw, 3)/(
        np.power(usr, 4)*np.power(rho*tcw, 2))
    xt1 = 1+np.power(xt, 3/4)
    # Saunders const  eq. 22
    ls = np.where(Q > 0, 6/np.power(xt1, 1/3), 6)
    delta = np.where(Q > 0, (ls*visw)/(np.sqrt(rho/rhow)*usr),
                     np.where((ls*visw)/(np.sqrt(rho/rhow)*usr) > 0.01, 0.01,
                              (ls*visw)/(np.sqrt(rho/rhow)*usr)))  # eq. 21
    # fraction of the solar radiation absorbed in layer delta
    fc = 0.065+11*delta-6.6e-5*(1-np.exp(-delta/0.0008))/delta
    Qs = fc*Rns
    Q = Rnl+shf+lhf+Qs
    dtc = Q*delta/tcw
    return Qs, dtc
