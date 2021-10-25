import numpy as np
from util_subs import (CtoK, kappa, gc, visc_air)

# ---------------------------------------------------------------------


def cdn_calc(u10n, usr, Ta, lat, meth):
    """
    Calculates neutral drag coefficient

    Parameters
    ----------
    u10n : float
        neutral 10m wind speed [m/s]
    usr : float
        friction velocity      [m/s]
    Ta   : float
        air temperature        [K]
    lat : float
        latitude               [degE]
    meth : str

    Returns
    -------
    cdn : float
    """
    cdn = np.zeros(Ta.shape)*np.nan
    if (meth == "S80"): # eq. 14 Smith 1980
        cdn = (0.61+0.063*u10n)*0.001
    elif (meth == "LP82"):
        #  Large & Pond 1981 u10n <11m/s & eq. 21 Large & Pond 1982
        cdn = np.where(u10n < 11, 1.2*0.001, (0.49+0.065*u10n)*0.001)
    elif (meth == "S88" or meth == "UA" or meth == "ecmwf" or meth == "C30" or
          meth == "C35" or meth == "Beljaars"): #  or meth == "C40"
        cdn = cdn_from_roughness(u10n, usr, Ta, lat, meth)
    elif (meth == "YT96"):
        # convert usr in eq. 21 to cdn to expand for low wind speeds
        cdn = np.power((0.10038+u10n*2.17e-3+np.power(u10n, 2)*2.78e-3 -
                        np.power(u10n, 3)*4.4e-5)/u10n, 2)
    elif (meth == "NCAR"): # eq. 11 Large and Yeager 2009
        cdn = np.where(u10n > 0.5, (0.142+2.7/u10n+u10n/13.09 -
                                    3.14807e-10*np.power(u10n, 6))*1e-3,
                       (0.142+2.7/0.5+0.5/13.09 -
                        3.14807e-10*np.power(0.5, 6))*1e-3)
        cdn = np.where(u10n > 33, 2.34e-3, np.copy(cdn))
        cdn = np.maximum(np.copy(cdn), 0.1e-3)
    else:
        raise ValueError("unknown method cdn: "+meth)
    
    return cdn
# ---------------------------------------------------------------------


def cdn_from_roughness(u10n, usr, Ta, lat, meth):
    """
    Calculates neutral drag coefficient from roughness length

    Parameters
    ----------
    u10n : float
        neutral 10m wind speed [m/s]
    usr : float
        friction velocity      [m/s]
    Ta   : float
        air temperature        [K]
    lat : float                [degE]
        latitude
    meth : str

    Returns
    -------
    cdn : float
    """
    g = gc(lat, None)
    cdn = (0.61+0.063*u10n)*0.001
    zo, zc, zs = np.zeros(Ta.shape), np.zeros(Ta.shape), np.zeros(Ta.shape)
    for it in range(5):
        if (meth == "S88"):
            # Charnock roughness length (eq. 4 in Smith 88)
            zc = 0.011*np.power(usr, 2)/g
            #  smooth surface roughness length (eq. 6 in Smith 88)
            zs = 0.11*visc_air(Ta)/usr
            zo = zc + zs  #  eq. 7 & 8 in Smith 88
        elif (meth == "UA"):
            # valid for 0<u<18m/s # Zeng et al. 1998 (24)
            zo = 0.013*np.power(usr, 2)/g+0.11*visc_air(Ta)/usr
        elif (meth == "C30"): # eq. 25 Fairall et al. 1996a
            a = 0.011*np.ones(Ta.shape)
            a = np.where(u10n > 10, 0.011+(u10n-10)*(0.018-0.011)/(18-10),
                         np.where(u10n > 18, 0.018, a))
            zo = a*np.power(usr, 2)/g+0.11*visc_air(Ta)/usr
        elif (meth == "C35"): # eq.6-11 Edson et al. (2013)
            zo = (0.11*visc_air(Ta)/usr +
                  np.minimum(0.0017*19-0.0050, 0.0017*u10n-0.0050) *
                  np.power(usr, 2)/g)
        elif ((meth == "ecmwf" or meth == "Beljaars")):
            # eq. (3.26) p.38 over sea IFS Documentation cy46r1
            zo = 0.018*np.power(usr, 2)/g+0.11*visc_air(Ta)/usr
        else:
            raise ValueError("unknown method for cdn_from_roughness "+meth)
            
        cdn = np.power(kappa/np.log(10/zo), 2)
    return cdn
# ---------------------------------------------------------------------


def cd_calc(cdn, hin, hout, psim):
    """
    Calculates drag coefficient at reference height

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


def ctcqn_calc(zol, cdn, usr, zo, Ta, meth):
    """
    Calculates neutral heat and moisture exchange coefficients

    Parameters
    ----------
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
    ctn : float
        neutral heat exchange coefficient
    cqn : float
        neutral moisture exchange coefficient
    """
    if (meth == "S80" or meth == "S88" or meth == "YT96"):
        cqn = np.ones(Ta.shape)*1.20*0.001  # from S88
        ctn = np.ones(Ta.shape)*1.00*0.001
    elif (meth == "LP82"):
        cqn = np.where((zol <= 0), 1.15*0.001, 1*0.001)
        ctn = np.where((zol <= 0), 1.13*0.001, 0.66*0.001)
    elif (meth == "NCAR"):
        cqn = np.maximum(34.6*0.001*np.sqrt(cdn), 0.1e-3)
        ctn = np.maximum(np.where(zol <= 0, 32.7*0.001*np.sqrt(cdn),
                                  18*0.001*np.sqrt(cdn)), 0.1e-3)
    elif (meth == "UA"):
        # Zeng et al. 1998 (25)
        rr = usr*zo/visc_air(Ta)
        zoq = zo/np.exp(2.67*np.power(rr, 1/4)-2.57)
        zot = np.copy(zoq)
        cqn = np.power(kappa, 2)/(np.log(10/zo)*np.log(10/zoq))
        ctn = np.power(kappa, 2)/(np.log(10/zo)*np.log(10/zoq))
    elif (meth == "C30"):
        rr = zo*usr/visc_air(Ta)
        zoq = np.minimum(5e-5/np.power(rr, 0.6), 1.15e-4)  # moisture roughness
        zot = np.copy(zoq)  # temperature roughness
        cqn = np.power(kappa, 2)/np.log(10/zo)/np.log(10/zoq)
        ctn = np.power(kappa, 2)/np.log(10/zo)/np.log(10/zot)
    elif (meth == "C35"):
        rr = zo*usr/visc_air(Ta)
        zoq = np.minimum(5.8e-5/np.power(rr, 0.72), 1.6e-4) # moisture roughness
        zot = np.copy(zoq)  # temperature roughness
        cqn = np.power(kappa, 2)/np.log(10/zo)/np.log(10/zoq)
        ctn = np.power(kappa, 2)/np.log(10/zo)/np.log(10/zot)
    elif (meth == "ecmwf" or meth == "Beljaars"):
        # eq. (3.26) p.38 over sea IFS Documentation cy46r1
        zot = 0.40*visc_air(Ta)/usr
        zoq = 0.62*visc_air(Ta)/usr
        cqn = np.power(kappa, 2)/np.log(10/zo)/np.log(10/zoq)
        ctn = np.power(kappa, 2)/np.log(10/zo)/np.log(10/zot)
    else:
        raise ValueError("unknown method ctcqn: "+meth)
        
    return ctn, cqn
# ---------------------------------------------------------------------


def ctcq_calc(cdn, cd, ctn, cqn, hin, hout, psit, psiq):
    """
    Calculates heat and moisture exchange coefficients at reference height

    Parameters
    ----------
    cdn : float
        neutral drag coefficient
    cd  : float
        drag coefficient at reference height
    ctn : float
        neutral heat exchange coefficient
    cqn : float
        neutral moisture exchange coefficient
    hin : float
        original temperature/humidity sensor height [m]
    hout : float
        reference height                   [m]
    psit : float
        heat stability function
    psiq : float
        moisture stability function

    Returns
    -------
    ct : float
       heat exchange coefficient
    cq : float
       moisture exchange coefficient
    """
    ct = (ctn*np.sqrt(cd/cdn) /
          (1+ctn*((np.log(hin[1]/hout[1])-psit)/(kappa*np.sqrt(cdn)))))

    cq = (cqn*np.sqrt(cd/cdn) /
          (1+cqn*((np.log(hin[2]/hout[2])-psiq)/(kappa*np.sqrt(cdn)))))

    return ct, cq
# ---------------------------------------------------------------------


def get_stabco(meth):
    """
    Gives the coefficients \\alpha, \\beta, \\gamma for stability functions

    Parameters
    ----------
    meth : str

    Returns
    -------
    coeffs : float
    """
    alpha, beta, gamma = 0, 0, 0
    if (meth == "S80" or meth == "S88" or meth == "NCAR" or
        meth == "UA" or meth == "ecmwf" or meth == "C30" or
        meth == "C35" or meth == "Beljaars"):
        alpha, beta, gamma = 16, 0.25, 5  # Smith 1980, from Dyer (1974)
    elif (meth == "LP82"):
        alpha, beta, gamma = 16, 0.25, 7
    elif (meth == "YT96"):
        alpha, beta, gamma = 20, 0.25, 5
    else:
        raise ValueError("unknown method stabco: "+meth)
    
    coeffs = np.zeros(3)
    coeffs[0] = alpha
    coeffs[1] = beta
    coeffs[2] = gamma
    return coeffs
# ---------------------------------------------------------------------


def psim_calc(zol, meth):
    """
    Calculates momentum stability function

    Parameters
    ----------
    zol : float
        height over MO length
    meth : str

    Returns
    -------
    psim : float
    """
    if (meth == "ecmwf"):
        psim = psim_ecmwf(zol)
    elif (meth == "C30" or meth == "C35"):
        psim = psiu_26(zol, meth)
    elif (meth == "Beljaars"): # Beljaars (1997) eq. 16, 17
        psim = np.where(zol < 0, psim_conv(zol, meth), psi_Bel(zol))
    else:
        psim = np.where(zol < 0, psim_conv(zol, meth),
                        psim_stab(zol, meth))
    return psim
# ---------------------------------------------------------------------


def psit_calc(zol, meth):
    """
    Calculates heat stability function

    Parameters
    ----------
    zol : float
        height over MO length
    meth : str
        parameterisation method

    Returns
    -------
    psit : float
    """
    if (meth == "ecmwf"):
        psit = np.where(zol < 0, psi_conv(zol, meth),
                        psi_ecmwf(zol))
    elif (meth == "C30" or meth == "C35"):
        psit = psit_26(zol)
    elif (meth == "Beljaars"): # Beljaars (1997) eq. 16, 17
        psit = np.where(zol < 0, psi_conv(zol, meth), psi_Bel(zol))
    else:
        psit = np.where(zol < 0, psi_conv(zol, meth),
                        psi_stab(zol, meth))
    return psit
# ---------------------------------------------------------------------


def psi_Bel(zol):
    """
    Calculates momentum/heat stability function

    Parameters
    ----------
    zol : float
        height over MO length
    meth : str
        parameterisation method

    Returns
    -------
    psit : float
    """
    a, b, c, d = 0.7, 0.75, 5, 0.35
    psi = -(a*zol+b*(zol-c/d)*np.exp(-d*zol)+b*c/d)
    return psi
# ---------------------------------------------------------------------


def psi_ecmwf(zol):
    """
    Calculates heat stability function for stable conditions
    for method ecmwf

    Parameters
    ----------
    zol : float
        height over MO length

    Returns
    -------
    psit : float
    """
    # eq (3.22) p. 37 IFS Documentation cy46r1
    a, b, c, d = 1, 2/3, 5, 0.35
    psit = -b*(zol-c/d)*np.exp(-d*zol)-np.power(1+(2/3)*a*zol, 1.5)-(b*c)/d+1
    return psit
# ---------------------------------------------------------------------


def psit_26(zol):
    """
    Computes temperature structure function as in C35

    Parameters
    ----------
    zol : float
        height over MO length

    Returns
    -------
    psi : float
    """
    b, d = 2/3, 0.35
    dzol = np.minimum(d*zol, 50)
    psi = -1*((1+b*zol)**1.5+b*(zol-14.28)*np.exp(-dzol)+8.525)
    k = np.where(zol < 0)
    x = np.sqrt(1-15*zol[k])
    psik = 2*np.log((1+x)/2)
    x = np.power(1-34.15*zol[k], 1/3)
    psic = (1.5*np.log((1+x+np.power(x, 2))/3)-np.sqrt(3) *
            np.arctan((1+2*x)/np.sqrt(3))+4*np.arctan(1)/np.sqrt(3))
    f = np.power(zol[k], 2)/(1+np.power(zol[k], 2))
    psi[k] = (1-f)*psik+f*psic
    return psi
# ---------------------------------------------------------------------


def psi_conv(zol, meth):
    """
    Calculates heat stability function for unstable conditions

    Parameters
    ----------
    zol : float
        height over MO length
    meth : str
        parameterisation method

    Returns
    -------
    psit : float
    """
    coeffs = get_stabco(meth)
    alpha, beta = coeffs[0], coeffs[1]
    xtmp = np.power(1-alpha*zol, beta)
    psit = 2*np.log((1+np.power(xtmp, 2))*0.5)
    return psit
# ---------------------------------------------------------------------


def psi_stab(zol, meth):
    """
    Calculates heat stability function for stable conditions

    Parameters
    ----------
    zol : float
        height over MO length
    meth : str
        parameterisation method

    Returns
    -------
    psit : float
    """
    coeffs = get_stabco(meth)
    gamma = coeffs[2]
    psit = -gamma*zol
    return psit
# ---------------------------------------------------------------------


def psim_ecmwf(zol):
    """
    Calculates momentum stability function for method ecmwf

    Parameters
    ----------
    zol : float
        height over MO length

    Returns
    -------
    psim : float
    """
    # eq (3.20, 3.22) p. 37 IFS Documentation cy46r1
    coeffs = get_stabco("ecmwf")
    alpha, beta = coeffs[0], coeffs[1]
    xtmp = np.power(1-alpha*zol, beta)
    a, b, c, d = 1, 2/3, 5, 0.35
    psim = np.where(zol < 0, np.pi/2-2*np.arctan(xtmp) +
                    np.log((np.power(1+xtmp, 2)*(1+np.power(xtmp, 2)))/8),
                    -b*(zol-c/d)*np.exp(-d*zol)-a*zol-(b*c)/d)
    return psim
# ---------------------------------------------------------------------


def psiu_26(zol, meth):
    """
    Computes velocity structure function C35

    Parameters
    ----------
    zol : float
        height over MO length

    Returns
    -------
    psi : float
    """
    if (meth == "C30"):
        dzol = np.minimum(0.35*zol, 50) # stable
        psi = -1*((1+zol)+0.6667*(zol-14.28)*np.exp(-dzol)+8.525)
        k = np.where(zol < 0) # unstable
        x = (1-15*zol[k])**0.25
        psik = (2*np.log((1+x)/2)+np.log((1+x*x)/2)-2*np.arctan(x) +
                2*np.arctan(1))
        x = (1-10.15*zol[k])**(1/3)
        psic = (1.5*np.log((1+x+x*x)/3) -
                np.sqrt(3)*np.arctan((1+2*x)/np.sqrt(3)) +
                4*np.arctan(1)/np.sqrt(3))
        f = zol[k]**2/(1+zol[k]**2)
        psi[k] = (1-f)*psik+f*psic
    elif (meth == "C35"): #  or meth == "C40"
        dzol = np.minimum(50, 0.35*zol)  # stable
        a, b, c, d = 0.7, 3/4, 5, 0.35
        psi = -1*(a*zol+b*(zol-c/d)*np.exp(-dzol)+b*c/d)
        k = np.where(zol < 0)  # unstable
        x = np.power(1-15*zol[k], 1/4)
        psik = 2*np.log((1+x)/2)+np.log((1+x*x)/2)-2*np.arctan(x)+2*np.arctan(1)
        x = np.power(1-10.15*zol[k], 1/3)
        psic = (1.5*np.log((1+x+np.power(x, 2))/3)-np.sqrt(3) *
                np.arctan((1+2*x)/np.sqrt(3))+4*np.arctan(1)/np.sqrt(3))
        f = np.power(zol[k], 2)/(1+np.power(zol[k], 2))
        psi[k] = (1-f)*psik+f*psic
    return psi
#------------------------------------------------------------------------------



def psim_conv(zol, meth):
    """
    Calculates momentum stability function for unstable conditions

    Parameters
    ----------
    zol : float
        height over MO length
    meth : str
        parameterisation method

    Returns
    -------
    psim : float
    """
    coeffs = get_stabco(meth)
    alpha, beta = coeffs[0], coeffs[1]
    xtmp = np.power(1-alpha*zol, beta)
    psim = (2*np.log((1+xtmp)*0.5)+np.log((1+np.power(xtmp, 2))*0.5) -
            2*np.arctan(xtmp)+np.pi/2)
    return psim
# ---------------------------------------------------------------------


def psim_stab(zol, meth):
    """
    Calculates momentum stability function for stable conditions

    Parameters
    ----------
    zol : float
        height over MO length
    meth : str
        parameterisation method

    Returns
    -------
    psim : float
    """
    coeffs = get_stabco(meth)
    gamma = coeffs[2]
    psim = -gamma*zol
    return psim
# ---------------------------------------------------------------------


def cs_C35(sst, qsea, rho, Rs, Rnl, cp, lv, delta, usr, tsr, qsr, lat):
    """
    Computes cool skin
    following COARE3.5 (Fairall et al. 1996, Edson et al. 2013)

    Parameters
    ----------
    sst : float
        sea surface temperature      [K]
    qsea : float
        specific humidity over sea   [g/kg]
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
    lat : float
       latitude                      [degE]

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
    g = gc(lat, None)
    if (np.nanmin(sst) > 200):  # if sst in Kelvin convert to Celsius
        sst = sst-CtoK
    # ************  cool skin constants  *******
    # density of water, specific heat capacity of water, water viscosity,
    # thermal conductivity of water
    rhow, cpw, visw, tcw = 1022, 4000, 1e-6, 0.6
    for i in range(4):
        aw = 2.1e-5*np.power(np.maximum(sst+3.2, 0), 0.79)
        bigc = 16*g*cpw*np.power(rhow*visw, 3)/(np.power(tcw, 2)*np.power(rho, 2))
        wetc = 0.622*lv*qsea/(287.1*np.power(sst+273.16, 2))
        Rns = 0.945*Rs  # albedo correction
        shf = rho*cp*usr*tsr
        lhf = rho*lv*usr*qsr
        Qnsol = shf+lhf+Rnl
        fs = 0.065+11*delta-6.6e-5/delta*(1-np.exp(-delta/8.0e-4))
        qcol = Qnsol+Rns*fs
        alq = aw*qcol+0.026*np.minimum(lhf, 0)*cpw/lv
        xlamx = 6*np.ones(sst.shape)
        xlamx = np.where(alq > 0, 6/(1+(bigc*alq/usr**4)**0.75)**0.333, 6)
        delta =  np.minimum(xlamx*visw/(np.sqrt(rho/rhow)*usr), 0.01)
        delta = np.where(alq > 0, xlamx*visw/(np.sqrt(rho/rhow)*usr), delta)
    dter = qcol*delta/tcw
    dqer = wetc*dter
    return dter, dqer, delta
# ---------------------------------------------------------------------


def delta(aw, Q, usr, lat):
    """
    Computes the thickness (m) of the viscous skin layer.
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
    lat : float
       latitude                      [degE]

    Returns
    -------
    delta : float
        the thickness (m) of the viscous skin layer

    """
    rhow, visw, tcw = 1025, 1e-6, 0.6
    # u* in the water
    usr_w = np.maximum(usr, 1e-4)*np.sqrt(1.2/rhow) # rhoa=1.2
    rcst_cs = 16*gc(lat)*np.power(visw, 3)/np.power(tcw, 2)
    lm = 6*(1+np.maximum(Q*aw*rcst_cs/np.power(usr_w, 4), 0)**0.75)**(-1/3)
    ztmp = visw/usr_w
    delta = np.where(Q > 0, np.minimum(6*ztmp, 0.007), lm*ztmp)
    return delta
# ---------------------------------------------------------------------


def cs_ecmwf(rho, Rs, Rnl, cp, lv, usr, tsr, qsr, sst, lat):
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
    lat : float
       latitude                  [degE]

    Returns
    -------
    dtc : float
        cool skin temperature correction [K]

    """
    if (np.nanmin(sst) < 200):  # if sst in Celsius convert to Kelvin
        sst = sst+CtoK
    aw = np.maximum(1e-5, 1e-5*(sst-CtoK))
    Rns = 0.945*Rs  # (net solar radiation (albedo correction)
    shf = rho*cp*usr*tsr
    lhf = rho*lv*usr*qsr
    Qnsol = shf+lhf+Rnl  # eq. 8.152
    d = delta(aw, Qnsol, usr, lat)
    for jc in range(4): # because implicit in terms of delta...
        # # fraction of the solar radiation absorbed in layer delta eq. 8.153
        # and Eq.(5) Zeng & Beljaars, 2005
        fs = 0.065+11*d-6.6e-5/d*(1-np.exp(-d/8e-4))
        Q = Qnsol+fs*Rns
        d = delta(aw, Q, usr, lat)
    dtc = Q*d/0.6  # (rhow*cw*kw)eq. 8.151
    return dtc
# ---------------------------------------------------------------------


def wl_ecmwf(rho, Rs, Rnl, cp, lv, usr, tsr, qsr, sst, skt, dtc, lat):
    """
    warm layer correction following IFS Documentation cy46r1
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
    lat : float
        latitude                    [degE]

    Returns
    -------
    dtwl : float
        warm layer correction       [K]

    """
    if (np.nanmin(sst) < 200):  # if sst in Celsius convert to Kelvin
        sst = sst+CtoK
    rhow, cpw, visw, rd0 = 1025, 4190, 1e-6, 3
    Rns = 0.945*Rs
    ## Previous value of dT / warm-layer, adapted to depth:
    aw = 2.1e-5*np.power(np.maximum(sst-CtoK+3.2, 0), 0.79) # thermal expansion coefficient of sea-water (SST accurate enough#)
    # *** Rd = Fraction of solar radiation absorbed in warm layer (-)
    a1, a2, a3 = 0.28, 0.27, 0.45
    b1, b2, b3 = -71.5, -2.8, -0.06 # [m-1]
    Rd = 1-(a1*np.exp(b1*rd0)+a2*np.exp(b2*rd0)+a3*np.exp(b3*rd0))
    shf = rho*cp*usr*tsr
    lhf = rho*lv*usr*qsr
    Qnsol = shf+lhf+Rnl
    usrw  = np.maximum(usr, 1e-4 )*np.sqrt(1.2/rhow)   # u* in the water
    zc3 = rd0*kappa*gc(lat)/np.power(1.2/rhow, 3/2)
    zc4 = (0.3+1)*kappa/rd0
    zc5 = (0.3+1)/(0.3*rd0)
    for jwl in range(10): # itteration to solve implicitely equation for warm layer
        dsst = skt-sst-dtc
        ## Buoyancy flux and stability parameter (zdl = -z/L) in water
        ZSRD = (Qnsol+Rns*Rd)/(rhow*cpw)
        ztmp = np.maximum(dsst, 0)
        zdl = np.where(ZSRD > 0, 2*(np.power(usrw, 2) *
                                    np.sqrt(ztmp/(5*rd0*gc(lat)*aw/visw)))+ZSRD,
                       np.power(usrw, 2)*np.sqrt(ztmp/(5*rd0*gc(lat)*aw/visw)))
        usr = np.maximum(usr, 1e-4 )
        zdL = zc3*aw*zdl/np.power(usr, 3)
        ## Stability function Phi_t(-z/L) (zdL is -z/L) :
        zphi = np.where(zdL > 0, (1+(5*zdL+4*np.power(zdL, 2)) /
                                  (1+3*zdL+0.25*np.power(zdL, 2)) +
                                  2/np.sqrt(1-16*(-np.abs(zdL)))),
                        1/np.sqrt(1-16*(-np.abs(zdL))))
        zz = zc4*(usrw)/zphi
        zz = np.maximum(np.abs(zz) , 1e-4)*np.sign(zz)
        dtwl = np.maximum(0, (zc5*ZSRD)/zz)
    return dtwl
#----------------------------------------------------------------------


def cs_Beljaars(rho, Rs, Rnl, cp, lv, usr, tsr, qsr, lat, Qs):
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
    lat : float
       latitude                  [degE]
    Qs : float
      radiation balance

    Returns
    -------
    Qs : float
      radiation balance
    dtc : float
        cool skin temperature correction [K]

    """
    g = gc(lat, None)
    tcw = 0.6       # thermal conductivity of water (at 20C) [W/m/K]
    visw = 1e-6     # kinetic viscosity of water [m^2/s]
    rhow = 1025     # Density of sea-water [kg/m^3]
    cpw = 4190      # specific heat capacity of water
    aw = 3e-4       # thermal expansion coefficient [K-1]
    Rns = 0.945*Rs  # net solar radiation (albedo correction)
    shf = rho*cp*usr*tsr
    lhf = rho*lv*usr*qsr
    Q = Rnl+shf+lhf+Qs
    xt = (16*Q*g*aw*cpw*np.power(rhow*visw, 3))/(np.power(usr, 4) *
                                                np.power(rho*tcw, 2))
    xt1 = 1+np.power(xt, 3/4)
    # Saunders const  eq. 22
    ls = np.where(Q > 0, 6/np.power(xt1, 1/3), 6)
    delta = np.where(Q > 0 , (ls*visw)/(np.sqrt(rho/rhow)*usr),
                     np.where((ls*visw)/(np.sqrt(rho/rhow)*usr) > 0.01, 0.01,
                              (ls*visw)/(np.sqrt(rho/rhow)*usr)))  # eq. 21
    # fraction of the solar radiation absorbed in layer delta
    fc = 0.065+11*delta-6.6e-5*(1-np.exp(-delta/0.0008))/delta
    Qs = fc*Rns
    Q = Rnl+shf+lhf+Qs
    dtc = Q*delta/tcw
    return Qs, dtc
#----------------------------------------------------------------------


def get_gust(beta, Ta, usr, tsrv, zi, lat):
    """
    Computes gustiness

    Parameters
    ----------
    beta : float
        constant
    Ta : float
        air temperature   [K]
    usr : float
        friction velocity [m/s]
    tsrv : float
        star virtual temperature of air [K]
    zi : int
        scale height of the boundary layer depth [m]
    lat : float
        latitude

    Returns
    -------
    ug : float        [m/s]
    """
    if (np.nanmax(Ta) < 200):  # convert to K if in Celsius
        Ta = Ta+273.16
    g = gc(lat, None)
    Bf = (-g/Ta)*usr*tsrv
    ug = np.ones(np.shape(Ta))*0.2
    ug = np.where(Bf > 0, beta*np.power(Bf*zi, 1/3), 0.2)
    return ug
# ---------------------------------------------------------------------


def get_L(L, lat, usr, tsr, qsr, hin, Ta, sst, qair, qsea, wind, monob, zo,
          zot, psim, meth):
    """
    calculates Monin-Obukhov length and virtual star temperature

    Parameters
    ----------
    L : str
        Monin-Obukhov length definition options
        "tsrv"  : default for S80, S88, LP82, YT96, UA, C30, C35 and NCAR
        "Rb" : following ecmwf (IFS Documentation cy46r1), default for ecmwf
               and Beljaars
    lat : float
        latitude
    usr : float
        friction wind speed (m/s)
    tsr : float
        star temperature (K)
    qsr : float
        star specific humidity (g/kg)
    hin : float
        sensor heights (m)
    Ta : float
        air temperature (K)
    sst : float
        sea surface temperature (K)
    qair : float
        air specific humidity (g/kg)
    qsea : float
        specific humidity at sea surface (g/kg)
    wind : float
        wind speed (m/s)
    monob : float
        Monin-Obukhov length from previous iteration step (m)
    zo   : float
        surface roughness       (m)
    zot   : float
        temperature roughness length       (m)
    psim : floast
        momentum stability function
    meth : str
        bulk parameterisation method option: "S80", "S88", "LP82", "YT96",
        "UA", "NCAR", "C30", "C35", "ecmwf", "Beljaars"

    Returns
    -------
    tsrv : float
        virtual star temperature (K)
    monob : float
        M-O length (m)
    Rb  : float
       Richardson number

    """
    g = gc(lat)
    Rb = np.empty(sst.shape)
    # as in aerobulk One_on_L in mod_phymbl.f90
    tsrv = tsr*(1+0.6077*qair)+0.6077*Ta*qsr
    # tsrv = tsr+0.6077*Ta*qsr
    # from eq. 3.24 ifs Cy46r1 pp. 37
    thvs = sst*(1+0.6077*qsea) # virtual SST
    dthv = (Ta-sst)*(1+0.6077*qair)+0.6077*Ta*(qair-qsea)
    tv = 0.5*(thvs+Ta*(1+0.6077*qair)) # estimate tv within surface layer
    # adjust wind to T sensor's height
    uz = (wind-usr/kappa*(np.log(hin[0]/hin[1])-psim +
                          psim_calc(hin[1]/monob, meth)))
    Rb = g*dthv*hin[1]/(tv*uz*uz)
    if (L == "tsrv"):
        tmp = (g*kappa*tsrv /
                np.maximum(np.power(usr, 2)*Ta*(1+0.6077*qair), 1e-9))
        tmp = np.minimum(np.abs(tmp), 200)*np.sign(tmp)
        monob = 1/np.copy(tmp)
    elif (L == "Rb"):
        zol = (Rb*(np.power(np.log((hin[1]+zo)/zo)-psim_calc((hin[1]+zo) /
                                                              monob, meth) +
                            psim_calc(zo/monob, meth), 2) /
                   (np.log((hin[1]+zo)/zot) -
                    psit_calc((hin[1]+zo)/monob, meth) +
                    psit_calc(zot/monob, meth))))
        monob = hin[1]/zol
    return tsrv, monob, Rb
#------------------------------------------------------------------------------


def get_strs(hin, monob, wind, zo, zot, zoq, dt, dq, dter, dqer, dtwl, ct, cq,
             cskin, wl, meth):
    """
    calculates star wind speed, temperature and specific humidity

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
    dter : float
        cskin temperature adjustment [K]
    dqer : float
        cskin q adjustment           [g/kg]
    dtwl : float
        warm layer temperature adjustment [K]
    ct : float
        temperature exchange coefficient
    cq : float
        moisture exchange coefficient
    cskin : int
        cool skin adjustment switch
    wl : int
        warm layer correction switch
    meth : str
        bulk parameterisation method option: "S80", "S88", "LP82", "YT96", "UA",
        "NCAR", "C30", "C35", "ecmwf", "Beljaars"

    Returns
    -------
    usr : float
        friction wind speed [m/s]
    tsr : float
        star temperature    [K]
    qsr : float
        star specific humidity [g/kg]

    """
    if (meth == "UA"):
        usr = np.where(hin[0]/monob <= -1.574, kappa*wind /
                       (np.log(-1.574*monob/zo)-psim_calc(-1.574, meth) +
                        psim_calc(zo/monob, meth) +
                        1.14*(np.power(-hin[0]/monob, 1/3) -
                        np.power(1.574, 1/3))),
                       np.where(hin[0]/monob < 0, kappa*wind /
                                (np.log(hin[0]/zo) -
                                 psim_calc(hin[0]/monob, meth) +
                                 psim_calc(zo/monob, meth)),
                                np.where(hin[0]/monob <= 1, kappa*wind /
                                         (np.log(hin[0]/zo) +
                                          5*hin[0]/monob-5*zo/monob),
                                         kappa*wind/(np.log(monob/zo)+5 -
                                                     5*zo/monob +
                                                     5*np.log(hin[0]/monob) +
                                                     hin[0]/monob-1))))
                                # Zeng et al. 1998 (7-10)
        tsr = np.where(hin[1]/monob < -0.465, kappa*(dt-dter*cskin-dtwl*wl) /
                       (np.log((-0.465*monob)/zot) -
                        psit_calc(-0.465, meth)+0.8*(np.power(0.465, -1/3) -
                        np.power(-hin[1]/monob, -1/3))),
                       np.where(hin[1]/monob < 0, kappa*(dt-dter*cskin-dtwl*wl) /
                                (np.log(hin[1]/zot) -
                                 psit_calc(hin[1]/monob, meth) +
                                 psit_calc(zot/monob, meth)),
                                np.where(hin[1]/monob <= 1,
                                         kappa*(dt-dter*cskin-dtwl*wl) /
                                         (np.log(hin[1]/zot) +
                                          5*hin[1]/monob-5*zot/monob),
                                         kappa*(dt-dter*cskin-dtwl*wl) /
                                         (np.log(monob/zot)+5 -
                                          5*zot/monob+5*np.log(hin[1]/monob) +
                                          hin[1]/monob-1))))
                                # Zeng et al. 1998 (11-14)
        qsr = np.where(hin[2]/monob < -0.465, kappa*(dq-dqer*cskin) /
                       (np.log((-0.465*monob)/zoq) -
                        psit_calc(-0.465, meth)+psit_calc(zoq/monob, meth) +
                        0.8*(np.power(0.465, -1/3) -
                             np.power(-hin[2]/monob, -1/3))),
                       np.where(hin[2]/monob < 0,
                                kappa*(dq-dqer*cskin)/(np.log(hin[1]/zot) -
                                psit_calc(hin[2]/monob, meth) +
                                psit_calc(zoq/monob, meth)),
                                np.where(hin[2]/monob <= 1,
                                         kappa*(dq-dqer*cskin) /
                                         (np.log(hin[1]/zoq)+5*hin[2]/monob -
                                          5*zoq/monob),
                                         kappa*(dq-dqer*cskin)/
                                         (np.log(monob/zoq)+5-5*zoq/monob +
                                          5*np.log(hin[2]/monob) +
                                          hin[2]/monob-1))))
    elif (meth == "C30" or meth == "C35"): #   or meth == "C40"
        usr = (wind*kappa/(np.log(hin[0]/zo)-psiu_26(hin[0]/monob, meth)))
        tsr = ((dt-dter*cskin-dtwl*wl)*(kappa/(np.log(hin[1]/zot) -
                                       psit_26(hin[1]/monob))))
        qsr = ((dq-dqer*cskin)*(kappa/(np.log(hin[2]/zoq) -
                                       psit_26(hin[2]/monob))))
    else:
        usr = (wind*kappa/(np.log(hin[0]/zo)-psim_calc(hin[0]/monob, meth)))
        tsr = ct*wind*(dt-dter*cskin-dtwl*wl)/usr
        qsr = cq*wind*(dq-dqer*cskin)/usr
    return usr, tsr, qsr
# ---------------------------------------------------------------------
