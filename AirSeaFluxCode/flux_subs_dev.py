import numpy as np
from util_subs import (kappa, visc_air)

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
            # temporary as in aerobulk
            zo = np.minimum(np.abs(zo), 0.001)
        else:
            raise ValueError("Unknown method for cdn_from_roughness "+meth)

        cdn = np.power(kappa/np.log(10/zo), 2)
        # temporary as in aerobulk
        # if meth == "ecmwf":
        #     cdn = np.maximum(cdn, 0.1e-3)
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
        # temporary as in aerobulk next 2lines
        # eq.3.26, Chap.3, p.34, IFS doc - Cy31r1
        zot = np.minimum(np.abs(zot), 0.001)
        zoq = np.minimum(np.abs(zoq), 0.001)
        cqn = np.power(kappa, 2)/np.log(10/zo)/np.log(10/zoq)
        ctn = np.power(kappa, 2)/np.log(10/zo)/np.log(10/zot)
        # temporary as in aerobulk
        # ctn = np.maximum(ctn, 0.1e-3)
        # cqn = np.maximum(cqn, 0.1e-3)
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


def get_stabco(meth):
    r"""
    Give the coefficients $\alpha$, $\beta$, $\gamma$ for stability functions.

    Parameters
    ----------
    meth : str

    Returns
    -------
    coeffs : float
    """
    alpha, beta, gamma = 0, 0, 0
    if meth in ["S80", "S88", "NCAR", "UA", "ecmwf", "C30", "C35", "Beljaars"]:
        alpha, beta, gamma = 16, 0.25, 5  # Smith 1980, from Dyer (1974)
    elif meth == "LP82":
        alpha, beta, gamma = 16, 0.25, 7
    elif meth == "YT96":
        alpha, beta, gamma = 20, 0.25, 5
    else:
        raise ValueError("Unknown method stabco: "+meth)
    coeffs = np.zeros(3)
    coeffs[0] = alpha
    coeffs[1] = beta
    coeffs[2] = gamma
    return coeffs
# ---------------------------------------------------------------------


def psim_calc(zol, meth):
    """
    Calculate momentum stability function.

    Parameters
    ----------
    zol : float
        height over MO length
    meth : str

    Returns
    -------
    psim : float
    """
    if meth == "ecmwf":
        psim = psim_ecmwf(zol)
    elif meth in ["C30", "C35"]:
        psim = psiu_26(zol, meth)
    elif meth == "Beljaars":  # Beljaars (1997) eq. 16, 17
        psim = np.where(zol < 0, psim_conv(zol, meth), psi_Bel(zol))
    else:
        psim = np.where(zol < 0, psim_conv(zol, meth),
                        psim_stab(zol, meth))
    return psim
# ---------------------------------------------------------------------


def psit_calc(zol, meth):
    """
    Calculate heat stability function.

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
    if meth == "ecmwf":
        # psit = np.where(zol < 0, psi_conv(zol, meth),
        #                 psi_ecmwf(zol))
        # temporary as in aerobulk
        psit = psi_ecmwf(zol)
    elif meth in ["C30", "C35"]:
        psit = psit_26(zol)
    elif meth == "Beljaars":  # Beljaars (1997) eq. 16, 17
        psit = np.where(zol < 0, psi_conv(zol, meth), psi_Bel(zol))
    else:
        psit = np.where(zol < 0, psi_conv(zol, meth),
                        psi_stab(zol, meth))
    return psit
# ---------------------------------------------------------------------


def psi_Bel(zol):
    """
    Calculate momentum/heat stability function.

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
    Calculate heat stability function for stable conditions.

    For method ecmwf

    Parameters
    ----------
    zol : float
        height over MO length

    Returns
    -------
    psit : float
    """
    # eq (3.22) p. 37 IFS Documentation cy46r1
    # a, b, c, d = 1, 2/3, 5, 0.35
    # psit = -b*(zol-c/d)*np.exp(-d*zol)-np.power(1+(2/3)*a*zol, 1.5)-(b*c)/d+1
    # temporary as in aerobulk
    a, b, c, d = 1, 2/3, 5, 0.35
    zol = np.minimum(np.copy(zol), 5)  # Very stable conditions (L>0 and big)
    # Unstable eq (3.20) p. 37 IFS Documentation cy46r1
    psi_unst = 2*np.log((1+np.sqrt(np.abs(1-16*zol)))/2)
    # Stable eq (3.22) p. 37 IFS Documentation cy46r1
    psi_stab = -b*(zol-c/d)*np.exp(-d*zol) - \
        np.power(np.abs(1+(2/3)*a*zol), 1.5)-b*c/d+1
    # Brodeau added np.abs() to avoid NaN values when unstable,
    # which contaminates the unstable solution...
    psit = np.where(zol > 0, psi_stab, psi_unst)
    return psit
# ---------------------------------------------------------------------


def psit_26(zol):
    """
    Compute temperature structure function as in C35.

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
    Calculate heat stability function for unstable conditions.

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
    Calculate heat stability function for stable conditions.

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
    Calculate momentum stability function for method ecmwf.

    Parameters
    ----------
    zol : float
        height over MO length

    Returns
    -------
    psim : float
    """
    # temporary as in aerobulk
    zol = np.minimum(np.copy(zol), 5)  # Very stable conditions (L>0 and big!)
    coeffs = get_stabco("ecmwf")
    alpha, beta = coeffs[0], coeffs[1]
    xtmp = np.power(1-alpha*zol, beta)
    psi_unst = np.pi/2-2 * \
        np.arctan(xtmp)+np.log((np.power(1+xtmp, 2)*(1+np.power(xtmp, 2)))/8)
    # Stable eq.3.22 p. 37 IFS Documentation cy46r1
    a, b, c, d = 1, 2/3, 5, 0.35
    psi_stab = -b*(zol-c/d)*np.exp(-d*zol)-a*zol-(b*c)/d
    psim = np.where(zol < 0, psi_unst, psi_stab)
    # eq (3.20, 3.22) p. 37 IFS Documentation cy46r1
    # coeffs = get_stabco("ecmwf")
    # alpha, beta = coeffs[0], coeffs[1]
    # xtmp = np.power(1-alpha*zol, beta)
    # a, b, c, d = 1, 2/3, 5, 0.35
    # psim = np.where(zol < 0, np.pi/2-2*np.arctan(xtmp) +
    #                 np.log((np.power(1+xtmp, 2)*(1+np.power(xtmp, 2)))/8),
    #                 -b*(zol-c/d)*np.exp(-d*zol)-a*zol-(b*c)/d)
    return psim
# ---------------------------------------------------------------------


def psiu_26(zol, meth):
    """
    Compute velocity structure function C35.

    Parameters
    ----------
    zol : float
        height over MO length

    Returns
    -------
    psi : float
    """
    if meth == "C30":
        dzol = np.minimum(0.35*zol, 50)  # stable
        psi = -1*((1+zol)+0.6667*(zol-14.28)*np.exp(-dzol)+8.525)
        k = np.where(zol < 0)  # unstable
        x = (1-15*zol[k])**0.25
        psik = (2*np.log((1+x)/2)+np.log((1+x*x)/2)-2*np.arctan(x) +
                2*np.arctan(1))
        x = (1-10.15*zol[k])**(1/3)
        psic = (1.5*np.log((1+x+x*x)/3) -
                np.sqrt(3)*np.arctan((1+2*x)/np.sqrt(3)) +
                4*np.arctan(1)/np.sqrt(3))
        f = zol[k]**2/(1+zol[k]**2)
        psi[k] = (1-f)*psik+f*psic
    elif meth == "C35":
        dzol = np.minimum(50, 0.35*zol)  # stable
        a, b, c, d = 0.7, 3/4, 5, 0.35
        psi = -1*(a*zol+b*(zol-c/d)*np.exp(-dzol)+b*c/d)
        k = np.where(zol < 0)  # unstable
        x = np.power(1-15*zol[k], 1/4)
        psik = 2*np.log((1+x)/2)+np.log((1+x*x)/2) - \
            2*np.arctan(x)+2*np.arctan(1)
        x = np.power(1-10.15*zol[k], 1/3)
        psic = (1.5*np.log((1+x+np.power(x, 2))/3)-np.sqrt(3) *
                np.arctan((1+2*x)/np.sqrt(3))+4*np.arctan(1)/np.sqrt(3))
        f = np.power(zol[k], 2)/(1+np.power(zol[k], 2))
        psi[k] = (1-f)*psik+f*psic

    return psi
# ----------------------------------------------------------------------------


def psim_conv(zol, meth):
    """
    Calculate momentum stability function for unstable conditions.

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
    Calculate momentum stability function for stable conditions.

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


def get_gust_old(beta, Ta, usr, tsrv, zi, grav):
    """
    Compute gustiness.

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
    ug = np.ones(np.shape(Ta))*0.2
    ug = np.where(Bf > 0, beta*np.power(Bf*zi, 1/3), 0.2)
    return ug
# ---------------------------------------------------------------------


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
        gust wind in stable conditions  [m/s]
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
    # ug = np.where(Bf > 0, beta*np.power(Bf*zi, 1/3), ustb)
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
    # 3. GF=1, 4. UA/ecmwf,  5. C35 code 
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
        # GustFact = np.empty([3, spd.shape[0]], dtype=float)*np.nan
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
        star temperature (K)
    qsr : float
        star specific humidity (g/kg)
    Ta : float
        air temperature (K)
    qair : float
        air specific humidity (g/kg)

    Returns
    -------
    tsrv : float
        virtual star temperature (K)

    """
    # NOTE: 0.6077 goes with mixing ratio, equiv kg/kg humidity
    # as in aerobulk One_on_L in mod_phymbl.f90
    # tsrv = tsr+0.6077*Ta*qsr
    tsrv = 0.001*(tsr*(1000+0.6077*qair)+0.6077*Ta*qsr)  # q [g/kg]
    return tsrv

# ---------------------------------------------------------------------


def get_Rb(grav, usr, hin_u, hin_t, tv, dtv, wind, monob, meth):
    """
    Calculate bulk Richardson number.

    Parameters
    ----------
    grav : float
        acceleration due to gravity (m/s2)
    usr : float
        friction wind speed (m/s)
    hin_u : float
        u sensor height (m)
    hin_t : float
        t sensor height (m)
    tv : float
        virtual temperature (K)
    dtv : float
        virtual temperature difference, air and sea (K)
    wind : float
        wind speed (m/s)
    monob : float
        Monin-Obukhov length from previous iteration step (m)
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
        t sensor height (m)
    monob : float
        Monin-Obukhov length from previous iteration step (m)
    zo   : float
        surface roughness       (m)
    zot   : float
        temperature roughness length       (m)
    meth : str
        bulk parameterisation method option: "S80", "S88", "LP82", "YT96",
        "UA", "NCAR", "C30", "C35", "ecmwf", "Beljaars"

    Returns
    -------
    monob : float
        M-O length (m)

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
        virtual star temperature (K)
    grav : float
        acceleration due to gravity (m/s2)
    tv : float
        virtual temperature (K)
    usr : float
        friction wind speed (m/s)

    Returns
    -------
    monob : float
        M-O length (m)

    """
    tsrv = np.maximum(np.abs(tsrv), 1e-9)*np.sign(tsrv)
    monob = (np.power(usr, 2)*tv)/(grav*kappa*tsrv)
    # temporary as in aerobulk
    # monob = np.maximum(np.power(usr, 2)*tv, 1e-9)/(grav*kappa*tsrv)
    # monob = 1/np.minimum(np.abs(1/monob), 200)*np.sign(1/monob)
    return monob

# ---------------------------------------------------------------------
