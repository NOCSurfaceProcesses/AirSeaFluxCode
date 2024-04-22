import numpy as np
import warnings
from util_subs import (CtoK)


def VaporPressure(temp, P, phase, meth):
    """
    Calculate the saturation vapor pressure.

    For temperatures above 0 deg C the vapor pressure over liquid water
    is calculated.

    The optional parameter 'liquid' changes the calculation to vapor pressure
    over liquid water over the entire temperature range.

    The current default fomulas are Hyland and Wexler for liquid and
    Goff Gratch for ice.

    Ported to Python and modified by S. Biri from Holger Voemel's original

    Parameters
    ----------
    temp : float
        Temperature [C]
    phase : str
        'liquid' : Calculate vapor pressure over liquid water or
        'ice' : Calculate vapor pressure over ice
    meth : str
        formula to be used
        Hardy               : vaporpressure formula from Hardy (1998)
        MagnusTetens        : vaporpressure formula from Magnus Tetens
        GoffGratch          : vaporpressure formula from Goff Gratch
        Buck                : vaporpressure formula from Buck (1981)
        Buck2               : vaporpressure formula from the Buck (2012)
        WMO                 : vaporpressure formula from WMO (1988)
        WMO2018             : vaporpressure formula from WMO (2018)
        Wexler              : vaporpressure formula from Wexler (1976)
        Sonntag             : vaporpressure formula from Sonntag (1994)
        Bolton              : vaporpressure formula from Bolton (1980)
        HylandWexler        : vaporpressure formula from Hyland and Wexler (1983)
        IAPWS               : vaporpressure formula from IAPWS (2002)
        Preining            : vaporpressure formula from Preining (2002)
        MurphyKoop          : vaporpressure formula from Murphy and Koop (2005)

    Returns
    -------
    P : float
        Saturation vapor pressure [hPa]
    """
    Psat = np.zeros(temp.size)*np.nan
    if np.nanmin(temp) > 200:  # if Ta in Kelvin convert to Celsius
        temp = temp-273.16
    T = np.copy(temp)+273.16  # Most formulas use T in [K]
    #  Formulas using [C] use the variable temp
    #  Calculate saturation pressure over liquid water
    if phase == 'liquid':
        if meth in (['HylandWexler', '']):
            """
            Source Hyland, R. W. and A. Wexler, Formulations for the
            Thermodynamic Properties of the saturated Phases of H2O from
            173.15K to 473.15K, ASHRAE Trans, 89(2A), 500-519, 1983."""
            Psat = np.exp(-0.58002206e4/T+0.13914993e1-0.48640239e-1*T +
                          0.41764768e-4*np.power(T, 2) -
                          0.14452093e-7*np.power(T, 3) +
                          0.65459673e1*np.log(T))/100
        elif meth == 'Hardy':
            """
            Source Hardy, B., 1998, ITS-90 Formulations for Vapor Pressure,
            Frostpoint Temperature, Dewpoint Temperature, and Enhancement
            Factors in the Range -100 to +100°C, The Proceedings of the Third
            International Symposium on Humidity & Moisture, London, England"""
            Psat = np.exp(-2.8365744e3/np.power(T, 2)-6.028076559e3/T +
                          1.954263612e1-2.737830188e-2*T +
                          1.6261698e-5*np.power(T, 2) +
                          7.0229056e-10*np.power(T, 3) -
                          1.8680009e-13*np.power(T, 4) +
                          2.7150305*np.log(T))/100
        elif meth == 'Preining':
            """Source : Vehkamaeki, H., M. Kulmala, I. Napari, K. E. J.
            Lehtinen, C.Timmreck, M. Noppel, and A. Laaksonen (2002),
            J. Geophys. Res., 107, doi:10.1029/2002JD002184."""
            Psat = np.exp(-7235.424651/T+77.34491296+5.7113e-3*T -
                          8.2*np.log(T))/100
        elif meth == 'Wexler':
            """Wexler, A., Vapor Pressure Formulation for Water in Range 0 to
            100 C. A Revision, Journal of Research of the National Bureau of
            Standards - A. Physics and Chemistry, September - December 1976,
            Vol. 80A, Nos.5 and 6, 775-785"""
            Psat = np.exp(-0.29912729e4*np.power(T, -2) -
                          0.60170128e4*np.power(T, -1) +
                          0.1887643854e2-0.28354721e-1*T +
                          0.17838301e-4*np.power(T, 2) -
                          0.84150417e-9*np.power(T, 3) +
                          0.44412543e-12*np.power(T, 4) +  # This line was corrected from '-' to '+' following the original citation. (HV 20140819). The change makes only negligible difference
                          2.858487*np.log(T))/100
        elif meth in (['GoffGratch', 'MartiMauersberger']):
            """Marti and Mauersberger don't have a vapor pressure curve over
            liquid. Using Goff Gratch instead; Goff Gratch formulation Source :
            Smithsonian Meteorological Tables, 5th edition, p. 350, 1984
            From original source: Goff & Gratch (1946), p. 107)
            """
            Ts = 373.16  # steam point temperature in K
            ews = 1013.246  # saturation pressure at steam point temperature
            Psat = np.power(10, -7.90298*(Ts/T-1)+5.02808*np.log10(Ts/T) -
                            1.3816e-7*(np.power(10, 11.344*(1-T/Ts))-1) +
                            8.1328e-3*(np.power(10, -3.49149*(Ts/T-1))-1) +
                            np.log10(ews))
        elif meth == 'MagnusTetens':
            """Source: Murray, F. W., On the computation of \
                         saturation vapor pressure, J. Appl. Meteorol., \
                         6, 203-204, 1967."""
            Psat = np.power(10, 7.5*(temp)/(temp+237.5)+0.7858)
            # Murray quotes this as the original formula and
            Psat = 6.1078*np.exp(17.269388*temp/(temp+237.3))
            # this as the mathematical aquivalent in the form of base e.
        elif meth == 'Buck':
            """Bucks vapor pressure formulation based on Tetens formula.
            Source: Buck, A. L., New equations for computing vapor pressure and
            enhancement factor, J. Appl. Meteorol., 20, 1527-1532, 1981."""
            Psat = (6.1121*np.exp(17.502*temp/(240.97+temp)) *
                    (1.0007+(3.46e-6*P)))
        elif meth == 'Buck2':
            """Bucks vapor pressure formulation based on Tetens formula.
            Source: Buck Research, Model CR-1A Hygrometer Operating Manual,
            May 2012"""
            Psat = (6.1121*np.exp((18.678-(temp)/234.5)*(temp)/(257.14+temp)) *
                    (1+1e-4*(7.2+P*(0.0320)+5.9e-6*np.power(T, 2))))
        elif meth == 'WMO':
            """Intended WMO formulation, originally published by Goff (1957)
            incorrectly referenced by WMO technical regulations, WMO-NO 49,
            Vol I, General Meteorological Standards and Recommended Practices,
            App. A, Corrigendum Aug 2000. and incorrectly referenced by WMO
            technical regulations, WMO-NO 49, Vol I, General Meteorological
            Standards and Recommended Practices, App. A, 1988."""
            Ts = 273.16  # triple point temperature in K
            Psat = np.power(10, 10.79574*(1-Ts/T)-5.028*np.log10(T/Ts) +
                            1.50475e-4*(1-10**(-8.2969*(T/Ts-1))) +
                            0.42873e-3*(10**(4.76955*(1-Ts/T))-1) +  # in eq. 13 is -4.76955; in aerobulk is like this
                            0.78614)
        elif meth == 'WMO2018':
            """WMO 2018 edition. Annex 4.B, eq. 4.B.1, 4.B.2, 4.B.5 """
            Psat = 6.112*np.exp(17.62*temp/(243.12+temp))*(1.0016+3.15e-6*P -
                                                           0.074/P)
        elif meth == 'Sonntag':
            """Source: Sonntag, D., Advancements in the field of hygrometry,
            Meteorol. Z., N. F., 3, 51-66, 1994."""
            Psat = np.exp(-6096.9385*np.power(T, -1)+16.635794 -
                          2.711193e-2*T+1.673952e-5*np.power(T, 2) +
                          2.433502*np.log(T))  # *(1.0016+P*3.15e-6-0.074/P)
        elif meth == 'Bolton':
            """Source: Bolton, D., The computation of equivalent potential
            temperature, Monthly Weather Report, 108, 1046-1053, 1980.
            equation (10)"""
            Psat = 6.112*np.exp(17.67*temp/(temp+243.5))
        elif meth == 'IAPWS':
            """Source: Wagner W. and A. Pruss (2002), The IAPWS formulation
            1995 for the thermodynamic properties of ordinary water substance
            for general and scientific use, J. Phys. Chem. Ref. Data, 31(2),
            387-535. This is the 'official' formulation from the International
            Association for the Properties of  Water and Steam The valid range
            of this formulation is 273.16 <= T <= 647.096 K and is based on the
            ITS90 temperature scale."""
            Tc = 647.096   # K   : Temperature at the critical point
            Pc = 22.064e4  #  hPa : Vapor pressure at the critical point
            nu = (1-T/Tc)
            a1, a2, a3 = -7.85951783, 1.84408259, -11.7866497
            a4, a5, a6 = 22.6807411, -15.9618719, 1.80122502
            Psat = (Pc*np.exp(Tc/T*(a1*nu+a2*np.power(nu, 1.5) +
                    a3*np.power(nu, 3)+a4*np.power(nu, 3.5) +
                    a5*np.power(nu, 4)+ a6*np.power(nu, 7.5))))
        elif meth == 'MurphyKoop':
            """Source : Murphy and Koop, Review of the vapour pressure of ice
            and supercooled water for atmospheric applications, Q. J. R.
            Meteorol. Soc (2005), 131, pp. 1539-1565."""
            Psat = np.exp(54.842763-6763.22/T-4.210*np.log(T)+0.000367*T +
                          np.tanh(0.0415*(T-218.8))*(53.878-1331.22/T -
                          9.44523*np.log(T)+0.014025*T))/100
    # Calculate saturation pressure over ice ----------------------------------
    elif phase == 'ice':
        """Default uses Goff Gratch over ice. There is little ambiguity in the
        ice saturation curve. Goff Gratch is widely used."""
        if meth == 'MartiMauersberger':
            """Source : Marti, J. and K Mauersberger, A survey and new
            measurements of ice vapor pressure at temperatures between 170 and
            250 K, GRL 20, 363-366, 1993."""
            Psat = np.power(10, -2663.5/T+12.537)/100
        elif meth == 'HylandWexler':
            """Source Hyland, R. W. and A. Wexler, Formulations for the
            Thermodynamic Properties of the saturated Phases of H2O from
            173.15K to 473.15K, ASHRAE Trans, 89(2A), 500-519, 1983."""
            Psat = np.exp(-0.56745359e4/T+0.63925247e1-0.96778430e-2*T +
                          0.62215701e-6*np.power(T, 2) +
                          0.20747825e-8*np.power(T, 3) -
                          0.9484024e-12*np.power(T, 4) +
                          0.41635019e1*np.log(T))/100
        elif meth == 'Wexler':
            """Wexler, A., Vapor pressure formulation for ice, Journal of
            Research of the National Bureau of Standards-A. 81A, 5-20, 1977."""
            Psat = np.exp(-0.58653696e4*np.power(T, -1)+0.22241033e2 +
                          0.13749042e-1*T-0.34031775e-4*np.power(T, 2) +
                          0.26967687e-7*np.power(T, 3) +
                          0.6918651*np.log(T))/100
        elif meth == 'Hardy':
            """Source Hardy, B., 1998, ITS-90 Formulations for Vapor Pressure,
            Frostpoint Temperature, Dewpoint Temperature, and Enhancement
            Factors in the Range -100 to +100°C, The Proceedings of the Third
            International Symposium on Humidity & Moisture, London, England.
            These coefficients are updated to ITS90 based on the work by
            Bob Hardy at Thunder Scientific: http://www.thunderscientific.com/
            tech_info/reflibrary/its90formulas.pdf """
            Psat = np.exp(-0.58666426e4*np.power(T, -1)+0.2232870244e2 +
                          0.139387003e-1*T-0.34262402e-4*np.power(T, 2) +
                          0.27040955e-7*np.power(T, 3) +
                          0.67063522e-1*np.log(T))/100
        elif meth in (['GoffGratch', '', 'IAPWS']):
            """IAPWS does not provide a vapor pressure formulation over ice use
            Goff Gratch instead.
            Source : Smithsonian Meteorological Tables, 5th edition, p. 350,
            1984"""
            ei0 = 6.1071  # mbar
            T0 = 273.16   # triple point in K
            Psat = np.power(10, -9.09718*(T0/T-1)-3.56654*np.log10(T0/T) +
                            0.876793*(1-T/T0)+np.log10(ei0))
        elif meth == 'MagnusTetens':
            """Source: Murray, F. W., On the computation of saturation vapor
            pressure, J. Appl. Meteorol., 6, 203-204, 1967."""
            Psat = np.power(10, 9.5*temp/(265.5+temp)+0.7858)
            #  Murray quotes this as the original formula and
            Psat = 6.1078*np.exp(21.8745584*(T-273.16)/(T-7.66))
            # this as the mathematical aquivalent in the form of base e.
        elif meth == 'Buck':
            """Bucks vapor pressure formulation based on Tetens formula.
            Source: Buck, A. L., New equations for computing vapor pressure and
            enhancement factor, J. Appl. Meteorol., 20, 1527-1532, 1981."""
            Psat = (6.1115*np.exp(22.452*temp/(272.55+temp)) *
                    (1.0003+(4.18e-6*P)))
        elif meth == 'Buck2':
            """Bucks vapor pressure formulation based on Tetens formula.
            Source: Buck Research, Model CR-1A Hygrometer Operating Manual,
            Sep 2001"""
            Psat = (6.1115*np.exp((23.036-temp/333.7)*temp/(279.82+temp)) *
                    (1+1e-4*(2.2+P*(0.0383+6.4e-6*np.power(T, 2)))))
        elif meth == 'CIMO':
            """Source: Annex 4B, Guide to Meteorological Instruments and
            Methods of Observation, WMO Publication No 8, 7th edition, Geneva,
            2008. (CIMO Guide)"""
            Psat = (6.112*np.exp(22.46*temp/(272.62+temp)) *
                    (1.0016+3.15e-6*P-0.074/P))
        elif meth in ('WMO', 'WMO2000'):
            """There is no typo issue in the WMO formulations for ice.
            WMO formulation, which is very similar to Goff & Gratch.
            Source : WMO technical regulations, WMO-NO 49, Vol I, General
            Meteorological Standards and Recommended Practices, Aug 2000,
            App. A."""
            T0 = 273.16  # triple point temperature in K
            Psat = np.power(10, -9.09685*(T0/T-1)-3.56654*np.log10(T0/T) +
                            0.87682*(1-T/T0)+0.78614)
        elif meth == 'Sonntag':
            """Source: Sonntag, D., Advancements in the field of hygrometry,
            Meteorol. Z., N. F., 3, 51-66, 1994."""
            Psat = np.exp(-6024.5282*np.power(T, -1)+24.721994+1.0613868e-2*T -
                          1.3198825e-5*np.power(T, 2)-0.49382577*np.log(T))
        elif meth == 'MurphyKoop':
            """Source : Murphy and Koop, Review of the vapour pressure of ice
            and supercooled water for atmospheric applications, Q. J. R.
            Meteorol. Soc (2005), 131, pp. 1539-1565."""
            Psat = np.exp(9.550426-5723.265/T+3.53068*np.log(T) -
                          0.00728332*T)/100

        # s = np.where(temp > 0)
        if np.where(temp > 0).size[0] >= 1:
            """Independent of the formula used for ice, use Hyland Wexler
            (water) for temperatures above freezing (see above).
            Source Hyland, R. W. and A. Wexler, Formulations for the
            Thermodynamic Properties of the saturated Phases of H2O from
            173.15K to 473.15K, ASHRAE Trans, 89(2A), 500-519, 1983."""
            Psat_w = np.exp(-0.58002206e4/T+0.13914993e1-0.48640239e-1*T +
                            0.41764768e-4*np.power(T, 2) -
                            0.14452093e-7*np.power(T, 3) +
                            0.65459673e1*np.log(T))/100
            Psat[np.where(temp > 0)] = Psat_w[np.where(temp > 0)]

    return Psat
# -----------------------------------------------------------------------------


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
