'''
The stratification functions :math:`\Psi_i` are integrals of the dimensionless profiles :math:`\Phi_i`, which are determined experimentally, and are applied as stablility corrections to the wind speed, temperature and humidity profiles.
They are a function of the stability parameter :math:`z/L` where :math:`L` is the Monin-Obukhov length.
'''
import numpy as np


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
        psit = np.where(zol < 0, psi_conv(zol, meth),
                        psi_ecmwf(zol))
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
    a, b, c, d = 1, 2/3, 5, 0.35
    psit = -b*(zol-c/d)*np.exp(-d*zol)-np.power(1+(2/3)*a*zol, 1.5)-(b*c)/d+1
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
