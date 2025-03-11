import numpy as np

CtoK = 273.16  # 273.15
r""" Conversion factor for $^\circ\,$C to K """

kappa = 0.4  # NOTE: 0.41
""" von Karman's constant """
# -----------------------------------------------------------------------------


def get_heights(h, dim_len):
    """ Reads input heights for velocity, temperature and humidity

    Parameters
    ----------
    h : float
        input heights [m]
    dim_len : int
        length dimension

    Returns
    -------
    hh : array
    """
    hh = np.zeros((3, dim_len))
    if isinstance(h, (float, int)):
        hh[0, :], hh[1, :], hh[2, :] = h, h, h
    elif (len(h) == 2 and np.ndim(h) == 1):
        hh[0, :], hh[1, :], hh[2, :] = h[0], h[1], h[1]
    elif (len(h) == 3 and np.ndim(h) == 1):
        hh[0, :], hh[1, :], hh[2, :] = h[0], h[1], h[2]
    elif (len(h) == 1 and np.ndim(h) == 2):
        hh = np.zeros((3, h.shape[1]))
        hh[0, :], hh[1, :], hh[2, :] = h[0, :], h[0, :], h[0, :]
    elif (len(h) == 2 and np.ndim(h) == 2):
        hh = np.zeros((3, h.shape[1]))
        hh[0, :], hh[1, :], hh[2, :] = h[0, :], h[1, :], h[1, :]
    elif (len(h) == 3 and np.ndim(h) == 2):
        hh = np.zeros((3, h.shape[1]))
        hh = np.copy(h)
    return hh
# ---------------------------------------------------------------------


def gc(lat, lon=None):
    r""" Computes gravity relative to latitude

    Parameters
    ----------
    lat : float
        latitude [$^\circ$]
    lon : float
        longitude [$^\circ$, optional]

    Returns
    -------
    gc : float
        gravity constant [m/s^2]
    """
    gamma = 9.7803267715
    c1 = 0.0052790414
    c2 = 0.0000232718
    c3 = 0.0000001262
    c4 = 0.0000000007
    if lon is not None:
        _, lat_m = np.meshgrid(lon, lat)
    else:
        lat_m = lat
    phi = lat_m*np.pi/180.
    xx = np.sin(phi)
    gc = (gamma*(1+c1*np.power(xx, 2)+c2*np.power(xx, 4)+c3*np.power(xx, 6) +
          c4*np.power(xx, 8)))
    return gc
# ---------------------------------------------------------------------


def visc_air(T):
    r""" Computes the kinematic viscosity of dry air as a function of air temp.
    following Andreas (1989), CRREL Report 89-11.

    Parameters
    ----------
    Ta : float
        air temperature [$^\circ$\,C]

    Returns
    -------
    visa : float
        kinematic viscosity [m^2/s]
    """
    T = np.asarray(T)
    if (np.nanmin(T) > 200):  # if Ta in Kelvin convert to Celsius
        T = T-273.16
    visa = 1.326e-5*(1+6.542e-3*T+8.301e-6*np.power(T, 2) -
                     4.84e-9*np.power(T, 3))
    return visa
# ---------------------------------------------------------------------


def set_flag(miss, rh, u10n, q10n, t10n, Rb, hin, monob, itera, out=0):
    """
    Set general flags.

    Parameters
    ----------
    miss : int
        mask of missing input points
    rh : float
        relative humidity             [%]
    u10n : float
        10m neutral wind speed        [ms^{-1}]
    q10n : float
        10m neutral specific humidity [g/kg]
    t10n : float
        10m neutral air temperature   [K]
    Rb : float
        bulk Richardson number
    hin : float
        measurement heights           [m]
    monob : float
        Monin-Obukhov length          [m]
    itera : int
        number of iteration
    out : int, optional
        output option for non converged points. The default is 0.

    Returns
    -------
    flag : str

    """
    # set maximum/minimum acceptable values
    u10max = 200
    q10max = 40  # [g/kg] (Equivalent to 0.04 kg/kg)
    t10min, t10max = 173, 373
    Rbmin, Rbmax = -0.5, 0.2
    flag = np.full(miss.shape, "n", dtype="object")
    flag = np.where(np.isnan(miss), "m", flag)

    # relative humidity flag
    flag = np.where(rh > 100, "r", flag)

    # u10n flag
    flag = np.where(((u10n < 0) | (u10n > u10max)) & (flag == "n"), "u",
                    np.where(((u10n < 0) | (u10n > u10max)) &
                             (np.char.find(flag.astype(str), 'u') == -1),
                             flag+[","]+["u"], flag))
    # q10n flag
    flag = np.where(((q10n < 0) | (q10n > q10max)) & (flag == "n"), "q",
                    np.where(((q10n < 0) | (q10n > q10max)) & (flag != "n"),
                             flag+[","]+["q"], flag))

    # t10n flag
    flag = np.where(((t10n < t10min) | (t10n > t10max)) & (flag == "n"), "t",
                    np.where(
                        ((t10n < t10min) | (t10n > t10max)) & (flag != "n"),
                        flag+[","]+["t"], flag))
    # stability flag
    flag = np.where(((Rb < Rbmin) | (Rb > Rbmax) |
                     ((hin[0]/monob) > 1000)) & (flag == "n"), "l",
                    np.where(((Rb < Rbmin) | (Rb > Rbmax) |
                              (np.abs(hin[0]/monob) > 1000)) &
                             (flag != "n"), flag+[","]+["l"], flag))

    if out == 1:
        flag = np.where((itera == -1) & (flag == "n"), "i", np.where(
            (itera == -1) & ((flag != "n") & (
                np.char.find(flag.astype(str), 'm') == -1)),
            flag+[","]+["i"], flag))
    else:
        flag = np.where((itera == -1) & (flag == "n"), "i", np.where(
            (itera == -1) & ((flag != "n") & (
                np.char.find(flag.astype(str), 'm') == -1) &
                (np.char.find(flag.astype(str), 'u') == -1)),
            flag+[","]+["i"], flag))

    return flag
# ---------------------------------------------------------------------


def get_outvars(out_var, cskin, gust):
    if out_var is None:  # full output
        if cskin == 1 and gust[0] == 0:  # skin ON and gust OFF
            res_vars = ("tau", "sensible", "latent", "monob", "cd", "cd10n",
                        "ct", "ct10n", "cq", "cq10n", "tsrv", "tsr", "qsr",
                        "usr", "psim", "psit", "psiq", "psim_ref", "psit_ref",
                        "psiq_ref", "u10n", "t10n", "q10n", "zo", "zot", "zoq",
                        "uref", "tref", "qref", "dter", "dqer", "dtwl", "tkt",
                        "Rl", "Rs", "Rnl", "qair", "qsea", "Rb", "rh", "rho",
                        "cp", "lv", "theta", "itera")
        elif cskin == 0 and gust[0] != 0:  # skin OFF and gust ON
            res_vars = ("tau", "sensible", "latent", "monob", "cd", "cd10n",
                        "ct", "ct10n", "cq", "cq10n", "tsrv", "tsr", "qsr",
                        "usr_gust", "ug", "GustFact",
                        "psim", "psit", "psiq", "psim_ref", "psit_ref",
                        "psiq_ref", "u10n", "t10n", "q10n", "zo", "zot", "zoq",
                        "uref", "tref", "qref", "qair", "qsea",  "Rb", "rh",
                        "rho", "cp", "lv", "theta", "itera")
        elif cskin == 0 and gust[0] == 0:
            res_vars = ("tau", "sensible", "latent", "monob", "cd", "cd10n",
                        "ct", "ct10n", "cq", "cq10n", "tsrv", "tsr", "qsr",
                        "usr", "psim", "psit", "psiq", "psim_ref", "psit_ref",
                        "psiq_ref", "u10n", "t10n", "q10n", "zo", "zot", "zoq",
                        "uref", "tref", "qref", "qair", "qsea", "Rb", "rh",
                        "rho", "cp", "lv", "theta", "itera")
        else:
            res_vars = ("tau", "sensible", "latent", "monob", "cd", "cd10n",
                        "ct", "ct10n", "cq", "cq10n", "tsrv", "tsr", "qsr",
                        "usr_gust", "ug", "GustFact",
                        "psim", "psit", "psiq", "psim_ref", "psit_ref",
                        "psiq_ref", "u10n", "t10n", "q10n", "zo", "zot", "zoq",
                        "uref", "tref", "qref", "dter", "dqer", "dtwl", "tkt",
                        "Rl", "Rs", "Rnl", "qair", "qsea", "Rb", "rh", "rho",
                        "cp", "lv", "theta", "itera")
    elif out_var == "limited":
        res_vars = ("tau", "sensible", "latent", "uref", "tref", "qref")
    else:
        res_vars = out_var
    return res_vars
# ---------------------------------------------------------------------


def rho_air(T, qair, p):
    """
    Compute density of (moist) air using the eq. of state of the atmosphere.

    as in aerobulk (https://github.com/brodeau/aerobulk/) Brodeau et al. (2016)

    Parameters
    ----------
    T : float
        absolute air temperature             [K]
    qair : float
        air specific humidity   [g/kg]
    p : float
        pressure in                [Pa]

    Returns
    -------
    rho_air : TYPE
        density of moist air   [kg/m^3]

    """
    rho_air = np.maximum(p/(287.05*T*(1+(461.495/287.05-1)*qair*0.001)), 0.8)
    return rho_air
