import pytest  # noqa: F401

import os
import numpy as np
import pandas as pd

import AirSeaFluxCode as asfc

DATA_PATH = os.path.join(asfc.__base__, "..", "Test_Data", "data_all.csv")
DF = pd.read_csv(DATA_PATH)
N = len(DF)
OUTVAR = ("tau", "sensible", "latent", "u10n", "t10n", "q10n")
LAT = np.asarray(DF["Latitude"])
SPD = np.asarray(DF["Wind speed"])
AT = np.asarray(DF["Air temperature"])
SST = np.asarray(DF["SST"])
RH = np.asarray(DF["RH"])
P = np.asarray(DF["P"])
SW = np.asarray(DF["Rs"])
HU = np.asarray(DF["zu"])
HT = np.asarray(DF["zt"])
H_IN = np.array([HU, HT, HT])
del HU, HT


@pytest.mark.parametrize(
    "meth, qmeth, ssfl, cskin, L, wl, gust",
    [
        ("S80", "Buck", "bulk", 0, "tsrv", 0, None),
        ("S80", "Buck2", "bulk", 0, "tsrv", 0, None),
        ("S88", "MagnusTetens", "bulk", 0, "tsrv", 0, None),
        ("LP82", "Hardy", "bulk", 0, "tsrv", 0, None),
        ("YT96", "GoffGratch", "bulk", 0, "tsrv", 0, None),
        ("YT96", "WMO", "bulk", 0, "tsrv", 1, None),
        ("UA", "WMO2018", "bulk", 0, "tsrv", 0, None),
        ("NCAR", "Wexler", "bulk", 0, "tsrv", 0, None),
        ("C30", "Sonntag", "bulk", 1, "tsrv", 0, None),
        ("C30", "Bolton", "skin", 0, "tsrv", 0, None),
        ("C35", "HylandWexler", "bulk", 1, "tsrv", 1, [1, 1.2, 500, 0.02]),
        ("ecmwf", "IAPWS", "bulk", 1, "tsrv", 0, None),
        ("Beljaars", "Preining", "bulk", 1, "Rb", 1, None),
        ("Beljaars", "MurphyKoop", "skin", 0, "Rb", 0, None),
    ],
)
def test_toy_asfc(meth, qmeth, ssfl, cskin, L, wl, gust) -> None:
    # TEST: Simple test to check that the main process runs

    # run AirSeaFluxCode
    res = asfc.AirSeaFluxCode(
        SPD,
        AT,
        SST,
        ssfl,
        meth=meth,
        lat=LAT,
        hin=H_IN,
        hum=["rh", RH],
        P=P,
        cskin=cskin,
        qmeth=qmeth,
        Rs=SW,
        tol=["all", 0.01, 0.01, 1e-05, 1e-3, 0.1, 0.1],
        L=L,
        out_var=OUTVAR,
        wl=wl,
        gust=gust,
    )

    assert isinstance(res, pd.DataFrame)
    assert res.shape[0] == N


@pytest.mark.parametrize(
    "meth, qmeth, ssfl, cskin, L, wl, gust",
    [
        ("UA", "Buck2", "bulk", 0, "tsrv", 0, None),
    ],
)
def test_toy_asfc_no_hum(meth, qmeth, ssfl, cskin, L, wl, gust) -> None:
    # TEST: Simple test to check that the main process runs

    # run AirSeaFluxCode
    res = asfc.AirSeaFluxCode(
        SPD,
        AT,
        SST,
        ssfl,
        meth=meth,
        lat=LAT,
        hin=H_IN,
        hum=None,
        P=P,
        cskin=cskin,
        qmeth=qmeth,
        Rs=SW,
        tol=["all", 0.01, 0.01, 1e-05, 1e-3, 0.1, 0.1],
        L=L,
        out_var=OUTVAR,
        wl=wl,
        gust=gust,
    )

    assert isinstance(res, pd.DataFrame)
    assert res.shape[0] == N
    assert np.isnan(res["latent"]).all()
    assert np.isnan(res["q10n"]).all()
