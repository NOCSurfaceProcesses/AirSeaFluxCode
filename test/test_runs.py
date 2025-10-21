import pytest  # noqa: F401

import os
import numpy as np
import pandas as pd

import AirSeaFluxCode as asfc


def test_toy_asfc() -> None:
    # TEST: Simple test to check that the main process runs
    data_path = os.path.join(asfc.__base__, "..", "Test_Data", "data_all.csv")
    inDt = pd.read_csv(data_path)
    n = len(inDt)

    lat = np.asarray(inDt["Latitude"])
    spd = np.asarray(inDt["Wind speed"])
    t = np.asarray(inDt["Air temperature"])
    sst = np.asarray(inDt["SST"])
    rh = np.asarray(inDt["RH"])
    p = np.asarray(inDt["P"])
    sw = np.asarray(inDt["Rs"])
    hu = np.asarray(inDt["zu"])
    ht = np.asarray(inDt["zt"])
    hin = np.array([hu, ht, ht])
    del hu, ht, inDt
    outvar = ("tau", "sensible", "latent", "u10n", "t10n", "q10n")
    # run AirSeaFluxCode
    res = asfc.AirSeaFluxCode(
        spd,
        t,
        sst,
        "bulk",
        meth="UA",
        lat=lat,
        hin=hin,
        hum=["rh", rh],
        P=p,
        cskin=0,
        Rs=sw,
        tol=["all", 0.01, 0.01, 1e-05, 1e-3, 0.1, 0.1],
        L="tsrv",
        out_var=outvar,
    )

    assert isinstance(res, pd.DataFrame)
    assert res.shape[0] == n
