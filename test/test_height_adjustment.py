import pytest  # noqa: F401

import os
import numpy as np
import pandas as pd

import AirSeaFluxCode as asfc
from AirSeaFluxCode.height_subs import (
    adjust_humidity,
    adjust_temperature,
    adjust_wind_speed,
)


def test_height_adjustment() -> None:
    # TEST: Simple test to check that the main process runs
    data_path = os.path.join(asfc.__base__, "..", "Test_Data", "data_all.csv")
    inDt = pd.read_csv(data_path, nrows=10)

    hin = 5
    hout = 25

    # INFO: Generate an initial set of temperatures, speed, humidity at 25m
    #       with Monob and Stars
    lat = np.asarray(inDt["Latitude"])
    spd = np.asarray(inDt["Wind speed"])
    t = np.asarray(inDt["Air temperature"])
    sst = np.asarray(inDt["SST"])
    rh = np.asarray(inDt["RH"])
    p = np.asarray(inDt["P"])
    sw = np.asarray(inDt["Rs"])
    hu = np.asarray(inDt["zu"])
    ht = np.asarray(inDt["zt"])
    del hu, ht, inDt
    outvar = (
        "tau",
        "sensible",
        "latent",
        "uref",
        "tref",
        "qref",
        "tsr",
        "usr",
        "qsr",
        "monob",
    )
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
        tol=["all", 0.01, 0.01, 1e-5, 1e-3, 0.1, 0.1],
        L="tsrv",
        out_var=outvar,
    )

    monob = res.loc[:, "monob"]
    ustr = res.loc[:, "usr"]
    tstr = res.loc[:, "tsr"]
    qstr = res.loc[:, "qsr"]
    u_in = res.loc[:, "uref"]
    t_in = res.loc[:, "tref"]
    q_in = res.loc[:, "qref"]

    # Adjust to 5m
    t5 = adjust_temperature(t_in, monob=monob, tsr=tstr, h_in=hout, h_out=hin)
    u5 = adjust_wind_speed(u_in, monob=monob, usr=ustr, h_in=hout, h_out=hin)
    q5 = adjust_humidity(q_in, monob=monob, qsr=qstr, h_in=hout, h_out=hin)

    # Readjust to 25m
    t25 = adjust_temperature(t5, monob=monob, tsr=tstr, h_in=hin, h_out=hout)
    u25 = adjust_wind_speed(u5, monob=monob, usr=ustr, h_in=hin, h_out=hout)
    q25 = adjust_humidity(q5, monob=monob, qsr=qstr, h_in=hin, h_out=hout)

    # Test is original
    assert np.allclose(t25, t_in)
    assert np.allclose(u25, u_in)
    assert np.allclose(q25, q_in)
