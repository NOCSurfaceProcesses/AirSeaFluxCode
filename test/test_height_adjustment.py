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


DATA_PATH = os.path.join(asfc.__base__, "..", "Test_Data", "data_all.csv")
DF = pd.read_csv(DATA_PATH, nrows=100)
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
OUTVAR = (
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
del HU, HT


@pytest.mark.parametrize(
    "hin, hout",
    [
        (5, 25),
        (2, 100),
        (15, 10),
        (25, 2),
    ],
)
def test_height_adjustment(hin, hout) -> None:
    # TEST: Simple test to check that the main process runs

    # INFO: Generate an initial set of temperatures, speed, humidity with Monob and
    #       Stars

    # run AirSeaFluxCode
    res = asfc.AirSeaFluxCode(
        SPD,
        AT,
        SST,
        "bulk",
        meth="UA",
        lat=LAT,
        hin=H_IN,
        hout=hin,
        hum=["rh", RH],
        P=P,
        cskin=0,
        Rs=SW,
        tol=["all", 0.01, 0.01, 1e-5, 1e-3, 0.1, 0.1],
        L="tsrv",
        out_var=OUTVAR,
    )

    monob = res.loc[:, "monob"]
    tstr = res.loc[:, "tsr"]
    ustr = res.loc[:, "usr"]
    qstr = res.loc[:, "qsr"]
    t_in = res.loc[:, "tref"]
    u_in = res.loc[:, "uref"]
    q_in = res.loc[:, "qref"]

    # Adjust to h-out
    t_out = adjust_temperature(
        t_in, monob=monob, tsr=tstr, h_in=hin, h_out=hout, meth="UA"
    )
    u_out = adjust_wind_speed(
        u_in, monob=monob, usr=ustr, h_in=hin, h_out=hout, meth="UA"
    )
    q_out = adjust_humidity(
        q_in, monob=monob, qsr=qstr, h_in=hin, h_out=hout, meth="UA"
    )

    # Readjust to h-in
    t_test = adjust_temperature(
        t_out, monob=monob, tsr=tstr, h_in=hout, h_out=hin, meth="UA"
    )
    u_test = adjust_wind_speed(
        u_out, monob=monob, usr=ustr, h_in=hout, h_out=hin, meth="UA"
    )
    q_test = adjust_humidity(
        q_out, monob=monob, qsr=qstr, h_in=hout, h_out=hin, meth="UA"
    )

    # TEST: is same as generated test input
    assert np.allclose(t_test, t_in)
    assert np.allclose(u_test, u_in)
    assert np.allclose(q_test, q_in)
