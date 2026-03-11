import pytest  # noqa: F401

from pathlib import Path
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


def test_temp_conv():
    flux = asfc.AirSeaFluxCode(
        spd=np.array([1.7]),
        T=np.array([27.6]),
        SST=np.array([28.219999]),
        SST_fl="skin",
        meth="C35",
        lat=np.array([0]),
        hum=["rh", np.array([80.4])],
        out_var=["sensible", "latent", "SST", "T"],
        hin=np.array([4, 3, 3]),
        convert=True,
    )
    assert all(flux["T"] > 200.0)
    assert all(flux["SST"] > 200.0)
    assert all(flux["SST"] != flux["T"])


def test_vs_coare():
    # TEST: against a test-file from COARE 3.5: https://github.com/NOAA-PSL/COARE-algorithm
    in_path = Path(os.path.dirname(__file__)) / "data"
    in_file = in_path / "test_35_data.txt"
    res_file = in_path / "test_35_output_py_082020.txt"
    assert in_file.exists() and res_file.exists(), "Missing input files"

    in_df = pd.read_csv(in_file, delimiter="\t")
    res_df = pd.read_csv(res_file, delimiter="\t")

    q = ["rh", np.asarray(in_df["rh"])]
    hgts = np.asarray(in_df[["zu", "zt", "zq"]]).T

    test_result = asfc.AirSeaFluxCode(
        spd=np.asarray(in_df["u"]),
        T=np.asarray(in_df["t"]),
        SST=np.asarray(in_df["ts"]),
        SST_fl="bulk",
        meth="C35",
        qmeth="Buck",
        lat=np.asarray(in_df["lat"]),
        hum=q,
        P=np.asarray(in_df["P"]),
        hin=hgts,
        hout=10,
        Rl=np.asarray(in_df["Rl"]),
        Rs=np.asarray(in_df["Rs"]),
        cskin=1,
        # skin="C35",
        gust=[5, 1.2, 600, 0.2],
        out_var=["usr", "tau", "sensible", "latent", "tsr", "qsr", "monob"],
        # maxiter=10,
    )

    assert np.allclose(test_result["usr"], res_df["# usr"], rtol=1e-2), "usr incorrect"
    assert np.allclose(test_result["tsr"], res_df["tsr"], atol=5e-4), (
        "tsr incorrect (0.0005)"
    )
    assert np.allclose(test_result["qsr"], res_df["qsr"], rtol=1e-2), "qsr incorrect"

    assert np.allclose(test_result["tau"], res_df["tau"], rtol=1e-2), "tau incorrect"
    assert np.allclose(test_result["latent"], -res_df["hlb"], rtol=1e-2), (
        "latent incorrect"
    )
    assert np.allclose(test_result["sensible"], -res_df["hsb"], atol=5e-2), (
        "sensible incorrect"
    )
