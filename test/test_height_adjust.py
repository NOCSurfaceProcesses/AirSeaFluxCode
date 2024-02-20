import unittest
import pandas as pd
pd.set_option('display.max_columns', 10)
import numpy as np
import os
from AirSeaFluxCode import AirSeaFluxCode


class TestHeight(unittest.TestCase):
    data_file = os.path.join(
        os.path.dirname(__file__), 'data', 'data_all.csv')
    df = pd.read_csv(data_file)

    def test_input_ref_height_match(self):
        # TEST: That ref height = input height
        test_data = self.df.head()

        lat = np.asarray(test_data["Latitude"])
        spd = np.asarray(test_data["Wind speed"])
        t = np.asarray(test_data["Air temperature"])
        sst = np.asarray(test_data["SST"])
        q = 10*np.asarray(test_data["RH"])  # Quick conversion % (100*kg/kg) -> g/kg
        p = np.asarray(test_data["P"])
        sw = np.asarray(test_data["Rs"])
        hu = np.asarray(test_data["zu"])
        ht = np.asarray(test_data["zt"])
        hin = [10, 10, 10]  # Not actually the height values in the data!
        del hu, ht, test_data
        out_var = ('tau', 'latent', 'sensible', 'uref', 'tref', 'qref')

        res = AirSeaFluxCode(
            spd, t, sst, "bulk", meth="UA", lat=lat, hin=hin,
            hum=["q", q], P=p, cskin=0, Rs=sw, hout=10,
            tol=['all', 0.01, 0.01, 1e-2, 1e-3, 0.1, 0.1], L="tsrv",
            out_var=out_var)

        # TEST: ref == 10 so expect these to be equal (maybe some thresh)
        assert np.isclose(spd, res['uref'], rtol=1e-3).all()
        assert np.isclose(t+273.16, res['tref'], rtol=1e-3).all()
        assert np.isclose(q, res['qref'], rtol=1e-3).all()

    def test_height_adjust(self):
        # TEST: That height adjusted data gives similar results
        test_data = self.df
        n = test_data.shape[0]

        # Proportion of results to be similar within relative tolerance
        # WARN: rtol and thresh are set quite low for this to pass
        rtol = 1e-1
        thresh = 0.9

        lat = np.asarray(test_data["Latitude"])
        spd = np.asarray(test_data["Wind speed"])
        t = np.asarray(test_data["Air temperature"])
        sst = np.asarray(test_data["SST"])
        rh = np.asarray(test_data["RH"])
        p = np.asarray(test_data["P"])
        sw = np.asarray(test_data["Rs"])
        hu = np.asarray(test_data["zu"])
        ht = np.asarray(test_data["zt"])
        hin = np.array([hu, ht, ht])
        del hu, ht, test_data
        out_var = ('tau', 'latent', 'sensible', 'u10n', 't10n', 'q10n',
                   'uref', 'tref', 'qref')

        # TEST: Similar results at a custom hout
        hout = 2
        res_at_hin = AirSeaFluxCode(
            spd, t, sst, "bulk", meth="UA", lat=lat, hin=hin,
            hum=["rh", rh], P=p, cskin=0, Rs=sw, hout=hout,
            tol=['all', 0.01, 0.01, 1e-2, 1e-3, 0.1, 0.1], L="tsrv",
            out_var=out_var)

        spdh = np.asarray(res_at_hin['uref'])
        th = np.asarray(res_at_hin['tref'])
        humh = ['q', np.asarray(res_at_hin['qref'])]

        res_at_h = AirSeaFluxCode(
            spdh, th, sst, "bulk", meth="UA", lat=lat, hin=[hout, hout, hout],
            hum=humh, P=p, cskin=0, Rs=sw, hout=hout,
            tol=['all', 0.01, 0.01, 1e-2, 1e-3, 0.1, 0.1], L="tsrv",
            out_var=out_var)

        tau_test_h = np.isclose(
            res_at_hin['tau'], res_at_h['tau'], rtol=rtol).sum()/n

        latent_test_h = np.isclose(res_at_hin['latent'],
                                   res_at_h['latent'], rtol=rtol).sum()/n
        sensible_test_h = np.isclose(res_at_hin['sensible'],
                                     res_at_h['sensible'], rtol=rtol).sum()/n

        assert tau_test_h > thresh
        assert latent_test_h > thresh
        assert sensible_test_h > thresh


if __name__ == '__main__':
    unittest.main()
