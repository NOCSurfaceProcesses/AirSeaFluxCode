import unittest
import pandas as pd
import numpy as np
import os
from AirSeaFluxCode import AirSeaFluxCode


class TestHeight(unittest.TestCase):
    data_file = os.path.join(
        os.path.dirname(__file__), 'data', 'data_all.csv')
    df = pd.read_csv(data_file)

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

        tau_test = np.isclose(
            res_at_hin['tau'], res_at_h['tau'], rtol=rtol).sum()/n

        latent_test = np.isclose(res_at_hin['latent'],
                                 res_at_h['latent'], rtol=rtol).sum()/n
        sensible_test = np.isclose(res_at_hin['sensible'],
                                   res_at_h['sensible'], rtol=rtol).sum()/n

        assert tau_test > thresh
        assert latent_test > thresh
        assert sensible_test > thresh

        # TEST: Similar results at 10m default
        spd10 = np.asarray(res_at_hin['u10n'])
        t10 = np.asarray(res_at_hin['t10n'])
        hum10 = ['q', np.asarray(res_at_hin['q10n'])]

        res_at_10 = AirSeaFluxCode(
            spd10, t10, sst, "bulk", meth="UA", lat=lat, hin=[10, 10, 10],
            hum=hum10, P=p, cskin=0, Rs=sw, hout=hout,
            tol=['all', 0.01, 0.01, 1e-2, 1e-3, 0.1, 0.1], L="tsrv",
            out_var=out_var)

        tau_test_2 = np.isclose(
            res_at_hin['tau'], res_at_10['tau'], rtol=rtol).sum()/n

        latent_test_2 = np.isclose(res_at_hin['latent'],
                                   res_at_10['latent'], rtol=rtol).sum()/n
        sensible_test_2 = np.isclose(res_at_hin['sensible'],
                                     res_at_10['sensible'], rtol=rtol).sum()/n

        assert tau_test_2 > thresh
        assert latent_test_2 > thresh
        assert sensible_test_2 > thresh


if __name__ == '__main__':
    unittest.main()
