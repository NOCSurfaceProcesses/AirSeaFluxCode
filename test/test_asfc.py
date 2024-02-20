from AirSeaFluxCode import AirSeaFluxCode
import numpy as np
import os
import unittest
import pandas as pd
pd.set_option('display.max_columns', 25)


class TestASFC(unittest.TestCase):
    # NOCS Flux dataset
    # https://rmets.onlinelibrary.wiley.com/doi/pdf/10.1002/joc.2059
    year = 2014
    data_file = os.path.join(
        os.path.dirname(__file__), 'data', f'nocs_flux_{year}.csv.gz')
    if not os.path.exists(data_file):
        raise ValueError("Don't have file")

    df = pd.read_csv(data_file)  # pandas can read gzip csv

    n_samps = 10_000
    df = df.sample(n_samps).reset_index(drop=True)

    def test_asfc_vs_nocflux(self):
        n = self.df.shape[0]

        # NOTE: NOCS Flux was built into a gridded flux dataset from ICOADS
        # point observations. My guess is that the gridding process means that
        # The flux values don't exactly correspond to the measurement values
        # at that location - i.e. the fluxes are 'averaged' in the same way as
        # the measured values. Hence we shouldn't expect perfect matching but
        # 'good enough'
        rtol = 1e-1
        thresh = 0.65

        lat = np.asarray(self.df['lat'])
        spd = np.asarray(self.df['wspd'])
        at = np.asarray(self.df['at'])
        sst = np.asarray(self.df['sst'])
        q = ['q', np.asarray(self.df['qair'])]  # qair is g/kg
        p = np.asarray(self.df['slp'])
        # sw = np.asarray(self.df['sw'])
        # lw = np.asarray(self.df['lw'])
        shf = np.asarray(self.df['shf'])
        lhf = np.asarray(self.df['lhf'])
        hin = 10
        hout = 10

        out_var = ('tau', 'latent', 'sensible', 'u10n', 't10n', 'q10n',
                   'uref', 'tref', 'qref')

        # NOCS Flux was built using S88.
        res = AirSeaFluxCode(
            spd, at, sst, "bulk", meth="S88", lat=lat, hin=hin,
            hum=q, P=p, cskin=0, hout=hout,  # Rs=sw, Rl=lw,
            tol=['all', 0.01, 0.01, 1e-2, 1e-3, 0.1, 0.1], L="tsrv",
            out_var=out_var)

        # Can only test against latent hf and sensible hf
        assert np.isclose(lhf, res['latent'], rtol=rtol).sum()/n >= thresh
        assert np.isclose(shf, res['sensible'], rtol=rtol).sum()/n >= thresh


if __name__ == '__main__':
    unittest.main()
