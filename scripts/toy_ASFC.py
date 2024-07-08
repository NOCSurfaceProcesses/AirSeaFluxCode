#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example of running AirSeaFluxCode with
1. R/V data (data_all.csv) or
2. one day era5 hourly data (era5_r360x180.nc)
compute fluxes
output NetCDF4 or csv
and statistics in "outS".txt

@author: sbiri
"""
#%% import packages
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
from AirSeaFluxCode import AirSeaFluxCode
import time
from tabulate import tabulate
#%%
def toy_ASFC(inF, outF, outS, sst_fl, gustIn, cskinIn, tolIn, meth, qmIn, LIn,
             stdIn):
    """
    Example routine of how to run AirSeaFluxCode with the test data given
    and save output either as .csv or NetCDF

    Parameters
    ----------
    inF : str
        input filename either data_all.csv or era5_r360x180.nc
    outF : str
        output filename
    outS : str
        output statistics filename
    gustIn : float
        gustiness option e.g. [1, 1.2, 800]
    cskinIn : int
        cool skin option input 0 or 1
    tolIn : float
        tolerance input option e.g. ['all', 0.01, 0.01, 1e-05, 1e-3, 0.1, 0.1]
    meth : str
        parametrisation method option

    Returns
    -------
    res : float
        AirSeaFluxCode output
    lon : float
        longitude from input netCDF file
    lat : float
        latitude from input netCDF file

    """
    out_var = ("tau", "sensible", "latent", "u10n", "t10n", "q10n")
    if (inF == "data_all.csv"):
        #%% load data_all
        inDt = pd.read_csv("data_all.csv")
        date = np.asarray(inDt["Date"])
        lon = np.asarray(inDt["Longitude"])
        lat = np.asarray(inDt["Latitude"])
        spd = np.asarray(inDt["Wind speed"])
        spd = spd+np.random.normal(0, stdIn[0], spd.shape)
        t = np.asarray(inDt["Air temperature"])
        t = t+np.random.normal(0, stdIn[1], t.shape)
        sst = np.asarray(inDt["SST"])
        sst = sst+np.random.normal(0, stdIn[2], sst.shape)
        rh = np.asarray(inDt["RH"])
        rh = rh+np.random.normal(0, stdIn[3], rh.shape)
        p = np.asarray(inDt["P"])
        sw = np.asarray(inDt["Rs"])
        hu = np.asarray(inDt["zu"])
        ht = np.asarray(inDt["zt"])
        hin = np.array([hu, ht, ht])
        del hu, ht, inDt
        #%% run AirSeaFluxCode
        res = AirSeaFluxCode(spd, t, sst, sst_fl, lat=lat, hum=['rh', rh], P=p,
                             hin=hin, Rs=sw, tol=tolIn, gust=gustIn,
                             cskin=cskinIn, meth=meth, qmeth=qmIn, L=LIn, 
                             maxiter=10, out_var = out_var)
        flg = res["flag"]

    elif (inF == 'era5_r360x180.nc'):
        #%% load era5_r360x180.nc
        fid = nc.Dataset(inF)
        lon = np.array(fid.variables["lon"])
        lat = np.array(fid.variables["lat"])
        tim = np.array(fid.variables["time"])
        lsm = np.array(fid.variables["lsm"])
        icon = np.array(fid.variables["siconc"])
        lsm = np.where(lsm > 0, np.nan, 1) # reverse 0 on land 1 over ocean
        icon = np.where(icon == 0, 1, np.nan) # keep only ice-free regions
        msk = lsm*icon
        T = np.array(fid.variables["t2m"])*msk
        T = T+np.random.normal(0, stdIn[1], T.shape)
        Td = np.array(fid.variables["d2m"])*msk
        Td = Td+np.random.normal(0, stdIn[3], T.shape)
        sst = np.array(fid.variables["sst"])
        sst = np.where(sst < -100, np.nan, sst)*msk
        sst = sst+np.random.normal(0, stdIn[2], sst.shape)
        p = np.array(fid.variables["msl"])*msk/100 # to set hPa
        lw = np.array(fid.variables["strd"])*msk/60/60
        sw = np.array(fid.variables["ssrd"])*msk/60/60
        u = np.array(fid.variables["u10"])
        v = np.array(fid.variables["v10"])
        fid.close()
        spd = np.sqrt(np.power(u, 2)+np.power(v, 2))*msk
        spd = spd+np.random.normal(0, stdIn[0], spd.shape)
        del u, v, fid
        hin = np.array([10, 2, 2])
        latIn = np.tile(lat, (len(lon), 1)).T.reshape(len(lon)*len(lat))
        date = np.copy(tim)

        #%% run AirSeaFluxCode
        res = np.zeros((len(tim),len(lon)*len(lat), 6))
        flg = np.empty((len(tim),len(lon)*len(lat)), dtype="object")
        # reshape input and run code
        for x in range(len(tim)):
            a = AirSeaFluxCode(spd.reshape(len(tim), len(lon)*len(lat))[x, :],
                               T.reshape(len(tim), len(lon)*len(lat))[x, :],
                               sst.reshape(len(tim), len(lon)*len(lat))[x, :],
                               sst_fl, lat=latIn,
                               hum=['Td', Td.reshape(len(tim), len(lon)*len(lat))[x, :]],
                               P=p.reshape(len(tim), len(lon)*len(lat))[x, :],
                               hin=hin,
                               Rs=sw.reshape(len(tim), len(lon)*len(lat))[x, :],
                               Rl=lw.reshape(len(tim), len(lon)*len(lat))[x, :],
                               gust=gustIn, cskin=cskinIn, tol=tolIn,
                               meth=meth, qmeth=qmIn, maxiter=30, L=LIn,
                               out_var = out_var)
            temp = a.iloc[:, :-1]
            temp = temp.to_numpy()
            flg[x, :] = a["flag"]
            res[x, :, :] = temp
            del a, temp
            n = np.shape(res)
            res = np.asarray([res[:, :, i]*msk.reshape(n[0], n[1])
                              for i in range(6)])
            res = np.moveaxis(res, 0, -1)
        flg = np.where(np.isnan(msk.reshape(len(tim), len(lon)*len(lat))),
                        'm', flg)
    if (outF[-3:] == '.nc'):
        if (inF == 'era5_r360x180.nc'):
            #%% save NetCDF4
            fid = nc.Dataset(outF,'w', format='NETCDF4')
            fid.createDimension('lon', len(lon))
            fid.createDimension('lat', len(lat))
            fid.createDimension('time', None)
            longitude = fid.createVariable('lon', 'f4', 'lon')
            latitude = fid.createVariable('lat', 'f4', 'lat')
            Date = fid.createVariable('Date', 'i4', 'time')
            tau = fid.createVariable('tau', 'f4', ('time','lat','lon'))
            sensible = fid.createVariable('shf', 'f4', ('time','lat','lon'))
            latent = fid.createVariable('lhf', 'f4', ('time','lat','lon'))
            monob = fid.createVariable('MO', 'f4', ('time','lat','lon'))
            u10n = fid.createVariable('u10n', 'f4', ('time','lat','lon'))
            t10n = fid.createVariable('t10n', 'f4', ('time','lat','lon'))
            q10n = fid.createVariable('q10n', 'f4', ('time','lat','lon'))
            flag = fid.createVariable('flag', 'U1', ('time','lat','lon'))

            longitude[:] = lon
            latitude[:] = lat
            Date[:] = tim
            tau[:] = res[:, :, 0].reshape((len(tim), len(lat), len(lon)))*msk
            sensible[:] = res[:, :, 1].reshape((len(tim), len(lat), len(lon)))*msk
            latent[:] = res[:, :, 2].reshape((len(tim), len(lat), len(lon)))*msk
            u10n[:] = res[:, :, 3].reshape((len(tim), len(lat), len(lon)))*msk
            t10n[:] = res[:, :, 4].reshape((len(tim), len(lat), len(lon)))*msk
            q10n[:] = res[:, :, 5].reshape((len(tim), len(lat), len(lon)))*msk
            flag[:] = flg.reshape((len(tim), len(lat), len(lon)))

            longitude.long_name = 'Longitude'
            longitude.units = 'degrees East'
            latitude.long_name = 'Latitude'
            latitude.units = 'degrees North'
            Date.long_name = "gregorian"
            Date.units = "hours since 1900-01-01 00:00:00.0"
            tau.long_name = 'Wind stress'
            tau.units = 'N/m^2'
            sensible.long_name = 'Sensible heat fluxe'
            sensible.units = 'W/m^2'
            latent.long_name = 'Latent heat flux'
            latent.units = 'W/m^2'
            u10n.long_name = '10m neutral wind speed'
            u10n.units = 'm/s'
            t10n.long_name = '10m neutral temperature'
            t10n.units = 'degrees Celsius'
            q10n.long_name = '10m neutral specific humidity'
            q10n.units = 'kgr/kgr'
            flag.long_name = ('flag "n" normal, "u": u10n < 0, "q": q10n < 0,'
                              '"l": zol<0.01, "m": missing, "i": points that'
                              'have not converged')
            fid.close()
            #%% delete variables
            # del longitude, latitude, Date, tau, sensible, latent, monob, cd, cdn
            # del ct, ctn, cq, cqn, tsrv, tsr, qsr, usr, psim, psit, psiq, u10n, t10n
            # del tv10n, q10n, zo, zot, zoq, urefs, trefs, qrefs, itera, dter, dqer
            # del qair, qsea, Rl, Rs, Rnl, dtwl
            # del tim, T, Td, p, lw, sw, lsm, spd, hin, latIn, icon, msk
        else:
            #%% save NetCDF4
            fid = nc.Dataset(outF,'w', format='NETCDF4')
            fid.createDimension('lon', len(lon))
            fid.createDimension('lat', len(lat))
            fid.createDimension('time', None)
            longitude = fid.createVariable('lon', 'f4', 'lon')
            latitude = fid.createVariable('lat', 'f4', 'lat')
            Date = fid.createVariable('Date', 'i4', 'time')
            tau = fid.createVariable('tau', 'f4', 'time')
            sensible = fid.createVariable('shf', 'f4', 'time')
            latent = fid.createVariable('lhf', 'f4', 'time')
            u10n = fid.createVariable('u10n', 'f4', 'time')
            t10n = fid.createVariable('t10n', 'f4', 'time')
            q10n = fid.createVariable('q10n', 'f4', 'time')
            flag = fid.createVariable('flag', 'U1', 'time')

            longitude[:] = lon
            latitude[:] = lat
            Date[:] = date
            tau[:] = res["tau"]
            sensible[:] = res["shf"]
            latent[:] = res["lhf"]
            u10n[:] = res["u10n"]
            t10n[:] = res["t10n"]
            q10n[:] = res["q10n"]
            flag[:] = res["flag"]

            longitude.long_name = 'Longitude'
            longitude.units = 'degrees East'
            latitude.long_name = 'Latitude'
            latitude.units = 'degrees North'
            Date.long_name = "calendar date"
            Date.units = "YYYYMMDD UTC"
            tau.long_name = 'Wind stress'
            tau.units = 'N/m^2'
            sensible.long_name = 'Sensible heat fluxe'
            sensible.units = 'W/m^2'
            latent.long_name = 'Latent heat flux'
            latent.units = 'W/m^2'
            u10n.long_name = '10m neutral wind speed'
            u10n.units = 'm/s'
            t10n.long_name = '10m neutral temperature'
            t10n.units = 'degrees Celsius'
            q10n.long_name = '10m neutral specific humidity'
            q10n.units = 'kgr/kgr'
            flag.long_name = ('flag "n" normal, "u": u10n < 0, "q": q10n < 0,'
                              '"l": zol<0.01, "m": missing, "i": points that'
                              'have not converged')
            fid.close()
            #%% delete variables
            # del longitude, latitude, Date, tau, sensible, latent, monob, cd, cdn
            # del ct, ctn, cq, cqn, tsrv, tsr, qsr, usr, psim, psit, psiq, u10n, t10n
            # del tv10n, q10n, zo, zot, zoq, urefs, trefs, qrefs, itera, dter, dqer
            # del qair, qsea, Rl, Rs, Rnl, ug, rh, Rib
            # del t, date, p, sw, spd, hin, sst
    else:
        #%% save as .csv
        res.insert(loc=0, column='date', value=date)
        res.insert(loc=1, column='lon', value=lon)
        res.insert(loc=2, column='lat', value=lat)
        res.to_csv(outF, float_format='%.3f')
    return res, lon, lat
#%% run function
start_time = time.perf_counter()
#------------------------------------------------------------------------------
inF = input("Give input file name (data_all.csv or era5_r360x180.nc): \n")
meth = input("Give prefered method: \n")
while meth not in ["S80", "S88", "LP82", "YT96", "UA", "NCAR", "C30", "C35",
                   "ecmwf","Beljaars"]:
    print("method unknown")
    meth = input("Give prefered method: \n")
else:
    meth = meth #[meth]
ext = meth+"_"
#------------------------------------------------------------------------------
sst_fl = input("Give SST flag (bulk/sin): \n")
#------------------------------------------------------------------------------
qmIn = input("Give prefered method for specific humidity: \n")
if (qmIn == ''):
    qmIn = 'Buck2' # default
while qmIn not in ["HylandWexler", "Hardy", "Preining", "Wexler",
                   "GoffGratch", "WMO", "MagnusTetens", "Buck", "Buck2",
                   "WMO2018", "Sonntag", "Bolton", "IAPWS", "MurphyKoop"]:
    print("method unknown")
    meth = input("Give prefered method: \n")
else:
    qmIn = qmIn
#------------------------------------------------------------------------------
gustIn = input("Give gustiness option (to switch it off enter 0;"
               " to set your own input use the form [1, B, zi, ugmin]"
               " i.e. [1, 1, 800, 0.01] or "
               "to use default press enter): \n")
if (gustIn == ''):
    gustIn = None
    ext = ext+'gust_'
else:
    gustIn = np.asarray(eval(gustIn), dtype=float)
    if ((np.all(gustIn) == 0)):
        ext = ext+'nogust_'
    else:
        ext = ext+'gust_'
#------------------------------------------------------------------------------
cskinIn = input("Give cool skin option (to use default press enter): \n")
if (cskinIn == ''):
    cskinIn = None
    if ((cskinIn == None) and (meth == "S80" or meth == "S88" or meth == "LP82"
                               or meth == "YT96" or meth == "UA"
                               or meth == "NCAR")):
        cskinIn = 0
        ext = ext+'noskin_'
    elif ((cskinIn == None) and (meth == "C30" or meth == "C35"
                                 or meth == "ecmwf" or meth == "Beljaars")):
        cskinIn = 1
        ext = ext+'skin_'
else:
    cskinIn = int(cskinIn)
    if (cskinIn == 0):
        ext = ext+'noskin_'
    elif (cskinIn == 1):
        ext = ext+'skin_'
#------------------------------------------------------------------------------
tolIn = input("Give tolerance option (to use default press enter): \n")
if (tolIn == ''):
    tolIn = ['all', 0.01, 0.01, 1e-05, 1e-3, 0.1, 0.1]
else:
    tolIn = eval(tolIn)
ext = ext+'tol'+tolIn[0]
#------------------------------------------------------------------------------
LIn = input("Give prefered method for L (tsrv or Rb): \n")
if (LIn == ''):
    LIn = 'tsrv' # default
elif LIn not in ["tsrv", "Rb"]:
    LIn = 'tsrv' # default
else:
    LIn = LIn
#------------------------------------------------------------------------------
stdIn = input("Give noise std for spd, T, SST and Td/rh \n (e.g. [0.01, 0, 0, 0]"
              " adds noise only to spd): \n")
if (stdIn == ''):
    stdIn = [0, 0, 0, 0] # no noise added
else:
    stdIn = eval(stdIn)
#------------------------------------------------------------------------------
outF = input("Give path and output file name: \n")
if ((outF == '') and (inF == "data_all.csv")):
    outF = "out_"+inF[:-4]+"_"+ext+".csv"
elif ((outF == '') and (inF == "era5_r360x180.nc")):
    outF = "out_"+inF[:-3]+"_"+ext+".nc"
elif ((outF[-4:] == '.csv') and (inF == 'era5_r360x180.nc')):
    outF = outF[:-4]+".nc"
elif ((outF[-3:] != '.nc') and (outF[-4:] != '.csv')):
    if (inF == "data_all.csv"):
        outF = outF+".csv"
    else:
        outF = outF+".nc"
else:
    outF = outF
#------------------------------------------------------------------------------
outS = input("Give path and statistics file name: \n")
if ((outS == '') and (inF == "data_all.csv")):
    outS = "RV_"+ext+"_stats.txt"
elif ((outS == '') and (inF == "era5_r360x180.nc")):
    outS = "era5_"+ext+"_stats.txt"
elif (outS[-4:] != '.txt'):
    outF = outS+".txt"

#------------------------------------------------------------------------------
print("\n run_ASFC.py, started for method "+meth)

res, lon, lat = toy_ASFC(inF, outF, outS, sst_fl, gustIn, cskinIn, tolIn, 
                         meth, qmIn, LIn, stdIn)
print("run_ASFC.py took ", np.round((time.perf_counter()-start_time)/60, 2),
      "minutes to run")

#%% generate flux plots
# if (inF == 'era5_r360x180.nc'):
#     cm = plt.cm.get_cmap('RdYlBu')
#     ttl = ["tau (Nm$^{-2}$)", "shf (Wm$^{-2}$)", "lhf (Wm$^{-2}$)"]
#     for i in range(3):
#         plt.figure()
#         plt.contourf(lon, lat,
#                       np.nanmean(res[:, :, i], axis=0).reshape(len(lat),
#                                                               len(lon)),
#                       100, cmap=cm)
#         plt.colorbar()
#         plt.tight_layout()
#         plt.xlabel("Longitude")
#         plt.ylabel("Latitude")
#         plt.title(meth+', '+ttl[i])
#         plt.savefig('./'+ttl[i][:3]+'_'+ext+'.png', dpi=300, bbox_inches='tight')
# elif (inF == "data_all.csv"):
#     ttl = ["tau (Nm$^{-2}$)", "shf (Wm$^{-2}$)", "lhf (Wm$^{-2}$)"]
#     for i in range(3):
#         plt.figure()
#         plt.plot(res[ttl[i][:3]],'.c', markersize=1)
#         plt.title(meth)
#         plt.xlabel("points")
#         plt.ylabel(ttl[i])
#         plt.savefig('./'+ttl[i][:3]+'_'+ext+'.png', dpi=300, bbox_inches='tight')

#%% generate txt file with statistics
if (cskinIn == None) and (meth in ["S80", "S88", "LP82", "YT96", "UA", "NCAR"]):
   cskinIn = 0
elif (cskinIn == None) and (meth in ["C30", "C35", "ecmwf", "Beljaars"]):
   cskinIn = 1
if np.all(gustIn == None) and (meth in ["C30", "C35"]):
    gustIn = [1, 1.2, 600, 0.2]
elif np.all(gustIn == None) and (meth in ["UA", "ecmwf"]):
    gustIn = [1, 1, 1000, 0.01]
elif np.all(gustIn == None):
    gustIn = [1, 1.2, 800, 0.01]
elif (np.size(gustIn) < 4) and (gustIn == 0):
    gust = [0, 0, 0, 0]
if tolIn == None:
    tolIn = ['all', 0.01, 0.01, 1e-05, 1e-3, 0.1, 0.1]


print("Input summary", file=open('./'+outS, 'a'))
print('input file name: {}, \n method: {}, \n gustiness: {}, \n cskin: {},'
      ' \n tolerance: {}, \n qmethod: {}, \n L: {}'.format(inF, meth, gustIn,
                                                           cskinIn, tolIn,
                                                           qmIn, LIn),
      file=open('./'+outS, 'a'))
ttl = np.asarray(["tau  ", "shf  ", "lhf  ", 
                  "u10n ", "t10n ", "q10n "])
header = ["var", "mean", "median", "min", "max", "5%", "95%"]
n = np.shape(res)
stats = np.copy(ttl)
if (inF == 'era5_r360x180.nc'):
    stats = np.c_[stats, np.nanmean(res.reshape(n[0]*n[1], n[2]), axis=0)]
    stats = np.c_[stats, np.nanmedian(res.reshape(n[0]*n[1], n[2]), axis=0)]
    stats = np.c_[stats, np.nanmin(res.reshape(n[0]*n[1], n[2]), axis=0)]
    stats = np.c_[stats, np.nanmax(res.reshape(n[0]*n[1], n[2]), axis=0)]
    stats = np.c_[stats, np.nanpercentile(res.reshape(n[0]*n[1], n[2]), 5,
                                          axis=0)]
    stats = np.c_[stats, np.nanpercentile(res.reshape(n[0]*n[1], n[2]), 95,
                                          axis=0)]
    print(tabulate(stats, headers=header, tablefmt="github", numalign="left",
                   floatfmt=("s", "2.2e", "2.2e", "2.2e", "2.2e", "2.2e",
                             "2.2e")), file=open('./'+outS, 'a'))
    print('-'*79+'\n', file=open('./'+outS, 'a'))
elif (inF == "data_all.csv"):
    a = res.loc[:, "tau":"q10n"].to_numpy(dtype="float64").T
    stats = np.c_[stats, np.nanmean(a, axis=1)]
    stats = np.c_[stats, np.nanmedian(a, axis=1)]
    stats = np.c_[stats, np.nanmin(a, axis=1)]
    stats = np.c_[stats, np.nanmax(a, axis=1)]
    stats = np.c_[stats, np.nanpercentile(a, 5, axis=1)]
    stats = np.c_[stats, np.nanpercentile(a, 95, axis=1)]
    print(tabulate(stats, headers=header, tablefmt="github", numalign="left",
                   floatfmt=("s", "2.2e", "2.2e", "2.2e", "2.2e", "2.2e",
                               "2.2e")),
          file=open('./'+outS, 'a'))
    print('-'*79+'\n', file=open('./'+outS, 'a'))
    del a

print('input file name: {}, \n method: {}, \n gustiness: {}, \n cskin: {},'
      ' \n tolerance: {}, \n output is written in: {}'.format(inF, meth,
                                                              gustIn, cskinIn,
                                                              tolIn, outF),
      file=open('./readme.txt', 'w'))
