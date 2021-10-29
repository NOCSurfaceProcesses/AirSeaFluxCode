import importlib
import sys
sys.path.insert(1, '/Users/ricorne/projects/orchestra/Code')
import   AirSeaFluxCode
importlib.reload(AirSeaFluxCode)
from AirSeaFluxCode import *
import pickle

inDt = pd.read_csv("~/projects/orchestra/Test_Data/data_all.csv")
date = np.asarray(inDt["Date"])
lon = np.asarray(inDt["Longitude"])
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
Rs = np.asarray(inDt["Rs"])

del hu, ht, inDt
# run AirSeaFluxCode
res1 = AirSeaFluxCode(spd, t, sst, lat=lat, hin=hin, P=p, maxiter=10,hum=None,cskin=0,wl=0,
                      tol=['all', 0.01, 0.01, 1e-05, 1e-3, 0.1, 0.1], L="Rb",meth="S80",gust=None)

# old version
pickle_off = open ("/Users/ricorne/projects/AirSeaFluxCode_master/orchestra/old_code.txt", "rb")
res = pickle.load(pickle_off)

for i in res1.columns:
    try:
        a=res1[i].round(4)
        b=res[i].round(4)
        print(a.equals(b))
    except:
        print(res1[i].equals(res[i]))
