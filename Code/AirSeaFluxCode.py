import warnings
import numpy as np
import pandas as pd
import logging
from hum_subs import (get_hum, gamma)
from util_subs import *
from flux_subs import *

class S88:
    
    def get_heights(self, hin, hout=10):
        self.hout = hout
        self.hin = hin
        self.h_in = get_heights(hin, len(self.spd))
        self.h_out = get_heights(self.hout,1)

    def get_specHumidity(self,qmeth="Buck2"):
        self.qair, self.qsea = get_hum(self.hum, self.T, self.SST, self.P, qmeth)
        if (np.all(np.isnan(self.qsea)) or np.all(np.isnan(self.qair))):
            raise ValueError("qsea and qair cannot be nan")
        self.dq = self.qair - self.qsea
        
        # Set lapse rate and Potential Temperature (now we have humdity)
        self._get_potentialT()
        self._get_lapse()

    def _get_potentialT(self):
        self.cp = 1004.67*(1+0.00084*self.qsea)
        self.th = np.where(self.T < 200, (np.copy(self.T)+CtoK) *
                  np.power(1000/self.P, 287.1/self.cp),
                  np.copy(self.T)*np.power(1000/self.P, 287.1/self.cp))  # potential T

    def _get_lapse(self):
        self.tlapse = gamma("dry", self.SST, self.T, self.qair/1000, self.cp)
        self.Ta = np.where(self.T < 200, np.copy(self.T)+CtoK+self.tlapse*self.h_in[1],
                      np.copy(self.T)+self.tlapse*self.h_in[1])  # convert to Kelvin if needed
        self.dt = self.Ta - self.SST

    def _fix_coolskin_warmlayer(self, wl, cskin, skin, Rl, Rs):
        assert wl in [0,1], "wl not valid"
        assert cskin in [0,1], "cskin not valid"
        assert skin in ["C35", "ecmwf" or "Beljaars"], "Skin value not valid"

        if ((cskin == 1 or wl == 1) and (np.all(Rl == None) or np.all(np.isnan(Rl)))
            and ((np.all(Rs == None) or np.all(np.isnan(Rs))))):
            print("Cool skin/warm layer is switched ON; Radiation input should not be empty")
            raise 
        
        self.wl = wl            
        self.cskin = cskin
        self.skin = skin
        self.Rs = np.ones(self.spd.shape)*np.nan if Rs is None else Rs
        self.Rl = np.ones(self.spd.shape)*np.nan if Rl is None else Rl

    def set_coolskin_warmlayer(self, wl=0, cskin=0, skin="C35", Rl=None, Rs=None):
        wl = 0 if wl is None else wl
        self._fix_coolskin_warmlayer(wl, cskin, skin, Rl, Rs)

    def _update_coolskin_warmlayer(self,ind):
        if ((self.cskin == 1) and (self.wl == 0)):
            if (self.skin == "C35"):
                self.dter[ind], self.dqer[ind], self.tkt[ind] = cs_C35(np.copy(self.SST[ind]),
                                                                       self.qsea[ind],
                                                                       self.rho[ind], self.Rs[ind],
                                                                       self.Rnl[ind],
                                                                       self.cp[ind], self.lv[ind],
                                                                       np.copy(self.tkt[ind]),
                                                                       self.usr[ind], self.tsr[ind],
                                                                       self.qsr[ind], self.lat[ind])
            elif (self.skin == "ecmwf"):
                self.dter[ind] = cs_ecmwf(self.rho[ind], self.Rs[ind], self.Rnl[ind], self.cp[ind],
                                          self.lv[ind], self.usr[ind], self.tsr[ind], self.qsr[ind],
                                          np.copy(self.SST[ind]), self.lat[ind])
                self.dqer[ind] = (self.dter[ind]*0.622*self.lv[ind]*self.qsea[ind] /
                                  (287.1*np.power(self.SST[ind], 2)))
            elif (self.skin == "Beljaars"):
                self.Qs[ind], self.dter[ind] = cs_Beljaars(self.rho[ind], self.Rs[ind], self.Rnl[ind],
                                                           self.cp[ind], self.lv[ind], self.usr[ind],
                                                           self.tsr[ind], self.qsr[ind], self.lat[ind],
                                                           np.copy(self.Qs[ind]))
                self.dqer = self.dter*0.622*self.lv*self.qsea/(287.1*np.power(self.SST, 2))
            self.skt = np.copy(self.SST)+self.dter
        elif ((self.cskin == 1) and (self.wl == 1)):
            if (self.skin == "C35"):
                self.dter[ind], self.dqer[ind], self.tkt[ind] = cs_C35(self.SST[ind], self.qsea[ind],
                                                                       self.rho[ind], self.Rs[ind],
                                                                       self.Rnl[ind],
                                                                       self.cp[ind], self.lv[ind],
                                                                       np.copy(self.tkt[ind]),
                                                                       self.usr[ind], self.tsr[ind],
                                                                       self.qsr[ind], self.lat[ind])
                self.dtwl[ind] = wl_ecmwf(self.rho[ind], self.Rs[ind], self.Rnl[ind], self.cp[ind],
                                          self.lv[ind], self.usr[ind], self.tsr[ind], self.qsr[ind],
                                          np.copy(self.SST[ind]), np.copy(self.skt[ind]),
                                          np.copy(self.dter[ind]), self.lat[ind])
                self.skt = np.copy(self.SST)+self.dter+self.dtwl
            elif (self.skin == "ecmwf"):
                self.dter[ind] = cs_ecmwf(self.rho[ind], self.Rs[ind], self.Rnl[ind], self.cp[ind],
                                          self.lv[ind], self.usr[ind], self.tsr[ind], self.qsr[ind],
                                          self.sst[ind], self.lat[ind])
                self.dtwl[ind] = wl_ecmwf(self.rho[ind], self.Rs[ind], self.Rnl[ind], self.cp[ind],
                                          self.lv[ind], self.usr[ind], self.tsr[ind], self.qsr[ind],
                                          np.copy(self.SST[ind]), np.copy(self.skt[ind]),
                                          np.copy(self.dter[ind]), self.lat[ind])
                self.skt = np.copy(self.SST)+self.dter+self.dtwl
                self.dqer[ind] = (self.dter[ind]*0.622*self.lv[ind]*self.qsea[ind] /
                             (287.1*np.power(self.skt[ind], 2)))
            elif (self.skin == "Beljaars"):
                self.Qs[ind], self.dter[ind] = cs_Beljaars(self.rho[ind], self.Rs[ind], self.Rnl[ind],
                                                           self.cp[ind], self.lv[ind], self.usr[ind],
                                                           self.tsr[ind], self.qsr[ind], self.lat[ind],
                                                           np.copy(self.Qs[ind]))
                self.dtwl[ind] = wl_ecmwf(self.rho[ind], self.Rs[ind], self.Rnl[ind], self.cp[ind],
                                     self.lv[ind], self.usr[ind], self.tsr[ind], self.qsr[ind],
                                     np.copy(self.SST[ind]), np.copy(self.skt[ind]),
                                     np.copy(self.dter[ind]), self.lat[ind])
                self.skt = np.copy(self.SST)+self.dter+self.dtwl
                self.dqer = self.dter*0.622*self.lv*self.qsea/(287.1*np.power(self.skt, 2))
        else:
            self.dter[ind] = np.zeros(self.SST[ind].shape)
            self.dqer[ind] = np.zeros(self.SST[ind].shape)
            self.tkt[ind] = 0.001*np.ones(self.T[ind].shape)

    def _first_guess(self):

        # reference height1
        self.ref_ht = 10

        #  first guesses
        self.t10n, self.q10n = np.copy(self.Ta), np.copy(self.qair)
        self.tv10n = self.t10n*(1+0.6077*self.q10n)

        #  Zeng et al. 1998
        tv = self.th*(1+0.6077*self.qair)   # virtual potential T
        self.dtv = self.dt*(1+0.6077*self.qair)+0.6077*self.th*self.dq

        # Rb eq. 11 Grachev & Fairall 1997
        Rb = self.g*10*(self.dtv)/(np.where(self.T < 200, np.copy(self.T)+CtoK, np.copy(self.T)) * np.power(self.wind, 2))
        self.monob = 1/Rb  # eq. 12 Grachev & Fairall 1997

        # ------------
        self.rho = self.P*100/(287.1*self.tv10n)
        self.lv = (2.501-0.00237*(self.SST-CtoK))*1e6  # J/kg
        
        self.dter = np.full(self.T.shape, -0.3)*self.msk
        self.tkt = np.full(self.T.shape, 0.001)*self.msk
        self.dqer = self.dter*0.622*self.lv*self.qsea/(287.1*np.power(self.SST, 2))
        self.Rnl = 0.97*(self.Rl-5.67e-8*np.power(self.SST-0.3*self.cskin, 4))
        self.Qs = 0.945*self.Rs
        self.dtwl = np.full(self.T.shape,0.3)*self.msk
        self.skt = np.copy(self.SST)

        # Apply the gustiness adjustment if defined for this class
        try:
            self._adjust_gust()
        except AttributeError:
            pass

        self.u10n = self.wind*np.log(10/1e-4)/np.log(self.hin[0]/1e-4)
        self.usr = 0.035*self.u10n
        self.cd10n = cdn_calc(self.u10n, self.usr, self.Ta, self.lat, self.meth)
        self.cd = cd_calc(self.cd10n, self.h_in[0], self.ref_ht, self.psim)
        self.usr = np.sqrt(self.cd*np.power(self.wind, 2))

        self.zo = np.full(self.arr_shp,1e-4)*self.msk
        self.zot, self.zoq = np.copy(self.zo), np.copy(self.zo)

        self.ct10n = np.power(kappa, 2)/(np.log(self.h_in[0]/self.zo)*np.log(self.h_in[1]/self.zot))
        self.cq10n = np.power(kappa, 2)/(np.log(self.h_in[0]/self.zo)*np.log(self.h_in[2]/self.zoq))
        
        self.ct = np.power(kappa, 2)/((np.log(self.h_in[0]/self.zo)-self.psim) * (np.log(self.h_in[1]/self.zot)-self.psit))
        self.cq = np.power(kappa, 2)/((np.log(self.h_in[0]/self.zo)-self.psim) * (np.log(self.h_in[2]/self.zoq)-self.psiq))
    
        self.tsr = (self.dt-self.dter*self.cskin-self.dtwl*self.wl)*kappa/(np.log(self.h_in[1]/self.zot) - psit_calc(self.h_in[1]/self.monob, self.meth))
        self.qsr = (self.dq-self.dqer*self.cskin)*kappa/(np.log(self.h_in[2]/self.zoq) - psit_calc(self.h_in[2]/self.monob, self.meth))

    def _zo_calc(self, ref_ht, cd10n):
        zo = ref_ht/np.exp(kappa/np.sqrt(cd10n))
        return zo
    
    def iterate(self,n=10, tol=None):

        if n < 5:
            warnings.warn("Iteration number <5 - resetting to 5.")
            n = 5

        # Decide which variables to use in tolerances based on tolerance specification
        tol = ['all', 0.01, 0.01, 1e-05, 1e-3, 0.1, 0.1] if tol is None else tol
        assert tol[0] in ['flux', 'ref', 'all'], "unknown tolerance input"

        old_vars = {"flux":["tau","sensible","latent"], "ref":["u10n","t10n","q10n"]}
        old_vars["all"] = old_vars["ref"] + old_vars["flux"]
        old_vars = old_vars[tol[0]]

        new_vars = {"flux":["tau","sensible","latent"], "ref":["utmp","t10n","q10n"]}
        new_vars["all"] = new_vars["ref"] + new_vars["flux"]
        new_vars = new_vars[tol[0]]

        ind = np.where(self.spd > 0)
        it = 0

        # Setup empty arrays
        self.itera = np.full(self.arr_shp,-1)*self.msk
        
        self.tsrv = np.zeros(self.arr_shp)*self.msk
        self.psim, self.psit, self.psiq = np.copy(self.tsrv), np.copy(self.tsrv), np.copy(self.tsrv)

        self.tau = np.full(self.arr_shp,0.05)*self.msk
        self.sensible = np.full(self.arr_shp,-5)*self.msk
        self.latent = np.full(self.arr_shp,-65)*self.msk

        # Generate the first guess values
        self._first_guess()

        #  iteration loop
        ii = True
        while ii:
            it += 1
            if it > n: break

            # Set the old variables (for comparison against "new")
            old = np.array([np.copy(getattr(self,i)) for i in old_vars])

            # Calculate cdn
            self.cd10n[ind] = cdn_calc(self.u10n[ind], self.usr[ind], self.Ta[ind], self.lat[ind], self.meth)
            
            if (np.all(np.isnan(self.cd10n))):
                break
                logging.info('break %s at iteration %s cd10n<0', meth, it)
                
            self.zo[ind] = self.ref_ht/np.exp(kappa/np.sqrt(self.cd10n[ind]))
            self.psim[ind] = psim_calc(self.h_in[0, ind]/self.monob[ind], self.meth)
            self.cd[ind] = cd_calc(self.cd10n[ind], self.h_in[0, ind], self.ref_ht, self.psim[ind])

            self.ct10n[ind], self.cq10n[ind] = ctcqn_calc(self.h_in[1, ind]/self.monob[ind], self.cd10n[ind],
                                                          self.usr[ind], self.zo[ind], self.Ta[ind], self.meth)

            self.zot[ind] = self.ref_ht/(np.exp(np.power(kappa, 2) / (self.ct10n[ind]*np.log(self.ref_ht/self.zo[ind]))))
            self.zoq[ind] = self.ref_ht/(np.exp(np.power(kappa, 2) / (self.cq10n[ind]*np.log(self.ref_ht/self.zo[ind]))))
            self.psit[ind] = psit_calc(self.h_in[1, ind]/self.monob[ind], self.meth)
            self.psiq[ind] = psit_calc(self.h_in[2, ind]/self.monob[ind], self.meth)
            self.ct[ind], self.cq[ind] = ctcq_calc(self.cd10n[ind], self.cd[ind], self.ct10n[ind],self.cq10n[ind], self.h_in[:, ind],
                                         [self.ref_ht, self.ref_ht, self.ref_ht], self.psit[ind], self.psiq[ind])

            # Some parameterizations set a minimum on parameters
            try:
                self._minimum_params()
            except AttributeError:
                pass
            
            self.usr[ind],self.tsr[ind], self.qsr[ind] = get_strs(self.h_in[:, ind], self.monob[ind],
                                                self.wind[ind], self.zo[ind], self.zot[ind],
                                                self.zoq[ind], self.dt[ind], self.dq[ind],
                                                self.dter[ind], self.dqer[ind],
                                                self.dtwl[ind], self.ct[ind], self.cq[ind],
                                                self.cskin, self.wl, self.meth)


            # Update CS/WL parameters
            self._update_coolskin_warmlayer(ind)
            
            # Logging output
            log_vars = {"dter":2,"dqer":7,"tkt":2,"Rnl":2,"usr":3,"tsr":4,"qsr":7}
            log_vars = [np.round(np.nanmedian(getattr(self,V)),R) for V,R in log_vars.items()]
            log_vars.insert(0,self.meth)
            logging.info('method {} | dter = {} | dqer = {} | tkt = {} | Rnl = {} | usr = {} | tsr = {} | qsr = {}'.format(*log_vars))

            self.Rnl[ind] = 0.97*(self.Rl[ind]-5.67e-8*np.power(self.SST[ind] + self.dter[ind]*self.cskin, 4))
            self.t10n[ind] = (self.Ta[ind] - self.tsr[ind]/kappa*(np.log(self.h_in[1, ind]/self.ref_ht)-self.psit[ind]))
            self.q10n[ind] = (self.qair[ind] - self.qsr[ind]/kappa*(np.log(self.h_in[2, ind]/self.ref_ht)-self.psiq[ind]))
            self.tv10n[ind] = self.t10n[ind]*(1+0.6077*self.q10n[ind])

            self.tsrv[ind], self.monob[ind], self.Rb[ind] = get_L(self.L, self.lat[ind], self.usr[ind], self.tsr[ind], self.qsr[ind], self.h_in[:, ind], self.Ta[ind],
                                                   (self.SST[ind]+self.dter[ind]*self.cskin + self.dtwl[ind]*self.wl), self.qair[ind], self.qsea[ind], self.wind[ind],
                                                   np.copy(self.monob[ind]), self.zo[ind], self.zot[ind], self.psim[ind], self.meth)

            self.psim[ind] = psim_calc(self.h_in[0, ind]/self.monob[ind], self.meth)
            self.psit[ind] = psit_calc(self.h_in[1, ind]/self.monob[ind], self.meth)
            self.psiq[ind] = psit_calc(self.h_in[2, ind]/self.monob[ind], self.meth)

            # gust[0] is asserted to be either 0 or 1
            if (self.gust[0] == 1):
                self.wind[ind] = np.sqrt(np.power(np.copy(self.spd[ind]), 2) + np.power(get_gust(self.gust[1], self.Ta[ind], self.usr[ind],
                                                                                                 self.tsrv[ind], self.gust[2], self.lat[ind]), 2))
            else:
                self.wind[ind] = np.copy(self.spd[ind])

                
            self.u10n[ind] = self.wind[ind]-self.usr[ind]/kappa*(np.log(self.h_in[0, ind]/10) - self.psim[ind])

            # make sure you allow small negative values convergence
            if (it < 4):  self.u10n = np.where(self.u10n < 0, 0.5, self.u10n)

            self.utmp = np.copy(self.u10n)
            self.utmp = np.where(self.utmp < 0, np.nan, self.utmp)
            self.itera[ind] = np.ones(1)*it
            self.tau = self.rho*np.power(self.usr, 2)*(self.spd/self.wind)
            self.sensible = self.rho*self.cp*self.usr*self.tsr
            self.latent = self.rho*self.lv*self.usr*self.qsr

            # Set the new variables (for comparison against "old")
            new = np.array([np.copy(getattr(self,i)) for i in new_vars])

            if (it > 2):  # force at least two iterations
                d = np.abs(new-old)
                if (tol[0] == 'flux'):
                    ind = np.where((d[0, :] > tol[1])+(d[1, :] > tol[2]) + (d[2, :] > tol[3]))
                elif (tol[0] == 'ref'):
                    ind = np.where((d[0, :] > tol[1])+(d[1, :] > tol[2]) + (d[2, :] > tol[3]))
                elif (tol[0] == 'all'):
                    ind = np.where((d[0, :] > tol[1])+(d[1, :] > tol[2]) + (d[2, :] > tol[3])+(d[3, :] > tol[4]) +
                                   (d[4, :] > tol[5])+(d[5, :] > tol[6]))

            self.ind = np.copy(ind)
            ii = False if (ind[0].size == 0) else True
            # End of iteration loop

        self.itera[ind] = -1
        self.itera = np.where(self.itera > n, -1, self.itera)
        logging.info('method %s | # of iterations:%s', self.meth, it)
        logging.info('method %s | # of points that did not converge :%s \n', self.meth, self.ind[0].size)


    def _get_humidity(self):
        "RH only used for flagging purposes"
        if ((self.hum[0] == 'rh') or (self.hum[0] == 'no')):
            self.rh = self.hum[1]
        elif (self.hum[0] == 'Td'):
            Td = self.hum[1]  # dew point temperature (K)
            Td = np.where(Td < 200, np.copy(Td)+CtoK, np.copy(Td))
            T = np.where(self.T < 200, np.copy(self.T)+CtoK, np.copy(self.T))
            #T = np.copy(self.T)
            esd = 611.21*np.exp(17.502*((Td-CtoK)/(Td-32.19)))
            es = 611.21*np.exp(17.502*((T-CtoK)/(T-32.19)))
            self.rh = 100*esd/es
        
    def _flag(self,out=0):
        "Set the general flags"
        
        flag = np.full(self.arr_shp, "n",dtype="object")

        if (self.hum[0] == 'no'):
            if (self.cskin == 1):
                flag = np.where(np.isnan(self.spd+self.T+self.SST+self.P+self.Rs+self.Rl), "m", flag)
            else:
                flag = np.where(np.isnan(self.spd+self.T+self.SST+self.P), "m", flag)
        else:
            if (self.cskin == 1):
                flag = np.where(np.isnan(self.spd+self.T+self.SST+self.hum[1]+self.P+self.Rs+self.Rl), "m", flag)
            else:
                flag = np.where(np.isnan(self.spd+self.T+self.SST+self.hum[1]+self.P), "m", flag)

            flag = np.where(self.rh > 100, "r", flag)
        

        # u10n flag
        flag = np.where(((self.u10n < 0)  | (self.u10n > 999)) & (flag == "n"), "u",
                             np.where((self.u10n < 0) &
                                      (np.char.find(flag.astype(str), 'u') == -1),
                                      flag+[","]+["u"], flag))
        # q10n flag
        flag = np.where(((self.q10n < 0) | (self.q10n > 999)) & (flag == "n"), "q",
                             np.where((self.q10n < 0) & (flag != "n"), flag+[","]+["q"],
                                      flag))

        # t10n flag (not currently used)
        flag = np.where((self.t10n < -999) & (flag == "n"), "t",
                             np.where((self.t10n < 0) & (flag != "n"), flag+[","]+["t"],
                                      flag))
        
        flag = np.where(((self.Rb < -0.5) | (self.Rb > 0.2) | ((self.hin[0]/self.monob) > 1000)) &
                             (flag == "n"), "l",
                             np.where(((self.Rb < -0.5) | (self.Rb > 0.2) |
                                       ((self.hin[0]/self.monob) > 1000)) &
                                      (flag != "n"), flag+[","]+["l"], flag))

        if (out == 1):
            flag = np.where((self.itera == -1) & (flag == "n"), "i",
                                 np.where((self.itera == -1) &
                                          ((flag != "n") &
                                           (np.char.find(flag.astype(str), 'm') == -1)),
                                          flag+[","]+["i"], flag))
        else:
            flag = np.where((self.itera == -1) & (flag == "n"), "i",
                                 np.where((self.itera == -1) &
                                          ((flag != "n") &
                                           (np.char.find(flag.astype(str), 'm') == -1) &
                                           (np.char.find(flag.astype(str), 'u') == -1)),
                                          flag+[","]+["i"], flag))
        self.flag = flag
            
    def get_output(self,out=0):

        assert out in [0,1], "out must be either 0 or 1"

        self._get_humidity() # Get the Relative humidity
        self._flag(out=out)  # Get flags
        
        # calculate output parameters
        rho = (0.34838*self.P)/(self.tv10n)
        self.t10n = self.t10n-(273.16+self.tlapse*self.ref_ht)
        
        # solve for zo from cd10n
        zo = self.ref_ht/np.exp(kappa/np.sqrt(self.cd10n))
        
        # adjust neutral cdn at any output height
        self.cdn = np.power(kappa/np.log(self.hout/zo), 2)
        self.cd = cd_calc(self.cdn, self.h_out[0], self.h_out[0], self.psim)

        # solve for zot, zoq from ct10n, cq10n
        zot = self.ref_ht/(np.exp(kappa**2/(self.ct10n*np.log(self.ref_ht/zo))))
        zoq = self.ref_ht/(np.exp(kappa**2/(self.cq10n*np.log(self.ref_ht/zo))))
        
        # adjust neutral ctn, cqn at any output height
        self.ctn = np.power(kappa, 2)/(np.log(self.h_out[0]/zo)*np.log(self.h_out[1]/zot))
        self.cqn = np.power(kappa, 2)/(np.log(self.h_out[0]/zo)*np.log(self.h_out[2]/zoq))
        self.ct, self.cq = ctcq_calc(self.cdn, self.cd, self.ctn, self.cqn, self.h_out, self.h_out, self.psit, self.psiq)
        self.uref = (self.spd-self.usr/kappa*(np.log(self.h_in[0]/self.h_out[0])-self.psim + psim_calc(self.h_out[0]/self.monob, self.meth)))
        tref = (self.Ta-self.tsr/kappa*(np.log(self.h_in[1]/self.h_out[1])-self.psit + psit_calc(self.h_out[0]/self.monob, self.meth)))
        self.tref = tref-(CtoK+self.tlapse*self.h_out[1])
        self.qref = (self.qair-self.qsr/kappa*(np.log(self.h_in[2]/self.h_out[2]) - self.psit+psit_calc(self.h_out[2]/self.monob, self.meth)))

        if (self.wl == 0): self.dtwl = np.zeros(self.T.shape)*self.msk  # reset to zero if not used

        # Do not calculate lhf if a measure of humidity is not input
        if (self.hum[0] == 'no'):
            self.latent = np.ones(self.SST.shape)*np.nan
            self.qsr = np.copy(self.latent)
            self.q10n = np.copy(self.latent)
            self.qref = np.copy(self.latent)
            self.qair = np.copy(self.latent)
            self.rh =  np.copy(self.latent)

        # Set the final wind speed values
        self.wind_spd = np.sqrt(np.power(self.wind, 2)-np.power(self.spd, 2))

        # Get class specific flags
        try:
            self._class_flag()
        except AttributeError:
            pass

        # Combine all output variables into a pandas array
        res_vars = ("tau","sensible","latent","monob","cd","cdn","ct","ctn","cq","cqn","tsrv","tsr","qsr","usr","psim","psit",
                    "psiq","u10n","t10n","tv10n","q10n","zo","zot","zoq","uref","tref","qref","itera","dter","dqer","dtwl",
                    "qair","qsea","Rl","Rs","Rnl","wind_spd","Rb","rh","tkt","lv")

        res = np.zeros((len(res_vars), len(self.spd)))
        for i, value in enumerate(res_vars): res[i][:] = getattr(self, value)

        if (out == 0):
            res[:, self.ind] = np.nan
            # set missing values where data have non acceptable values
            if (self.hum[0] != 'no'): res = np.asarray([np.where(self.q10n < 0, np.nan, res[i][:]) for i in range(len(res_vars))]) # FIXME: why 41?
            res = np.asarray([np.where(self.u10n < 0, np.nan, res[i][:]) for i in range(len(res_vars))])
        else:
            warnings.warn("Warning: the output will contain values for points that have not converged and negative values (if any) for u10n/q10n")

        resAll = pd.DataFrame(data=res.T, index=range(self.nlen), columns=res_vars)
    
        resAll["flag"] = self.flag

        return resAll

    def add_variables(self, spd, T, SST, lat=None, hum=None, P=None, L=None):

        # Add the mandatory variables
        assert type(spd)==type(T)==type(SST)==np.ndarray, "input type of spd, T and SST should be numpy.ndarray"
        self.L="tsrv" if L is None else L
        self.arr_shp = spd.shape
        self.nlen = len(spd)
        self.spd = spd
        self.T = T
        self.hum = ['no', np.full(SST.shape,80)] if hum is None else hum
        self.SST = np.where(SST < 200, np.copy(SST)+CtoK, np.copy(SST))
        self.lat = np.full(self.arr_shp,45) if lat is None else lat
        self.g = gc(self.lat)
        self.P = np.full(n, 1013) if P is None else P

        # mask to preserve missing values when initialising variables
        self.msk=np.empty(SST.shape)
        self.msk = np.where(np.isnan(spd+T+SST), np.nan, 1)
        self.Rb = np.empty(SST.shape)*self.msk
        self.dtwl = np.full(T.shape,0.3)*self.msk

        # Set the wind array
        # gust[0] is asserted to be either 0 or 1
        if (self.gust[0] == 1):
            self.wind = np.sqrt(np.power(np.copy(self.spd), 2)+np.power(0.5, 2))
        else:
            self.wind = np.copy(spd)
        
    def add_gust(self,gust=None):

        if np.all(gust == None):
            try:
                gust = self.default_gust
            except AttributeError:
                gust = [1,1.2,800]
        elif ((np.size(gust) < 3) and (gust == 0)):
            gust = [0, 0, 0]
            
        assert np.size(gust) == 3, "gust input must be a 3x1 array"
        assert gust[0] in [0,1], "gust at position 0 must be 0 or 1"
        self.gust = gust

    def _class_flag(self):
        "A flag specific to this class - only used for certain classes where utmp_lo and utmp_hi are defined"
        self.flag = np.where(((self.utmp < self.utmp_lo) | (self.utmp > self.utmp_hi)) & (self.flag == "n"), "o",
                             np.where(((self.utmp < self.utmp_lo) | (self.utmp > self.utmp_hi)) &
                                      ((self.flag != "n") &
                                       (np.char.find(self.flag.astype(str), 'u') == -1) &
                                       (np.char.find(self.flag.astype(str), 'q') == -1)),
                                      self.flag+[","]+["o"], self.flag))

    def __init__(self):
        self.meth = "S88"

class S80(S88):

    def __init__(self):
        self.meth = "S80"
        self.utmp_lo = 6
        self.utmp_hi = 22
        
class YT96(S88):

    def __init__(self):
        self.meth = "YT96"
        self.utmp_lo = 0
        self.utmp_hi = 26

class LP82(S88):
    
    def __init__(self):
        self.meth = "LP82"
        self.utmp_lo = 3
        self.utmp_hi = 25


class NCAR(S88):

    def _minimum_params(self):
        self.cd = np.maximum(np.copy(self.cd), 1e-4)
        self.ct = np.maximum(np.copy(self.ct), 1e-4)
        self.cq = np.maximum(np.copy(self.cq), 1e-4)
        self.zo = np.minimum(np.copy(self.zo), 0.0025)

    def _zo_calc(self, ref_ht, cd10n):
        "Special z0 calculation for NCAR"
        zo = ref_ht/np.exp(kappa/np.sqrt(cd10n))
        zo = np.minimum(np.copy(zo), 0.0025)
        return zo
    
    def __init__(self):
        self.meth = "NCAR"
        self.utmp_lo = 0.5
        self.utmp_hi = 999


class UA(S88):
    
    def _adjust_gust(self):
        # gustiness adjustment
        if (self.gust[0] == 1):
            self.wind = np.where(self.dtv >= 0, np.where(self.spd > 0.1, self.spd, 0.1),
                                 np.sqrt(np.power(np.copy(self.spd), 2)+np.power(0.5, 2)))

    def __init__(self):
        self.meth = "UA"
        self.default_gust = [1,1,1000]
        self.utmp_lo = 18
        self.utmp_hi = 999


class C30(S88):
    def set_coolskin_warmlayer(self, wl=0, cskin=1, skin="C35", Rl=None, Rs=None):
        self._fix_coolskin_warmlayer(wl, cskin, skin, Rl, Rs)

    def __init__(self):
        self.meth = "C30"
        self.default_gust = [1,1.2,600]

class C35(C30):
    def __init__(self):
        self.meth = "C35"
        self.default_gust = [1,1.2,600]

class ecmwf(C30):
    def set_coolskin_warmlayer(self, wl=0, cskin=1, skin="ecmwf", Rl=None, Rs=None):
        self._fix_coolskin_warmlayer(wl, cskin, skin, Rl, Rs)

    def __init__(self):
        self.meth = "ecmwf"
        self.default_gust = [1,1,1000]

class Beljaars(C30):
    def set_coolskin_warmlayer(self, wl=0, cskin=1, skin="Beljaars", Rl=None, Rs=None):
        self._fix_coolskin_warmlayer(wl, cskin, skin, Rl, Rs)
        
    def __init__(self):
        self.meth = "Beljaars"
        self.default_gust = [1,1,1000]
        
def AirSeaFluxCode(spd, T, SST, lat=None, hum=None, P=None, hin=18, hout=10,
                   Rl=None, Rs=None, cskin=None, skin="C35", wl=0, gust=None,
                   meth="S88", qmeth="Buck2", tol=None, n=10, out=0, L=None):
    """
    Calculates turbulent surface fluxes using different parameterizations
    Calculates height adjusted values for spd, T, q

    Parameters
    ----------
        spd : float
            relative wind speed in m/s (is assumed as magnitude difference
            between wind and surface current vectors)
        T : float
            air temperature in K (will convert if < 200)
        SST : float
            sea surface temperature in K (will convert if < 200)
        lat : float
            latitude (deg), default 45deg
        hum : float
            humidity input switch 2x1 [x, values] default is relative humidity
            x='rh' : relative humidity in %
            x='q' : specific humidity (g/kg)
            x='Td' : dew point temperature (K)
        P : float
            air pressure (hPa), default 1013hPa
        hin : float
            sensor heights in m (array 3x1 or 3xn), default 18m
        hout : float
            output height, default is 10m
        Rl : float
            downward longwave radiation (W/m^2)
        Rs : float
            downward shortwave radiation (W/m^2)
        cskin : int
            0 switch cool skin adjustment off, else 1
            default is 1
        skin : str
            cool skin method option "C35", "ecmwf" or "Beljaars"
        wl : int
            warm layer correction default is 0, to switch on set to 1
        gust : int
            3x1 [x, beta, zi] x=1 to include the effect of gustiness, else 0
            beta gustiness parameter, beta=1 for UA, beta=1.2 for COARE
            zi PBL height (m) 600 for COARE, 1000 for UA and ecmwf, 800 default
            default for COARE [1, 1.2, 600]
            default for UA, ecmwf [1, 1, 1000]
            default else [1, 1.2, 800]
        meth : str
            "S80", "S88", "LP82", "YT96", "UA", "NCAR", "C30", "C35",
            "ecmwf", "Beljaars"
        qmeth : str
            is the saturation evaporation method to use amongst
            "HylandWexler","Hardy","Preining","Wexler","GoffGratch","WMO",
            "MagnusTetens","Buck","Buck2","WMO2018","Sonntag","Bolton",
            "IAPWS","MurphyKoop"]
            default is Buck2
        tol : float
           4x1 or 7x1 [option, lim1-3 or lim1-6]
           option : 'flux' to set tolerance limits for fluxes only lim1-3
           option : 'ref' to set tolerance limits for height adjustment lim-1-3
           option : 'all' to set tolerance limits for both fluxes and height
                    adjustment lim1-6
           default is tol=['all', 0.01, 0.01, 1e-05, 1e-3, 0.1, 0.1]
        n : int
            number of iterations (defautl = 10)
        out : int
            set 0 to set points that have not converged, negative values of
                  u10n and q10n to missing (default)
            set 1 to keep points
        L : str
           Monin-Obukhov length definition options
           "tsrv"  : default for "S80", "S88", "LP82", "YT96", "UA", "NCAR",
                     "C30", "C35"
           "Rb" : following ecmwf (IFS Documentation cy46r1), default for
                  "ecmwf", "Beljaars"
    Returns
    -------
        res : array that contains
                       1. momentum flux       (N/m^2)
                       2. sensible heat       (W/m^2)
                       3. latent heat         (W/m^2)
                       4. Monin-Obhukov length (m)
                       5. drag coefficient (cd)
                       6. neutral drag coefficient (cdn)
                       7. heat exchange coefficient (ct)
                       8. neutral heat exchange coefficient (ctn)
                       9. moisture exhange coefficient (cq)
                       10. neutral moisture exchange coefficient (cqn)
                       11. star virtual temperatcure (tsrv)
                       12. star temperature (tsr)
                       13. star specific humidity (qsr)
                       14. star wind speed (usr)
                       15. momentum stability function (psim)
                       16. heat stability function (psit)
                       17. moisture stability function (psiq)
                       18. 10m neutral wind speed (u10n)
                       19. 10m neutral temperature (t10n)
                       20. 10m neutral virtual temperature (tv10n)
                       21. 10m neutral specific humidity (q10n)
                       22. surface roughness length (zo)
                       23. heat roughness length (zot)
                       24. moisture roughness length (zoq)
                       25. wind speed at reference height (uref)
                       26. temperature at reference height (tref)
                       27. specific humidity at reference height (qref)
                       28. number of iterations until convergence
                       29. cool-skin temperature depression (dter)
                       30. cool-skin humidity depression (dqer)
                       31. warm layer correction (dtwl)
                       32. specific humidity of air (qair)
                       33. specific humidity at sea surface (qsea)
                       34. downward longwave radiation (Rl)
                       35. downward shortwave radiation (Rs)
                       36. downward net longwave radiation (Rnl)
                       37. gust wind speed (ug)
                       38. Bulk Richardson number (Rib)
                       39. relative humidity (rh)
                       40. thickness of the viscous layer (delta)
                       41. lv latent heat of vaporization (Jkg−1)
                       42. flag ("n": normal, "o": out of nominal range,
                                 "u": u10n<0, "q":q10n<0
                                 "m": missing,
                                 "l": Rib<-0.5 or Rib>0.2 or z/L>1000,
                                 "r" : rh>100%,
                                 "i": convergence fail at n)

    2021 / Author S. Biri
    """
    logging.basicConfig(filename='flux_calc.log', filemode="w",
                        format='%(asctime)s %(message)s', level=logging.INFO)
    logging.captureWarnings(True)

    iclass = globals()[meth]()
    iclass.add_gust(gust=gust)
    iclass.add_variables(spd, T, SST, lat=lat, hum=hum, P=P, L=L)
    iclass.get_heights(hin, hout)
    iclass.get_specHumidity(qmeth=qmeth)
    iclass.set_coolskin_warmlayer(wl=wl, cskin=cskin,skin=skin,Rl=Rl,Rs=Rs)
    iclass.iterate(tol=tol,n=n)
    resAll = iclass.get_output(out=out)

    return resAll
