===========
Users Guide
===========

Introduction
============

The flux calculation code was implemented in order to provide a useful, easy to use and straightforward "roadmap" of when and why to use different bulk formulae for the calculation of surface turbulent fluxes.

Differences in the calculations between different methods can be found in:

- the way they compute specific humidity from relative humidity, temperature and pressure
- the way they parameterise the exchange coefficients
- the inclusion of heat and moisture roughness lengths
- the inclusion of cool skin/warm layer correction instead of the bulk sea surface temperature
- the inclusion of gustiness in the wind speed, and
- the momentum, heat and moisture stability functions definitions


The available parameterizations in AirSeaFluxCode provided in order to calculate the momentum, sensible heat and latent heat fluxes are implemented following:

- [Smith1980]_ as S80: the surface drag coefficient is related to 10m wind speed (u\ :sub:`10`\), surface heat and moisture exchange coefficients are constant. The stability parameterizations are based on the Monin-Obukhov similarity theory for stable and unstable condition which modify the wind, temperature and humidity profiles and derives surface turbulent fluxes in open ocean conditions (valid for wind speeds from 6 to 22 ms\ :sup:`-1`).

- [Smith1988]_ as S88: is an improvement of the S80 parameterization in the sense that it provides the surface drag coefficient in relation to surface roughness over smooth and viscous surface and otherwise derives surface turbulent fluxes in open ocean conditions as described for S80.

- [LargePond1981]_, [LargePond1982]_ as LP82: the surface drag coefficient is computed in relation to u\ :sub:`10` and has different parameterization for different ranges of wind speed. The heat and moisture exchange coefficients are constant for wind speeds<11ms\ :sup:`-1` and a function of u\ :sub:`10` for wind speeds between 11 and 25ms\ :sup:`-1`. The stability parameterizations are based on the Monin-Obukhov similarity theory for stable and unstable condition.

- [YellandTaylor1996]_, [Yelland1998]_ as YT96: the surface drag coefficient is a function of u\ :sub:`*`\. The heat and moisture exchange coefficients are considered constant as in the cases of S80 and S88.

- [Zeng1998]_ as UA: the drag coefficient is given as a function of roughness length over smooth and viscous surface. The parameterization includes the effect of gustiness. The heat and moisture exchange coefficients are a function of heat and moisture roughness lengths and are valid in the range of 0.5 and 18 ms\ :sup:`-1`.

- [LargeYeager2004]_, [LargeYeager2009]_ as NCAR: the surface drag coefficient is computed in relation to wind speed for u\ :sub:`10` \>0.5 ms\ :sup:`-1`. The heat exchange coefficient is given as a function of the drag coefficient (one for stable and one for unstable conditions) and the moisture exchange coefficient is also a function of the drag coefficient.

- [Fairall1996]_, [Fairall2003]_, [Edson2013]_ as C30, and C35: is based on data collected from four expeditions in order to improve the drag and exchange coefficients parameterizations relative to surface roughness. It includes the effects of "cool skin", and gustiness. The effects of waves and sea state are neglected in order to keep the software as simple as possible, without compromising the integrity of the outputs though.

- [ECMWF2019]_ as ecmwf: the drag, heat and moisture coefficients parameterizations are computed relative to surface roughness estimates. It includes gustiness in the computation of wind speed.

- [Beljaars1995a]_, [Beljaars1995b]_, [ZengBeljaars2005]_ as Beljaars: the drag, heat and moisture coefficients parameterizations are computed relative to surface roughness estimates. It includes gustiness in the computation of wind speed.

.. role::  raw-html(raw)
    :format: html

Description of AirSeaFluxCode
=============================

In AirSeaFluxCode we use a consistent calculation approach across all algorithms; where this requires changes from published descriptions the effect of those changes are quantified and shown to be small compared to the significance levels we set in Table 1. The AirSeaFluxCode software calculates air-sea flux of momentum, sensible heat and latent heat fluxes from bulk meteorological variables (wind speed (spd), air temperature (T), and relative humidity (RH)) provided at a certain height (hin) above the surface and sea surface temperature (SST) and height adjusted values for wind speed, air temperature and specific humidity of air at a user specified reference height (default is 10 m). 

Additionally, non essential parameters can be given as inputs, such as: downward long/shortwave radiation (Rl, Rs), latitude (lat), reference output height (hout),  cool skin (cskin), cool skin correction method (skin, following either  [Fairall1996b]_ (default for C30, and C35), [ZengBeljaars2005]_ (default for Beljaars), [ECMWF2019]_ (default for ecmwf)), warm layer correction (wl), gustiness (gust) and boundary layer height (zi), choice of bulk algorithm method (meth), the choice of saturation vapour pressure function (qmeth), tolerance limits (tol), choice of Monin-Obukhov length function (L), and the maximum number of iterations (maxiter). Note that all input variables need to be loaded as numpy.ndarray.

The air and sea surface specific humidity are calculated using the functions qsat\_air(T, P, RH, qmeth) and qsat\_sea(SST, P, qmeth) , which call functions contained in VaporPressure.py to calculate saturation vapour pressure following a chosen method (default is [Buck2012]_).

- The air temperature is converted to air temperature for adiabatic expansion following: Ta = T + 273.16 + :math:`\Gamma \cdot`\hin

- The density of air is defined as :math:`\rho`\= (0.34838\ :math:`\cdot`\P)/T\ :sub:`v10n`

- The specific heat at constant pressure is defined  as c\ :sub:`p`\= 1004.67(1 + 0.00084\ :math:`\cdot`\q\ :sub:`sea`\)

- The latent heat of vapourization is defined as L\ :sub:`v`\ = (2.501-0.00237\ :math:`\cdot`\SST)\ :raw-html:`&#183;`\10\ :sup:`6` (SST in \ :raw-html:`&deg;`\C)

Initial values for the exchange coefficients and friction velocity are calculated assuming neutral stability. The program iterates to calculate the temperature and humidity fluxes and the virtual temperature as T\ :sub:`v`\=T\ :sub:`a`\(1+0.61q\ :sub:`air`\) , then the stability parameter z/L either as,

.. math::
   :label: zol

   \frac{z}{L}=\frac{z(g \cdot k \cdot T_{*v})}{T_{v10n} \cdot u_{*}^{2}}

or  as a function of the Richardson number as described by [ECMWF2019]_ [their equations 3.23--3.25]; hence a new value for u\ :sub:`10n`\, hence new transfer coefficients, hence new flux values until convergence is obtained (Table 1).  At every iteration step if there are points where the neutral 10 m wind speed (u\ :sub:`10n`\) becomes negative the wind speed value at these points is set to NaN.
The values for air density, specific heat at constant volume, and the latent heat of vaporisation are used in converting the scaled fluxes u\ :sub:`*`\, T\ :sub:`*`\, and q\ :sub:`*`\ (:eq:`strs`, for UA we retain their equations 7-14) to flux values in Nm\ :sup:`-2` and Wm\ :sup:`-2`, respectively.

.. math::
   :label: strs

   \begin{array}{l}
     u_{\ast} = \frac{k\cdot u_{z}}{\log(\frac{z}{z_{om}})-\Psi_{m}(\frac{z}{L})+\Psi_{m}(\frac{z_{om}}{L})} \\
     t_{\ast} = \frac{k\cdot (T-SST)}{\log(\frac{z}{z_{oh}})-\Psi_{h}(\frac{z}{L})+\Psi_{h}(\frac{z_{oh}}{L})} \\
     q_{\ast} = \frac{k\cdot (q_{air}-q_{sea})}{\log(\frac{z}{z_{oq}})-\Psi_{q}(\frac{z}{L})+\Psi_{q}(\frac{z_{oq}}{L})} 
   \end{array}  
    


AirSeaFluxCode is set up to test for convergence between the i\ :sup:`th` and (i-1)\ :sup:`th` iteration according to the tolerance limits shown in Table 1 for six variables in total, of which three are relative to the height adjustment (u\ :sub:`10`\, t\ :sub:`10`\, q\ :sub:`10`\) and three to the flux calculation (:math:`\tau`, shf, lhf) respectively. The tolerance limits are set according to the maximum accuracy that can be feasible for each variable. The user can choose to allow for convergence either only for the fluxes (default), or only for height adjustment or for both (all six variables). Values that have not converged are by default set to missing, but the number of iterations until convergence is provided as an output (this number is set to -1 for non convergent points).
A set of flags are provided as an output that signify: "m" where input values are missing; "o" where the wind speed for this point is outside the nominal range for the used parameterization; "u" or "q" for points that produce unphysical values for u\ :sub:`10n` \ or q\ :sub:`10n`\ respectively during the iteration loop; "r" where relative humidity is greater than 100%; "l" where the bulk Richardson number is below -0.5 or above 0.2 or z/L is greater than 1000; "i" where the value failed to converge after n number of iterations, if the points converged normally they are flagged with "n". The user should expect NaN values if out is set to zero (namely output only values that have converged) for values that have not converged after the set number of iterations (default is ten) or if they produced unphysical values for u\ :sub:`10n` \ or q\ :sub:`10n`\.

.. table:: Table 1: Tolerance and significance limits
   :widths: auto

   =============================  =============  =============
   Variable                       Tolerance      Significance
   =============================  =============  =============
   u\ :sub:`10n` [ms\ :sup:`-1`]  0.01           0.1
   T\ :sub:`10n` [K]              0.01           0.1 
   q\ :sub:`10n` [g/kg]           10\ :sup:`-2`  10\ :sup:`-1`
   :math:`\tau` [Nm\ :sup:`-2`]   10\ :sup:`-3`  10\ :sup:`-2`
   shf [Wm\ :sup:`-2`]            0.1            2 
   lhf [Wm\ :sup:`-2`]            0.1            2 
   =============================  =============  =============

AirSeaFluxCode module
=====================

.. automodule:: AirSeaFluxCode
   :members: AirSeaFluxCode

Description of Sub-Routines
===========================

This section provides a description of the constants and sub-routines that are called in AirSeaFluxCode.

Drag Coefficient Functions
--------------------------

.. automodule:: AirSeaFluxCode.flux_subs
   :members: cdn_calc, cdn_from_roughness, cd_calc

Heat and Moisture Exchange Coefficient Functions
------------------------------------------------

.. automodule:: AirSeaFluxCode.flux_subs
   :no-index:
   :members: ctqn_calc, ctq_calc

Stratification Functions
------------------------

The stratification functions :math:`\Psi_i` are integrals of the dimensionless profiles :math:`\Phi_i`, which are determined experimentally, and are applied as stablility corrections to the wind speed, temperature and humidity profiles.
They are a function of the stability parameter :math:`z/L` where :math:`L` is the Monin-Obukhov length.

.. automodule:: AirSeaFluxCode.flux_subs
   :no-index:
   :members: get_stabco, psim_calc, psit_calc, psi_Bel, psi_ecmwf, psit_26, psi_conv, psi_stab, psim_ecmwf, psiu_26, psim_conv, psim_stab

Other Flux Functions
--------------------

.. automodule:: AirSeaFluxCode.flux_subs
   :no-index:
   :members: apply_GF, get_gust, get_strs, get_tsrv, get_Rb, get_Ltsrv, get_LRb

Cool-skin/Warm-layer Functions
------------------------------

.. automodule:: AirSeaFluxCode.cs_wl_subs
   :members:

Humidity Functions
------------------

.. automodule:: AirSeaFluxCode.hum_subs
   :members:

Utility Functions
-----------------

.. automodule:: AirSeaFluxCode.util_subs
   :members:

.. [Beljaars1995a] Beljaars, A. C. M. (1995a). The impact of some aspects of the boundary layer scheme in the ecmwf model. Proc. Seminar on Parameterization of Sub-Grid Scale Physical Processes, Reading, United Kingdom, ECMWF.
.. [Beljaars1995b] Beljaars, A. C. M. (1995b). The parameterization of surface fluxes in large scale models under free convection. Quart. J. Roy. Meteor. Soc., 121:255–270. 
.. [Buck2012] Buck, A. L. (2012). Buck research instruments, LLC, chapter Appendix I, pages 20–21. unknown, Boulder, CO 80308.
.. [ECMWF2019] ECMWF, 2019. “Part IV: Physical processes,” in Turbulent transport and interactions with the surface. IFS documentation CY46R1 (Reading, RG2 9AX, England: ECMWF), 33–58. Available at: https://www.ecmwf.int/node/19308.
.. [Edson2013] Edson, J. B., Jampana, V., Weller, R. A., Bigorre, S. P., Plueddemann, A. J., Fairall, C. W., Miller, S. D., Mahrt, L., Vickers, D., and Hersbach, H. (2013). On the exchange of momentum over the open ocean. Journal of Physical Oceanography, 43.
.. [Fairall1996] Fairall, C. W., Bradley, E. F., Godfrey, J. S., Wick, G. A., Edson, J. B., and Young, G. S. (1996). Cool-skin and warm-layer effects on sea surface temperature. Journal of Geophysical Research, 101(C1):1295–1308.
.. [Fairall1996b] Fairall, C. W., Bradley, E. F., Rogers, D. P., Edson, J. B., and Young, G. S. (1996b). Bulk parameterization of air-sea fluxes for tropical ocean global atmosphere coupled-ocean atmosphere response experiment. Journal of Geophysical Research, 101(C2):3747–3764.
.. [Fairall2003] Fairall, C. W., Bradley, E. F., Hare, J. E., Grachev, A. A., and Edson, J. B. (2003). Bulk parameterization of air-sea fluxes: updates and verification for the coare algorithm. Journal of Climate, 16:571–591.
.. [Hersbach2020] Hersbach, H., Bell, B., Berrisford, P., Hirahara, S., Horányi, A., Muñoz-Sabater, J., et al. (2020). The ERA5 global reanalysis. Q. J. R. Meteorological Soc. 146, 1999–2049. doi: 10.1002/qj.3803
.. [LargePond1981] Large, W. G. and Pond, S. (1981). Open ocean momentum flux measurements in moderate to strong winds. Journal of Physical Oceanography, 11(324–336).
.. [LargePond1982] Large, W. G. and Pond, S. (1982). Sensible and latent heat flux measurements over the ocean. Journal of Physical Oceanography, 12:464–482.
.. [LargeYeager2004] Large, W. G. and Yeager, S. (2004). Diurnal to decadal global forcing for ocean and sea-ice models: The data sets and flux climatologies. University Corporation for Atmospheric Research.
.. [LargeYeager2009] Large, W. G. and Yeager, S. (2009). The global climatology of an interannually varying air–sea flux data set. Climate Dyn., 33:341–364.
.. [Schulzweida2022] Schulzweida, Uwe. (2022). CDO User Guide (2.1.0). Zenodo. https://doi.org/10.5281/zenodo.7112925
.. [Smith1980] Smith, S. D. (1980). Wind stress and heat flux over the ocean in gale force winds. Journal of Physical Oceanography, 10:709–726.
.. [Smith1988] Smith, S. D. (1988). Coefficients for sea surface wind stress, heat flux, and wind profiles as a function of wind speed and temperature. Journal of Geophysical Research, 93(C12):15467–15472.
.. [Smith2018] Smith, S. R., Briggs, K., Bourassa, M. A., Elya, J., and Paver, C. R. (2018). Shipboard automated meteorological and oceanographic system data archive: 2005–2017. Geoscience Data Journal, 5:73–86.
.. [Smith2019] Smith, S. R., Rolph, J. J., Briggs, K., and Bourassa, M. A. (2019). Quality Controlled Shipboard Automated Meteo- rological and Oceanographic System (SAMOS) data. Center for Ocean-Atmospheric Prediction Studies, pages The Florida State University, Tallahassee, FL, USA, http://samos.coaps.fsu.edu.
.. [YellandTaylor1996] Yelland, M. and Taylor, P. K. (1996). Wind stress measurements from the open ocean. Journal of Physical Oceanography, 26:541–558.
.. [Yelland1998] Yelland, M., Moat, B. I., Taylor, P. K., Pascal, R. W., Hutchings, J., and Cornell, V. C. (1998). Wind stress measurements from the open ocean corrected for airflow distortion by the ship. Journal of Physical Oceanography, 28:1511–1526.
.. [ZengBeljaars2005] Zeng, X. and Beljaars, A. (2005). A prognostic scheme of sea surface skin temperature for modeling and data assimilation. Geophys. Res. Lett., 32(L14605).
.. [Zeng1998] Zeng, X., Zhao, M., and Dickinson, R. (1998). Intercomparison of bulk aerodynamic algorithms for the computation of sea surface fluxes using toga coare and tao data. J. Climate, 11:2628–2644.

