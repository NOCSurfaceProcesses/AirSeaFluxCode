===============
Getting Started
===============

AirSeaFluxCode.py is a Python 3.6+ module designed to process data (input as numpy ndarray float number type) to calculate surface turbulent fluxes, flux product estimates and to provide height adjusted values for wind speed, air temperature and specific humidity of air at a user defined reference height from a minimum number of meteorological parameters (wind speed, air temperature, and sea surface temperature) and for a variety of different bulk algorithms (at the time of the release amount to ten). 

Several optional parameters can be input such as: an estimate of humidity (relative humidity, specific humidity or dew point temperature) is required in the case an output of latent heat flux is requested; atmospheric pressure. If cool skin/warm layer adjustments are switched on then shortwave/longwave radiations should be provided as input. Other options the user can define on input are the height on to which the output parameters would be adjusted, the function of the cool skin adjustment provided that the option for applying the adjustment is switched on, the option to consider the effect of convective gustiness. The user can: choose from a wide variety of saturation vapour pressure function in order to compute specific humidity from relative humidity or dew point temperature, provide user defined tolerance limits, user define the maximum number of iterations.

For recommendations or bug reports, please visit https://github.com/NOCSurfaceProcesses/AirSeaFluxCode

.. role::  raw-html(raw)
    :format: html

Description of test data
========================

A suite of data is provided for testing, containing values for air temperature, sea surface temperature, wind speed, air pressure, relative humidity, shortwave radiation, longitude and latitude.

The first test data set (data\_all.csv) is developed as daily averages from minute data provided by the Shipboard Automated Meteorological and Oceanographic System SAMOS ([Smith2019]_ , [Smith2018]_ ); it contains a synthesis of various conditions from meteorological and surface oceanographic data from research vessels and three that increase the accuracy of the flux estimate (atmospheric pressure, relative humidity, shortwave radiation). We use quality control level three (research level quality), and we only keep variables flagged as Z (good data) (for details on flag definitions see [Smith2018]_). The input sensors' heights vary by ship and sometimes by cruise. The data contain wind speeds ranging between 0.015 and 18.5ms\ :sup:`-1`, air temperatures ranging from -3 to 9.7\ :raw-html:`&deg;`\C and air-sea temperature differences (T-T\ :sub:`0`\, hereafter :math:`\Delta`\T) from around -3 to 3\ :raw-html:`&deg;`\C. A sample output file is given (data\_all\_out.csv and its statistics in data\_all\_stats.txt) run with default options (see data\_all\_stats.txt for the input summary); note that deviations from the output values might occur due to floating point errors. 

The second test data set contained in era5\_r360x180.nc contains ERA5 ([Hersbach2020]_, [ECMWF2019]_) hourly data for one sample day (15/07/2019) remapped to 1\ :raw-html:`&deg;`\x1\ :raw-html:`&deg;`\ regular grid resolution using cdo ([Schulzweida2022]_). In this case all essential and optional input SSVs are available. For the calculation of TSFs we only consider values over the ice-free ocean by applying the available land mask and sea-ice concentration (equal to zero) and setting values over land or ice to missing (flag="m"). The data contain wind speeds ranging from 0.01 to 24.9 ms\ :sup:`-1`, air temperatures ranging from -17.2 to 35.4\ :raw-html:`&deg;`\C and :math:`\Delta`\T from around -16.2 to 8\ :raw-html:`&deg;`\C.



Description of sample code
==========================

In the AirSeaFluxCode `repository`_ we provide two types of sample routines to aid the user running the code. The first is the routine toy\_ASFC.py which is an example of running AirSeaFluxCode either with one-dimensional data sets (like a subset of R/V data) loading the necessary parameters from the test data (data\_all.csv) or gridded 3D data sampled in era5\_r360x180.nc.

The routine first loads the data in the appropriate format (numpy.ndarray, type float), then calls AirSeaFluxCode loads the data as input, and finally saves the output as  text or as a NetCDF file and at the same time generates a table of statistics for all the output parameters and figures of the mean values of the turbulent surface fluxes.

Second a jupyter notebook (ASFC\_notebook.ipynb) is provided as a step by step guide on how to run AirSeaFluxCode, starting from the libraries the user would need to import. It also provides an example on how to run AirSeaFluxCode with the research vessel data as input and generate basic plots of momentum and (sensible and latent) heat fluxes. The user can launch the `Jupyter Notebook App`_ by clicking on *Jupyter Notebook* icon in Anaconda start menu, this will launch a new browser window in your browser of choice (more details can be found `here`_).

.. _repository: https://github.com/NOCSurfaceProcesses/AirSeaFluxCode
.. _Jupyter Notebook App: https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what\_is\_jupyter.html
.. _here: https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html

