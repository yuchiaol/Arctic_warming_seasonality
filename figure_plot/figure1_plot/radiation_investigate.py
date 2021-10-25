flag_run = 1
# ================================================================
# Yu-Chiao @ Fort Lee, NJ Oct 20, 2020
# examination on  surface air temperature in cesm1-cam5 simulations
# ================================================================

# ================================================================
# import functions
# ================================================================
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
from math import isnan, radians
#from mpl_toolkits.basemap import Basemap
from IPython import get_ipython
import sys, os
#import cmocean
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.io.img_tiles as cimgt
from cartopy.io.img_tiles import StamenTerrain
from scipy import stats
import matplotlib.path as mpath
#from sklearn.utils import resample
import seaborn as sns

sys.path.append('/home/yliang/lib/python_functions/data_process/')
import data_process_f

# ================================================================
# define functions 
# ================================================================
def perform_ttest_1d_here(exp1_var,exp2_var,sig_level):
    [xxx, pvalue] = stats.ttest_ind(exp1_var,exp2_var)
    ttest_map = np.nan
    pvalue_map = pvalue
    if pvalue < sig_level:
       ttest_map = 1.
       pvalue_map = pvalue

    return ttest_map, pvalue_map

if flag_run == 1:
# ================================================================
# read simulations
# ================================================================
# read sat
   varname = 'FSNS'
   year1 = 1850
   year2 = 1999
   year_N = int(year2 - year1 + 1)
   tt = np.linspace(1,year_N,year_N)

   factor0 = -1.

# read grid basics
   dirname = '/data1/yliang/co2_experiments/processed_data/'
   filename = 'FSNS_annual_mean_temp_output.nc'
   f = Dataset(dirname + filename, 'r')
   lon = f.variables['lon'][:].data
   lat = f.variables['lat'][:].data
   area = data_process_f.area_calculate_nonuniform(lon,lat)
   sat_control = f.variables['var_co2x1'][:,:,:].data*factor0
   sat_co2xp25 = f.variables['var_co2xp25'][:,:,:].data*factor0
   sat_co2xp5 = f.variables['var_co2xp5'][:,:,:].data*factor0
   sat_co2x1 = f.variables['var_co2x1'][:,:,:].data*factor0
   sat_co2x2 = f.variables['var_co2x2'][:,:,:].data*factor0
   sat_co2x3 = f.variables['var_co2x3'][:,:,:].data*factor0
   sat_co2x4 = f.variables['var_co2x4'][:,:,:].data*factor0
   sat_co2x5 = f.variables['var_co2x5'][:,:,:].data*factor0
   sat_co2x6 = f.variables['var_co2x6'][:,:,:].data*factor0
   sat_co2x7 = f.variables['var_co2x7'][:,:,:].data*factor0
   sat_co2x8 = f.variables['var_co2x8'][:,:,:].data*factor0
   f.close()

   dirname = '/data1/yliang/co2_experiments/processed_data/'
   filename = 'FLNS_annual_mean_temp_output.nc'
   f = Dataset(dirname + filename, 'r')
   lon = f.variables['lon'][:].data
   lat = f.variables['lat'][:].data
   area = data_process_f.area_calculate_nonuniform(lon,lat)
   fx_control = f.variables['var_co2x1'][:,:,:].data
   fx_co2xp25 = f.variables['var_co2xp25'][:,:,:].data
   fx_co2xp5 = f.variables['var_co2xp5'][:,:,:].data
   fx_co2x1 = f.variables['var_co2x1'][:,:,:].data
   fx_co2x2 = f.variables['var_co2x2'][:,:,:].data
   fx_co2x3 = f.variables['var_co2x3'][:,:,:].data
   fx_co2x4 = f.variables['var_co2x4'][:,:,:].data
   fx_co2x5 = f.variables['var_co2x5'][:,:,:].data
   fx_co2x6 = f.variables['var_co2x6'][:,:,:].data
   fx_co2x7 = f.variables['var_co2x7'][:,:,:].data
   fx_co2x8 = f.variables['var_co2x8'][:,:,:].data
   f.close()

   dirname = '/data1/yliang/co2_experiments/processed_data/'
   filename = 'FSNS_annual_mean_temp_output_e_case.nc'
   f = Dataset(dirname + filename, 'r')
   esat_co2x1 = f.variables['var_co2x1'][-30:,:,:].data*factor0
   esat_co2x2 = f.variables['var_co2x2'][-30:,:,:].data*factor0
   esat_co2x3 = f.variables['var_co2x3'][-30:,:,:].data*factor0
   esat_co2x4 = f.variables['var_co2x4'][-30:,:,:].data*factor0
   esat_co2x5 = f.variables['var_co2x5'][-30:,:,:].data*factor0
   esat_co2x6 = f.variables['var_co2x6'][-30:,:,:].data*factor0
   f.close()

   n_e = esat_co2x6.shape[0]

   dirname = '/data1/yliang/co2_experiments/processed_data/'
   filename = 'FLNS_annual_mean_temp_output_e_case.nc'
   f = Dataset(dirname + filename, 'r')
   lon = f.variables['lon'][:].data
   lat = f.variables['lat'][:].data
   area = data_process_f.area_calculate_nonuniform(lon,lat)
   efx_co2x1 = f.variables['var_co2x1'][:,:,:].data
   efx_co2x2 = f.variables['var_co2x2'][:,:,:].data
   efx_co2x3 = f.variables['var_co2x3'][:,:,:].data
   efx_co2x4 = f.variables['var_co2x4'][:,:,:].data
   efx_co2x5 = f.variables['var_co2x5'][:,:,:].data
   efx_co2x6 = f.variables['var_co2x6'][:,:,:].data
   f.close()

   ny = len(lat)
   nx = len(lon)

   mask_var = (sat_control[1,:,:]/sat_control[1,:,:]).copy()

# select arctic region
   [x1, x2, y1, y2] = data_process_f.find_lon_lat_index(0,360,60,89,lon,lat)

# ================================================================
# calculate time series
# ================================================================
# simulated sat
   ts_sat_control_arctic = np.zeros((year_N))
   ts_sat_co2xp25_arctic = np.zeros((year_N))
   ts_sat_co2xp5_arctic = np.zeros((year_N))
   ts_sat_co2x1_arctic = np.zeros((year_N))
   ts_sat_co2x2_arctic = np.zeros((year_N))
   ts_sat_co2x3_arctic = np.zeros((year_N))
   ts_sat_co2x4_arctic = np.zeros((year_N))
   ts_sat_co2x5_arctic = np.zeros((year_N))
   ts_sat_co2x6_arctic = np.zeros((year_N))
   ts_sat_co2x7_arctic = np.zeros((year_N))
   ts_sat_co2x8_arctic = np.zeros((year_N))

   ts_fx_control_arctic = np.zeros((year_N))
   ts_fx_co2xp25_arctic = np.zeros((year_N))
   ts_fx_co2xp5_arctic = np.zeros((year_N))
   ts_fx_co2x1_arctic = np.zeros((year_N))
   ts_fx_co2x2_arctic = np.zeros((year_N))
   ts_fx_co2x3_arctic = np.zeros((year_N))
   ts_fx_co2x4_arctic = np.zeros((year_N))
   ts_fx_co2x5_arctic = np.zeros((year_N))
   ts_fx_co2x6_arctic = np.zeros((year_N))
   ts_fx_co2x7_arctic = np.zeros((year_N))
   ts_fx_co2x8_arctic = np.zeros((year_N))

   ts_esat_co2x1_arctic = np.zeros((n_e))
   ts_esat_co2x2_arctic = np.zeros((n_e))
   ts_esat_co2x3_arctic = np.zeros((n_e))
   ts_esat_co2x4_arctic = np.zeros((n_e))
   ts_esat_co2x5_arctic = np.zeros((n_e))
   ts_esat_co2x6_arctic = np.zeros((n_e))

   ts_efx_co2x1_arctic = np.zeros((n_e))
   ts_efx_co2x2_arctic = np.zeros((n_e))
   ts_efx_co2x3_arctic = np.zeros((n_e))
   ts_efx_co2x4_arctic = np.zeros((n_e))
   ts_efx_co2x5_arctic = np.zeros((n_e))
   ts_efx_co2x6_arctic = np.zeros((n_e))

   factor0 = 1.

   for NT in range(year_N):
       print('year:' + str(NT))

       ts_sat_control_arctic[NT] = np.nansum(sat_control[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_sat_co2xp25_arctic[NT] = np.nansum(sat_co2xp25[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_sat_co2xp5_arctic[NT] = np.nansum(sat_co2xp5[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_sat_co2x1_arctic[NT] = np.nansum(sat_co2x1[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_sat_co2x2_arctic[NT] = np.nansum(sat_co2x2[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_sat_co2x3_arctic[NT] = np.nansum(sat_co2x3[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_sat_co2x4_arctic[NT] = np.nansum(sat_co2x4[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_sat_co2x5_arctic[NT] = np.nansum(sat_co2x5[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_sat_co2x6_arctic[NT] = np.nansum(sat_co2x6[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_sat_co2x7_arctic[NT] = np.nansum(sat_co2x7[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_sat_co2x8_arctic[NT] = np.nansum(sat_co2x8[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])

       ts_fx_control_arctic[NT] = np.nansum(fx_control[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_fx_co2xp25_arctic[NT] = np.nansum(fx_co2xp25[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_fx_co2xp5_arctic[NT] = np.nansum(fx_co2xp5[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_fx_co2x1_arctic[NT] = np.nansum(fx_co2x1[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_fx_co2x2_arctic[NT] = np.nansum(fx_co2x2[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_fx_co2x3_arctic[NT] = np.nansum(fx_co2x3[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_fx_co2x4_arctic[NT] = np.nansum(fx_co2x4[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_fx_co2x5_arctic[NT] = np.nansum(fx_co2x5[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_fx_co2x6_arctic[NT] = np.nansum(fx_co2x6[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_fx_co2x7_arctic[NT] = np.nansum(fx_co2x7[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_fx_co2x8_arctic[NT] = np.nansum(fx_co2x8[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])

   for NT in range(n_e):
       print('year:' + str(NT))

       ts_esat_co2x1_arctic[NT] = np.nansum(esat_co2x1[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_esat_co2x2_arctic[NT] = np.nansum(esat_co2x2[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_esat_co2x3_arctic[NT] = np.nansum(esat_co2x3[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_esat_co2x4_arctic[NT] = np.nansum(esat_co2x4[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_esat_co2x5_arctic[NT] = np.nansum(esat_co2x5[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_esat_co2x6_arctic[NT] = np.nansum(esat_co2x6[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])

       ts_efx_co2x1_arctic[NT] = np.nansum(efx_co2x1[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_efx_co2x2_arctic[NT] = np.nansum(efx_co2x2[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_efx_co2x3_arctic[NT] = np.nansum(efx_co2x3[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_efx_co2x4_arctic[NT] = np.nansum(efx_co2x4[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_efx_co2x5_arctic[NT] = np.nansum(efx_co2x5[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_efx_co2x6_arctic[NT] = np.nansum(efx_co2x6[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])

# ================================================================
# plot figures
# ================================================================
if True:

   plt.close('all')
   fig = plt.figure(1)
   fig.set_size_inches(10, 10, forward=True)
   ax1 = fig.add_axes([0.08, 0.55, 0.4, 0.35])   

   ttt = np.linspace(1,9,9)

   ts_interval = np.zeros((9,2))
   ts_test1 = (ts_sat_co2xp25_arctic-ts_sat_control_arctic)[-30:].mean()
   ts_test = (ts_sat_co2xp25_arctic-ts_sat_control_arctic)[-30:].copy()
   ts_interval[0,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))  
   ts_test2 = (ts_sat_co2xp5_arctic-ts_sat_control_arctic)[-30:].mean()
   ts_test = (ts_sat_co2xp5_arctic-ts_sat_control_arctic)[-30:].copy()
   ts_interval[1,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test3 = (ts_sat_co2x2_arctic-ts_sat_control_arctic)[-30:].mean()
   ts_test = (ts_sat_co2x2_arctic-ts_sat_control_arctic)[-30:].copy()
   ts_interval[2,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test4 = (ts_sat_co2x3_arctic-ts_sat_control_arctic)[-30:].mean()
   ts_test = (ts_sat_co2x3_arctic-ts_sat_control_arctic)[-30:].copy()
   ts_interval[3,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test5 = (ts_sat_co2x4_arctic-ts_sat_control_arctic)[-30:].mean()
   ts_test = (ts_sat_co2x4_arctic-ts_sat_control_arctic)[-30:].copy()
   ts_interval[4,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test6 = (ts_sat_co2x5_arctic-ts_sat_control_arctic)[-30:].mean()
   ts_test = (ts_sat_co2x5_arctic-ts_sat_control_arctic)[-30:].copy()
   ts_interval[5,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test7 = (ts_sat_co2x6_arctic-ts_sat_control_arctic)[-30:].mean()
   ts_test = (ts_sat_co2x6_arctic-ts_sat_control_arctic)[-30:].copy()
   ts_interval[6,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test8 = (ts_sat_co2x7_arctic-ts_sat_control_arctic)[-30:].mean()
   ts_test = (ts_sat_co2x7_arctic-ts_sat_control_arctic)[-30:].copy()
   ts_interval[7,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test9 = (ts_sat_co2x8_arctic-ts_sat_control_arctic)[-30:].mean()
   ts_test = (ts_sat_co2x8_arctic-ts_sat_control_arctic)[-30:].copy()
   ts_interval[8,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))

   for II in range(7):
       plt.plot([ttt[2+II]]*2,ts_interval[2+II,:],'k-')
       plt.plot([ttt[2+II]-0.18,ttt[2+II]+0.18],[ts_interval[2+II,0]]*2,'k-')
       plt.plot([ttt[2+II]-0.18,ttt[2+II]+0.18],[ts_interval[2+II,1]]*2,'k-')
   plt.plot(ttt[2:],[ts_test3,ts_test4,ts_test5,ts_test6,ts_test7,ts_test8,ts_test9], 'ko-',label='fully coupled model',markersize=9)

   ts_interval = np.zeros((5,2))
   ts_test3 = (ts_esat_co2x2_arctic-ts_esat_co2x1_arctic)[-30:].mean()
   ts_test = (ts_esat_co2x2_arctic-ts_esat_co2x1_arctic)[-30:].copy()
   ts_interval[0,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test4 = (ts_esat_co2x3_arctic-ts_esat_co2x1_arctic)[-30:].mean()
   ts_test = (ts_esat_co2x3_arctic-ts_esat_co2x1_arctic)[-30:].copy()
   ts_interval[1,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test5 = (ts_esat_co2x4_arctic-ts_esat_co2x1_arctic)[-30:].mean()
   ts_test = (ts_esat_co2x4_arctic-ts_esat_co2x1_arctic)[-30:].copy()
   ts_interval[2,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test6 = (ts_esat_co2x5_arctic-ts_esat_co2x1_arctic)[-30:].mean()
   ts_test = (ts_esat_co2x5_arctic-ts_esat_co2x1_arctic)[-30:].copy()
   ts_interval[3,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test7 = (ts_esat_co2x6_arctic-ts_esat_co2x1_arctic)[-30:].mean()
   ts_test = (ts_esat_co2x6_arctic-ts_esat_co2x1_arctic)[-30:].copy()
   ts_interval[4,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))

   for II in range(5):
       plt.plot([ttt[2+II]]*2,ts_interval[II,:],'k-')
       plt.plot([ttt[2+II]-0.18,ttt[2+II]+0.18],[ts_interval[II,0]]*2,'k-')
       plt.plot([ttt[2+II]-0.18,ttt[2+II]+0.18],[ts_interval[II,1]]*2,'k-')
   plt.plot(ttt[2:7],[ts_test3,ts_test4,ts_test5,ts_test6,ts_test7], 'k^--',label='slab ocean model', markersize=9)

   plt.xticks(ttt,['CO2x0.25','CO2x0.5','2xCO2','3xCO2','4xCO2','5xCO2','6xCO2','7xCO2','8xCO2'], rotation=0)

   plt.xlim(2.5,9.5)
#   plt.ylim(5,25)

   plt.legend()
   plt.ylabel('W/m$^2$')
   plt.title('(a) Arctic Surface Shortwave Flux Response')

   ax1 = fig.add_axes([0.56, 0.55, 0.4, 0.35])
   ts_interval = np.zeros((9,2))
   ts_test1 = (ts_fx_co2xp25_arctic-ts_fx_control_arctic)[-30:].mean()
   ts_test = (ts_fx_co2xp25_arctic-ts_fx_control_arctic)[-30:].copy()
   ts_interval[0,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test2 = (ts_fx_co2xp5_arctic-ts_fx_control_arctic)[-30:].mean()
   ts_test = (ts_fx_co2xp5_arctic-ts_fx_control_arctic)[-30:].copy()
   ts_interval[1,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test3 = (ts_fx_co2x2_arctic-ts_fx_control_arctic)[-30:].mean()
   ts_test = (ts_fx_co2x2_arctic-ts_fx_control_arctic)[-30:].copy()
   ts_interval[2,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test4 = (ts_fx_co2x3_arctic-ts_fx_control_arctic)[-30:].mean()
   ts_test = (ts_fx_co2x3_arctic-ts_fx_control_arctic)[-30:].copy()
   ts_interval[3,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test5 = (ts_fx_co2x4_arctic-ts_fx_control_arctic)[-30:].mean()
   ts_test = (ts_fx_co2x4_arctic-ts_fx_control_arctic)[-30:].copy()
   ts_interval[4,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test6 = (ts_fx_co2x5_arctic-ts_fx_control_arctic)[-30:].mean()
   ts_test = (ts_fx_co2x5_arctic-ts_fx_control_arctic)[-30:].copy()
   ts_interval[5,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test7 = (ts_fx_co2x6_arctic-ts_fx_control_arctic)[-30:].mean()
   ts_test = (ts_fx_co2x6_arctic-ts_fx_control_arctic)[-30:].copy()
   ts_interval[6,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test8 = (ts_fx_co2x7_arctic-ts_fx_control_arctic)[-30:].mean()
   ts_test = (ts_fx_co2x7_arctic-ts_fx_control_arctic)[-30:].copy()
   ts_interval[7,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test9 = (ts_fx_co2x8_arctic-ts_fx_control_arctic)[-30:].mean()
   ts_test = (ts_fx_co2x8_arctic-ts_fx_control_arctic)[-30:].copy()
   ts_interval[8,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))

   for II in range(7):
       plt.plot([ttt[2+II]]*2,ts_interval[2+II,:],'k-')
       plt.plot([ttt[2+II]-0.18,ttt[2+II]+0.18],[ts_interval[2+II,0]]*2,'k-')
       plt.plot([ttt[2+II]-0.18,ttt[2+II]+0.18],[ts_interval[2+II,1]]*2,'k-')
   plt.plot(ttt[2:],[ts_test3,ts_test4,ts_test5,ts_test6,ts_test7,ts_test8,ts_test9], 'ko-',label='fully coupled model',markersize=9)

   ts_interval = np.zeros((5,2))
   ts_test3 = (ts_efx_co2x2_arctic-ts_efx_co2x1_arctic)[-30:].mean()
   ts_test = (ts_efx_co2x2_arctic-ts_efx_co2x1_arctic)[-30:].copy()
   ts_interval[0,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test4 = (ts_efx_co2x3_arctic-ts_efx_co2x1_arctic)[-30:].mean()
   ts_test = (ts_efx_co2x3_arctic-ts_efx_co2x1_arctic)[-30:].copy()
   ts_interval[1,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test5 = (ts_efx_co2x4_arctic-ts_efx_co2x1_arctic)[-30:].mean()
   ts_test = (ts_efx_co2x4_arctic-ts_efx_co2x1_arctic)[-30:].copy()
   ts_interval[2,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test6 = (ts_efx_co2x5_arctic-ts_efx_co2x1_arctic)[-30:].mean()
   ts_test = (ts_efx_co2x5_arctic-ts_efx_co2x1_arctic)[-30:].copy()
   ts_interval[3,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test7 = (ts_efx_co2x6_arctic-ts_efx_co2x1_arctic)[-30:].mean()
   ts_test = (ts_efx_co2x6_arctic-ts_efx_co2x1_arctic)[-30:].copy()
   ts_interval[4,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))

   for II in range(5):
       plt.plot([ttt[2+II]]*2,ts_interval[II,:],'k-')
       plt.plot([ttt[2+II]-0.18,ttt[2+II]+0.18],[ts_interval[II,0]]*2,'k-')
       plt.plot([ttt[2+II]-0.18,ttt[2+II]+0.18],[ts_interval[II,1]]*2,'k-')
   plt.plot(ttt[2:7],[ts_test3,ts_test4,ts_test5,ts_test6,ts_test7], 'k^--',label='slab ocean model', markersize=9)

   plt.xticks(ttt,['CO2x0.25','CO2x0.5','2xCO2','3xCO2','4xCO2','5xCO2','6xCO2','7xCO2','8xCO2'], rotation=0)
   
   plt.legend()
   plt.xlim(2.5,9.5)
#   plt.ylim(0,25)
   plt.ylabel('W/m$^2$')
   plt.title('(b) Arctic Surface Longwave Flux Response')

   plt.savefig('trend_tmp_plot.jpg', format='jpeg', dpi=200)

   plt.show()

   sys.exit()


