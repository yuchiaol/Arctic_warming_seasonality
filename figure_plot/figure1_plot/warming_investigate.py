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
   varname = 'TREFHT'
   year1 = 1850
   year2 = 1999
   year_N = int(year2 - year1 + 1)
   tt = np.linspace(1,year_N,year_N)

   factor0 = 1.

# read grid basics
   dirname = '/data1/yliang/co2_experiments/processed_data/'
   filename = 'TREFHT_annual_mean_temp_output.nc'
   f = Dataset(dirname + filename, 'r')
   lon = f.variables['lon'][:].data
   lat = f.variables['lat'][:].data
   area = data_process_f.area_calculate_nonuniform(lon,lat)
   sat_control = f.variables['var_co2x1'][:,:,:].data
   sat_co2xp25 = f.variables['var_co2xp25'][:,:,:].data
   sat_co2xp5 = f.variables['var_co2xp5'][:,:,:].data
   sat_co2x1 = f.variables['var_co2x1'][:,:,:].data
   sat_co2x2 = f.variables['var_co2x2'][:,:,:].data
   sat_co2x3 = f.variables['var_co2x3'][:,:,:].data
   sat_co2x4 = f.variables['var_co2x4'][:,:,:].data
   sat_co2x5 = f.variables['var_co2x5'][:,:,:].data
   sat_co2x6 = f.variables['var_co2x6'][:,:,:].data
   sat_co2x7 = f.variables['var_co2x7'][:,:,:].data
   sat_co2x8 = f.variables['var_co2x8'][:,:,:].data
   f.close()

   dirname = '/data1/yliang/co2_experiments/processed_data/'
   filename = 'LHFLX_annual_mean_temp_output.nc'
   f = Dataset(dirname + filename, 'r')
   lon = f.variables['lon'][:].data
   lat = f.variables['lat'][:].data
   area = data_process_f.area_calculate_nonuniform(lon,lat)
   lfx_control = f.variables['var_co2x1'][:,:,:].data
   lfx_co2xp25 = f.variables['var_co2xp25'][:,:,:].data
   lfx_co2xp5 = f.variables['var_co2xp5'][:,:,:].data
   lfx_co2x1 = f.variables['var_co2x1'][:,:,:].data
   lfx_co2x2 = f.variables['var_co2x2'][:,:,:].data
   lfx_co2x3 = f.variables['var_co2x3'][:,:,:].data
   lfx_co2x4 = f.variables['var_co2x4'][:,:,:].data
   lfx_co2x5 = f.variables['var_co2x5'][:,:,:].data
   lfx_co2x6 = f.variables['var_co2x6'][:,:,:].data
   lfx_co2x7 = f.variables['var_co2x7'][:,:,:].data
   lfx_co2x8 = f.variables['var_co2x8'][:,:,:].data
   f.close()

   dirname = '/data1/yliang/co2_experiments/processed_data/'
   filename = 'SHFLX_annual_mean_temp_output.nc'
   f = Dataset(dirname + filename, 'r')
   lon = f.variables['lon'][:].data
   lat = f.variables['lat'][:].data
   area = data_process_f.area_calculate_nonuniform(lon,lat)
   sfx_control = f.variables['var_co2x1'][:,:,:].data
   sfx_co2xp25 = f.variables['var_co2xp25'][:,:,:].data
   sfx_co2xp5 = f.variables['var_co2xp5'][:,:,:].data
   sfx_co2x1 = f.variables['var_co2x1'][:,:,:].data
   sfx_co2x2 = f.variables['var_co2x2'][:,:,:].data
   sfx_co2x3 = f.variables['var_co2x3'][:,:,:].data
   sfx_co2x4 = f.variables['var_co2x4'][:,:,:].data
   sfx_co2x5 = f.variables['var_co2x5'][:,:,:].data
   sfx_co2x6 = f.variables['var_co2x6'][:,:,:].data
   sfx_co2x7 = f.variables['var_co2x7'][:,:,:].data
   sfx_co2x8 = f.variables['var_co2x8'][:,:,:].data
   f.close()

   fx_control = lfx_control + sfx_control
   fx_co2xp25 = lfx_co2xp25 + sfx_co2xp25
   fx_co2xp5 = lfx_co2xp5 + sfx_co2xp5
   fx_co2x1 = lfx_co2x1 + sfx_co2x1
   fx_co2x2 = lfx_co2x2 + sfx_co2x2
   fx_co2x3 = lfx_co2x3 + sfx_co2x3
   fx_co2x4 = lfx_co2x4 + sfx_co2x4
   fx_co2x5 = lfx_co2x5 + sfx_co2x5
   fx_co2x6 = lfx_co2x6 + sfx_co2x6
   fx_co2x7 = lfx_co2x7 + sfx_co2x7
   fx_co2x8 = lfx_co2x8 + sfx_co2x8

   dirname = '/data1/yliang/co2_experiments/processed_data/'
   filename = 'ICEFRAC_annual_mean_temp_output.nc'
   f = Dataset(dirname + filename, 'r')
   sic_control = f.variables['var_co2x1'][:,:,:].data*factor0
   sic_co2xp25 = f.variables['var_co2xp25'][:,:,:].data*factor0
   sic_co2xp5 = f.variables['var_co2xp5'][:,:,:].data*factor0
   sic_co2x1 = f.variables['var_co2x1'][:,:,:].data*factor0
   sic_co2x2 = f.variables['var_co2x2'][:,:,:].data*factor0
   sic_co2x3 = f.variables['var_co2x3'][:,:,:].data*factor0
   sic_co2x4 = f.variables['var_co2x4'][:,:,:].data*factor0
   sic_co2x5 = f.variables['var_co2x5'][:,:,:].data*factor0
   sic_co2x6 = f.variables['var_co2x6'][:,:,:].data*factor0
   sic_co2x7 = f.variables['var_co2x7'][:,:,:].data*factor0
   sic_co2x8 = f.variables['var_co2x8'][:,:,:].data*factor0
   f.close()

   sic_control[sic_control<0.15] = 0.
   sic_co2xp25[sic_co2xp25<0.15] = 0.
   sic_co2xp5[sic_co2xp5<0.15] = 0.
   sic_co2x1[sic_co2x1<0.15] = 0.
   sic_co2x2[sic_co2x2<0.15] = 0.
   sic_co2x3[sic_co2x3<0.15] = 0.
   sic_co2x4[sic_co2x4<0.15] = 0.
   sic_co2x5[sic_co2x5<0.15] = 0.
   sic_co2x6[sic_co2x6<0.15] = 0.
   sic_co2x7[sic_co2x7<0.15] = 0.
   sic_co2x8[sic_co2x8<0.15] = 0.

   dirname = '/data1/yliang/co2_experiments/processed_data/'
   filename = 'TREFHT_annual_mean_temp_output_f_case.nc'
   f = Dataset(dirname + filename, 'r')
   fsat_control = f.variables['var_co2x1'][-30:,:,:].data
   fsat_co2xp25 = f.variables['var_co2xp25'][-30:,:,:].data
   fsat_co2xp5 = f.variables['var_co2xp5'][-30:,:,:].data
   fsat_co2x1 = f.variables['var_co2x1'][-30:,:,:].data
   fsat_co2x2 = f.variables['var_co2x2'][-30:,:,:].data
   fsat_co2x3 = f.variables['var_co2x3'][-30:,:,:].data
   fsat_co2x4 = f.variables['var_co2x4'][-30:,:,:].data
   fsat_co2x5 = f.variables['var_co2x5'][-30:,:,:].data
   fsat_co2x6 = f.variables['var_co2x6'][-30:,:,:].data
   fsat_co2x7 = f.variables['var_co2x7'][-30:,:,:].data
   fsat_co2x8 = f.variables['var_co2x8'][-30:,:,:].data
   f.close()

   n_f = fsat_co2x8.shape[0]

   dirname = '/data1/yliang/co2_experiments/processed_data/'
   filename = 'TREFHT_annual_mean_temp_output_e_case.nc'
   f = Dataset(dirname + filename, 'r')
   esat_co2x1 = f.variables['var_co2x1'][-30:,:,:].data
   esat_co2x2 = f.variables['var_co2x2'][-30:,:,:].data
   esat_co2x3 = f.variables['var_co2x3'][-30:,:,:].data
   esat_co2x4 = f.variables['var_co2x4'][-30:,:,:].data
   esat_co2x5 = f.variables['var_co2x5'][-30:,:,:].data
   esat_co2x6 = f.variables['var_co2x6'][-30:,:,:].data
   f.close()

   n_e = esat_co2x6.shape[0]

   dirname = '/data1/yliang/co2_experiments/processed_data/'
   filename = 'ICEFRAC_annual_mean_temp_output_e_case.nc'
   f = Dataset(dirname + filename, 'r')
   esic_co2x1 = f.variables['var_co2x1'][-30:,:,:].data*factor0
   esic_co2x2 = f.variables['var_co2x2'][-30:,:,:].data*factor0
   esic_co2x3 = f.variables['var_co2x3'][-30:,:,:].data*factor0
   esic_co2x4 = f.variables['var_co2x4'][-30:,:,:].data*factor0
   esic_co2x5 = f.variables['var_co2x5'][-30:,:,:].data*factor0
   esic_co2x6 = f.variables['var_co2x6'][-30:,:,:].data*factor0
   f.close()

   esic_co2x1[esic_co2x1<0.15] = 0.
   esic_co2x2[esic_co2x2<0.15] = 0.
   esic_co2x3[esic_co2x3<0.15] = 0.
   esic_co2x4[esic_co2x4<0.15] = 0.
   esic_co2x5[esic_co2x5<0.15] = 0.
   esic_co2x6[esic_co2x6<0.15] = 0.

   dirname = '/data1/yliang/co2_experiments/processed_data/'
   filename = 'LHFLX_annual_mean_temp_output_e_case.nc'
   f = Dataset(dirname + filename, 'r')
   lon = f.variables['lon'][:].data
   lat = f.variables['lat'][:].data
   area = data_process_f.area_calculate_nonuniform(lon,lat)
   elfx_co2x1 = f.variables['var_co2x1'][:,:,:].data
   elfx_co2x2 = f.variables['var_co2x2'][:,:,:].data
   elfx_co2x3 = f.variables['var_co2x3'][:,:,:].data
   elfx_co2x4 = f.variables['var_co2x4'][:,:,:].data
   elfx_co2x5 = f.variables['var_co2x5'][:,:,:].data
   elfx_co2x6 = f.variables['var_co2x6'][:,:,:].data
   f.close()

   dirname = '/data1/yliang/co2_experiments/processed_data/'
   filename = 'SHFLX_annual_mean_temp_output_e_case.nc'
   f = Dataset(dirname + filename, 'r')
   lon = f.variables['lon'][:].data
   lat = f.variables['lat'][:].data
   area = data_process_f.area_calculate_nonuniform(lon,lat)
   esfx_co2x1 = f.variables['var_co2x1'][:,:,:].data
   esfx_co2x2 = f.variables['var_co2x2'][:,:,:].data
   esfx_co2x3 = f.variables['var_co2x3'][:,:,:].data
   esfx_co2x4 = f.variables['var_co2x4'][:,:,:].data
   esfx_co2x5 = f.variables['var_co2x5'][:,:,:].data
   esfx_co2x6 = f.variables['var_co2x6'][:,:,:].data
   f.close()

   efx_co2x1 = elfx_co2x1 + esfx_co2x1
   efx_co2x2 = elfx_co2x2 + esfx_co2x2
   efx_co2x3 = elfx_co2x3 + esfx_co2x3
   efx_co2x4 = elfx_co2x4 + esfx_co2x4
   efx_co2x5 = elfx_co2x5 + esfx_co2x5
   efx_co2x6 = elfx_co2x6 + esfx_co2x6

   ny = len(lat)
   nx = len(lon)

   mask_var = (sat_control[1,:,:]/sat_control[1,:,:]).copy()

# select arctic region
   [x1, x2, y1, y2] = data_process_f.find_lon_lat_index(0,360,60,89,lon,lat)

# ================================================================
# calculate time series
# ================================================================
# simulated sat
   ts_sat_control_global = np.zeros((year_N))
   ts_sat_co2xp25_global = np.zeros((year_N))
   ts_sat_co2xp5_global = np.zeros((year_N))
   ts_sat_co2x1_global = np.zeros((year_N))
   ts_sat_co2x2_global = np.zeros((year_N))
   ts_sat_co2x3_global = np.zeros((year_N))
   ts_sat_co2x4_global = np.zeros((year_N))
   ts_sat_co2x5_global = np.zeros((year_N))
   ts_sat_co2x6_global = np.zeros((year_N))
   ts_sat_co2x7_global = np.zeros((year_N))
   ts_sat_co2x8_global = np.zeros((year_N))

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

   ts_sic_control_arctic = np.zeros((year_N))
   ts_sic_co2xp25_arctic = np.zeros((year_N))
   ts_sic_co2xp5_arctic = np.zeros((year_N))
   ts_sic_co2x1_arctic = np.zeros((year_N))
   ts_sic_co2x2_arctic = np.zeros((year_N))
   ts_sic_co2x3_arctic = np.zeros((year_N))
   ts_sic_co2x4_arctic = np.zeros((year_N))
   ts_sic_co2x5_arctic = np.zeros((year_N))
   ts_sic_co2x6_arctic = np.zeros((year_N))
   ts_sic_co2x7_arctic = np.zeros((year_N))
   ts_sic_co2x8_arctic = np.zeros((year_N))

   ts_fsat_co2xp25_global = np.zeros((n_f))
   ts_fsat_co2xp5_global = np.zeros((n_f))
   ts_fsat_co2x1_global = np.zeros((n_f))
   ts_fsat_co2x2_global = np.zeros((n_f))
   ts_fsat_co2x3_global = np.zeros((n_f))
   ts_fsat_co2x4_global = np.zeros((n_f))
   ts_fsat_co2x5_global = np.zeros((n_f))
   ts_fsat_co2x6_global = np.zeros((n_f))
   ts_fsat_co2x7_global = np.zeros((n_f))
   ts_fsat_co2x8_global = np.zeros((n_f))

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

   ts_fsat_co2xp25_arctic = np.zeros((n_f))
   ts_fsat_co2xp5_arctic = np.zeros((n_f))
   ts_fsat_co2x1_arctic = np.zeros((n_f))
   ts_fsat_co2x2_arctic = np.zeros((n_f))
   ts_fsat_co2x3_arctic = np.zeros((n_f))
   ts_fsat_co2x4_arctic = np.zeros((n_f))
   ts_fsat_co2x5_arctic = np.zeros((n_f))
   ts_fsat_co2x6_arctic = np.zeros((n_f))
   ts_fsat_co2x7_arctic = np.zeros((n_f))
   ts_fsat_co2x8_arctic = np.zeros((n_f))

   ts_esat_co2x1_global = np.zeros((n_e))
   ts_esat_co2x2_global = np.zeros((n_e))
   ts_esat_co2x3_global = np.zeros((n_e))
   ts_esat_co2x4_global = np.zeros((n_e))
   ts_esat_co2x5_global = np.zeros((n_e))
   ts_esat_co2x6_global = np.zeros((n_e))

   ts_esat_co2x1_arctic = np.zeros((n_e))
   ts_esat_co2x2_arctic = np.zeros((n_e))
   ts_esat_co2x3_arctic = np.zeros((n_e))
   ts_esat_co2x4_arctic = np.zeros((n_e))
   ts_esat_co2x5_arctic = np.zeros((n_e))
   ts_esat_co2x6_arctic = np.zeros((n_e))

   ts_esic_co2x1_arctic = np.zeros((n_e))
   ts_esic_co2x2_arctic = np.zeros((n_e))
   ts_esic_co2x3_arctic = np.zeros((n_e))
   ts_esic_co2x4_arctic = np.zeros((n_e))
   ts_esic_co2x5_arctic = np.zeros((n_e))
   ts_esic_co2x6_arctic = np.zeros((n_e))

   ts_efx_co2x1_arctic = np.zeros((n_e))
   ts_efx_co2x2_arctic = np.zeros((n_e))
   ts_efx_co2x3_arctic = np.zeros((n_e))
   ts_efx_co2x4_arctic = np.zeros((n_e))
   ts_efx_co2x5_arctic = np.zeros((n_e))
   ts_efx_co2x6_arctic = np.zeros((n_e))

   factor0 = 1./1000000000000.

   for NT in range(year_N):
       print('year:' + str(NT))
       ts_sat_control_global[NT] = np.nansum(sat_control[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)
       ts_sat_co2xp25_global[NT] = np.nansum(sat_co2xp25[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)
       ts_sat_co2xp5_global[NT] = np.nansum(sat_co2xp5[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)
       ts_sat_co2x1_global[NT] = np.nansum(sat_co2x1[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)
       ts_sat_co2x2_global[NT] = np.nansum(sat_co2x2[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)
       ts_sat_co2x3_global[NT] = np.nansum(sat_co2x3[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)
       ts_sat_co2x4_global[NT] = np.nansum(sat_co2x4[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)
       ts_sat_co2x5_global[NT] = np.nansum(sat_co2x5[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)
       ts_sat_co2x6_global[NT] = np.nansum(sat_co2x6[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)
       ts_sat_co2x7_global[NT] = np.nansum(sat_co2x7[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)
       ts_sat_co2x8_global[NT] = np.nansum(sat_co2x8[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)

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

       ts_sic_control_arctic[NT] = np.nansum(sic_control[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])*factor0
       ts_sic_co2xp25_arctic[NT] = np.nansum(sic_co2xp25[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])*factor0
       ts_sic_co2xp5_arctic[NT] = np.nansum(sic_co2xp5[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])*factor0
       ts_sic_co2x1_arctic[NT] = np.nansum(sic_co2x1[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])*factor0
       ts_sic_co2x2_arctic[NT] = np.nansum(sic_co2x2[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])*factor0
       ts_sic_co2x3_arctic[NT] = np.nansum(sic_co2x3[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])*factor0
       ts_sic_co2x4_arctic[NT] = np.nansum(sic_co2x4[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])*factor0
       ts_sic_co2x5_arctic[NT] = np.nansum(sic_co2x5[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])*factor0
       ts_sic_co2x6_arctic[NT] = np.nansum(sic_co2x6[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])*factor0
       ts_sic_co2x7_arctic[NT] = np.nansum(sic_co2x7[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])*factor0
       ts_sic_co2x8_arctic[NT] = np.nansum(sic_co2x8[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])*factor0

   for NT in range(n_f):
       print('year:' + str(NT))

       ts_fsat_co2xp25_global[NT] = np.nansum(fsat_co2xp25[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)
       ts_fsat_co2xp5_global[NT] = np.nansum(fsat_co2xp5[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)
       ts_fsat_co2x1_global[NT] = np.nansum(fsat_co2x1[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)
       ts_fsat_co2x2_global[NT] = np.nansum(fsat_co2x2[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)
       ts_fsat_co2x3_global[NT] = np.nansum(fsat_co2x3[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)
       ts_fsat_co2x4_global[NT] = np.nansum(fsat_co2x4[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)
       ts_fsat_co2x5_global[NT] = np.nansum(fsat_co2x5[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)
       ts_fsat_co2x6_global[NT] = np.nansum(fsat_co2x6[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)
       ts_fsat_co2x7_global[NT] = np.nansum(fsat_co2x7[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)
       ts_fsat_co2x8_global[NT] = np.nansum(fsat_co2x8[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)

       ts_fsat_co2xp25_arctic[NT] = np.nansum(fsat_co2xp25[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_fsat_co2xp5_arctic[NT] = np.nansum(fsat_co2xp5[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_fsat_co2x1_arctic[NT] = np.nansum(fsat_co2x1[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_fsat_co2x2_arctic[NT] = np.nansum(fsat_co2x2[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_fsat_co2x3_arctic[NT] = np.nansum(fsat_co2x3[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_fsat_co2x4_arctic[NT] = np.nansum(fsat_co2x4[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_fsat_co2x5_arctic[NT] = np.nansum(fsat_co2x5[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_fsat_co2x6_arctic[NT] = np.nansum(fsat_co2x6[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_fsat_co2x7_arctic[NT] = np.nansum(fsat_co2x7[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_fsat_co2x8_arctic[NT] = np.nansum(fsat_co2x8[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])

   for NT in range(n_e):
       print('year:' + str(NT))

       ts_esat_co2x1_global[NT] = np.nansum(esat_co2x1[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)
       ts_esat_co2x2_global[NT] = np.nansum(esat_co2x2[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)
       ts_esat_co2x3_global[NT] = np.nansum(esat_co2x3[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)
       ts_esat_co2x4_global[NT] = np.nansum(esat_co2x4[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)
       ts_esat_co2x5_global[NT] = np.nansum(esat_co2x5[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)
       ts_esat_co2x6_global[NT] = np.nansum(esat_co2x6[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)

       ts_esat_co2x1_arctic[NT] = np.nansum(esat_co2x1[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_esat_co2x2_arctic[NT] = np.nansum(esat_co2x2[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_esat_co2x3_arctic[NT] = np.nansum(esat_co2x3[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_esat_co2x4_arctic[NT] = np.nansum(esat_co2x4[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_esat_co2x5_arctic[NT] = np.nansum(esat_co2x5[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_esat_co2x6_arctic[NT] = np.nansum(esat_co2x6[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])

       ts_esic_co2x1_arctic[NT] = np.nansum(esic_co2x1[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])*factor0
       ts_esic_co2x2_arctic[NT] = np.nansum(esic_co2x2[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])*factor0
       ts_esic_co2x3_arctic[NT] = np.nansum(esic_co2x3[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])*factor0
       ts_esic_co2x4_arctic[NT] = np.nansum(esic_co2x4[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])*factor0
       ts_esic_co2x5_arctic[NT] = np.nansum(esic_co2x5[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])*factor0
       ts_esic_co2x6_arctic[NT] = np.nansum(esic_co2x6[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])*factor0

       ts_efx_co2x1_arctic[NT] = np.nansum(efx_co2x1[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_efx_co2x2_arctic[NT] = np.nansum(efx_co2x2[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_efx_co2x3_arctic[NT] = np.nansum(efx_co2x3[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_efx_co2x4_arctic[NT] = np.nansum(efx_co2x4[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_efx_co2x5_arctic[NT] = np.nansum(efx_co2x5[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
       ts_efx_co2x6_arctic[NT] = np.nansum(efx_co2x6[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])

# read cmip6 data
   dirname = '/home/yliang/research/ecs_aa/cmip6/'
   filename = 'cmip6_ts_temp_output_2xco2_4xco2_pi.nc'
   f = Dataset(dirname + filename, 'r')
   ts_2xco2_arctic_cmip6 = np.nanmean(f.variables['ts_2xco2_arctic'][:,-30:].data, axis=1)
   ts_4xco2_arctic_cmip6 = np.nanmean(f.variables['ts_4xco2_arctic'][:,-30:].data, axis=1)
   ts_pi_arctic_cmip6 = np.nanmean(f.variables['ts_pi_arctic'][:,-30:].data, axis=1)
   f.close()

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

   color_code = ['b','c','g','y','m','r'] 
   for II in range(len(ts_2xco2_arctic_cmip6)):
       plt.plot([ttt[2+2]]*2, [ts_4xco2_arctic_cmip6[II]-ts_pi_arctic_cmip6[II]]*2, 'o',color=color_code[II],markersize=4)
       plt.plot([ttt[2]]*2, [ts_2xco2_arctic_cmip6[II]-ts_pi_arctic_cmip6[II]]*2, 'o',color=color_code[II], markersize=4)

   plt.xticks(ttt,['CO2x0.25','CO2x0.5','2xCO2','3xCO2','4xCO2','5xCO2','6xCO2','7xCO2','8xCO2'], rotation=0)

   plt.xlim(2.5,9.5)
   plt.ylim(0,25)

   plt.legend()
   plt.ylabel('K')
   plt.title('(a) Arctic SAT Response')

# SIC
   ax1 = fig.add_axes([0.56, 0.55, 0.4, 0.35])
   ts_interval = np.zeros((9,2))
   ts_test1 = (ts_sic_co2xp25_arctic-ts_sic_control_arctic)[-30:].mean()
   ts_test = (ts_sic_co2xp25_arctic-ts_sic_control_arctic)[-30:].copy()
   ts_interval[0,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test2 = (ts_sic_co2xp5_arctic-ts_sic_control_arctic)[-30:].mean()
   ts_test = (ts_sic_co2xp5_arctic-ts_sic_control_arctic)[-30:].copy()
   ts_interval[1,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test3 = (ts_sic_co2x2_arctic-ts_sic_control_arctic)[-30:].mean()
   ts_test = (ts_sic_co2x2_arctic-ts_sic_control_arctic)[-30:].copy()
   ts_interval[2,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test4 = (ts_sic_co2x3_arctic-ts_sic_control_arctic)[-30:].mean()
   ts_test = (ts_sic_co2x3_arctic-ts_sic_control_arctic)[-30:].copy()
   ts_interval[3,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test5 = (ts_sic_co2x4_arctic-ts_sic_control_arctic)[-30:].mean()
   ts_test = (ts_sic_co2x4_arctic-ts_sic_control_arctic)[-30:].copy()
   ts_interval[4,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test6 = (ts_sic_co2x5_arctic-ts_sic_control_arctic)[-30:].mean()
   ts_test = (ts_sic_co2x5_arctic-ts_sic_control_arctic)[-30:].copy()
   ts_interval[5,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test7 = (ts_sic_co2x6_arctic-ts_sic_control_arctic)[-30:].mean()
   ts_test = (ts_sic_co2x6_arctic-ts_sic_control_arctic)[-30:].copy()
   ts_interval[6,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test8 = (ts_sic_co2x7_arctic-ts_sic_control_arctic)[-30:].mean()
   ts_test = (ts_sic_co2x7_arctic-ts_sic_control_arctic)[-30:].copy()
   ts_interval[7,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test9 = (ts_sic_co2x8_arctic-ts_sic_control_arctic)[-30:].mean()
   ts_test = (ts_sic_co2x8_arctic-ts_sic_control_arctic)[-30:].copy()
   ts_interval[8,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))

   for II in range(7):
       plt.plot([ttt[2+II]]*2,ts_interval[2+II,:],'k-')
       plt.plot([ttt[2+II]-0.18,ttt[2+II]+0.18],[ts_interval[2+II,0]]*2,'k-')
       plt.plot([ttt[2+II]-0.18,ttt[2+II]+0.18],[ts_interval[2+II,1]]*2,'k-')
   plt.plot(ttt[2:],[ts_test3,ts_test4,ts_test5,ts_test6,ts_test7,ts_test8,ts_test9], 'ko-',label='fully coupled model',markersize=9)

   ts_interval = np.zeros((5,2))
   ts_test3 = (ts_esic_co2x2_arctic-ts_esic_co2x1_arctic)[-30:].mean()
   ts_test = (ts_esic_co2x2_arctic-ts_esic_co2x1_arctic)[-30:].copy()
   ts_interval[0,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test4 = (ts_esic_co2x3_arctic-ts_esic_co2x1_arctic)[-30:].mean()
   ts_test = (ts_esic_co2x3_arctic-ts_esic_co2x1_arctic)[-30:].copy()
   ts_interval[1,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test5 = (ts_esic_co2x4_arctic-ts_esic_co2x1_arctic)[-30:].mean()
   ts_test = (ts_esic_co2x4_arctic-ts_esic_co2x1_arctic)[-30:].copy()
   ts_interval[2,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test6 = (ts_esic_co2x5_arctic-ts_esic_co2x1_arctic)[-30:].mean()
   ts_test = (ts_esic_co2x5_arctic-ts_esic_co2x1_arctic)[-30:].copy()
   ts_interval[3,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test7 = (ts_esic_co2x6_arctic-ts_esic_co2x1_arctic)[-30:].mean()
   ts_test = (ts_esic_co2x6_arctic-ts_esic_co2x1_arctic)[-30:].copy()
   ts_interval[4,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))

   for II in range(5):
       plt.plot([ttt[2+II]]*2,ts_interval[II,:],'k-')
       plt.plot([ttt[2+II]-0.18,ttt[2+II]+0.18],[ts_interval[II,0]]*2,'k-')
       plt.plot([ttt[2+II]-0.18,ttt[2+II]+0.18],[ts_interval[II,1]]*2,'k-')
   plt.plot(ttt[2:7],[ts_test3,ts_test4,ts_test5,ts_test6,ts_test7], 'k^--',label='slab ocean model', markersize=9)

   plt.xticks(ttt,['CO2x0.25','CO2x0.5','2xCO2','3xCO2','4xCO2','5xCO2','6xCO2','7xCO2','8xCO2'], rotation=0)

   plt.xlim(2.5,9.5)
   plt.ylim(-1.15e1,-3)

#   ax1.invert_yaxis()

   plt.legend()
   plt.ylabel('km$^2$ x 10$^6$')
   plt.title('(b) Arctic SIE Response')

   ax1 = fig.add_axes([0.08, 0.10, 0.4, 0.35])
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
   plt.ylim(0,25)
   plt.ylabel('W/m$^2$')
   plt.title('(c) Arctic Turbulent Heat Flux Response')

# plot thickness
# read sat
   varname = 'Z3'
   year1 = 1850
   year2 = 1879
   year_N = int(year2 - year1 + 1)
   tt = np.linspace(year1,year2,year_N)
   time_sel = -30
   lev_sel = [0, 9]

# read grid basics
   dirname = '/data1/yliang/co2_experiments/processed_data/'
   filename = varname + '_annual_polar_cap_mean_temp_output.nc'
   f = Dataset(dirname + filename, 'r')
   lev = f.variables['lev'][lev_sel].data
   var_co2xp25 = f.variables['var_co2xp25'][time_sel:,lev_sel].data
   var_co2xp5 = f.variables['var_co2xp5'][time_sel:,lev_sel].data
   var_co2x1 = f.variables['var_co2x1'][time_sel:,lev_sel].data
   var_co2x2 = f.variables['var_co2x2'][time_sel:,lev_sel].data
   var_co2x3 = f.variables['var_co2x3'][time_sel:,lev_sel].data
   var_co2x4 = f.variables['var_co2x4'][time_sel:,lev_sel].data
   var_co2x5 = f.variables['var_co2x5'][time_sel:,lev_sel].data
   var_co2x6 = f.variables['var_co2x6'][time_sel:,lev_sel].data
   var_co2x7 = f.variables['var_co2x7'][time_sel:,lev_sel].data
   var_co2x8 = f.variables['var_co2x8'][time_sel:,lev_sel].data
   f.close()

   filename = varname + '_annual_polar_cap_mean_temp_output_f_case.nc'
   f = Dataset(dirname + filename, 'r')
   lev = f.variables['lev'][lev_sel].data
   fvar_co2xp25 = f.variables['var_co2xp25'][time_sel:,lev_sel].data
   fvar_co2xp5 = f.variables['var_co2xp5'][time_sel:,lev_sel].data
   fvar_co2x1 = f.variables['var_co2x1'][time_sel:,lev_sel].data
   fvar_co2x2 = f.variables['var_co2x2'][time_sel:,lev_sel].data
   fvar_co2x3 = f.variables['var_co2x3'][time_sel:,lev_sel].data
   fvar_co2x4 = f.variables['var_co2x4'][time_sel:,lev_sel].data
   fvar_co2x5 = f.variables['var_co2x5'][time_sel:,lev_sel].data
   fvar_co2x6 = f.variables['var_co2x6'][time_sel:,lev_sel].data
   fvar_co2x7 = f.variables['var_co2x7'][time_sel:,lev_sel].data
   fvar_co2x8 = f.variables['var_co2x8'][time_sel:,lev_sel].data
   f.close()

   filename = varname + '_annual_polar_cap_mean_temp_output_e_case.nc'
   f = Dataset(dirname + filename, 'r')
   lev = f.variables['lev'][lev_sel].data
   evar_co2x1 = f.variables['var_co2x1'][time_sel:,lev_sel].data
   evar_co2x2 = f.variables['var_co2x2'][time_sel:,lev_sel].data
   evar_co2x3 = f.variables['var_co2x3'][time_sel:,lev_sel].data
   evar_co2x4 = f.variables['var_co2x4'][time_sel:,lev_sel].data
   evar_co2x5 = f.variables['var_co2x5'][time_sel:,lev_sel].data
   evar_co2x6 = f.variables['var_co2x6'][time_sel:,lev_sel].data
   f.close()

   nz = len(lev)

   ax1 = fig.add_axes([0.56, 0.10, 0.4, 0.35])
   ts_interval = np.zeros((9,2))
   var_std = (var_co2x1[:,-1] - var_co2x1[:,0]).copy()
   var_test = (var_co2xp25[:,-1] - var_co2xp25[:,0]).copy()
   ts_test = (var_test-var_std).copy()
   ts_interval[0,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test1 = ts_test.copy()
   var_test = (var_co2xp5[:,-1] - var_co2xp5[:,0]).copy()
   ts_test = (var_test-var_std).copy()
   ts_interval[1,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test2 = ts_test.copy()
   var_test = (var_co2x2[:,-1] - var_co2x2[:,0]).copy()
   ts_test = (var_test-var_std).copy()
   ts_interval[2,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test3 = ts_test.copy()
   var_test = (var_co2x3[:,-1] - var_co2x3[:,0]).copy()
   ts_test = (var_test-var_std).copy()
   ts_interval[3,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test4 = ts_test.copy()
   var_test = (var_co2x4[:,-1] - var_co2x4[:,0]).copy()
   ts_test = (var_test-var_std).copy()
   ts_interval[4,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test5 = ts_test.copy()
   var_test = (var_co2x5[:,-1] - var_co2x5[:,0]).copy()
   ts_test = (var_test-var_std).copy()
   ts_interval[5,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test6 = ts_test.copy()
   var_test = (var_co2x6[:,-1] - var_co2x6[:,0]).copy()
   ts_test = (var_test-var_std).copy()
   ts_interval[6,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test7 = ts_test.copy()
   var_test = (var_co2x7[:,-1] - var_co2x7[:,0]).copy()
   ts_test = (var_test-var_std).copy()
   ts_interval[7,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test8 = ts_test.copy()
   var_test = (var_co2x8[:,-1] - var_co2x8[:,0]).copy()
   ts_test = (var_test-var_std).copy()
   ts_interval[8,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test9 = ts_test.copy()

   for II in range(7):
       plt.plot([ttt[2+II]]*2,ts_interval[2+II,:],'k-')
       plt.plot([ttt[2+II]-0.18,ttt[2+II]+0.18],[ts_interval[2+II,0]]*2,'k-')
       plt.plot([ttt[2+II]-0.18,ttt[2+II]+0.18],[ts_interval[2+II,1]]*2,'k-')
   plt.plot(ttt[2:],[ts_test3.mean(),ts_test4.mean(),ts_test5.mean(),ts_test6.mean(),ts_test7.mean(),ts_test8.mean(),ts_test9.mean()], 'ko-',label='fully coupled model',markersize=9)

   ts_interval = np.zeros((5,2))
   var_test = (evar_co2x2[:,-1] - evar_co2x2[:,0]).copy()
   ts_test = (var_test-var_std).copy()
   ts_interval[0,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test1 = ts_test.copy()
   var_test = (evar_co2x3[:,-1] - evar_co2x3[:,0]).copy()
   ts_test = (var_test-var_std).copy()
   ts_interval[1,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test2 = ts_test.copy()
   var_test = (evar_co2x4[:,-1] - evar_co2x4[:,0]).copy()
   ts_test = (var_test-var_std).copy()
   ts_interval[2,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test3 = ts_test.copy()
   var_test = (evar_co2x5[:,-1] - evar_co2x5[:,0]).copy()
   ts_test = (var_test-var_std).copy()
   ts_interval[3,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test4 = ts_test.copy()
   var_test = (evar_co2x6[:,-1] - evar_co2x6[:,0]).copy()
   ts_test = (var_test-var_std).copy()
   ts_interval[4,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test5 = ts_test.copy()

   for II in range(5):
       plt.plot([ttt[2+II]]*2,ts_interval[II,:],'k-')
       plt.plot([ttt[2+II]-0.18,ttt[2+II]+0.18],[ts_interval[II,0]]*2,'k-')
       plt.plot([ttt[2+II]-0.18,ttt[2+II]+0.18],[ts_interval[II,1]]*2,'k-')
   plt.plot(ttt[2:7],[ts_test1.mean(),ts_test2.mean(),ts_test3.mean(),ts_test4.mean(),ts_test5.mean()], 'k^--',label='slab ocean model',markersize=9)

   plt.xticks(ttt,['CO2x0.25','CO2x0.5','2xCO2','3xCO2','4xCO2','5xCO2','6xCO2','7xCO2','8xCO2'], rotation=0)

   plt.legend()
   plt.xlim(2.5,9.5)
   plt.ylim(0,350)
   plt.ylabel('m')
   plt.title('(d) Arctic 1000-500 hPa Thickness Response')


   plt.savefig('trend_tmp_plot.jpg', format='jpeg', dpi=200)

   plt.show()

   sys.exit()


