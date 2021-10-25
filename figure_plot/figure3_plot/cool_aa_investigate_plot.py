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

# read grid basics
#   dirname = '/data1/yliang/co2_experiments/processed_data/'
   dirname = '/home/yliang/research/aa_co2/data_process/'
   filename = 'TREFHT_annual_mean_temp_output.nc'
   f = Dataset(dirname + filename, 'r')
   lon = f.variables['lon'][:].data
   lat = f.variables['lat'][:].data
   area = data_process_f.area_calculate_nonuniform(lon,lat)
   sat_control = f.variables['var_co2x1'][:,:,:].data
   sat_co2xp125 = f.variables['var_co2xp125'][:,:,:].data
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


   ny = len(lat)
   nx = len(lon)

# select north atlantic warming hole
#   [x1_wh, x2_wh, y1_wh, y2_wh] = data_process_f.find_lon_lat_index(300,360,40,75,lon,lat)  
#   sat_control[:,y1_wh:y2_wh+1,x1_wh:] = np.nan
#   sat_co2xp25[:,y1_wh:y2_wh+1,x1_wh:] = np.nan
#   sat_co2xp5[:,y1_wh:y2_wh+1,x1_wh:] = np.nan
#   sat_co2x1[:,y1_wh:y2_wh+1,x1_wh:] = np.nan
#   sat_co2x2[:,y1_wh:y2_wh+1,x1_wh:] = np.nan
#   sat_co2x3[:,y1_wh:y2_wh+1,x1_wh:] = np.nan
#   sat_co2x4[:,y1_wh:y2_wh+1,x1_wh:] = np.nan
#   sat_co2x5[:,y1_wh:y2_wh+1,x1_wh:] = np.nan
#   sat_co2x6[:,y1_wh:y2_wh+1,x1_wh:] = np.nan
#   sat_co2x7[:,y1_wh:y2_wh+1,x1_wh:] = np.nan
#   sat_co2x8[:,y1_wh:y2_wh+1,x1_wh:] = np.nan

#   [x1_wh, x2_wh, y1_wh, y2_wh] = data_process_f.find_lon_lat_index(0,45,40,75,lon,lat)  
#   sat_control[:,y1_wh:y2_wh+1,x1_wh:x2_wh+1] = np.nan
#   sat_co2xp25[:,y1_wh:y2_wh+1,x1_wh:x2_wh+1] = np.nan
#   sat_co2xp5[:,y1_wh:y2_wh+1,x1_wh:x2_wh+1] = np.nan
#   sat_co2x1[:,y1_wh:y2_wh+1,x1_wh:x2_wh+1] = np.nan
#   sat_co2x2[:,y1_wh:y2_wh+1,x1_wh:x2_wh+1] = np.nan
#   sat_co2x3[:,y1_wh:y2_wh+1,x1_wh:x2_wh+1] = np.nan
#   sat_co2x4[:,y1_wh:y2_wh+1,x1_wh:x2_wh+1] = np.nan
#   sat_co2x5[:,y1_wh:y2_wh+1,x1_wh:x2_wh+1] = np.nan
#   sat_co2x6[:,y1_wh:y2_wh+1,x1_wh:x2_wh+1] = np.nan
#   sat_co2x7[:,y1_wh:y2_wh+1,x1_wh:x2_wh+1] = np.nan
#   sat_co2x8[:,y1_wh:y2_wh+1,x1_wh:x2_wh+1] = np.nan

   mask_var = (sat_control[1,:,:]/sat_control[1,:,:]).copy()

#   for JJ in range(ny):
#       for II in range(nx):
#           if (sat_co2x4[-30:,JJ,II].mean() - sat_control[-30:,JJ,II].mean()) < 2:
#              mask_var[JJ,II] = np.nan

# select arctic region
   [x1, x2, y1, y2] = data_process_f.find_lon_lat_index(0,360,60,89,lon,lat)

# ================================================================
# calculate time series
# ================================================================
# simulated sat
   ts_sat_control_global = np.zeros((year_N))
   ts_sat_co2xp125_global = np.zeros((year_N))
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
   ts_sat_co2xp125_arctic = np.zeros((year_N))
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

   for NT in range(year_N):
       print('year:' + str(NT))
       ts_sat_control_global[NT] = np.nansum(sat_control[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)
       ts_sat_co2xp125_global[NT] = np.nansum(sat_co2xp125[NT,:,:]*area*mask_var)/np.nansum(area*mask_var)
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
       ts_sat_co2xp125_arctic[NT] = np.nansum(sat_co2xp125[NT,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
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

# ================================================================
# calculate AAF
# ================================================================
   aa_f_co2xp25 = (ts_fsat_co2xp25_arctic-ts_fsat_co2x1_arctic)/(ts_fsat_co2xp25_global-ts_fsat_co2x1_global)
   aa_f_co2xp5 = (ts_fsat_co2xp5_arctic-ts_fsat_co2x1_arctic)/(ts_fsat_co2xp5_global-ts_fsat_co2x1_global)
   aa_f_co2x2 = (ts_fsat_co2x2_arctic-ts_fsat_co2x1_arctic)/(ts_fsat_co2x2_global-ts_fsat_co2x1_global)
   aa_f_co2x3 = (ts_fsat_co2x3_arctic-ts_fsat_co2x1_arctic)/(ts_fsat_co2x3_global-ts_fsat_co2x1_global)
   aa_f_co2x4 = (ts_fsat_co2x4_arctic-ts_fsat_co2x1_arctic)/(ts_fsat_co2x4_global-ts_fsat_co2x1_global)
   aa_f_co2x5 = (ts_fsat_co2x5_arctic-ts_fsat_co2x1_arctic)/(ts_fsat_co2x5_global-ts_fsat_co2x1_global)
   aa_f_co2x6 = (ts_fsat_co2x6_arctic-ts_fsat_co2x1_arctic)/(ts_fsat_co2x6_global-ts_fsat_co2x1_global)
   aa_f_co2x7 = (ts_fsat_co2x7_arctic-ts_fsat_co2x1_arctic)/(ts_fsat_co2x7_global-ts_fsat_co2x1_global)
   aa_f_co2x8 = (ts_fsat_co2x8_arctic-ts_fsat_co2x1_arctic)/(ts_fsat_co2x8_global-ts_fsat_co2x1_global)

   aa_e_co2x2 = (ts_esat_co2x2_arctic-ts_esat_co2x1_arctic)/(ts_esat_co2x2_global-ts_esat_co2x1_global)
   aa_e_co2x3 = (ts_esat_co2x3_arctic-ts_esat_co2x1_arctic)/(ts_esat_co2x3_global-ts_esat_co2x1_global)
   aa_e_co2x4 = (ts_esat_co2x4_arctic-ts_esat_co2x1_arctic)/(ts_esat_co2x4_global-ts_esat_co2x1_global)
   aa_e_co2x5 = (ts_esat_co2x5_arctic-ts_esat_co2x1_arctic)/(ts_esat_co2x5_global-ts_esat_co2x1_global)
   aa_e_co2x6 = (ts_esat_co2x6_arctic-ts_esat_co2x1_arctic)/(ts_esat_co2x6_global-ts_esat_co2x1_global)

# ================================================================
# plot figures
# ================================================================
if True:
#   aa_sat_ctr_trend = (arctic_sat_ctr_trend)
#   aa_sat_xod_trend = (arctic_sat_ctr_trend-arctic_sat_xod_trend)
#   aa_sat_xc2_trend = (arctic_sat_ctr_trend-arctic_sat_xc2_trend)

   plt.close('all')
   fig = plt.figure(1)
   fig.set_size_inches(10, 10, forward=True)

   ax1 = fig.add_axes([0.1, 0.58, 0.8, 0.3])
#   plt.plot(tt,ts_sat_control_arctic,'k--',label='pi-control')
#   plt.plot(tt,ts_sat_co2xp25_arctic,'--',color='blue',label='co2x0.25')
#   plt.plot(tt,ts_sat_co2xp5_arctic,'--',color='blueviolet',label='co2x0.5')   
#   plt.plot(tt,ts_sat_co2x1_arctic,'--',color='rosybrown',label='co2x1')
#   plt.plot(tt,ts_sat_co2x2_arctic,'--',color='orange',label='co2x2')
#   plt.plot(tt,ts_sat_co2x3_arctic,'--',color='gold',label='co2x3')
#   plt.plot(tt,ts_sat_co2x4_arctic,'--',color='magenta',label='co2x4')
#   plt.plot(tt,ts_sat_co2x5_arctic,'--',color='brown',label='co2x5')
#   plt.plot(tt,ts_sat_co2x6_arctic,'--',color='tomato',label='co2x6')
#   plt.plot(tt,ts_sat_co2x7_arctic,'--',color='salmon',label='co2x7')
#   plt.plot(tt,ts_sat_co2x8_arctic,'--',color='red',label='co2x8')

#   plt.plot(tt,ts_sat_control_global,'k-')   
#   plt.plot(tt,ts_sat_co2xp25_global,'-',color='blue')
#   plt.plot(tt,ts_sat_co2xp5_global,'-',color='blueviolet')  
#   plt.plot(tt,ts_sat_co2x1_global,'-',color='rosybrown')
#   plt.plot(tt,ts_sat_co2x2_global,'-',color='orange')
#   plt.plot(tt,ts_sat_co2x3_global,'-',color='gold')
#   plt.plot(tt,ts_sat_co2x4_global,'-',color='magenta')
#   plt.plot(tt,ts_sat_co2x5_global,'-',color='brown')
#   plt.plot(tt,ts_sat_co2x6_global,'-',color='tomato')
#   plt.plot(tt,ts_sat_co2x7_global,'-',color='salmon')
#   plt.plot(tt,ts_sat_co2x8_global,'-',color='red')

#   plt.title('(a) Global and Arctic SAT')
#   plt.xlim(1,150)

#   plt.plot([121,121],[1.5,3.5],'k-')
   plt.fill_between([121,150], [1,1], [4,4], where=[4,4]>= [1,1], facecolor='grey', interpolate=True, alpha=0.3)

   ts_test = ((ts_sat_co2xp125_arctic-ts_sat_control_arctic)/(ts_sat_co2xp125_global-ts_sat_control_global)).copy()
   aa_co2xp125 = ts_test[-30:].copy()
   plt.plot(tt,ts_test,'-',color='turquoise',label='0.125xCO2')
   ts_test = ((ts_sat_co2xp25_arctic-ts_sat_control_arctic)/(ts_sat_co2xp25_global-ts_sat_control_global)).copy()
   aa_co2xp25 = ts_test[-30:].copy()
   plt.plot(tt,ts_test,'-',color='cadetblue',label='0.25xCO2')
   ts_test = ((ts_sat_co2xp5_arctic-ts_sat_control_arctic)/(ts_sat_co2xp5_global-ts_sat_control_global)).copy()
   aa_co2xp5 = ts_test[-30:].copy()
   plt.plot(tt,ts_test,'-',color='royalblue',label='0.5xCO2')
#   ts_test = ((ts_sat_co2x1_arctic-ts_sat_control_arctic)/(ts_sat_co2x1_global-ts_sat_control_global)).copy()
#   plt.plot(tt,ts_test,'-',color='rosybrown',label='co2x1')
   ts_test = ((ts_sat_co2x2_arctic-ts_sat_control_arctic)/(ts_sat_co2x2_global-ts_sat_control_global)).copy()
   aa_co2x2 = ts_test[-30:].copy()
   plt.plot(tt,ts_test,'-',color='b',label='2xCO2')
   ts_test = ((ts_sat_co2x3_arctic-ts_sat_control_arctic)/(ts_sat_co2x3_global-ts_sat_control_global)).copy()
   aa_co2x3 = ts_test[-30:].copy()
   plt.plot(tt,ts_test,'-',color='dodgerblue',label='3xCO2')
   ts_test = ((ts_sat_co2x4_arctic-ts_sat_control_arctic)/(ts_sat_co2x4_global-ts_sat_control_global)).copy()
   aa_co2x4 = ts_test[-30:].copy()
   plt.plot(tt,ts_test,'-',color='springgreen',label='4xCO2')
   ts_test = ((ts_sat_co2x5_arctic-ts_sat_control_arctic)/(ts_sat_co2x5_global-ts_sat_control_global)).copy()
   aa_co2x5 = ts_test[-30:].copy()
   plt.plot(tt,ts_test,'-',color='gold',label='5xCO2')
   ts_test = ((ts_sat_co2x6_arctic-ts_sat_control_arctic)/(ts_sat_co2x6_global-ts_sat_control_global)).copy()
   aa_co2x6 = ts_test[-30:].copy()
   plt.plot(tt,ts_test,'-',color='orange',label='6xCO2')
   ts_test = ((ts_sat_co2x7_arctic-ts_sat_control_arctic)/(ts_sat_co2x7_global-ts_sat_control_global)).copy()
   aa_co2x7 = ts_test[-30:].copy()
   plt.plot(tt,ts_test,'-',color='tomato',label='7xCO2')
   ts_test = ((ts_sat_co2x8_arctic-ts_sat_control_arctic)/(ts_sat_co2x8_global-ts_sat_control_global)).copy()
   aa_co2x8 = ts_test[-30:].copy()
   plt.plot(tt,ts_test,'-',color='red',label='8xCO2')

   plt.legend(ncol=6, loc='upper right')
   plt.title('(a) Arctic Amplificaton Factor Evolution')
   plt.xlim(1,150)
   plt.ylim(1.5,3.5)

   plt.xlabel('year')

   plt.xticks([1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150],['1','','','30','','','60','','','90','','','120','','','150'])

# AAF plot
   ax1 = fig.add_axes([0.1, 0.17, 0.36, 0.32])

   ttt = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

   ts_interval = np.zeros((10,2))
   ts_test = aa_co2xp125.copy()
   ts_interval[0,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test = aa_co2xp25.copy()
   ts_interval[1,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test = aa_co2xp5.copy()
   ts_interval[2,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test = aa_co2x2.copy()
   ts_interval[3,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test = aa_co2x3.copy()
   ts_interval[4,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test = aa_co2x4.copy()
   ts_interval[5,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test = aa_co2x5.copy()
   ts_interval[6,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test = aa_co2x6.copy()
   ts_interval[7,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test = aa_co2x7.copy()
   ts_interval[8,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test = aa_co2x8.copy()
   ts_interval[9,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))

   for II in range(10):
       plt.plot([ttt[II]]*2,ts_interval[II,:],'k-')
       plt.plot([ttt[II]-0.19,ttt[II]+0.19],[ts_interval[II,0]]*2,'k-')
       plt.plot([ttt[II]-0.19,ttt[II]+0.19],[ts_interval[II,1]]*2,'k-')

   plt.plot(ttt[:],[aa_co2xp125.mean(),aa_co2xp25.mean(),aa_co2xp5.mean(),aa_co2x2.mean(),aa_co2x3.mean(),aa_co2x4.mean(),\
                 aa_co2x5.mean(),aa_co2x6.mean(),aa_co2x7.mean(),aa_co2x8.mean()], 'ko-', label='fully coupled model',markersize=9)

   ts_interval = np.zeros((5,2))
   ts_test = aa_e_co2x2.copy()
   ts_interval[0,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test = aa_e_co2x3.copy()
   ts_interval[1,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test = aa_e_co2x4.copy()
   ts_interval[2,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test = aa_e_co2x5.copy()
   ts_interval[3,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test = aa_e_co2x6.copy()
   ts_interval[4,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))

   for II in range(5):
       plt.plot([ttt[II+3]]*2,ts_interval[II,:],'k-')
       plt.plot([ttt[II+3]-0.19,ttt[II+3]+0.19],[ts_interval[II,0]]*2,'k-')
       plt.plot([ttt[II+3]-0.19,ttt[II+3]+0.19],[ts_interval[II,1]]*2,'k-')

   plt.plot(ttt[3:8],[aa_e_co2x2.mean(),aa_e_co2x3.mean(),aa_e_co2x4.mean(),aa_e_co2x5.mean(),aa_e_co2x6.mean()], 'k^--', label='slab ocean model', markersize=9)

   plt.xticks(ttt,['0.125xCO2','0.25xCO2','0.5xCO2','2xCO2','3xCO2','4xCO2','5xCO2','6xCO2','7xCO2','8xCO2'], rotation=45)
#   plt.yticks([1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5])
   plt.ylim(1.5,3.2)
   plt.xlim(.5,10.5)
   plt.legend()

   plt.title('(b) Last 30-year Mean AAF')

# plot annual amplitude
# read sat
   varname = 'TREFHT'
   year1 = 1850
   year2 = 1999
   year_N = int(year2 - year1 + 1)
   tt = np.linspace(1,year_N,year_N)

# read grid basics
   dirname = '/data1/yliang/co2_experiments/processed_data/'
   filename = varname + '_annual_mean_temp_output.nc'
   f = Dataset(dirname + filename, 'r')
   lon = f.variables['lon'][:].data
   lat = f.variables['lat'][:].data
   area = data_process_f.area_calculate_nonuniform(lon,lat)
   f.close()

   ny = len(lat)
   nx = len(lon)

   sat_control = np.zeros((year_N,12,ny,nx))
   sat_co2xp25 = np.zeros((year_N,12,ny,nx))
   sat_co2xp5 = np.zeros((year_N,12,ny,nx))
   sat_co2x1 = np.zeros((year_N,12,ny,nx))
   sat_co2x2 = np.zeros((year_N,12,ny,nx))
   sat_co2x3 = np.zeros((year_N,12,ny,nx))
   sat_co2x4 = np.zeros((year_N,12,ny,nx))
   sat_co2x5 = np.zeros((year_N,12,ny,nx))
   sat_co2x6 = np.zeros((year_N,12,ny,nx))
   sat_co2x7 = np.zeros((year_N,12,ny,nx))
   sat_co2x8 = np.zeros((year_N,12,ny,nx))

   for NM in range(12):
       dirname = '/data1/yliang/co2_experiments/processed_data/'
       filename = varname + '_month' + str(NM+1) + '_mean_temp_output.nc'
       print(filename)
       f = Dataset(dirname + filename, 'r')
       sat_control[:,NM,:,:] = f.variables['var_co2x1'][:,:,:].data
       sat_co2xp25[:,NM,:,:] = f.variables['var_co2xp25'][:,:,:].data
       sat_co2xp5[:,NM,:,:] = f.variables['var_co2xp5'][:,:,:].data
       sat_co2x1[:,NM,:,:] = f.variables['var_co2x1'][:,:,:].data
       sat_co2x2[:,NM,:,:] = f.variables['var_co2x2'][:,:,:].data
       sat_co2x3[:,NM,:,:] = f.variables['var_co2x3'][:,:,:].data
       sat_co2x4[:,NM,:,:] = f.variables['var_co2x4'][:,:,:].data
       sat_co2x5[:,NM,:,:] = f.variables['var_co2x5'][:,:,:].data
       sat_co2x6[:,NM,:,:] = f.variables['var_co2x6'][:,:,:].data
       sat_co2x7[:,NM,:,:] = f.variables['var_co2x7'][:,:,:].data
       sat_co2x8[:,NM,:,:] = f.variables['var_co2x8'][:,:,:].data
       f.close()

   esat_co2x1 = np.zeros((30,12,ny,nx))
   esat_co2x2 = np.zeros((30,12,ny,nx))
   esat_co2x3 = np.zeros((30,12,ny,nx))
   esat_co2x4 = np.zeros((30,12,ny,nx))
   esat_co2x5 = np.zeros((30,12,ny,nx))
   esat_co2x6 = np.zeros((30,12,ny,nx))

   for NM in range(12):
       dirname = '/data1/yliang/co2_experiments/processed_data/'
       filename = varname + '_month' + str(NM+1) + '_mean_temp_output_e_case.nc'
       print(filename)
       f = Dataset(dirname + filename, 'r')
       esat_co2x1[:,NM,:,:] = f.variables['var_co2x1'][-30:,:,:].data
       esat_co2x2[:,NM,:,:] = f.variables['var_co2x2'][-30:,:,:].data
       esat_co2x3[:,NM,:,:] = f.variables['var_co2x3'][-30:,:,:].data
       esat_co2x4[:,NM,:,:] = f.variables['var_co2x4'][-30:,:,:].data
       esat_co2x5[:,NM,:,:] = f.variables['var_co2x5'][-30:,:,:].data
       esat_co2x6[:,NM,:,:] = f.variables['var_co2x6'][-30:,:,:].data
       f.close()

   n_e = esat_co2x6.shape[0]
# select arctic region
   [x1, x2, y1, y2] = data_process_f.find_lon_lat_index(0,360,60,89,lon,lat)

# create mask array
   mask_var = (area/area).copy()

# ================================================================
# calculate time series
# ================================================================
# simulated sat
   ts_sat_control_global = np.zeros((year_N,12))
   ts_sat_co2xp25_global = np.zeros((year_N,12))
   ts_sat_co2xp5_global = np.zeros((year_N,12))
   ts_sat_co2x1_global = np.zeros((year_N,12))
   ts_sat_co2x2_global = np.zeros((year_N,12))
   ts_sat_co2x3_global = np.zeros((year_N,12))
   ts_sat_co2x4_global = np.zeros((year_N,12))
   ts_sat_co2x5_global = np.zeros((year_N,12))
   ts_sat_co2x6_global = np.zeros((year_N,12))
   ts_sat_co2x7_global = np.zeros((year_N,12))
   ts_sat_co2x8_global = np.zeros((year_N,12))

   ts_sat_control_arctic = np.zeros((year_N,12))
   ts_sat_co2xp25_arctic = np.zeros((year_N,12))
   ts_sat_co2xp5_arctic = np.zeros((year_N,12))
   ts_sat_co2x1_arctic = np.zeros((year_N,12))
   ts_sat_co2x2_arctic = np.zeros((year_N,12))
   ts_sat_co2x3_arctic = np.zeros((year_N,12))
   ts_sat_co2x4_arctic = np.zeros((year_N,12))
   ts_sat_co2x5_arctic = np.zeros((year_N,12))
   ts_sat_co2x6_arctic = np.zeros((year_N,12))
   ts_sat_co2x7_arctic = np.zeros((year_N,12))
   ts_sat_co2x8_arctic = np.zeros((year_N,12))

   for NT in range(year_N):
       for NM in range(12):
           print(NT, NM)
           ts_sat_control_global[NT,NM] = np.nansum(sat_control[NT,NM,:,:]*area*mask_var)/np.nansum(area*mask_var)
           ts_sat_co2xp25_global[NT,NM] = np.nansum(sat_co2xp25[NT,NM,:,:]*area*mask_var)/np.nansum(area*mask_var)
           ts_sat_co2xp5_global[NT,NM] = np.nansum(sat_co2xp5[NT,NM,:,:]*area*mask_var)/np.nansum(area*mask_var)
           ts_sat_co2x1_global[NT,NM] = np.nansum(sat_co2x1[NT,NM,:,:]*area*mask_var)/np.nansum(area*mask_var)
           ts_sat_co2x2_global[NT,NM] = np.nansum(sat_co2x2[NT,NM,:,:]*area*mask_var)/np.nansum(area*mask_var)
           ts_sat_co2x3_global[NT,NM] = np.nansum(sat_co2x3[NT,NM,:,:]*area*mask_var)/np.nansum(area*mask_var)
           ts_sat_co2x4_global[NT,NM] = np.nansum(sat_co2x4[NT,NM,:,:]*area*mask_var)/np.nansum(area*mask_var)
           ts_sat_co2x5_global[NT,NM] = np.nansum(sat_co2x5[NT,NM,:,:]*area*mask_var)/np.nansum(area*mask_var)
           ts_sat_co2x6_global[NT,NM] = np.nansum(sat_co2x6[NT,NM,:,:]*area*mask_var)/np.nansum(area*mask_var)
           ts_sat_co2x7_global[NT,NM] = np.nansum(sat_co2x7[NT,NM,:,:]*area*mask_var)/np.nansum(area*mask_var)
           ts_sat_co2x8_global[NT,NM] = np.nansum(sat_co2x8[NT,NM,:,:]*area*mask_var)/np.nansum(area*mask_var)

           ts_sat_control_arctic[NT,NM] = np.nansum(sat_control[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_sat_co2xp25_arctic[NT,NM] = np.nansum(sat_co2xp25[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_sat_co2xp5_arctic[NT,NM] = np.nansum(sat_co2xp5[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_sat_co2x1_arctic[NT,NM] = np.nansum(sat_co2x1[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_sat_co2x2_arctic[NT,NM] = np.nansum(sat_co2x2[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_sat_co2x3_arctic[NT,NM] = np.nansum(sat_co2x3[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_sat_co2x4_arctic[NT,NM] = np.nansum(sat_co2x4[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_sat_co2x5_arctic[NT,NM] = np.nansum(sat_co2x5[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_sat_co2x6_arctic[NT,NM] = np.nansum(sat_co2x6[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_sat_co2x7_arctic[NT,NM] = np.nansum(sat_co2x7[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_sat_co2x8_arctic[NT,NM] = np.nansum(sat_co2x8[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])

   ts_esat_co2x1_global = np.zeros((n_e,12))
   ts_esat_co2x2_global = np.zeros((n_e,12))
   ts_esat_co2x3_global = np.zeros((n_e,12))
   ts_esat_co2x4_global = np.zeros((n_e,12))
   ts_esat_co2x5_global = np.zeros((n_e,12))
   ts_esat_co2x6_global = np.zeros((n_e,12))

   ts_esat_co2x1_arctic = np.zeros((n_e,12))
   ts_esat_co2x2_arctic = np.zeros((n_e,12))
   ts_esat_co2x3_arctic = np.zeros((n_e,12))
   ts_esat_co2x4_arctic = np.zeros((n_e,12))
   ts_esat_co2x5_arctic = np.zeros((n_e,12))
   ts_esat_co2x6_arctic = np.zeros((n_e,12))

   for NT in range(n_e):
       for NM in range(12):
           print(NT, NM)
           ts_esat_co2x1_global[NT,NM] = np.nansum(esat_co2x1[NT,NM,:,:]*area*mask_var)/np.nansum(area*mask_var)
           ts_esat_co2x2_global[NT,NM] = np.nansum(esat_co2x2[NT,NM,:,:]*area*mask_var)/np.nansum(area*mask_var)
           ts_esat_co2x3_global[NT,NM] = np.nansum(esat_co2x3[NT,NM,:,:]*area*mask_var)/np.nansum(area*mask_var)
           ts_esat_co2x4_global[NT,NM] = np.nansum(esat_co2x4[NT,NM,:,:]*area*mask_var)/np.nansum(area*mask_var)
           ts_esat_co2x5_global[NT,NM] = np.nansum(esat_co2x5[NT,NM,:,:]*area*mask_var)/np.nansum(area*mask_var)
           ts_esat_co2x6_global[NT,NM] = np.nansum(esat_co2x6[NT,NM,:,:]*area*mask_var)/np.nansum(area*mask_var)

           ts_esat_co2x1_arctic[NT,NM] = np.nansum(esat_co2x1[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_esat_co2x2_arctic[NT,NM] = np.nansum(esat_co2x2[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_esat_co2x3_arctic[NT,NM] = np.nansum(esat_co2x3[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_esat_co2x4_arctic[NT,NM] = np.nansum(esat_co2x4[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_esat_co2x5_arctic[NT,NM] = np.nansum(esat_co2x5[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_esat_co2x6_arctic[NT,NM] = np.nansum(esat_co2x6[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])

   ts_test_tmp = ((ts_sat_co2xp25_arctic - ts_sat_control_arctic)).copy()/((ts_sat_co2xp25_global - ts_sat_control_global)).copy()
   ts_test_xp25 = ts_test_tmp.copy()
   ts_test_xp25[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_xp25[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2xp5_arctic - ts_sat_control_arctic)).copy()/((ts_sat_co2xp5_global - ts_sat_control_global)).copy()
   ts_test_xp5 = ts_test_tmp.copy()
   ts_test_xp5[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_xp5[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x2_arctic - ts_sat_control_arctic)).copy()/((ts_sat_co2x2_global - ts_sat_control_global)).copy()
   ts_test_x2 = ts_test_tmp.copy()
   ts_test_x2[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x2[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x3_arctic - ts_sat_control_arctic)).copy()/((ts_sat_co2x3_global - ts_sat_control_global)).copy()
   ts_test_x3 = ts_test_tmp.copy()
   ts_test_x3[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x3[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x4_arctic - ts_sat_control_arctic)).copy()/((ts_sat_co2x4_global - ts_sat_control_global)).copy()
   ts_test_x4 = ts_test_tmp.copy()
   ts_test_x4[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x4[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x5_arctic - ts_sat_control_arctic)).copy()/((ts_sat_co2x5_global - ts_sat_control_global)).copy()
   ts_test_x5 = ts_test_tmp.copy()
   ts_test_x5[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x5[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x6_arctic - ts_sat_control_arctic)).copy()/((ts_sat_co2x6_global - ts_sat_control_global)).copy()
   ts_test_x6 = ts_test_tmp.copy()
   ts_test_x6[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x6[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x7_arctic - ts_sat_control_arctic)).copy()/((ts_sat_co2x7_global - ts_sat_control_global)).copy()
   ts_test_x7 = ts_test_tmp.copy()
   ts_test_x7[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x7[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x8_arctic - ts_sat_control_arctic)).copy()/((ts_sat_co2x8_global - ts_sat_control_global)).copy()
   ts_test_x8 = ts_test_tmp.copy()
   ts_test_x8[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x8[:,6:] = ts_test_tmp[:,0:6].copy()

   tse_test_tmp = ((ts_esat_co2x2_arctic - ts_esat_co2x1_arctic)/(ts_esat_co2x2_global - ts_esat_co2x1_global)).copy()
   tse_test_x2 = tse_test_tmp.copy()
   tse_test_x2[:,0:6] = tse_test_tmp[:,6:].copy()
   tse_test_x2[:,6:] = tse_test_tmp[:,0:6].copy()
   tse_test_tmp = ((ts_esat_co2x3_arctic - ts_esat_co2x1_arctic)/(ts_esat_co2x3_global - ts_esat_co2x1_global)).copy()
   tse_test_x3 = tse_test_tmp.copy()
   tse_test_x3[:,0:6] = tse_test_tmp[:,6:].copy()
   tse_test_x3[:,6:] = tse_test_tmp[:,0:6].copy()
   tse_test_tmp = ((ts_esat_co2x4_arctic - ts_esat_co2x1_arctic)/(ts_esat_co2x4_global - ts_esat_co2x1_global)).copy()
   tse_test_x4 = tse_test_tmp.copy()
   tse_test_x4[:,0:6] = tse_test_tmp[:,6:].copy()
   tse_test_x4[:,6:] = tse_test_tmp[:,0:6].copy()
   tse_test_tmp = ((ts_esat_co2x5_arctic - ts_esat_co2x1_arctic)/(ts_esat_co2x5_global - ts_esat_co2x1_global)).copy()
   tse_test_x5 = tse_test_tmp.copy()
   tse_test_x5[:,0:6] = tse_test_tmp[:,6:].copy()
   tse_test_x5[:,6:] = tse_test_tmp[:,0:6].copy()
   tse_test_tmp = ((ts_esat_co2x6_arctic - ts_esat_co2x1_arctic)/(ts_esat_co2x6_global - ts_esat_co2x1_global)).copy()
   tse_test_x6 = tse_test_tmp.copy()
   tse_test_x6[:,0:6] = tse_test_tmp[:,6:].copy()
   tse_test_x6[:,6:] = tse_test_tmp[:,0:6].copy()

# seasonal cycle evolution
   ts_max_xp25 = np.zeros((year_N))
   ts_min_xp25 = np.zeros((year_N))
   ts_max_xp5 = np.zeros((year_N))
   ts_min_xp5 = np.zeros((year_N))
   ts_max_x2 = np.zeros((year_N))
   ts_min_x2 = np.zeros((year_N))
   ts_max_x3 = np.zeros((year_N))
   ts_min_x3 = np.zeros((year_N))
   ts_max_x4 = np.zeros((year_N))
   ts_min_x4 = np.zeros((year_N))
   ts_max_x5 = np.zeros((year_N))
   ts_min_x5 = np.zeros((year_N))
   ts_max_x6 = np.zeros((year_N))
   ts_min_x6 = np.zeros((year_N))
   ts_max_x7 = np.zeros((year_N))
   ts_min_x7 = np.zeros((year_N))
   ts_max_x8 = np.zeros((year_N))
   ts_min_x8 = np.zeros((year_N))

   for NT in range(year_N):
       ts_max_xp25[NT] = np.nanmax(ts_test_xp25[NT,:])
       ts_min_xp25[NT] = np.nanmin(ts_test_xp25[NT,:])
       ts_max_xp5[NT] = np.nanmax(ts_test_xp5[NT,:])
       ts_min_xp5[NT] = np.nanmin(ts_test_xp5[NT,:])
       ts_max_x2[NT] = np.nanmax(ts_test_x2[NT,:])
       ts_min_x2[NT] = np.nanmin(ts_test_x2[NT,:])
       ts_max_x3[NT] = np.nanmax(ts_test_x3[NT,:])
       ts_min_x3[NT] = np.nanmin(ts_test_x3[NT,:])
       ts_max_x4[NT] = np.nanmax(ts_test_x4[NT,:])
       ts_min_x4[NT] = np.nanmin(ts_test_x4[NT,:])
       ts_max_x5[NT] = np.nanmax(ts_test_x5[NT,:])
       ts_min_x5[NT] = np.nanmin(ts_test_x5[NT,:])
       ts_max_x6[NT] = np.nanmax(ts_test_x6[NT,:])
       ts_min_x6[NT] = np.nanmin(ts_test_x6[NT,:])
       ts_max_x7[NT] = np.nanmax(ts_test_x7[NT,:])
       ts_min_x7[NT] = np.nanmin(ts_test_x7[NT,:])
       ts_max_x8[NT] = np.nanmax(ts_test_x8[NT,:])
       ts_min_x8[NT] = np.nanmin(ts_test_x8[NT,:])

   tse_max_x2 = np.zeros((n_e))
   tse_min_x2 = np.zeros((n_e))
   tse_max_x3 = np.zeros((n_e))
   tse_min_x3 = np.zeros((n_e))
   tse_max_x4 = np.zeros((n_e))
   tse_min_x4 = np.zeros((n_e))
   tse_max_x5 = np.zeros((n_e))
   tse_min_x5 = np.zeros((n_e))
   tse_max_x6 = np.zeros((n_e))
   tse_min_x6 = np.zeros((n_e))

   for NT in range(n_e):
       tse_max_x2[NT] = np.nanmax(tse_test_x2[NT,:])
       tse_min_x2[NT] = np.nanmin(tse_test_x2[NT,:])
       tse_max_x3[NT] = np.nanmax(tse_test_x3[NT,:])
       tse_min_x3[NT] = np.nanmin(tse_test_x3[NT,:])
       tse_max_x4[NT] = np.nanmax(tse_test_x4[NT,:])
       tse_min_x4[NT] = np.nanmin(tse_test_x4[NT,:])
       tse_max_x5[NT] = np.nanmax(tse_test_x5[NT,:])
       tse_min_x5[NT] = np.nanmin(tse_test_x5[NT,:])
       tse_max_x6[NT] = np.nanmax(tse_test_x6[NT,:])
       tse_min_x6[NT] = np.nanmin(tse_test_x6[NT,:])

   ax1 = fig.add_axes([0.54, 0.17, 0.36, 0.32])

   ts_interval = np.zeros((9,2))
   ts_test = (ts_max_xp25-ts_min_xp25)[-30:].copy()
   ts_interval[0,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test = (ts_max_xp5-ts_min_xp5)[-30:].copy()
   ts_interval[1,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test = (ts_max_x2-ts_min_x2)[-30:].copy()
   ts_interval[2,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test = (ts_max_x3-ts_min_x3)[-30:].copy()
   ts_interval[3,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test = (ts_max_x4-ts_min_x4)[-30:].copy()
   ts_interval[4,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test = (ts_max_x5-ts_min_x5)[-30:].copy()
   ts_interval[5,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test = (ts_max_x6-ts_min_x6)[-30:].copy()
   ts_interval[6,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test = (ts_max_x7-ts_min_x7)[-30:].copy()
   ts_interval[7,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test = (ts_max_x8-ts_min_x8)[-30:].copy()
   ts_interval[8,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))

   for II in range(7):
       plt.plot([ttt[II+2]]*2,ts_interval[II+2,:],'k-')
       plt.plot([ttt[II+2]-0.1,ttt[II+2]+0.1],[ts_interval[II+2,0]]*2,'k-')
       plt.plot([ttt[II+2]-0.1,ttt[II+2]+0.1],[ts_interval[II+2,1]]*2,'k-')

   test_array = [(ts_max_x2-ts_min_x2)[-30:].mean(), (ts_max_x3-ts_min_x3)[-30:].mean(), (ts_max_x4-ts_min_x4)[-30:].mean(),\
                 (ts_max_x5-ts_min_x5)[-30:].mean(), (ts_max_x6-ts_min_x6)[-30:].mean(), (ts_max_x7-ts_min_x7)[-30:].mean(),\
                 (ts_max_x8-ts_min_x8)[-30:].mean()]

   plt.plot(ttt[2:9],test_array,'o-',color='k',label='fully coupled model',markersize=9)

   ts_interval = np.zeros((5,2))
   ts_test = (tse_max_x2-tse_min_x2)[-30:].copy()
   ts_interval[0,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test = (tse_max_x3-tse_min_x3)[-30:].copy()
   ts_interval[1,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test = (tse_max_x4-tse_min_x4)[-30:].copy()
   ts_interval[2,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test = (tse_max_x5-tse_min_x5)[-30:].copy()
   ts_interval[3,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
   ts_test = (tse_max_x6-tse_min_x6)[-30:].copy()
   ts_interval[4,:] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))

   for II in range(5):
       plt.plot([ttt[II+2]]*2,ts_interval[II,:],'k-')
       plt.plot([ttt[II+2]-0.15,ttt[II+2]+0.15],[ts_interval[II,0]]*2,'k-')
       plt.plot([ttt[II+2]-0.15,ttt[II+2]+0.15],[ts_interval[II,1]]*2,'k-')

   test_array = [(tse_max_x2-tse_min_x2)[-30:].mean(), (tse_max_x3-tse_min_x3)[-30:].mean(), (tse_max_x4-tse_min_x4)[-30:].mean(),\
                 (tse_max_x5-tse_min_x5)[-30:].mean(), (tse_max_x6-tse_min_x6)[-30:].mean()]

   plt.plot(ttt[2:7],test_array,'^--',color='k',label='slab ocean model',markersize=9)

   plt.legend()

   plt.xticks(np.linspace(0,8,9),['co2x0.25','co2x0.5','2xCO2','3xCO2','4xCO2','5xCO2','6xCO2','7xCO2','8xCO2'], rotation=0)

   plt.xlim(1.5,8.5)


   plt.title('(c) Annual Amplitude of AAF')

   plt.savefig('trend_tmp_plot.jpg', format='jpeg', dpi=200)

   plt.show()

   sys.exit()


# scatter plot

   ax1 = fig.add_axes([0.1, 0.17, 0.36, 0.32])

   plt.plot([-30, 30],[-30, 30],'--', color='gray')
#   plt.plot([0, 0],[-30, 30],'-', color='k')
#   plt.plot([-30, 30],[0,0],'-', color='k')

#   y_test = (ts_sat_co2xp25_arctic-ts_sat_control_arctic)[-30:].copy()
#   x_test = (ts_sat_co2xp25_global-ts_sat_control_global)[-30:].copy()
#   p_test_res = stats.theilslopes(y_test,x_test)
#   text_test = 'co2x0.25, m=' + str(round(p_test_res[0],2)) + '$\pm$' + str(round(abs(p_test_res[0]-p_test_res[2]), 2))
#   text_test = 'co2x0.25'
#   plt.plot(x_test,y_test,'o',color='blue', markersize=2, alpha=0.5)
#   plt.plot([x_test.mean()]*2,[y_test.mean()]*2,'o',color='blue',markersize=12, label=text_test)

#   y_test = (ts_sat_co2xp5_arctic-ts_sat_control_arctic)[-30:].copy()
#   x_test = (ts_sat_co2xp5_global-ts_sat_control_global)[-30:].copy()
#   text_test = 'co2x0.5'
#   plt.plot(x_test,y_test,'o',color='blueviolet',markersize=2, alpha=0.5)  
#   plt.plot([x_test.mean()]*2,[y_test.mean()]*2,'o',color='blueviolet',markersize=12, label=text_test)

   y_test = (ts_sat_co2x2_arctic-ts_sat_control_arctic)[-30:].copy()
   x_test = (ts_sat_co2x2_global-ts_sat_control_global)[-30:].copy()
   text_test = '2xCO2'
   plt.plot(x_test,y_test,'o',color='b',markersize=2, alpha=0.5)
   plt.plot([x_test.mean()]*2,[y_test.mean()]*2,'o',color='b',markersize=12, label=text_test, mec='grey')

   y_test = (ts_sat_co2x3_arctic-ts_sat_control_arctic)[-30:].copy()
   x_test = (ts_sat_co2x3_global-ts_sat_control_global)[-30:].copy()
   text_test = '3xCO2'
   plt.plot(x_test,y_test,'o',color='dodgerblue',markersize=2, alpha=0.5)
   plt.plot([x_test.mean()]*2,[y_test.mean()]*2,'o',color='dodgerblue',markersize=12, label=text_test, mec='grey')

   y_test = (ts_sat_co2x4_arctic-ts_sat_control_arctic)[-30:].copy()
   x_test = (ts_sat_co2x4_global-ts_sat_control_global)[-30:].copy()
   text_test = '4xCO2'
   plt.plot(x_test,y_test,'o',color='springgreen',markersize=2, alpha=0.5)
   plt.plot([x_test.mean()]*2,[y_test.mean()]*2,'o',color='springgreen',markersize=12, label=text_test, mec='grey')

   y_test = (ts_sat_co2x5_arctic-ts_sat_control_arctic)[-30:].copy()
   x_test = (ts_sat_co2x5_global-ts_sat_control_global)[-30:].copy()
   text_test = '5xCO2'
   plt.plot(x_test,y_test,'o',color='gold',markersize=2, alpha=0.5)
   plt.plot([x_test.mean()]*2,[y_test.mean()]*2,'o',color='gold',markersize=12, label=text_test, mec='grey')

   y_test = (ts_sat_co2x6_arctic-ts_sat_control_arctic)[-30:].copy()
   x_test = (ts_sat_co2x6_global-ts_sat_control_global)[-30:].copy()
   text_test = '6xCO2'
   plt.plot(x_test,y_test,'o',color='orange',markersize=2, alpha=0.5)
   plt.plot([x_test.mean()]*2,[y_test.mean()]*2,'o',color='orange',markersize=12, label=text_test, mec='grey')

   y_test = (ts_sat_co2x7_arctic-ts_sat_control_arctic)[-30:].copy()
   x_test = (ts_sat_co2x7_global-ts_sat_control_global)[-30:].copy()
   text_test = '7xCO2'
   plt.plot(x_test,y_test,'o',color='tomato',markersize=2, alpha=0.5)
   plt.plot([x_test.mean()]*2,[y_test.mean()]*2,'o',color='tomato',markersize=12, label=text_test, mec='grey')

   y_test = (ts_sat_co2x8_arctic-ts_sat_control_arctic)[-30:].copy()
   x_test = (ts_sat_co2x8_global-ts_sat_control_global)[-30:].copy()
   text_test = '8xCO2'
   plt.plot(x_test,y_test,'o',color='red',markersize=2, alpha=0.5)
   plt.plot([x_test.mean()]*2,[y_test.mean()]*2,'o',color='red',markersize=12, label=text_test, mec='grey')

# SOM
   y_test = (ts_esat_co2x2_arctic-ts_esat_co2x1_arctic)[-30:].copy()
   x_test = (ts_esat_co2x2_global-ts_esat_co2x1_global)[-30:].copy()
   text_test = '2xCO2'
   plt.plot(x_test,y_test,'o',color='b',markersize=2, alpha=0.5)
   plt.plot([x_test.mean()]*2,[y_test.mean()]*2,'^',color='b',markersize=12, label=text_test, mec='grey')

   y_test = (ts_esat_co2x3_arctic-ts_esat_co2x1_arctic)[-30:].copy()
   x_test = (ts_esat_co2x3_global-ts_esat_co2x1_global)[-30:].copy()
   text_test = '3xCO2'
   plt.plot(x_test,y_test,'o',color='dodgerblue',markersize=2, alpha=0.5)
   plt.plot([x_test.mean()]*2,[y_test.mean()]*2,'^',color='dodgerblue',markersize=12, label=text_test, mec='grey')

   y_test = (ts_esat_co2x4_arctic-ts_esat_co2x1_arctic)[-30:].copy()
   x_test = (ts_esat_co2x4_global-ts_esat_co2x1_global)[-30:].copy()
   text_test = '4xCO2'
   plt.plot(x_test,y_test,'o',color='springgreen',markersize=2, alpha=0.5)
   plt.plot([x_test.mean()]*2,[y_test.mean()]*2,'^',color='springgreen',markersize=12, label=text_test, mec='grey')

   y_test = (ts_esat_co2x5_arctic-ts_esat_co2x1_arctic)[-30:].copy()
   x_test = (ts_esat_co2x5_global-ts_esat_co2x1_global)[-30:].copy()
   text_test = '5xCO2'
   plt.plot(x_test,y_test,'o',color='gold',markersize=2, alpha=0.5)
   plt.plot([x_test.mean()]*2,[y_test.mean()]*2,'^',color='gold',markersize=12, label=text_test, mec='grey')

   y_test = (ts_esat_co2x6_arctic-ts_esat_co2x1_arctic)[-30:].copy()
   x_test = (ts_esat_co2x6_global-ts_esat_co2x1_global)[-30:].copy()
   text_test = '6xCO2'
   plt.plot(x_test,y_test,'o',color='orange',markersize=2, alpha=0.5)
   plt.plot([x_test.mean()]*2,[y_test.mean()]*2,'^',color='orange',markersize=12, label=text_test, mec='grey')

   plt.ylabel('Arctic SAT change (K)')
   plt.xlabel('Global SAT change (K)')
   plt.axis([-0, 12, 0, 25])

   plt.legend(ncol=2, loc='upper left', fontsize='small')

   plt.title('(b) Arctic Warming vs Global Warming')
  
