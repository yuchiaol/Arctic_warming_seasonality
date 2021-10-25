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

   fsat_control = np.zeros((30,12,ny,nx))
   fsat_co2xp25 = np.zeros((30,12,ny,nx))
   fsat_co2xp5 = np.zeros((30,12,ny,nx))
   fsat_co2x1 = np.zeros((30,12,ny,nx))
   fsat_co2x2 = np.zeros((30,12,ny,nx))
   fsat_co2x3 = np.zeros((30,12,ny,nx))
   fsat_co2x4 = np.zeros((30,12,ny,nx))
   fsat_co2x5 = np.zeros((30,12,ny,nx))
   fsat_co2x6 = np.zeros((30,12,ny,nx))
   fsat_co2x7 = np.zeros((30,12,ny,nx))
   fsat_co2x8 = np.zeros((30,12,ny,nx))

   for NM in range(12):
       dirname = '/data1/yliang/co2_experiments/processed_data/'
       filename = varname + '_month' + str(NM+1) + '_mean_temp_output_f_case.nc'
       print(filename)
       f = Dataset(dirname + filename, 'r')
       fsat_co2xp25[:,NM,:,:] = f.variables['var_co2xp25'][-30:,:,:].data
       fsat_co2xp5[:,NM,:,:] = f.variables['var_co2xp5'][-30:,:,:].data
       fsat_co2x1[:,NM,:,:] = f.variables['var_co2x1'][-30:,:,:].data
       fsat_co2x2[:,NM,:,:] = f.variables['var_co2x2'][-30:,:,:].data
       fsat_co2x3[:,NM,:,:] = f.variables['var_co2x3'][-30:,:,:].data
       fsat_co2x4[:,NM,:,:] = f.variables['var_co2x4'][-30:,:,:].data
       fsat_co2x5[:,NM,:,:] = f.variables['var_co2x5'][-30:,:,:].data
       fsat_co2x6[:,NM,:,:] = f.variables['var_co2x6'][-30:,:,:].data
       fsat_co2x7[:,NM,:,:] = f.variables['var_co2x7'][-30:,:,:].data
       fsat_co2x8[:,NM,:,:] = f.variables['var_co2x8'][-30:,:,:].data
       f.close()

   n_f = fsat_co2x8.shape[0]

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

   ts_fsat_co2xp25_global = np.zeros((n_f,12))
   ts_fsat_co2xp5_global = np.zeros((n_f,12))
   ts_fsat_co2x1_global = np.zeros((n_f,12))
   ts_fsat_co2x2_global = np.zeros((n_f,12))
   ts_fsat_co2x3_global = np.zeros((n_f,12))
   ts_fsat_co2x4_global = np.zeros((n_f,12))
   ts_fsat_co2x5_global = np.zeros((n_f,12))
   ts_fsat_co2x6_global = np.zeros((n_f,12))
   ts_fsat_co2x7_global = np.zeros((n_f,12))
   ts_fsat_co2x8_global = np.zeros((n_f,12))

   ts_fsat_co2xp25_arctic = np.zeros((n_f,12))
   ts_fsat_co2xp5_arctic = np.zeros((n_f,12))
   ts_fsat_co2x1_arctic = np.zeros((n_f,12))
   ts_fsat_co2x2_arctic = np.zeros((n_f,12))
   ts_fsat_co2x3_arctic = np.zeros((n_f,12))
   ts_fsat_co2x4_arctic = np.zeros((n_f,12))
   ts_fsat_co2x5_arctic = np.zeros((n_f,12))
   ts_fsat_co2x6_arctic = np.zeros((n_f,12))
   ts_fsat_co2x7_arctic = np.zeros((n_f,12))
   ts_fsat_co2x8_arctic = np.zeros((n_f,12))

   for NT in range(n_f):
       for NM in range(12):
           print(NT, NM)
           ts_fsat_co2xp25_global[NT,NM] = np.nansum(fsat_co2xp25[NT,NM,:,:]*area*mask_var)/np.nansum(area*mask_var)
           ts_fsat_co2xp5_global[NT,NM] = np.nansum(fsat_co2xp5[NT,NM,:,:]*area*mask_var)/np.nansum(area*mask_var)
           ts_fsat_co2x1_global[NT,NM] = np.nansum(fsat_co2x1[NT,NM,:,:]*area*mask_var)/np.nansum(area*mask_var)
           ts_fsat_co2x2_global[NT,NM] = np.nansum(fsat_co2x2[NT,NM,:,:]*area*mask_var)/np.nansum(area*mask_var)
           ts_fsat_co2x3_global[NT,NM] = np.nansum(fsat_co2x3[NT,NM,:,:]*area*mask_var)/np.nansum(area*mask_var)
           ts_fsat_co2x4_global[NT,NM] = np.nansum(fsat_co2x4[NT,NM,:,:]*area*mask_var)/np.nansum(area*mask_var)
           ts_fsat_co2x5_global[NT,NM] = np.nansum(fsat_co2x5[NT,NM,:,:]*area*mask_var)/np.nansum(area*mask_var)
           ts_fsat_co2x6_global[NT,NM] = np.nansum(fsat_co2x6[NT,NM,:,:]*area*mask_var)/np.nansum(area*mask_var)
           ts_fsat_co2x7_global[NT,NM] = np.nansum(fsat_co2x7[NT,NM,:,:]*area*mask_var)/np.nansum(area*mask_var)
           ts_fsat_co2x8_global[NT,NM] = np.nansum(fsat_co2x8[NT,NM,:,:]*area*mask_var)/np.nansum(area*mask_var)

           ts_fsat_co2xp25_arctic[NT,NM] = np.nansum(fsat_co2xp25[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_fsat_co2xp5_arctic[NT,NM] = np.nansum(fsat_co2xp5[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_fsat_co2x1_arctic[NT,NM] = np.nansum(fsat_co2x1[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_fsat_co2x2_arctic[NT,NM] = np.nansum(fsat_co2x2[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_fsat_co2x3_arctic[NT,NM] = np.nansum(fsat_co2x3[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_fsat_co2x4_arctic[NT,NM] = np.nansum(fsat_co2x4[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_fsat_co2x5_arctic[NT,NM] = np.nansum(fsat_co2x5[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_fsat_co2x6_arctic[NT,NM] = np.nansum(fsat_co2x6[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_fsat_co2x7_arctic[NT,NM] = np.nansum(fsat_co2x7[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_fsat_co2x8_arctic[NT,NM] = np.nansum(fsat_co2x8[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])

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


# ================================================================
# plot figures
# ================================================================
if True:

   plt.close('all')
   fig = plt.figure(1)
   fig.set_size_inches(10, 10, forward=True)

   ttt = np.linspace(1,year_N,year_N)
   tt = np.linspace(1,12,12)

# AAF

# averaged seasonal cycle
   tt = np.linspace(1,12,12)
   ts_test_tmp = ((ts_sat_co2xp25_arctic - ts_sat_control_arctic)/(ts_sat_co2xp25_global - ts_sat_control_global)).copy()
   ts_test_xp25 = ts_test_tmp.copy()
   ts_test_xp25[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_xp25[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2xp5_arctic - ts_sat_control_arctic)/(ts_sat_co2xp5_global - ts_sat_control_global)).copy()
   ts_test_xp5 = ts_test_tmp.copy()
   ts_test_xp5[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_xp5[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x2_arctic - ts_sat_control_arctic)/(ts_sat_co2x2_global - ts_sat_control_global)).copy()
   ts_test_x2 = ts_test_tmp.copy()
   ts_test_x2[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x2[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x3_arctic - ts_sat_control_arctic)/(ts_sat_co2x3_global - ts_sat_control_global)).copy()
   ts_test_x3 = ts_test_tmp.copy()
   ts_test_x3[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x3[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x4_arctic - ts_sat_control_arctic)/(ts_sat_co2x4_global - ts_sat_control_global)).copy()
   ts_test_x4 = ts_test_tmp.copy()
   ts_test_x4[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x4[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x5_arctic - ts_sat_control_arctic)/(ts_sat_co2x5_global - ts_sat_control_global)).copy()
   ts_test_x5 = ts_test_tmp.copy()
   ts_test_x5[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x5[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x6_arctic - ts_sat_control_arctic)/(ts_sat_co2x6_global - ts_sat_control_global)).copy()
   ts_test_x6 = ts_test_tmp.copy()
   ts_test_x6[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x6[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x7_arctic - ts_sat_control_arctic)/(ts_sat_co2x7_global - ts_sat_control_global)).copy()
   ts_test_x7 = ts_test_tmp.copy()
   ts_test_x7[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x7[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x8_arctic - ts_sat_control_arctic)/(ts_sat_co2x8_global - ts_sat_control_global)).copy()
   ts_test_x8 = ts_test_tmp.copy()
   ts_test_x8[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x8[:,6:] = ts_test_tmp[:,0:6].copy()

   ts_interval = np.zeros((9,2,12))
   for III in range(12):
       ts_test = ts_test_xp25[-30:,III].copy()-ts_test_xp25[-30:,0].mean()*0
       ts_interval[0,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_xp5[-30:,III].copy()-ts_test_xp5[-30:,0].mean()*0
       ts_interval[1,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_x2[-30:,III].copy()-ts_test_x2[-30:,0].mean()*0
       ts_interval[2,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_x3[-30:,III].copy()-ts_test_x3[-30:,0].mean()*0
       ts_interval[3,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_x4[-30:,III].copy()-ts_test_x4[-30:,0].mean()*0
       ts_interval[4,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_x5[-30:,III].copy()-ts_test_x5[-30:,0].mean()*0
       ts_interval[5,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_x6[-30:,III].copy()-ts_test_x6[-30:,0].mean()*0
       ts_interval[6,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_x7[-30:,III].copy()-ts_test_x7[-30:,0].mean()*0
       ts_interval[7,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_x8[-30:,III].copy()-ts_test_x8[-30:,0].mean()*0
       ts_interval[8,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))

   ax1 = fig.add_axes([0.1, 0.47, 0.36, 0.32])
   ts_interval = abs(ts_interval)
   for II in range(7):
       for JJ in range(12):
           plt.plot([tt[JJ]]*2,ts_interval[II+2,:,JJ],'k-', alpha=0.3)
           plt.plot([tt[JJ]-0.1,tt[JJ]+0.1],[ts_interval[II+2,0,JJ]]*2,'k-',alpha=0.3)
           plt.plot([tt[JJ]-0.1,tt[JJ]+0.1],[ts_interval[II+2,1,JJ]]*2,'k-',alpha=0.3)

#   plt.plot(tt,np.nanmean(ts_test_xp25[-30:,:], axis=0)-np.nanmean(ts_test_xp25[-30:,:], axis=0)[0],'o-',color='blue',label='x0.25', alpha=0.3)
#   plt.plot(tt,np.nanmean(ts_test_xp5[-30:,:], axis=0)-np.nanmean(ts_test_xp5[-30:,:], axis=0)[0],'o-',color='blueviolet',label='x0.5', alpha=0.3)
   plt.plot(tt,np.nanmean(ts_test_x2[-30:,:], axis=0)-np.nanmean(ts_test_x2[-30:,:], axis=0)[0]*0,'o-',color='b',label='x2')
   plt.plot(tt,np.nanmean(ts_test_x3[-30:,:], axis=0)-np.nanmean(ts_test_x3[-30:,:], axis=0)[0]*0,'o-',color='dodgerblue',label='x3')
   plt.plot(tt,np.nanmean(ts_test_x4[-30:,:], axis=0)-np.nanmean(ts_test_x4[-30:,:], axis=0)[0]*0,'o-',color='springgreen',label='x4')
   plt.plot(tt,np.nanmean(ts_test_x5[-30:,:], axis=0)-np.nanmean(ts_test_x5[-30:,:], axis=0)[0]*0,'o-',color='gold',label='x5')
   plt.plot(tt,np.nanmean(ts_test_x6[-30:,:], axis=0)-np.nanmean(ts_test_x6[-30:,:], axis=0)[0]*0,'o-',color='orange',label='x6')
   plt.plot(tt,np.nanmean(ts_test_x7[-30:,:], axis=0)-np.nanmean(ts_test_x7[-30:,:], axis=0)[0]*0,'o-',color='tomato',label='x7')
   plt.plot(tt,np.nanmean(ts_test_x8[-30:,:], axis=0)-np.nanmean(ts_test_x8[-30:,:], axis=0)[0]*0,'o-',color='red',label='x8')
   plt.xticks(tt,['Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun'], fontsize=10)
   plt.legend(ncol=1,fontsize='small',loc='upper left')
   plt.ylim(0.9,3.7)
   plt.title('(a) FOM AAF')


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

# SOM case
   ts_interval = np.zeros((5,2,12))
   for III in range(12):
       ts_test = tse_test_x2[-30:,III].copy()-tse_test_x2[-30:,0].mean()*0
       ts_interval[0,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = tse_test_x3[-30:,III].copy()-tse_test_x3[-30:,0].mean()*0
       ts_interval[1,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = tse_test_x4[-30:,III].copy()-tse_test_x4[-30:,0].mean()*0
       ts_interval[2,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = tse_test_x5[-30:,III].copy()-tse_test_x5[-30:,0].mean()*0
       ts_interval[3,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = tse_test_x6[-30:,III].copy()-tse_test_x6[-30:,0].mean()*0
       ts_interval[4,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))

   ax1 = fig.add_axes([0.53, 0.47, 0.36, 0.32])
   ts_interval = abs(ts_interval)
   for II in range(5):
       for JJ in range(12):
           plt.plot([tt[JJ]]*2,ts_interval[II,:,JJ],'k-',alpha=0.3)
           plt.plot([tt[JJ]-0.1,tt[JJ]+0.1],[ts_interval[II,0,JJ]]*2,'k-',alpha=0.3)
           plt.plot([tt[JJ]-0.1,tt[JJ]+0.1],[ts_interval[II,1,JJ]]*2,'k-',alpha=0.3)

   plt.plot(tt,np.nanmean(tse_test_x2[-30:,:], axis=0)-np.nanmean(tse_test_x2[-30:,:], axis=0)[0]*0,'^-',color='b',label='x2')
   plt.plot(tt,np.nanmean(tse_test_x3[-30:,:], axis=0)-np.nanmean(tse_test_x3[-30:,:], axis=0)[0]*0,'^-',color='dodgerblue',label='x3')
   plt.plot(tt,np.nanmean(tse_test_x4[-30:,:], axis=0)-np.nanmean(tse_test_x4[-30:,:], axis=0)[0]*0,'^-',color='springgreen',label='x4')
   plt.plot(tt,np.nanmean(tse_test_x5[-30:,:], axis=0)-np.nanmean(tse_test_x5[-30:,:], axis=0)[0]*0,'^-',color='gold',label='x5')
   plt.plot(tt,np.nanmean(tse_test_x6[-30:,:], axis=0)-np.nanmean(tse_test_x6[-30:,:], axis=0)[0]*0,'^-',color='orange',label='x6')
   plt.xticks(tt,['Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun'], fontsize=10)
   plt.legend(ncol=1,fontsize='small',loc='upper left')
   plt.ylim(0.9,3.7)
   plt.title('(b) SOM AAF')

   plt.savefig('trend_tmp_plot.jpg', format='jpeg', dpi=200)

   plt.show()

   sys.exit()


