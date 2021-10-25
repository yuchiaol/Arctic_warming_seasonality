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

# select arctic region
   [x1, x2, y1, y2] = data_process_f.find_lon_lat_index(0,360,60,89,lon,lat)

# create mask array
   mask_var = (area/area).copy()

# ================================================================
# set mask_var
# ================================================================
# land-only
   dirname = '/data2/im2527/CESM-LE/8xCO2.B1850LENS.n21.f09_g16/atm/hist/'
   filename = '8xCO2.B1850LENS.n21.f09_g16.cam.h0.1904-12.nc'
   f = Dataset(dirname + filename, 'r')
   LANDFRAC = f.variables['LANDFRAC'][:,:,:].data.squeeze()  
   f.close()

   mask_var = LANDFRAC.copy()
   mask_var[mask_var<1.] = -100000
   mask_var[mask_var==1.] = np.nan
   mask_var[mask_var<0.] = 1.

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
           ts_sat_control_global[NT,NM] = np.nansum(sat_control[NT,NM,:,:]*area)/np.nansum(area)
           ts_sat_co2xp25_global[NT,NM] = np.nansum(sat_co2xp25[NT,NM,:,:]*area)/np.nansum(area)
           ts_sat_co2xp5_global[NT,NM] = np.nansum(sat_co2xp5[NT,NM,:,:]*area)/np.nansum(area)
           ts_sat_co2x1_global[NT,NM] = np.nansum(sat_co2x1[NT,NM,:,:]*area)/np.nansum(area)
           ts_sat_co2x2_global[NT,NM] = np.nansum(sat_co2x2[NT,NM,:,:]*area)/np.nansum(area)
           ts_sat_co2x3_global[NT,NM] = np.nansum(sat_co2x3[NT,NM,:,:]*area)/np.nansum(area)
           ts_sat_co2x4_global[NT,NM] = np.nansum(sat_co2x4[NT,NM,:,:]*area)/np.nansum(area)
           ts_sat_co2x5_global[NT,NM] = np.nansum(sat_co2x5[NT,NM,:,:]*area)/np.nansum(area)
           ts_sat_co2x6_global[NT,NM] = np.nansum(sat_co2x6[NT,NM,:,:]*area)/np.nansum(area)
           ts_sat_co2x7_global[NT,NM] = np.nansum(sat_co2x7[NT,NM,:,:]*area)/np.nansum(area)
           ts_sat_co2x8_global[NT,NM] = np.nansum(sat_co2x8[NT,NM,:,:]*area)/np.nansum(area)

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

# ================================================================
# plot figures
# ================================================================
if True:

   plt.close('all')
   fig = plt.figure(1)
   fig.set_size_inches(10, 10, forward=True)

   ttt = np.linspace(1,year_N,year_N)
   tt = np.linspace(1,12,12)

   ts_test_tmp = ((ts_sat_co2xp25_arctic - ts_sat_control_arctic)).copy()
   ts_test_xp25 = ts_test_tmp.copy()
   ts_test_xp25[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_xp25[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2xp5_arctic - ts_sat_control_arctic)).copy()
   ts_test_xp5 = ts_test_tmp.copy()
   ts_test_xp5[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_xp5[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x2_arctic - ts_sat_control_arctic)).copy()
   ts_test_x2 = ts_test_tmp.copy()
   ts_test_x2[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x2[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x3_arctic - ts_sat_control_arctic)).copy()
   ts_test_x3 = ts_test_tmp.copy()
   ts_test_x3[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x3[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x4_arctic - ts_sat_control_arctic)).copy()
   ts_test_x4 = ts_test_tmp.copy()
   ts_test_x4[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x4[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x5_arctic - ts_sat_control_arctic)).copy()
   ts_test_x5 = ts_test_tmp.copy()
   ts_test_x5[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x5[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x6_arctic - ts_sat_control_arctic)).copy()
   ts_test_x6 = ts_test_tmp.copy()
   ts_test_x6[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x6[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x7_arctic - ts_sat_control_arctic)).copy()
   ts_test_x7 = ts_test_tmp.copy()
   ts_test_x7[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x7[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x8_arctic - ts_sat_control_arctic)).copy()
   ts_test_x8 = ts_test_tmp.copy()
   ts_test_x8[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x8[:,6:] = ts_test_tmp[:,0:6].copy()

   ts_interval = np.zeros((9,2,12))
   for III in range(12):
       ts_test = ts_test_xp25[-30:,III].copy()-ts_test_xp25[-30:,0].mean()
       ts_interval[0,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_xp5[-30:,III].copy()-ts_test_xp5[-30:,0].mean()
       ts_interval[1,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_x2[-30:,III].copy()-ts_test_x2[-30:,0].mean()
       ts_interval[2,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_x3[-30:,III].copy()-ts_test_x3[-30:,0].mean()
       ts_interval[3,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_x4[-30:,III].copy()-ts_test_x4[-30:,0].mean()
       ts_interval[4,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_x5[-30:,III].copy()-ts_test_x5[-30:,0].mean()
       ts_interval[5,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_x6[-30:,III].copy()-ts_test_x6[-30:,0].mean()
       ts_interval[6,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_x7[-30:,III].copy()-ts_test_x7[-30:,0].mean()
       ts_interval[7,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_x8[-30:,III].copy()-ts_test_x8[-30:,0].mean()
       ts_interval[8,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))

   ax1 = fig.add_axes([0.3, 0.7, 0.36, 0.27])
   ts_interval = (ts_interval)
   for II in range(7):
       for JJ in range(12):
           plt.plot([tt[JJ]]*2,ts_interval[II+2,:,JJ],'k-', alpha=0.3)
           plt.plot([tt[JJ]-0.1,tt[JJ]+0.1],[ts_interval[II+2,0,JJ]]*2,'k-',alpha=0.3)
           plt.plot([tt[JJ]-0.1,tt[JJ]+0.1],[ts_interval[II+2,1,JJ]]*2,'k-',alpha=0.3)

#   plt.plot(tt,np.nanmean(ts_test_xp25[-30:,:], axis=0),'o-',color='blue',label='x0.25', alpha=0.3)
#   plt.plot(tt,np.nanmean(ts_test_xp5[-30:,:], axis=0),'o-',color='blueviolet',label='x0.5', alpha=0.3)
   plt.plot(tt,np.nanmean(ts_test_x2[-30:,:], axis=0)-np.nanmean(ts_test_x2[-30:,:], axis=0)[0],'o-',color='b',label='2x')
   plt.plot(tt,np.nanmean(ts_test_x3[-30:,:], axis=0)-np.nanmean(ts_test_x3[-30:,:], axis=0)[0],'o-',color='dodgerblue',label='3x')
   plt.plot(tt,np.nanmean(ts_test_x4[-30:,:], axis=0)-np.nanmean(ts_test_x4[-30:,:], axis=0)[0],'o-',color='springgreen',label='4x')
   plt.plot(tt,np.nanmean(ts_test_x5[-30:,:], axis=0)-np.nanmean(ts_test_x5[-30:,:], axis=0)[0],'o-',color='gold',label='5x')
   plt.plot(tt,np.nanmean(ts_test_x6[-30:,:], axis=0)-np.nanmean(ts_test_x6[-30:,:], axis=0)[0],'o-',color='orange',label='6x')
   plt.plot(tt,np.nanmean(ts_test_x7[-30:,:], axis=0)-np.nanmean(ts_test_x7[-30:,:], axis=0)[0],'o-',color='tomato',label='7x')
   plt.plot(tt,np.nanmean(ts_test_x8[-30:,:], axis=0)-np.nanmean(ts_test_x8[-30:,:], axis=0)[0],'o-',color='red',label='8x')
   plt.xticks(tt,['Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun'], fontsize=10)
   plt.legend(ncol=1,fontsize='small',loc='upper left')
   plt.ylim(-2,18)
   plt.title('(a) Ocean-only Arctic SAT Response')
   plt.ylabel('K')

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

   ax1 = fig.add_axes([0.3, 0.37, 0.36, 0.27])
   ts_interval = (ts_interval)
   for II in range(7):
       for JJ in range(12):
           plt.plot([tt[JJ]]*2,ts_interval[II+2,:,JJ],'k-', alpha=0.3)
           plt.plot([tt[JJ]-0.1,tt[JJ]+0.1],[ts_interval[II+2,0,JJ]]*2,'k-',alpha=0.3)
           plt.plot([tt[JJ]-0.1,tt[JJ]+0.1],[ts_interval[II+2,1,JJ]]*2,'k-',alpha=0.3)

#   plt.plot(tt,np.nanmean(ts_test_xp25[-30:,:], axis=0)-np.nanmean(ts_test_xp25[-30:,:], axis=0)[0],'o-',color='blue',label='x0.25', alpha=0.3)
#   plt.plot(tt,np.nanmean(ts_test_xp5[-30:,:], axis=0)-np.nanmean(ts_test_xp5[-30:,:], axis=0)[0],'o-',color='blueviolet',label='x0.5', alpha=0.3)
   plt.plot(tt,np.nanmean(ts_test_x2[-30:,:], axis=0)-np.nanmean(ts_test_x2[-30:,:], axis=0)[0]*0,'o-',color='b',label='2x')
   plt.plot(tt,np.nanmean(ts_test_x3[-30:,:], axis=0)-np.nanmean(ts_test_x3[-30:,:], axis=0)[0]*0,'o-',color='dodgerblue',label='3x')
   plt.plot(tt,np.nanmean(ts_test_x4[-30:,:], axis=0)-np.nanmean(ts_test_x4[-30:,:], axis=0)[0]*0,'o-',color='springgreen',label='4x')
   plt.plot(tt,np.nanmean(ts_test_x5[-30:,:], axis=0)-np.nanmean(ts_test_x5[-30:,:], axis=0)[0]*0,'o-',color='gold',label='5x')
   plt.plot(tt,np.nanmean(ts_test_x6[-30:,:], axis=0)-np.nanmean(ts_test_x6[-30:,:], axis=0)[0]*0,'o-',color='orange',label='6x')
   plt.plot(tt,np.nanmean(ts_test_x7[-30:,:], axis=0)-np.nanmean(ts_test_x7[-30:,:], axis=0)[0]*0,'o-',color='tomato',label='7x')
   plt.plot(tt,np.nanmean(ts_test_x8[-30:,:], axis=0)-np.nanmean(ts_test_x8[-30:,:], axis=0)[0]*0,'o-',color='red',label='8x')
   plt.xticks(tt,['Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun'], fontsize=10)
   plt.legend(ncol=1,fontsize='small',loc='upper left')
   plt.ylim(0.5,4.5)
   plt.title('(b) Ocean-only AAF')

# global SAT response
   ts_test_tmp = ((ts_sat_co2xp25_global - ts_sat_control_global)).copy()
   ts_test_xp25 = ts_test_tmp.copy()
   ts_test_xp25[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_xp25[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2xp5_global - ts_sat_control_global)).copy()
   ts_test_xp5 = ts_test_tmp.copy()
   ts_test_xp5[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_xp5[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x2_global - ts_sat_control_global)).copy()
   ts_test_x2 = ts_test_tmp.copy()
   ts_test_x2[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x2[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x3_global - ts_sat_control_global)).copy()
   ts_test_x3 = ts_test_tmp.copy()
   ts_test_x3[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x3[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x4_global - ts_sat_control_global)).copy()
   ts_test_x4 = ts_test_tmp.copy()
   ts_test_x4[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x4[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x5_global - ts_sat_control_global)).copy()
   ts_test_x5 = ts_test_tmp.copy()
   ts_test_x5[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x5[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x6_global - ts_sat_control_global)).copy()
   ts_test_x6 = ts_test_tmp.copy()
   ts_test_x6[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x6[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x7_global - ts_sat_control_global)).copy()
   ts_test_x7 = ts_test_tmp.copy()
   ts_test_x7[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x7[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x8_global - ts_sat_control_global)).copy()
   ts_test_x8 = ts_test_tmp.copy()
   ts_test_x8[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x8[:,6:] = ts_test_tmp[:,0:6].copy()

   ts_interval = np.zeros((9,2,12))
   for III in range(12):
       ts_test = ts_test_xp25[-30:,III].copy()-ts_test_xp25[-30:,0].mean()
       ts_interval[0,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_xp5[-30:,III].copy()-ts_test_xp5[-30:,0].mean()
       ts_interval[1,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_x2[-30:,III].copy()-ts_test_x2[-30:,0].mean()
       ts_interval[2,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_x3[-30:,III].copy()-ts_test_x3[-30:,0].mean()
       ts_interval[3,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_x4[-30:,III].copy()-ts_test_x4[-30:,0].mean()
       ts_interval[4,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_x5[-30:,III].copy()-ts_test_x5[-30:,0].mean()
       ts_interval[5,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_x6[-30:,III].copy()-ts_test_x6[-30:,0].mean()
       ts_interval[6,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_x7[-30:,III].copy()-ts_test_x7[-30:,0].mean()
       ts_interval[7,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_x8[-30:,III].copy()-ts_test_x8[-30:,0].mean()
       ts_interval[8,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))

#   ax1 = fig.add_axes([0.1, 0.37, 0.36, 0.27])
#   ts_interval = (ts_interval)
#   for II in range(7):
#       for JJ in range(12):
#           plt.plot([tt[JJ]]*2,ts_interval[II+2,:,JJ],'k-', alpha=0.3)
#           plt.plot([tt[JJ]-0.1,tt[JJ]+0.1],[ts_interval[II+2,0,JJ]]*2,'k-',alpha=0.3)
#           plt.plot([tt[JJ]-0.1,tt[JJ]+0.1],[ts_interval[II+2,1,JJ]]*2,'k-',alpha=0.3)

#   plt.plot(tt,np.nanmean(ts_test_x2[-30:,:], axis=0)-np.nanmean(ts_test_x2[-30:,:], axis=0)[0],'o-',color='b',label='2x')
#   plt.plot(tt,np.nanmean(ts_test_x3[-30:,:], axis=0)-np.nanmean(ts_test_x3[-30:,:], axis=0)[0],'o-',color='dodgerblue',label='3x')
#   plt.plot(tt,np.nanmean(ts_test_x4[-30:,:], axis=0)-np.nanmean(ts_test_x4[-30:,:], axis=0)[0],'o-',color='springgreen',label='4x')
#   plt.plot(tt,np.nanmean(ts_test_x5[-30:,:], axis=0)-np.nanmean(ts_test_x5[-30:,:], axis=0)[0],'o-',color='gold',label='5x')
#   plt.plot(tt,np.nanmean(ts_test_x6[-30:,:], axis=0)-np.nanmean(ts_test_x6[-30:,:], axis=0)[0],'o-',color='orange',label='6x')
#   plt.plot(tt,np.nanmean(ts_test_x7[-30:,:], axis=0)-np.nanmean(ts_test_x7[-30:,:], axis=0)[0],'o-',color='tomato',label='7x')
#   plt.plot(tt,np.nanmean(ts_test_x8[-30:,:], axis=0)-np.nanmean(ts_test_x8[-30:,:], axis=0)[0],'o-',color='red',label='8x')
#   plt.xticks(tt,['Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun'], fontsize=10)
#   plt.legend(ncol=1,fontsize='small',loc='upper left')
#   plt.ylim(-0.4,0.6)
#   plt.title('(c) Global SAT Response')
#   plt.ylabel('K')

# plot sat response outside arctic
# select arctic region
   [x1, x2, y1, y2] = data_process_f.find_lon_lat_index(0,360,-90,60,lon,lat)

# create mask array
   mask_var = (area/area).copy()

# ================================================================
# set mask_var
# ================================================================
# land-only
   dirname = '/data2/im2527/CESM-LE/8xCO2.B1850LENS.n21.f09_g16/atm/hist/'
   filename = '8xCO2.B1850LENS.n21.f09_g16.cam.h0.1904-12.nc'
   f = Dataset(dirname + filename, 'r')
   LANDFRAC = f.variables['LANDFRAC'][:,:,:].data.squeeze()
   f.close()

   mask_var = LANDFRAC.copy()
   mask_var[mask_var<1.] = -100000
   mask_var[mask_var==1.] = np.nan
   mask_var[mask_var<0.] = 1.

# simulated sat
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
           ts_sat_control_arctic[NT,NM] = np.nansum(sat_control[NT,NM,:y2,:]*area[:y2,:]*mask_var[:y2,:])/np.nansum(area[:y2,:]*mask_var[:y2,:])
           ts_sat_co2xp25_arctic[NT,NM] = np.nansum(sat_co2xp25[NT,NM,:y2,:]*area[:y2,:]*mask_var[:y2,:])/np.nansum(area[:y2,:]*mask_var[:y2,:])
           ts_sat_co2xp5_arctic[NT,NM] = np.nansum(sat_co2xp5[NT,NM,:y2,:]*area[:y2,:]*mask_var[:y2,:])/np.nansum(area[:y2,:]*mask_var[:y2,:])
           ts_sat_co2x1_arctic[NT,NM] = np.nansum(sat_co2x1[NT,NM,:y2,:]*area[:y2,:]*mask_var[:y2,:])/np.nansum(area[:y2,:]*mask_var[:y2,:])
           ts_sat_co2x2_arctic[NT,NM] = np.nansum(sat_co2x2[NT,NM,:y2,:]*area[:y2,:]*mask_var[:y2,:])/np.nansum(area[:y2,:]*mask_var[:y2,:])
           ts_sat_co2x3_arctic[NT,NM] = np.nansum(sat_co2x3[NT,NM,:y2,:]*area[:y2,:]*mask_var[:y2,:])/np.nansum(area[:y2,:]*mask_var[:y2,:])
           ts_sat_co2x4_arctic[NT,NM] = np.nansum(sat_co2x4[NT,NM,:y2,:]*area[:y2,:]*mask_var[:y2,:])/np.nansum(area[:y2,:]*mask_var[:y2,:])
           ts_sat_co2x5_arctic[NT,NM] = np.nansum(sat_co2x5[NT,NM,:y2,:]*area[:y2,:]*mask_var[:y2,:])/np.nansum(area[:y2,:]*mask_var[:y2,:])
           ts_sat_co2x6_arctic[NT,NM] = np.nansum(sat_co2x6[NT,NM,:y2,:]*area[:y2,:]*mask_var[:y2,:])/np.nansum(area[:y2,:]*mask_var[:y2,:])
           ts_sat_co2x7_arctic[NT,NM] = np.nansum(sat_co2x7[NT,NM,:y2,:]*area[:y2,:]*mask_var[:y2,:])/np.nansum(area[:y2,:]*mask_var[:y2,:])
           ts_sat_co2x8_arctic[NT,NM] = np.nansum(sat_co2x8[NT,NM,:y2,:]*area[:y2,:]*mask_var[:y2,:])/np.nansum(area[:y2,:]*mask_var[:y2,:])

   ts_test_tmp = ((ts_sat_co2xp25_arctic - ts_sat_control_arctic)).copy()
   ts_test_xp25 = ts_test_tmp.copy()
   ts_test_xp25[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_xp25[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2xp5_arctic - ts_sat_control_arctic)).copy()
   ts_test_xp5 = ts_test_tmp.copy()
   ts_test_xp5[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_xp5[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x2_arctic - ts_sat_control_arctic)).copy()
   ts_test_x2 = ts_test_tmp.copy()
   ts_test_x2[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x2[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x3_arctic - ts_sat_control_arctic)).copy()
   ts_test_x3 = ts_test_tmp.copy()
   ts_test_x3[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x3[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x4_arctic - ts_sat_control_arctic)).copy()
   ts_test_x4 = ts_test_tmp.copy()
   ts_test_x4[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x4[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x5_arctic - ts_sat_control_arctic)).copy()
   ts_test_x5 = ts_test_tmp.copy()
   ts_test_x5[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x5[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x6_arctic - ts_sat_control_arctic)).copy()
   ts_test_x6 = ts_test_tmp.copy()
   ts_test_x6[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x6[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x7_arctic - ts_sat_control_arctic)).copy()
   ts_test_x7 = ts_test_tmp.copy()
   ts_test_x7[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x7[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x8_arctic - ts_sat_control_arctic)).copy()
   ts_test_x8 = ts_test_tmp.copy()
   ts_test_x8[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x8[:,6:] = ts_test_tmp[:,0:6].copy()

   ts_interval = np.zeros((9,2,12))
   for III in range(12):
       ts_test = ts_test_xp25[-30:,III].copy()-ts_test_xp25[-30:,0].mean()
       ts_interval[0,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_xp5[-30:,III].copy()-ts_test_xp5[-30:,0].mean()
       ts_interval[1,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_x2[-30:,III].copy()-ts_test_x2[-30:,0].mean()
       ts_interval[2,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_x3[-30:,III].copy()-ts_test_x3[-30:,0].mean()
       ts_interval[3,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_x4[-30:,III].copy()-ts_test_x4[-30:,0].mean()
       ts_interval[4,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_x5[-30:,III].copy()-ts_test_x5[-30:,0].mean()
       ts_interval[5,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_x6[-30:,III].copy()-ts_test_x6[-30:,0].mean()
       ts_interval[6,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_x7[-30:,III].copy()-ts_test_x7[-30:,0].mean()
       ts_interval[7,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))
       ts_test = ts_test_x8[-30:,III].copy()-ts_test_x8[-30:,0].mean()
       ts_interval[8,:,III] = stats.t.interval(0.95, len(ts_test) - 1, loc=ts_test.mean(), scale= np.std(ts_test) / np.sqrt(len(ts_test)))

#   ax1 = fig.add_axes([0.55, 0.37, 0.36, 0.27])
#   ts_interval = (ts_interval)
#   for II in range(7):
#       for JJ in range(12):
#           plt.plot([tt[JJ]]*2,ts_interval[II+2,:,JJ],'k-', alpha=0.3)
#           plt.plot([tt[JJ]-0.1,tt[JJ]+0.1],[ts_interval[II+2,0,JJ]]*2,'k-',alpha=0.3)
#           plt.plot([tt[JJ]-0.1,tt[JJ]+0.1],[ts_interval[II+2,1,JJ]]*2,'k-',alpha=0.3)

#   plt.plot(tt,np.nanmean(ts_test_x2[-30:,:], axis=0)-np.nanmean(ts_test_x2[-30:,:], axis=0)[0],'o-',color='b',label='2x')
#   plt.plot(tt,np.nanmean(ts_test_x3[-30:,:], axis=0)-np.nanmean(ts_test_x3[-30:,:], axis=0)[0],'o-',color='dodgerblue',label='3x')
#   plt.plot(tt,np.nanmean(ts_test_x4[-30:,:], axis=0)-np.nanmean(ts_test_x4[-30:,:], axis=0)[0],'o-',color='springgreen',label='4x')
#   plt.plot(tt,np.nanmean(ts_test_x5[-30:,:], axis=0)-np.nanmean(ts_test_x5[-30:,:], axis=0)[0],'o-',color='gold',label='5x')
#   plt.plot(tt,np.nanmean(ts_test_x6[-30:,:], axis=0)-np.nanmean(ts_test_x6[-30:,:], axis=0)[0],'o-',color='orange',label='6x')
#   plt.plot(tt,np.nanmean(ts_test_x7[-30:,:], axis=0)-np.nanmean(ts_test_x7[-30:,:], axis=0)[0],'o-',color='tomato',label='7x')
#   plt.plot(tt,np.nanmean(ts_test_x8[-30:,:], axis=0)-np.nanmean(ts_test_x8[-30:,:], axis=0)[0],'o-',color='red',label='8x')
#   plt.xticks(tt,['Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun'], fontsize=10)
#   plt.legend(ncol=1,fontsize='small',loc='lower left')
#   plt.ylim(-0.6,0.2)
#   plt.title('(d) Extra-Arctic SAT Response')
#   plt.ylabel('K')

# read sic
   varname = 'ICEFRAC'
   year1 = 1850
   year2 = 1999
   year_N = int(year2 - year1 + 1)
   tt = np.linspace(1,year_N,year_N)
   tt = np.linspace(1,12,12)

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

   factor0 = 1./1000000000000.

   for NM in range(12):
       dirname = '/data1/yliang/co2_experiments/processed_data/'
       filename = varname + '_month' + str(NM+1) + '_mean_temp_output.nc'
       print(filename)
       f = Dataset(dirname + filename, 'r')
       sat_control[:,NM,:,:] = f.variables['var_co2x1'][:,:,:].data*factor0
       sat_co2xp25[:,NM,:,:] = f.variables['var_co2xp25'][:,:,:].data*factor0
       sat_co2xp5[:,NM,:,:] = f.variables['var_co2xp5'][:,:,:].data*factor0
       sat_co2x1[:,NM,:,:] = f.variables['var_co2x1'][:,:,:].data*factor0
       sat_co2x2[:,NM,:,:] = f.variables['var_co2x2'][:,:,:].data*factor0
       sat_co2x3[:,NM,:,:] = f.variables['var_co2x3'][:,:,:].data*factor0
       sat_co2x4[:,NM,:,:] = f.variables['var_co2x4'][:,:,:].data*factor0
       sat_co2x5[:,NM,:,:] = f.variables['var_co2x5'][:,:,:].data*factor0
       sat_co2x6[:,NM,:,:] = f.variables['var_co2x6'][:,:,:].data*factor0
       sat_co2x7[:,NM,:,:] = f.variables['var_co2x7'][:,:,:].data*factor0
       sat_co2x8[:,NM,:,:] = f.variables['var_co2x8'][:,:,:].data*factor0
       f.close()

   sat_control[sat_control<0.15*factor0] = 0.
   sat_co2x1[sat_co2x1<0.15*factor0] = 0.
   sat_co2x2[sat_co2x2<0.15*factor0] = 0.
   sat_co2x3[sat_co2x3<0.15*factor0] = 0.
   sat_co2x4[sat_co2x4<0.15*factor0] = 0.
   sat_co2x5[sat_co2x5<0.15*factor0] = 0.
   sat_co2x6[sat_co2x6<0.15*factor0] = 0.
   sat_co2x7[sat_co2x7<0.15*factor0] = 0.
   sat_co2x8[sat_co2x8<0.15*factor0] = 0.


   sat1_control = np.zeros((year_N,12,ny,nx))
   sat1_co2xp25 = np.zeros((year_N,12,ny,nx))
   sat1_co2xp5 = np.zeros((year_N,12,ny,nx))
   sat1_co2x1 = np.zeros((year_N,12,ny,nx))
   sat1_co2x2 = np.zeros((year_N,12,ny,nx))
   sat1_co2x3 = np.zeros((year_N,12,ny,nx))
   sat1_co2x4 = np.zeros((year_N,12,ny,nx))
   sat1_co2x5 = np.zeros((year_N,12,ny,nx))
   sat1_co2x6 = np.zeros((year_N,12,ny,nx))
   sat1_co2x7 = np.zeros((year_N,12,ny,nx))
   sat1_co2x8 = np.zeros((year_N,12,ny,nx))

   for NM in range(12):
       dirname = '/data1/yliang/co2_experiments/processed_data/'
       filename = 'SHFLX_month' + str(NM+1) + '_mean_temp_output.nc'
       print(filename)
       f = Dataset(dirname + filename, 'r')
       sat1_control[:,NM,:,:] = f.variables['var_co2x1'][:,:,:].data
       sat1_co2xp25[:,NM,:,:] = f.variables['var_co2xp25'][:,:,:].data
       sat1_co2xp5[:,NM,:,:] = f.variables['var_co2xp5'][:,:,:].data
       sat1_co2x1[:,NM,:,:] = f.variables['var_co2x1'][:,:,:].data
       sat1_co2x2[:,NM,:,:] = f.variables['var_co2x2'][:,:,:].data
       sat1_co2x3[:,NM,:,:] = f.variables['var_co2x3'][:,:,:].data
       sat1_co2x4[:,NM,:,:] = f.variables['var_co2x4'][:,:,:].data
       sat1_co2x5[:,NM,:,:] = f.variables['var_co2x5'][:,:,:].data
       sat1_co2x6[:,NM,:,:] = f.variables['var_co2x6'][:,:,:].data
       sat1_co2x7[:,NM,:,:] = f.variables['var_co2x7'][:,:,:].data
       sat1_co2x8[:,NM,:,:] = f.variables['var_co2x8'][:,:,:].data
       f.close()

   sat2_control = np.zeros((year_N,12,ny,nx))
   sat2_co2xp25 = np.zeros((year_N,12,ny,nx))
   sat2_co2xp5 = np.zeros((year_N,12,ny,nx))
   sat2_co2x1 = np.zeros((year_N,12,ny,nx))
   sat2_co2x2 = np.zeros((year_N,12,ny,nx))
   sat2_co2x3 = np.zeros((year_N,12,ny,nx))
   sat2_co2x4 = np.zeros((year_N,12,ny,nx))
   sat2_co2x5 = np.zeros((year_N,12,ny,nx))
   sat2_co2x6 = np.zeros((year_N,12,ny,nx))
   sat2_co2x7 = np.zeros((year_N,12,ny,nx))
   sat2_co2x8 = np.zeros((year_N,12,ny,nx))

   for NM in range(12):
       dirname = '/data1/yliang/co2_experiments/processed_data/'
       filename = 'LHFLX_month' + str(NM+1) + '_mean_temp_output.nc'
       print(filename)
       f = Dataset(dirname + filename, 'r')
       sat2_control[:,NM,:,:] = f.variables['var_co2x1'][:,:,:].data
       sat2_co2xp25[:,NM,:,:] = f.variables['var_co2xp25'][:,:,:].data
       sat2_co2xp5[:,NM,:,:] = f.variables['var_co2xp5'][:,:,:].data
       sat2_co2x1[:,NM,:,:] = f.variables['var_co2x1'][:,:,:].data
       sat2_co2x2[:,NM,:,:] = f.variables['var_co2x2'][:,:,:].data
       sat2_co2x3[:,NM,:,:] = f.variables['var_co2x3'][:,:,:].data
       sat2_co2x4[:,NM,:,:] = f.variables['var_co2x4'][:,:,:].data
       sat2_co2x5[:,NM,:,:] = f.variables['var_co2x5'][:,:,:].data
       sat2_co2x6[:,NM,:,:] = f.variables['var_co2x6'][:,:,:].data
       sat2_co2x7[:,NM,:,:] = f.variables['var_co2x7'][:,:,:].data
       sat2_co2x8[:,NM,:,:] = f.variables['var_co2x8'][:,:,:].data
       f.close()

   flx_control = sat2_control + sat1_control
   flx_co2xp25 = sat2_co2xp25 + sat1_co2xp25
   flx_co2xp5 = sat2_co2xp5 + sat1_co2xp5
   flx_co2x1 = sat2_co2x1 + sat1_co2x1
   flx_co2x2 = sat2_co2x2 + sat1_co2x2
   flx_co2x3 = sat2_co2x3 + sat1_co2x3
   flx_co2x4 = sat2_co2x4 + sat1_co2x4
   flx_co2x5 = sat2_co2x5 + sat1_co2x5
   flx_co2x6 = sat2_co2x6 + sat1_co2x6
   flx_co2x7 = sat2_co2x7 + sat1_co2x7
   flx_co2x8 = sat2_co2x8 + sat1_co2x8

# select arctic region
   [x1, x2, y1, y2] = data_process_f.find_lon_lat_index(0,360,60,89,lon,lat)

# create mask array
   mask_var = (area/area).copy()

# ================================================================
# set mask_var
# ================================================================
# land-only
   dirname = '/data2/im2527/CESM-LE/8xCO2.B1850LENS.n21.f09_g16/atm/hist/'
   filename = '8xCO2.B1850LENS.n21.f09_g16.cam.h0.1904-12.nc'
   f = Dataset(dirname + filename, 'r')
   LANDFRAC = f.variables['LANDFRAC'][:,:,:].data.squeeze()
   f.close()

   mask_var = LANDFRAC.copy()
   mask_var[mask_var<1.] = -100000
   mask_var[mask_var==1.] = np.nan
   mask_var[mask_var<0.] = 1.

   ts_flx_control_global = np.zeros((year_N,12))
   ts_flx_co2xp25_global = np.zeros((year_N,12))
   ts_flx_co2xp5_global = np.zeros((year_N,12))
   ts_flx_co2x1_global = np.zeros((year_N,12))
   ts_flx_co2x2_global = np.zeros((year_N,12))
   ts_flx_co2x3_global = np.zeros((year_N,12))
   ts_flx_co2x4_global = np.zeros((year_N,12))
   ts_flx_co2x5_global = np.zeros((year_N,12))
   ts_flx_co2x6_global = np.zeros((year_N,12))
   ts_flx_co2x7_global = np.zeros((year_N,12))
   ts_flx_co2x8_global = np.zeros((year_N,12))

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
           ts_flx_control_global[NT,NM] = np.nansum(flx_control[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_flx_co2xp25_global[NT,NM] = np.nansum(flx_co2xp25[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_flx_co2xp5_global[NT,NM] = np.nansum(flx_co2xp5[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_flx_co2x1_global[NT,NM] = np.nansum(flx_co2x1[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_flx_co2x2_global[NT,NM] = np.nansum(flx_co2x2[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_flx_co2x3_global[NT,NM] = np.nansum(flx_co2x3[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_flx_co2x4_global[NT,NM] = np.nansum(flx_co2x4[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_flx_co2x5_global[NT,NM] = np.nansum(flx_co2x5[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_flx_co2x6_global[NT,NM] = np.nansum(flx_co2x6[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_flx_co2x7_global[NT,NM] = np.nansum(flx_co2x7[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_flx_co2x8_global[NT,NM] = np.nansum(flx_co2x8[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])/np.nansum(area[y1:,:]*mask_var[y1:,:])

           ts_sat_control_arctic[NT,NM] = np.nansum(sat_control[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])#/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_sat_co2xp25_arctic[NT,NM] = np.nansum(sat_co2xp25[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])#/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_sat_co2xp5_arctic[NT,NM] = np.nansum(sat_co2xp5[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])#/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_sat_co2x1_arctic[NT,NM] = np.nansum(sat_co2x1[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])#/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_sat_co2x2_arctic[NT,NM] = np.nansum(sat_co2x2[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])#/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_sat_co2x3_arctic[NT,NM] = np.nansum(sat_co2x3[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])#/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_sat_co2x4_arctic[NT,NM] = np.nansum(sat_co2x4[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])#/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_sat_co2x5_arctic[NT,NM] = np.nansum(sat_co2x5[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])#/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_sat_co2x6_arctic[NT,NM] = np.nansum(sat_co2x6[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])#/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_sat_co2x7_arctic[NT,NM] = np.nansum(sat_co2x7[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])#/np.nansum(area[y1:,:]*mask_var[y1:,:])
           ts_sat_co2x8_arctic[NT,NM] = np.nansum(sat_co2x8[NT,NM,y1:,:]*area[y1:,:]*mask_var[y1:,:])#/np.nansum(area[y1:,:]*mask_var[y1:,:])

   ts_test_tmp = ((ts_sat_co2xp25_arctic - ts_sat_control_arctic)).copy()
   ts_test_xp25 = ts_test_tmp.copy()
   ts_test_xp25[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_xp25[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2xp5_arctic - ts_sat_control_arctic)).copy()
   ts_test_xp5 = ts_test_tmp.copy()
   ts_test_xp5[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_xp5[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x2_arctic - ts_sat_control_arctic)).copy()
   ts_test_x2 = ts_test_tmp.copy()
   ts_test_x2[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x2[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x3_arctic - ts_sat_control_arctic)).copy()
   ts_test_x3 = ts_test_tmp.copy()
   ts_test_x3[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x3[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x4_arctic - ts_sat_control_arctic)).copy()
   ts_test_x4 = ts_test_tmp.copy()
   ts_test_x4[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x4[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x5_arctic - ts_sat_control_arctic)).copy()
   ts_test_x5 = ts_test_tmp.copy()
   ts_test_x5[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x5[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x6_arctic - ts_sat_control_arctic)).copy()
   ts_test_x6 = ts_test_tmp.copy()
   ts_test_x6[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x6[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x7_arctic - ts_sat_control_arctic)).copy()
   ts_test_x7 = ts_test_tmp.copy()
   ts_test_x7[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x7[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_sat_co2x8_arctic - ts_sat_control_arctic)).copy()
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

#   ax1 = fig.add_axes([0.1, 0.04, 0.36, 0.27])
#   ts_interval = abs(ts_interval)
#   for II in range(7):
#       for JJ in range(12):
#           plt.plot([tt[JJ]]*2,-ts_interval[II+2,:,JJ],'k-', alpha=0.3)
#           plt.plot([tt[JJ]-0.1,tt[JJ]+0.1],[-ts_interval[II+2,0,JJ]]*2,'k-',alpha=0.3)
#           plt.plot([tt[JJ]-0.1,tt[JJ]+0.1],[-ts_interval[II+2,1,JJ]]*2,'k-',alpha=0.3)

#   plt.plot(tt,np.nanmean(ts_test_x2[-30:,:], axis=0)-np.nanmean(ts_test_x2[-30:,:], axis=0)[0]*0,'o-',color='b',label='2x')
#   plt.plot(tt,np.nanmean(ts_test_x3[-30:,:], axis=0)-np.nanmean(ts_test_x3[-30:,:], axis=0)[0]*0,'o-',color='dodgerblue',label='3x')
#   plt.plot(tt,np.nanmean(ts_test_x4[-30:,:], axis=0)-np.nanmean(ts_test_x4[-30:,:], axis=0)[0]*0,'o-',color='springgreen',label='4x')
#   plt.plot(tt,np.nanmean(ts_test_x5[-30:,:], axis=0)-np.nanmean(ts_test_x5[-30:,:], axis=0)[0]*0,'o-',color='gold',label='5x')
#   plt.plot(tt,np.nanmean(ts_test_x6[-30:,:], axis=0)-np.nanmean(ts_test_x6[-30:,:], axis=0)[0]*0,'o-',color='orange',label='6x')
#   plt.plot(tt,np.nanmean(ts_test_x7[-30:,:], axis=0)-np.nanmean(ts_test_x7[-30:,:], axis=0)[0]*0,'o-',color='tomato',label='7x')
#   plt.plot(tt,np.nanmean(ts_test_x8[-30:,:], axis=0)-np.nanmean(ts_test_x8[-30:,:], axis=0)[0]*0,'o-',color='red',label='8x')
#   plt.xticks(tt,['Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun'], fontsize=10)
#   plt.legend(ncol=2,fontsize='small',loc='lower left')
#   plt.ylim(-1,13)
#   plt.title('(e) Arctic Land-only SIE Response')
#   plt.ylabel('km$^2$ x 10$^6$')

# heat flux
   tt = np.linspace(1,12,12)
   ts_test_tmp = ((ts_flx_co2xp25_global - ts_flx_control_global)).copy()
   ts_test_xp25 = ts_test_tmp.copy()
   ts_test_xp25[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_xp25[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_flx_co2xp5_global - ts_flx_control_global)).copy()
   ts_test_xp5 = ts_test_tmp.copy()
   ts_test_xp5[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_xp5[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_flx_co2x2_global - ts_flx_control_global)).copy()
   ts_test_x2 = ts_test_tmp.copy()
   ts_test_x2[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x2[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_flx_co2x3_global - ts_flx_control_global)).copy()
   ts_test_x3 = ts_test_tmp.copy()
   ts_test_x3[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x3[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_flx_co2x4_global - ts_flx_control_global)).copy()
   ts_test_x4 = ts_test_tmp.copy()
   ts_test_x4[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x4[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_flx_co2x5_global - ts_flx_control_global)).copy()
   ts_test_x5 = ts_test_tmp.copy()
   ts_test_x5[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x5[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_flx_co2x6_global - ts_flx_control_global)).copy()
   ts_test_x6 = ts_test_tmp.copy()
   ts_test_x6[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x6[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_flx_co2x7_global - ts_flx_control_global)).copy()
   ts_test_x7 = ts_test_tmp.copy()
   ts_test_x7[:,0:6] = ts_test_tmp[:,6:].copy()
   ts_test_x7[:,6:] = ts_test_tmp[:,0:6].copy()
   ts_test_tmp = ((ts_flx_co2x8_global - ts_flx_control_global)).copy()
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

   ax1 = fig.add_axes([0.3, 0.04, 0.36, 0.27])
   ts_interval = (ts_interval)
   for II in range(7):
       for JJ in range(12):
           plt.plot([tt[JJ]]*2,ts_interval[II+2,:,JJ],'k-', alpha=0.3)
           plt.plot([tt[JJ]-0.1,tt[JJ]+0.1],[ts_interval[II+2,0,JJ]]*2,'k-',alpha=0.3)
           plt.plot([tt[JJ]-0.1,tt[JJ]+0.1],[ts_interval[II+2,1,JJ]]*2,'k-',alpha=0.3)

   plt.plot(tt,np.nanmean(ts_test_x2[-30:,:], axis=0)-np.nanmean(ts_test_x2[-30:,:], axis=0)[0]*0,'o-',color='b',label='2x')
   plt.plot(tt,np.nanmean(ts_test_x3[-30:,:], axis=0)-np.nanmean(ts_test_x3[-30:,:], axis=0)[0]*0,'o-',color='dodgerblue',label='3x')
   plt.plot(tt,np.nanmean(ts_test_x4[-30:,:], axis=0)-np.nanmean(ts_test_x4[-30:,:], axis=0)[0]*0,'o-',color='springgreen',label='4x')
   plt.plot(tt,np.nanmean(ts_test_x5[-30:,:], axis=0)-np.nanmean(ts_test_x5[-30:,:], axis=0)[0]*0,'o-',color='gold',label='5x')
   plt.plot(tt,np.nanmean(ts_test_x6[-30:,:], axis=0)-np.nanmean(ts_test_x6[-30:,:], axis=0)[0]*0,'o-',color='orange',label='6x')
   plt.plot(tt,np.nanmean(ts_test_x7[-30:,:], axis=0)-np.nanmean(ts_test_x7[-30:,:], axis=0)[0]*0,'o-',color='tomato',label='7x')
   plt.plot(tt,np.nanmean(ts_test_x8[-30:,:], axis=0)-np.nanmean(ts_test_x8[-30:,:], axis=0)[0]*0,'o-',color='red',label='8x')
   plt.xticks(tt,['Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun'], fontsize=10)
   plt.legend(ncol=2,fontsize='small',loc='upper left')
#   plt.ylim(-5,35)
   plt.ylabel('W/m$^2$')
   plt.title('(c) Ocean-only Turbulent Heat Flux Response')


   plt.savefig('trend_tmp_plot.jpg', format='jpeg', dpi=200)

   plt.show()

   sys.exit()


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

   plt.figure(5)
   plt.plot(ttt,ts_max_xp25-ts_min_xp25,'-',color='blue',label='co2x0.25')
   plt.plot(ttt,ts_max_xp5-ts_min_xp5,'-',color='blueviolet',label='co2x0.5')
   plt.plot(ttt,ts_max_x2-ts_min_x2,'-',color='orange',label='co2x2')
   plt.plot(ttt,ts_max_x3-ts_min_x3,'-',color='gold',label='co2x3')
   plt.plot(ttt,ts_max_x4-ts_min_x4,'-',color='magenta',label='co2x4')
   plt.plot(ttt,ts_max_x5-ts_min_x5,'-',color='brown',label='co2x5')
   plt.plot(ttt,ts_max_x6-ts_min_x6,'-',color='tomato',label='co2x6')
   plt.plot(ttt,ts_max_x7-ts_min_x7,'-',color='salmon',label='co2x7')
   plt.plot(ttt,ts_max_x8-ts_min_x8,'-',color='red',label='co2x8')
   plt.xticks(np.linspace(10,150,15), fontsize=8)
   plt.xlim(31,150)
   plt.ylim(1,4)
#   plt.legend()

# seasonal cycle amplitude plot

   plt.figure(6)

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

   for II in range(9):
       plt.plot([ttt[II]]*2,ts_interval[II,:],'k-')
       plt.plot([ttt[II]-0.1,ttt[II]+0.1],[ts_interval[II,0]]*2,'k-')
       plt.plot([ttt[II]-0.1,ttt[II]+0.1],[ts_interval[II,1]]*2,'k-')

   plt.plot([1]*2,[(ts_max_xp25-ts_min_xp25)[-30:].mean()]*2,'o',color='blue',label='co2x0.25')
   plt.plot([2]*2,[(ts_max_xp5-ts_min_xp5)[-30:].mean()]*2,'o',color='blueviolet',label='co2x0.5')
   plt.plot([3]*2,[(ts_max_x2-ts_min_x2)[-30:].mean()]*2,'o',color='orange',label='co2x2')
   plt.plot([4]*2,[(ts_max_x3-ts_min_x3)[-30:].mean()]*2,'o',color='gold',label='co2x3')
   plt.plot([5]*2,[(ts_max_x4-ts_min_x4)[-30:].mean()]*2,'o',color='magenta',label='co2x4')
   plt.plot([6]*2,[(ts_max_x5-ts_min_x5)[-30:].mean()]*2,'o',color='brown',label='co2x5')
   plt.plot([7]*2,[(ts_max_x6-ts_min_x6)[-30:].mean()]*2,'o',color='tomato',label='co2x6')
   plt.plot([8]*2,[(ts_max_x7-ts_min_x7)[-30:].mean()]*2,'o',color='salmon',label='co2x7')
   plt.plot([9]*2,[(ts_max_x8-ts_min_x8)[-30:].mean()]*2,'o',color='red',label='co2x8')

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
       plt.plot([ttt[II+2]-0.1,ttt[II+2]+0.1],[ts_interval[II,0]]*2,'k-')
       plt.plot([ttt[II+2]-0.1,ttt[II+2]+0.1],[ts_interval[II,1]]*2,'k-')

   plt.plot([3]*2,[(tse_max_x2-tse_min_x2)[-30:].mean()]*2,'^',color='orange')
   plt.plot([4]*2,[(tse_max_x3-tse_min_x3)[-30:].mean()]*2,'^',color='gold')
   plt.plot([5]*2,[(tse_max_x4-tse_min_x4)[-30:].mean()]*2,'^',color='magenta')
   plt.plot([6]*2,[(tse_max_x5-tse_min_x5)[-30:].mean()]*2,'^',color='brown')
   plt.plot([7]*2,[(tse_max_x6-tse_min_x6)[-30:].mean()]*2,'^',color='tomato')

   plt.xticks(np.linspace(1,9,9),['co2x0.25','co2x0.5','co2x2','co2x3','co2x4','co2x5','co2x6','co2x7','co2x8'], rotation=60)


   plt.savefig('trend_tmp_plot.jpg', format='jpeg', dpi=200)

   plt.show()



