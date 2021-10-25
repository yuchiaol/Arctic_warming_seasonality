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

sys.path.append('/home/yliang/lib/python_functions/data_process/')
import data_process_f

# ================================================================
# define functions 
# ================================================================
def perform_ttest_here(exp1_var,exp2_var,ny,nx,sig_level):
    ttest_map = np.zeros((ny,nx))*np.nan
    pvalue_map = np.zeros((ny,nx))*np.nan
    for JJ in range(ny):
        for II in range(nx):
            [xxx, pvalue] = stats.ttest_ind(exp1_var[:,JJ,II],exp2_var[:,JJ,II])
            if pvalue < sig_level:
               ttest_map[JJ,II] = 1.
            pvalue_map[JJ,II] = pvalue

    return ttest_map, pvalue_map

def plot_box(ax,lon1,lon2,lat1,lat2, color_txt):

    ax.plot(np.linspace(lon1,lon1,100), np.linspace(lat1, lat2, 100), transform=ccrs.PlateCarree(), color=color_txt, linewidth=0.6)
    ax.plot(np.linspace(lon2,lon2,100), np.linspace(lat1, lat2, 100), transform=ccrs.PlateCarree(), color=color_txt, linewidth=0.6)
    ax.plot(np.linspace(lon1,lon2,100), np.linspace(lat1, lat1, 100), transform=ccrs.PlateCarree(), color=color_txt, linewidth=0.6)
    ax.plot(np.linspace(lon1,lon2,100), np.linspace(lat2, lat2, 100), transform=ccrs.PlateCarree(), color=color_txt, linewidth=0.6)

def plot_here(lat,lev,map_2d,ttest_map_in,clevel):

    im1 = plt.contourf(lat, (lev), map_2d*ttest_map_in, levels=clevel, extend='both', cmap='RdBu_r')
#    map_2d_tmp = map_2d.copy()
#    sig_map_tmp = ttest_map_in.copy()
#    sig_map = sig_map_tmp.copy()
#    for II in range(len(lev[:])):
#        plt.plot(lat[:],(lev[II]*sig_map[II,:]),'ko', markersize=0.5)
 
#    plt.plot([60,90],[500,500],'c-')
    plt.xlim(20,90)
    plt.xticks([20,40,60,80],['20N','40N','60N','80N'], fontsize=9)
#    plt.yticks((np.array([1000,850,600,500,300,200,150,100,70,50,30,20,10])),['1000','850','600','500','300','200','150','100','70','50','30','20','10'])
    plt.yticks((np.array([1000,950,900,850,800,700,600,500,400,300,200,100,10])),['1000','950','900','850','800','700','600','500','400','300','200','100','10'], fontsize=9)
    plt.gca().invert_yaxis()

    plt.xlim(20,90)

    return im1

if flag_run == 1:
# ================================================================
# read simulations
# ================================================================
# read sat
   varname = 'T'
   year1 = 1850
   year2 = 1999
   year_N = int(year2 - year1 + 1)
   tt = np.linspace(year1,year2,year_N)  
#   lat_sl = 95
   lat_sl = 0

# read grid basics
   dirname = '/data1/yliang/co2_experiments/processed_data/'
   filename = varname + '_annual_zonal_mean_temp_output.nc'
   f = Dataset(dirname + filename, 'r')
   lev = f.variables['lev'][:].data
   lat = f.variables['lat'][lat_sl:].data
#   var_co2xp25 = f.variables['var_co2xp25'][-30:,:,lat_sl:].data
#   var_co2xp5 = f.variables['var_co2xp5'][-30:,:,lat_sl:].data
   var_co2x1 = f.variables['var_co2x1'][-30:,:,lat_sl:].data
   var_co2x2 = f.variables['var_co2x2'][-30:,:,lat_sl:].data
   var_co2x3 = f.variables['var_co2x3'][-30:,:,lat_sl:].data
   var_co2x4 = f.variables['var_co2x4'][-30:,:,lat_sl:].data
   var_co2x5 = f.variables['var_co2x5'][-30:,:,lat_sl:].data
   var_co2x6 = f.variables['var_co2x6'][-30:,:,lat_sl:].data
   var_co2x7 = f.variables['var_co2x7'][-30:,:,lat_sl:].data
   var_co2x8 = f.variables['var_co2x8'][-30:,:,lat_sl:].data
   f.close()

#   var_co2xp25[abs(var_co2xp25)>1.e20] = np.nan
#   var_co2xp5[abs(var_co2xp5)>1.e20] = np.nan
   var_co2x1[abs(var_co2x1)>1.e20] = np.nan
   var_co2x2[abs(var_co2x2)>1.e20] = np.nan
   var_co2x3[abs(var_co2x3)>1.e20] = np.nan
   var_co2x4[abs(var_co2x4)>1.e20] = np.nan
   var_co2x5[abs(var_co2x5)>1.e20] = np.nan
   var_co2x6[abs(var_co2x6)>1.e20] = np.nan
   var_co2x7[abs(var_co2x7)>1.e20] = np.nan
   var_co2x8[abs(var_co2x8)>1.e20] = np.nan

   ny = len(lat)
   nz = len(lev)
   nt = var_co2x5.shape[0]

# ================================================================
# calculate AAF
# ================================================================
   dirname = '/data1/yliang/co2_experiments/processed_data/'
   filename = 'TREFHT_annual_mean_temp_output.nc'
   f = Dataset(dirname + filename, 'r')
   lon = f.variables['lon'][:].data
   lat = f.variables['lat'][:].data
   area = data_process_f.area_calculate_nonuniform(lon,lat)
   f.close()
 
#   aa_co2xp25 = np.zeros((nt,nz))
#   aa_co2xp5 = np.zeros((nt,nz))
   aa_co2x2 = np.zeros((nt,nz))
   aa_co2x3 = np.zeros((nt,nz))
   aa_co2x4 = np.zeros((nt,nz))
   aa_co2x5 = np.zeros((nt,nz))
   aa_co2x6 = np.zeros((nt,nz))
   aa_co2x7 = np.zeros((nt,nz))
   aa_co2x8 = np.zeros((nt,nz))

   lat_arctic = 160

#   a_co2xp25 = np.zeros((nt,nz))
#   a_co2xp5 = np.zeros((nt,nz))
   a_co2x2 = np.zeros((nt,nz))
   a_co2x3 = np.zeros((nt,nz))
   a_co2x4 = np.zeros((nt,nz))
   a_co2x5 = np.zeros((nt,nz))
   a_co2x6 = np.zeros((nt,nz))
   a_co2x7 = np.zeros((nt,nz))
   a_co2x8 = np.zeros((nt,nz))

   for NT in range(nt):
       for KK in range(nz):
#           a_co2xp25[NT,KK] = np.nanmean((var_co2xp25[NT,KK,lat_arctic:]-var_co2x1[NT,KK,lat_arctic:]))
#           a_co2xp5[NT,KK] = np.nanmean((var_co2xp5[NT,KK,lat_arctic:]-var_co2x1[NT,KK,lat_arctic:]))
           a_co2x2[NT,KK] = np.nanmean((var_co2x2[NT,KK,lat_arctic:]-var_co2x1[NT,KK,lat_arctic:]))
           a_co2x3[NT,KK] = np.nanmean((var_co2x3[NT,KK,lat_arctic:]-var_co2x1[NT,KK,lat_arctic:]))
           a_co2x4[NT,KK] = np.nanmean((var_co2x4[NT,KK,lat_arctic:]-var_co2x1[NT,KK,lat_arctic:]))
           a_co2x5[NT,KK] = np.nanmean((var_co2x5[NT,KK,lat_arctic:]-var_co2x1[NT,KK,lat_arctic:]))
           a_co2x6[NT,KK] = np.nanmean((var_co2x6[NT,KK,lat_arctic:]-var_co2x1[NT,KK,lat_arctic:]))
           a_co2x7[NT,KK] = np.nanmean((var_co2x7[NT,KK,lat_arctic:]-var_co2x1[NT,KK,lat_arctic:]))
           a_co2x8[NT,KK] = np.nanmean((var_co2x8[NT,KK,lat_arctic:]-var_co2x1[NT,KK,lat_arctic:]))

# ================================================================
# perform statistical significance
# ================================================================
   sig_level = 0.05

#   test_map1 = var_co2xp25.copy()
#   test_map2 = var_co2x1.copy()
#   [ttest_map_xp25, pvalue_map] = perform_ttest_here(test_map1,test_map2,nz,ny,sig_level)

#   test_map1 = var_co2xp5.copy()
#   test_map2 = var_co2x1.copy()
#   [ttest_map_xp5, pvalue_map] = perform_ttest_here(test_map1,test_map2,nz,ny,sig_level)

   test_map1 = var_co2x2.copy()
   test_map2 = var_co2x1.copy()
   [ttest_map_x2, pvalue_map] = perform_ttest_here(test_map1,test_map2,nz,ny,sig_level)

   test_map1 = var_co2x3.copy()
   test_map2 = var_co2x1.copy()
   [ttest_map_x3, pvalue_map] = perform_ttest_here(test_map1,test_map2,nz,ny,sig_level)

   test_map1 = var_co2x4.copy()
   test_map2 = var_co2x1.copy()
   [ttest_map_x4, pvalue_map] = perform_ttest_here(test_map1,test_map2,nz,ny,sig_level)

   test_map1 = var_co2x5.copy()
   test_map2 = var_co2x1.copy()
   [ttest_map_x5, pvalue_map] = perform_ttest_here(test_map1,test_map2,nz,ny,sig_level)

   test_map1 = var_co2x6.copy()
   test_map2 = var_co2x1.copy()
   [ttest_map_x6, pvalue_map] = perform_ttest_here(test_map1,test_map2,nz,ny,sig_level)

   test_map1 = var_co2x7.copy()
   test_map2 = var_co2x1.copy()
   [ttest_map_x7, pvalue_map] = perform_ttest_here(test_map1,test_map2,nz,ny,sig_level)

   test_map1 = var_co2x8.copy()
   test_map2 = var_co2x1.copy()
   [ttest_map_x8, pvalue_map] = perform_ttest_here(test_map1,test_map2,nz,ny,sig_level)

# ================================================================
# plot figures
# ================================================================
if True:
   
   plt.close('all')

   fig = plt.figure()
   fig.set_size_inches(10, 10, forward=True)

# plot sat
   clevel = np.linspace(-15,15,41)

#   fig.add_axes([0.05, 0.6, 0.17, 0.32])
#   map_2d = np.nanmean(var_co2x8-var_co2x1, axis=0)
#   im = plot_here(lat,lev,map_2d,ttest_map_xp25,clevel)
#   plt.title('(a) CO2x8', fontsize=12)  

#   fig.add_axes([0.27, 0.6, 0.17, 0.32])
#   map_2d = np.nanmean(var_co2xp5-var_co2x1, axis=0)
#   im = plot_here(lat,lev,map_2d,ttest_map_xp25,clevel)
#   plt.title('(b) CO2x0.5', fontsize=12)

   fig.add_axes([0.05, 0.6, 0.175, 0.3])
   map_2d = np.nanmean(var_co2x2-var_co2x1, axis=0)
   im = plot_here(lat,lev,map_2d,ttest_map_x2,clevel)
   plt.title('(a) 2xCO2', fontsize=12)

   fig.add_axes([0.28, 0.6, 0.175, 0.3])
   map_2d = np.nanmean(var_co2x3-var_co2x1, axis=0)
   im = plot_here(lat,lev,map_2d,ttest_map_x3,clevel)
   plt.title('(b) 3xCO2', fontsize=12)

   fig.add_axes([0.51, 0.6, 0.175, 0.3])
   map_2d = np.nanmean(var_co2x4-var_co2x1, axis=0)
   im = plot_here(lat,lev,map_2d,ttest_map_x4,clevel)
   plt.title('(c) 4xCO2', fontsize=12)

   fig.add_axes([0.74, 0.6, 0.175, 0.3])
   map_2d = np.nanmean(var_co2x5-var_co2x1, axis=0)
   im = plot_here(lat,lev,map_2d,ttest_map_x5,clevel)
   plt.title('(d) 5xCO2', fontsize=12)

   fig.add_axes([0.05, 0.22, 0.175, 0.3])
   map_2d = np.nanmean(var_co2x6-var_co2x1, axis=0)
   im = plot_here(lat,lev,map_2d,ttest_map_x6,clevel)
   plt.title('(e) 6xCO2', fontsize=12)

   fig.add_axes([0.28, 0.22, 0.175, 0.3])
   map_2d = np.nanmean(var_co2x7-var_co2x1, axis=0)
   im = plot_here(lat,lev,map_2d,ttest_map_x7,clevel)
   plt.title('(f) 7xCO2', fontsize=12)

   fig.add_axes([0.51, 0.22, 0.175, 0.3])
   map_2d = np.nanmean(var_co2x8-var_co2x1, axis=0)
   im = plot_here(lat,lev,map_2d,ttest_map_x7,clevel)
   plt.title('(g) 8xCO2', fontsize=12)

   cbaxes = fig.add_axes([0.05, 0.165, 0.865, 0.01])
   cbar = plt.colorbar(im, cax=cbaxes, orientation='horizontal', ticks=np.linspace(clevel[0],clevel[-1],11))
   cbar.set_label('K', rotation=0)

   fig.add_axes([0.74, 0.22, 0.175, 0.3])
   factor = 1.
#   plt.plot(np.nanmean(a_co2xp25, axis=0)*factor,lev,'-',color='blue',label='co2x0.25')
#   plt.plot(np.nanmean(a_co2xp5, axis=0)*factor,lev,'-',color='blueviolet',label='co2x0.5')
#   plt.plot(np.nansum(aa_co2x2, axis=0),lev,'-',color='rosybrown')
   plt.plot(np.nanmean(a_co2x2, axis=0)*factor,lev,'-',color='b',label='2x')
   plt.plot(np.nanmean(a_co2x3, axis=0)*factor,lev,'-',color='dodgerblue',label='3x')
   plt.plot(np.nanmean(a_co2x4, axis=0)*factor,lev,'-',color='springgreen',label='4x')
   plt.plot(np.nanmean(a_co2x5, axis=0)*factor,lev,'-',color='gold',label='5x')
   plt.plot(np.nanmean(a_co2x6, axis=0)*factor,lev,'-',color='orange',label='6x')
   plt.plot(np.nanmean(a_co2x7, axis=0)*factor,lev,'-',color='tomato',label='7x')
   plt.plot(np.nanmean(a_co2x8, axis=0)*factor,lev,'-',color='red',label='8x')
   plt.yticks((np.array([1000,950,900,850,800,700,600,500,400,300,200,100,10])),['1000','950','900','850','800','700','600','500','400','300','200','100','10'], fontsize=9)
   plt.xticks([-5,0,5,10,15,20], fontsize=9)
   plt.legend(fontsize='small',ncol=1)
   plt.axis([-5, 22, 10, 1000])
   plt.gca().invert_yaxis()
   plt.title('(h) Arctic T Response')  

   plt.savefig('trend_tmp_plot.jpg', format='jpeg', dpi=200)

   plt.show()

   sys.exit()




