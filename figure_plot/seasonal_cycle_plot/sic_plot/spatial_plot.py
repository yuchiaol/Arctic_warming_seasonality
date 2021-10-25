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

def plot_here(ax1,lon_sel,lat_sel,sat_ctr_last,sat_ctr_first,ttest_map_ctr,clevel):

    ax1.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())
    map_2d = (np.nanmean(sat_ctr_last, axis=0)-np.nanmean(sat_ctr_first, axis=0))*1
    [x_out, lon_x] = data_process_f.extend_longitude(map_2d,lat_sel,lon_sel)
    im1 = ax1.contourf(lon_x, lat_sel, x_out, levels=clevel, extend='both', transform=ccrs.PlateCarree(), cmap='RdBu')
    sig_map = ttest_map_ctr.copy()
    ax1.contourf(lon,lat_sel,sig_map,colors = 'none', hatches=['//'], transform=ccrs.PlateCarree())
    ax1.coastlines('110m',color='k',linewidth=0.7)
    ax1.set_boundary(circle, transform=ax1.transAxes)
    ax1.set_aspect('auto')

    return im1

def plot_here2(ax1,lon_sel,lat_sel,sat_ctr_last,sat_ctr_first,clevel):

    ax1.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())
    map_2d = (np.nanmean(sat_ctr_last, axis=0)-np.nanmean(sat_ctr_first, axis=0))*1
    [x_out, lon_x] = data_process_f.extend_longitude(map_2d,lat_sel,lon_sel)
    im1 = ax1.contourf(lon_x, lat_sel, x_out, levels=clevel, extend='both', transform=ccrs.PlateCarree(), cmap='RdBu')
    ax1.coastlines('110m',color='k',linewidth=0.7)
    ax1.set_boundary(circle, transform=ax1.transAxes)
    ax1.set_aspect('auto')

    return im1

if flag_run == 1:
# ================================================================
# read simulations
# ================================================================
# read sat
   varname = 'ICEFRAC'
   year1 = 1850
   year2 = 1999
   year_N = int(year2 - year1 + 1)
   tt = np.linspace(1,year_N,year_N)

   lat_sel = 160

# read grid basics
   dirname = '/data1/yliang/co2_experiments/processed_data/'
   filename = varname + '_annual_mean_temp_output.nc'
   f = Dataset(dirname + filename, 'r')
   lon = f.variables['lon'][:].data
   lat = f.variables['lat'][lat_sel:].data
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
       sat_control[:,NM,:,:] = f.variables['var_co2x1'][:,lat_sel:,:].data*factor0
       sat_co2xp25[:,NM,:,:] = f.variables['var_co2xp25'][:,lat_sel:,:].data*factor0
       sat_co2xp5[:,NM,:,:] = f.variables['var_co2xp5'][:,lat_sel:,:].data*factor0
       sat_co2x1[:,NM,:,:] = f.variables['var_co2x1'][:,lat_sel:,:].data*factor0
       sat_co2x2[:,NM,:,:] = f.variables['var_co2x2'][:,lat_sel:,:].data*factor0
       sat_co2x3[:,NM,:,:] = f.variables['var_co2x3'][:,lat_sel:,:].data*factor0
       sat_co2x4[:,NM,:,:] = f.variables['var_co2x4'][:,lat_sel:,:].data*factor0
       sat_co2x5[:,NM,:,:] = f.variables['var_co2x5'][:,lat_sel:,:].data*factor0
       sat_co2x6[:,NM,:,:] = f.variables['var_co2x6'][:,lat_sel:,:].data*factor0
       sat_co2x7[:,NM,:,:] = f.variables['var_co2x7'][:,lat_sel:,:].data*factor0
       sat_co2x8[:,NM,:,:] = f.variables['var_co2x8'][:,lat_sel:,:].data*factor0
       f.close()

#   sat_control[sat_control<0.15*factor0] = 0.
#   sat_co2x1[sat_co2x1<0.15*factor0] = 0.
#   sat_co2x2[sat_co2x2<0.15*factor0] = 0.
#   sat_co2x3[sat_co2x3<0.15*factor0] = 0.
#   sat_co2x4[sat_co2x4<0.15*factor0] = 0.
#   sat_co2x5[sat_co2x5<0.15*factor0] = 0.
#   sat_co2x6[sat_co2x6<0.15*factor0] = 0.
#   sat_co2x7[sat_co2x7<0.15*factor0] = 0.
#   sat_co2x8[sat_co2x8<0.15*factor0] = 0.


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
       esat_co2x1[:,NM,:,:] = f.variables['var_co2x1'][-30:,lat_sel:,:].data*factor0
       esat_co2x2[:,NM,:,:] = f.variables['var_co2x2'][-30:,lat_sel:,:].data*factor0
       esat_co2x3[:,NM,:,:] = f.variables['var_co2x3'][-30:,lat_sel:,:].data*factor0
       esat_co2x4[:,NM,:,:] = f.variables['var_co2x4'][-30:,lat_sel:,:].data*factor0
       esat_co2x5[:,NM,:,:] = f.variables['var_co2x5'][-30:,lat_sel:,:].data*factor0
       esat_co2x6[:,NM,:,:] = f.variables['var_co2x6'][-30:,lat_sel:,:].data*factor0
       f.close()

#   esat_co2x1[esat_co2x1<0.15*factor0] = 0.
#   esat_co2x2[esat_co2x2<0.15*factor0] = 0.
#   esat_co2x3[esat_co2x3<0.15*factor0] = 0.
#   esat_co2x4[esat_co2x4<0.15*factor0] = 0.
#   esat_co2x5[esat_co2x5<0.15*factor0] = 0.
#   esat_co2x6[esat_co2x6<0.15*factor0] = 0.

   n_e = esat_co2x6.shape[0]

# ================================================================
# plot figures
# ================================================================
if True:

   month_sel = 8
   sig_level = 0.05
   clevel = np.linspace(-0.005,0.005,21)

   plt.close('all')
   fig = plt.figure(1)
   fig.set_size_inches(10, 10, forward=True)

   theta = np.linspace(0, 2*np.pi, 100)
   center, radius = [0.5, 0.5], 0.5
   verts = np.vstack([np.sin(theta), np.cos(theta)]).T
   circle = mpath.Path(verts * radius + center)

   ax1 = fig.add_axes([0.05, 0.65, 0.22, 0.22], projection=ccrs.Orthographic(0, 90))
   sat_ctr_last = sat_co2x2[-30:,month_sel,:,:].copy()*area
   sat_ctr_first = sat_co2x1[-30:,month_sel,:,:].copy()*area
   [ttest_map, xxx] = perform_ttest_here(sat_ctr_last,sat_ctr_first,ny,nx,sig_level)
   im1 = plot_here(ax1,lon,lat,sat_ctr_last,sat_ctr_first,ttest_map,clevel)
   plt.title('(a) 2xCO2', fontsize=10)

   ax1 = fig.add_axes([0.29, 0.65, 0.22, 0.22], projection=ccrs.Orthographic(0, 90))
   sat_ctr_last = sat_co2x3[-30:,month_sel,:,:].copy()*area
   sat_ctr_first = sat_co2x1[-30:,month_sel,:,:].copy()*area
   [ttest_map, xxx] = perform_ttest_here(sat_ctr_last,sat_ctr_first,ny,nx,sig_level)
   im1 = plot_here(ax1,lon,lat,sat_ctr_last,sat_ctr_first,ttest_map,clevel)
   plt.title('(b) 3xCO2', fontsize=10)

   ax1 = fig.add_axes([0.53, 0.65, 0.22, 0.22], projection=ccrs.Orthographic(0, 90))
   sat_ctr_last = sat_co2x4[-30:,month_sel,:,:].copy()*area
   sat_ctr_first = sat_co2x1[-30:,month_sel,:,:].copy()*area
   [ttest_map, xxx] = perform_ttest_here(sat_ctr_last,sat_ctr_first,ny,nx,sig_level)
   im1 = plot_here(ax1,lon,lat,sat_ctr_last,sat_ctr_first,ttest_map,clevel)
   plt.title('(c) 4xCO2', fontsize=10)

   ax1 = fig.add_axes([0.77, 0.65, 0.22, 0.22], projection=ccrs.Orthographic(0, 90))
   sat_ctr_last = sat_co2x5[-30:,month_sel,:,:].copy()*area
   sat_ctr_first = sat_co2x1[-30:,month_sel,:,:].copy()*area
   [ttest_map, xxx] = perform_ttest_here(sat_ctr_last,sat_ctr_first,ny,nx,sig_level)
   im1 = plot_here(ax1,lon,lat,sat_ctr_last,sat_ctr_first,ttest_map,clevel)
   plt.title('(d) 5xCO2', fontsize=10)

   ax1 = fig.add_axes([0.05, 0.38, 0.22, 0.22], projection=ccrs.Orthographic(0, 90))
   sat_ctr_last = sat_co2x6[-30:,month_sel,:,:].copy()*area
   sat_ctr_first = sat_co2x1[-30:,month_sel,:,:].copy()*area
   [ttest_map, xxx] = perform_ttest_here(sat_ctr_last,sat_ctr_first,ny,nx,sig_level)
   im1 = plot_here(ax1,lon,lat,sat_ctr_last,sat_ctr_first,ttest_map,clevel)
   plt.title('(e) 6xCO2', fontsize=10)

   ax1 = fig.add_axes([0.29, 0.38, 0.22, 0.22], projection=ccrs.Orthographic(0, 90))
   sat_ctr_last = sat_co2x7[-30:,month_sel,:,:].copy()*area
   sat_ctr_first = sat_co2x1[-30:,month_sel,:,:].copy()*area
   [ttest_map, xxx] = perform_ttest_here(sat_ctr_last,sat_ctr_first,ny,nx,sig_level)
   im1 = plot_here(ax1,lon,lat,sat_ctr_last,sat_ctr_first,ttest_map,clevel)
   plt.title('(f) 7xCO2', fontsize=10)

   ax1 = fig.add_axes([0.53, 0.38, 0.22, 0.22], projection=ccrs.Orthographic(0, 90))
   sat_ctr_last = sat_co2x8[-30:,month_sel,:,:].copy()*area
   sat_ctr_first = sat_co2x1[-30:,month_sel,:,:].copy()*area
   [ttest_map, xxx] = perform_ttest_here(sat_ctr_last,sat_ctr_first,ny,nx,sig_level)
   im1 = plot_here(ax1,lon,lat,sat_ctr_last,sat_ctr_first,ttest_map,clevel)
   plt.title('(g) 8xCO2', fontsize=10)

   ax1 = fig.add_axes([0.77, 0.38, 0.22, 0.22], projection=ccrs.Orthographic(0, 90))
   sat_ctr_last = sat_co2x1[-30:,month_sel,:,:].copy()*area
   sat_ctr_first = sat_co2x1[-30:,month_sel,:,:].copy()*area*0
   [ttest_map, xxx] = perform_ttest_here(sat_ctr_last,sat_ctr_first,ny,nx,sig_level)
   im1 = plot_here2(ax1,lon,lat,sat_ctr_last,sat_ctr_first,clevel)
   plt.title('(h) Reference', fontsize=10)


   cbaxes = fig.add_axes([0.16, 0.33, 0.72, 0.01])
   cbar = plt.colorbar(im1, cax=cbaxes, orientation='horizontal', ticks=np.linspace(clevel[0],clevel[-1],11))
   cbar.set_label('km$^{2}$', rotation=0)

   plt.savefig('trend_tmp_plot.jpg', format='jpeg', dpi=200)

   plt.show()

   sys.exit()



