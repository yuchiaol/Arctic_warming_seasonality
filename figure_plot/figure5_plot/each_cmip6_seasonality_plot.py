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

def plot_here(tt,dtas_in,plot_style,text_in,color_in):

    ts_tmp2 = dtas_in.copy()
    ts_tmp2[:,:,:6] = (dtas_in[:,:,6:]).copy()
    ts_tmp2[:,:,6:] = (dtas_in[:,:,:6]).copy()
    ts_mean = np.nanmean(ts_tmp2[:,:,:], axis=1)
    ts_mean = ts_mean - ts_mean[:,0].mean()
    plt.plot(tt,np.nanmean(ts_mean, axis=0),plot_style+'o-', label=text_in, color=color_in)
    max_index = np.argmax(np.nanmean(ts_mean, axis=0))
    plt.plot([tt[max_index]]*2,[np.nanmean(ts_mean, axis=0)[max_index]]*2, plot_style+'*', markersize=15, color=color_in)
    plt.fill_between(tt, np.nanmean(ts_mean, axis=0)-np.nanstd(ts_mean, axis=0),np.nanmean(ts_mean, axis=0)+np.nanstd(ts_mean, axis=0), color=color_in,alpha=0.3)

def plot_here2(tt,dtas_in,plot_style,text_in,color_in):

    ts_tmp2 = dtas_in.copy()
    ts_tmp2[:,:,:6] = (dtas_in[:,:,6:]).copy()
    ts_tmp2[:,:,6:] = (dtas_in[:,:,:6]).copy()
    ts_mean = np.nanmean(ts_tmp2[:,:,:], axis=1)
    plt.plot(tt,np.nanmean(ts_mean, axis=0),plot_style+'o-', label=text_in, color=color_in)
    max_index = np.argmax(np.nanmean(ts_mean, axis=0))
    plt.plot([tt[max_index]]*2,[np.nanmean(ts_mean, axis=0)[max_index]]*2, plot_style+'*', markersize=15, color=color_in)
    plt.fill_between(tt, np.nanmean(ts_mean, axis=0)-np.nanstd(ts_mean, axis=0),np.nanmean(ts_mean, axis=0)+np.nanstd(ts_mean, axis=0), color=color_in,alpha=0.3)


if flag_run == 1:
# ================================================================
# read simulations
# ================================================================

# ================================================================
# plot figures
# ================================================================
   model_name = ['ACCESS-CM2(1)','ACCESS-ESM1-5(1)','AWI-CM-1-1-MR(0)','BCC-CSM2-MR(0)','CAMS-CSM1-0(0)','CanESM5(0)',\
                 'CESM2(1)','CESM2-WACCM(2)','CIESM(1)','CMCC-CM2-SR5(1)','CNRM-CM6-1-HR(1)','CNRM-CM6-1(0)','CNRM-ESM2-1(1)',\
                 'E3SM-1-1(1)','EC-Earth3(1)','EC-Earth3-Veg(1)','FGOALS-f3-L(0)','FGOALS-g3(0)','FIO-ESM-2-0(1)',\
                 'GFDL-CM4(1)','GFDL-ESM4(1)','GISS-E2-1-G(-1)','HadGEM3-GC31-LL(1)','HadGEM3-GC31-MM(1)','IITM-ESM(0)','INM-CM4-8(0)','INM-CM5-0(0)',\
                 'IPSL-CM6A-LR(1)','KACE-1-0-G(2)','MCM-UA-1-0(1)','MIROC6(1)','MIROC-ES2L(1)','MPI-ESM1-2-HR(0)','MPI-ESM1-2-LR(0)','MRI-ESM2-0(0)','NESM3(2)',\
                 'NorESM2-LM(0)','NorESM2-MM(0)','TaiESM1(1)','UKESM1-0-LL(1)','']

   plt.close('all')

   tt = np.linspace(1,13,13)

# plot Arctic warming
   fig = plt.figure(1)
   fig.set_size_inches(10, 10, forward=True)

# read cmip6 results
# case1
   case_sel = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,\
               21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
   dirname = '/home/yliang/research/aa_co2/data_process/cmip6/'
   filename = 'cmip6_tas_temp_output_for_aa_2070_2099.nc'
   f = Dataset(dirname + filename, 'r')
   tas_global_hist = f.variables['tas_global_hist'][case_sel,:].data
   tas_arctic_hist = f.variables['tas_arctic_hist'][case_sel,:].data
   tas_global_future = f.variables['tas_global_future'][case_sel,:].data
   tas_arctic_future = f.variables['tas_arctic_future'][case_sel,:].data
   tas_global_pi = f.variables['tas_global_pi'][case_sel,:].data
   tas_arctic_pi = f.variables['tas_arctic_pi'][case_sel,:].data
   f.close()

   n_model = tas_global_hist.shape[0]
   year_N = int(tas_global_hist.shape[1]/12)

   tas_global_hist_month = np.zeros((n_model,year_N,12))
   tas_arctic_hist_month = np.zeros((n_model,year_N,12))
   tas_global_future_month = np.zeros((n_model,year_N,12))
   tas_arctic_future_month = np.zeros((n_model,year_N,12))
#   tas_global_pi_month = np.zeros((n_model,12))
#   tas_arctic_pi_month = np.zeros((n_model,12))

   for NM in range(n_model):
       tas_global_hist_month[NM,:,:] = (tas_global_hist[NM,:]-tas_global_pi[NM,:]).reshape((year_N,12))
       tas_arctic_hist_month[NM,:,:] = (tas_arctic_hist[NM,:]-tas_arctic_pi[NM,:]).reshape((year_N,12))
       tas_global_future_month[NM,:,:] = (tas_global_future[NM,:]-tas_global_pi[NM,:]).reshape((year_N,12))
       tas_arctic_future_month[NM,:,:] = (tas_arctic_future[NM,:]-tas_arctic_pi[NM,:]).reshape((year_N,12))
#       tas_global_pi_month[NM,:] = np.nanmean(tas_global_pi[NM,:].reshape((year_N,12)), axis=0)
#       tas_arctic_pi_month[NM,:] = np.nanmean(tas_arctic_pi[NM,:].reshape((year_N,12)), axis=0)

   dtas_global_hist_month = tas_global_hist_month
   dtas_arctic_hist_month = tas_arctic_hist_month
   dtas_global_future_month = tas_global_future_month
   dtas_arctic_future_month = tas_arctic_future_month

# case 2
   dirname = '/home/yliang/research/aa_co2/data_process/cmip6/'
   filename = 'cmip6_tas_temp_output_for_aa_2015_2044.nc'
   f = Dataset(dirname + filename, 'r')
   tas_global_hist2 = f.variables['tas_global_hist'][case_sel,:].data
   tas_arctic_hist2 = f.variables['tas_arctic_hist'][case_sel,:].data
   tas_global_future2 = f.variables['tas_global_future'][case_sel,:].data
   tas_arctic_future2 = f.variables['tas_arctic_future'][case_sel,:].data
   tas_global_pi2 = f.variables['tas_global_pi'][case_sel,:].data
   tas_arctic_pi2 = f.variables['tas_arctic_pi'][case_sel,:].data
   f.close()

   n_model = tas_global_hist2.shape[0]
   year_N = int(tas_global_hist2.shape[1]/12)

   tas_global_hist_month2 = np.zeros((n_model,year_N,12))
   tas_arctic_hist_month2 = np.zeros((n_model,year_N,12))
   tas_global_future_month2 = np.zeros((n_model,year_N,12))
   tas_arctic_future_month2 = np.zeros((n_model,year_N,12))
#   tas_global_pi_month = np.zeros((n_model,12))
#   tas_arctic_pi_month = np.zeros((n_model,12))

   for NM in range(n_model):
       tas_global_hist_month2[NM,:,:] = (tas_global_hist2[NM,:]-tas_global_pi2[NM,:]).reshape((year_N,12))
       tas_arctic_hist_month2[NM,:,:] = (tas_arctic_hist2[NM,:]-tas_arctic_pi2[NM,:]).reshape((year_N,12))
       tas_global_future_month2[NM,:,:] = (tas_global_future2[NM,:]-tas_global_pi2[NM,:]).reshape((year_N,12))
       tas_arctic_future_month2[NM,:,:] = (tas_arctic_future2[NM,:]-tas_arctic_pi2[NM,:]).reshape((year_N,12))
#       tas_global_pi_month[NM,:] = np.nanmean(tas_global_pi[NM,:].reshape((year_N,12)), axis=0)
#       tas_arctic_pi_month[NM,:] = np.nanmean(tas_arctic_pi[NM,:].reshape((year_N,12)), axis=0)

   dtas_global_hist_month2 = tas_global_hist_month2
   dtas_arctic_hist_month2 = tas_arctic_hist_month2
   dtas_global_future_month2 = tas_global_future_month2
   dtas_arctic_future_month2 = tas_arctic_future_month2

   ax1 = fig.add_axes([0.16, 0.2, 0.25, 0.64])

   ts_tmp2 = dtas_arctic_hist_month.copy()
   ts_tmp2[:,:,:6] = (dtas_arctic_hist_month[:,:,6:]).copy()
   ts_tmp2[:,:,6:] = (dtas_arctic_hist_month[:,:,:6]).copy()
   ts_mean = np.nanmean(ts_tmp2[:,:,:], axis=1)
   ts_mean_ext = np.zeros((n_model+1,13))
   ts_mean_ext[:n_model,:12] = ts_mean.copy()

   im = plt.pcolor(tt,np.linspace(1,n_model+1,n_model+1),ts_mean_ext, cmap='YlOrRd', vmin=0, vmax=15)
   for II in range(n_model+1):
       arg = np.argmax(ts_mean_ext[II,:]) 
       plt.plot([tt[arg]+0.5]*2, [II+1.5]*2,'b*')

   plt.xticks(tt+0.5,['Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun'], fontsize=9, rotation=45)
   plt.yticks(np.linspace(1,n_model+1,n_model+1)+0.5,model_name, fontsize=9)
   plt.xlim(1,12)
   plt.ylim(1,41)
   plt.title('(a) CMIP6 1985-2014 Arctic SAT (40)', fontsize=10)

   ax1 = fig.add_axes([0.43, 0.2, 0.25, 0.64])

   ts_tmp2 = dtas_arctic_hist_month.copy()
   ts_tmp2[:,:,:6] = (dtas_arctic_future_month2[:,:,6:]).copy()
   ts_tmp2[:,:,6:] = (dtas_arctic_future_month2[:,:,:6]).copy()
   ts_mean = np.nanmean(ts_tmp2[:,:,:], axis=1)
   ts_mean_ext = np.zeros((n_model+1,13))
   ts_mean_ext[:n_model,:12] = ts_mean.copy()

   im = plt.pcolor(tt,np.linspace(1,n_model+1,n_model+1),ts_mean_ext, cmap='YlOrRd', vmin=0, vmax=15)
   for II in range(n_model+1):
       arg = np.argmax(ts_mean_ext[II,:])
       plt.plot([tt[arg]+0.5]*2, [II+1.5]*2,'b*')

   plt.xticks(tt+0.5,['Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun'], fontsize=9, rotation=45)
   plt.yticks(np.linspace(1,n_model+1,n_model+1)+0.5,[], fontsize=9)
   plt.xlim(1,12)
   plt.ylim(1,41)
   plt.title('(b) CMIP6 2015-2044 Arctic SAT (40)', fontsize=10)

   ax1 = fig.add_axes([0.7, 0.2, 0.25, 0.64])

   ts_tmp2 = dtas_arctic_hist_month.copy()
   ts_tmp2[:,:,:6] = (dtas_arctic_future_month[:,:,6:]).copy()
   ts_tmp2[:,:,6:] = (dtas_arctic_future_month[:,:,:6]).copy()
   ts_mean = np.nanmean(ts_tmp2[:,:,:], axis=1)
   ts_mean_ext = np.zeros((n_model+1,13))
   ts_mean_ext[:n_model,:12] = ts_mean.copy()

   im = plt.pcolor(tt,np.linspace(1,n_model+1,n_model+1),ts_mean_ext, cmap='YlOrRd', vmin=0, vmax=15)
   for II in range(n_model+1):
       arg = np.argmax(ts_mean_ext[II,:])
       plt.plot([tt[arg]+0.5]*2, [II+1.5]*2,'b*')

   plt.xticks(tt+0.5,['Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun'], fontsize=9, rotation=45)
   plt.yticks(np.linspace(1,n_model+1,n_model+1)+0.5,[], fontsize=9)
   plt.xlim(1,12)
   plt.ylim(1,41)
   plt.title('(c) CMIP6 2070-2099 Arctic SAT (40)', fontsize=10)

   cbaxes = fig.add_axes([0.16, 0.14, 0.79, 0.01])
   cbar = plt.colorbar(im, cax=cbaxes, orientation='horizontal', ticks=np.linspace(0,15,16))
   cbar.set_label('K', rotation=0)



   plt.savefig('trend_tmp_plot.jpg', format='jpeg', dpi=200)

   plt.show()

   sys.exit()

   ax1 = fig.add_axes([0.55, 0.4, 0.36, 0.27])
   ts_tmp2 = (dtas_arctic_future_month/dtas_global_future_month).copy()
   ts_tmp2[:,:,:6] = (dtas_arctic_future_month[:,:,6:]/dtas_global_future_month[:,:,6:]).copy()
   ts_tmp2[:,:,6:] = (dtas_arctic_future_month[:,:,:6]/dtas_global_future_month[:,:,:6]).copy()
   ts_mean = np.nanmean(ts_tmp2[:,:,:], axis=1)
#   plt.plot(tt,np.nanmean(ts_mean, axis=0)-np.nanmean(ts_mean, axis=0)[0]*0,'ro-', label='2070-2099')
#   plt.plot(tt[max_index],np.nanmean(ts_mean, axis=0)[max_index]-np.nanmean(ts_mean, axis=0)[0]*0,'r*',markersize=15) 
#   plt.fill_between(tt, np.nanmean(ts_mean, axis=0)-np.nanstd(ts_mean, axis=0)-np.nanmean(ts_mean, axis=0)[0]*0,\
#                        np.nanmean(ts_mean, axis=0)+np.nanstd(ts_mean, axis=0)-np.nanmean(ts_mean, axis=0)[0]*0, color='r',alpha=0.3)

#   ts_tmp2 = (dtas_arctic_future_month2/dtas_global_future_month2).copy()
#   ts_tmp2[:,:,:6] = (dtas_arctic_future_month2[:,:,6:]/dtas_global_future_month2[:,:,6:]).copy()
#   ts_tmp2[:,:,6:] = (dtas_arctic_future_month2[:,:,:6]/dtas_global_future_month2[:,:,:6]).copy()
#   ts_mean = np.nanmean(ts_tmp2[:,:,:], axis=1)
#   plt.plot(tt,np.nanmean(ts_mean, axis=0)-np.nanmean(ts_mean, axis=0)[0]*0,'go-', label='2015-2044')
#   plt.plot(tt[max_index],np.nanmean(ts_mean, axis=0)[max_index]-np.nanmean(ts_mean, axis=0)[0]*0,'g*',markersize=15)
#   plt.fill_between(tt, np.nanmean(ts_mean, axis=0)-np.nanstd(ts_mean, axis=0)-np.nanmean(ts_mean, axis=0)[0]*0,\
#                        np.nanmean(ts_mean, axis=0)+np.nanstd(ts_mean, axis=0)-np.nanmean(ts_mean, axis=0)[0]*0, color='g',alpha=0.3)

#   plt.ylim(0,6)

   plt.xticks(tt,['Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun'], fontsize=10)
   plt.legend(loc='upper right', fontsize='small')
   plt.xlim(1,12)
   plt.title('(f) CMIP6 AAF (40)', fontsize=10)

   cbaxes = fig.add_axes([0.05, 0.165, 0.865, 0.01])
   cbar = plt.colorbar(im, cax=cbaxes, orientation='horizontal', ticks=np.linspace(clevel[0],clevel[-1],11))
   cbar.set_label('K', rotation=0)

   plt.savefig('trend_tmp_plot.jpg', format='jpeg', dpi=200)

   plt.show()













