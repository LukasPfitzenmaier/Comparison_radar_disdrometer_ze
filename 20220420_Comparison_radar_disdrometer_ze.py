#!/usr/bin/env python3
# coding: utf-8

# authors: Leonardo Porcacchia, Lukas Pfitzenmaier and Alexander Pschera
##############################################
#Neccessary modules:

import argparse
from scipy.interpolate import interp1d
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.dates import DateFormatter
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import xarray as xr
import glob
import netCDF4 as nc
import matplotlib.dates as mdates # Changes of data formates can be made
from datetime import datetime, timedelta
import os
import csv
import pytz

#from raincoat.disdrometer.read_parsivel import readPars
from raincoat.FWD_sim import FWD_sim
import raincoat.disdrometer.pars_class as pc
from raincoat.radarFunctions import getVarTimeRange, getRadarVar
from raincoat.statistical_analysis.stat_anal_core import calculate_offset
from raincoat.functions import check_consistency_of_droplet_diameter
from raincoat.functions import nor_LOC2UTC
from raincoat.functions import derive_mean_temperature
from raincoat.functions import getRadarVar_v2
from raincoat.functions import readPars00
from raincoat.functions import t_matrix_LUT
from raincoat.functions import OutPutDirectoryCeck
###############################################################################
#define argparser:
parser = argparse.ArgumentParser(description="Add a site, date and number of days to process (going backwards), to distinguish reflectivity offset from Radar to Disdrometer")
parser.add_argument("-s","--site",type=str,help="observation site for Radar and Disdrometer: e.g. jue, nor, mag", default='jue')
parser.add_argument("-y","--year",type=int,help="year of the observation, usually 4 digits, e.g. yyyy", default='2019')
parser.add_argument("-m","--month",type=int,help="month of the observation, usually 2 digits, e.g. mm", default='09')
parser.add_argument("-d","--day",type=int,help="day of the observation, usually 2 digits, e.g. dd", default='30')
parser.add_argument("-db","--days_back",type=int,help="How many days before chosen date shall be processed",default=6)
parser.add_argument("-op", "--output_path", type=str, help="", default="/work/")
parser.add_argument("-edp", "--enable_diagnostic_plots", type=str,\
    help="Additional plots will be saved for insights into processing.", default=False)
args = parser.parse_args()

# change into working directory
os.chdir('.')
home_cwd = os.getcwd()
enable_diagnostic_plots=args.enable_diagnostic_plots
if enable_diagnostic_plots:
    cancellation_file = open("./reasons_for_processing_abortion_by_date.txt","w")
    cancellation_file.write(str(args.site)+":\n")
else:
    cancellation_file = "sss"
 
if args.site=='jue':
    ### for JÜLICH
    path_output         = args.output_path+'test_output'
    path_output_plots   = args.output_path+'test_output'#'/home/lpfitzen/Plot_summery/frm4radar_disdrometer_Ze' 
    path_output_data    = args.output_path+'test_output'# '/data/frm4radar/juelich_folder/disdrometer_DATA/l1'
    path_inputput_disdo = '/data/obs/site/jue/parsivel/l1'
    path_inputput_radar = '/data/obs/site/jue/mirac-a/l1'
    radar    = 'mirac'
    site     = 'jue'
    location = 'JOYCE'
elif args.site=='nor':
    ### for NORUNDA
    path_output         = args.output_path+'smhi'
    path_output_plots   = args.output_path+'smhi/plot'
    path_output_data    = args.output_path+'smhi/data'
    path_inputput_disdo = '/data/frm4radar/smhi_folder/Disdrometer_DATA/netcdf'
    path_inputput_radar = '/data/frm4radar/smhi_folder/Radar_DATA/l1'
    radar    = 'smhi'
    site     = 'nor'
    location = 'Norunda'
elif args.site=='mag':
    ### for MAGURELE
    path_output         = args.output_path+'inoe'
    path_output_plots   = args.output_path+'inoe/plot' 
    path_output_data    = args.output_path+'inoe/data'
    #Disdrometer path still needs to be set and also be adjusted to file structure further down:
    #path_inputput_disdo = '/data/frm4radar/inoe_folder/disdrometer_DATA' #will not work so far because of different data type .log
    path_inputput_radar = '/data/frm4radar/inoe_folder/radar_DATA/l1'
    radar    = 'inoe'
    site     = 'Maguerele'
    location = 'Bucharest'
 

# how many data points have to be selected to cover 10 min of measurements (disdrometer measurements)
points5min = 10 
# Height definition between the radar data are read in the program
height_bot = 0
height_top = 1000
# Height definition between the radar data are averaged for Ze-comparisson
height_av1bot = 120
height_av1top = 150
# Height definition between the radar data are averaged for Ze-comparisson
height_av2bot = 250
height_av2top = 300

###Parameters
rr_lowth = 0.1  #mm/Hr
nD_lowth = 30   #particles/sampling interval
ZeR_upth = 15.0 #dBZ

# definde start and end time between the data should be plotted
date_end    = datetime(int(args.year),int(args.month),int(args.day))
date_start  = date_end - timedelta(days=args.days_back)

print(date_start, date_end)

dd1     = date_start.strftime('%Y%m%d')    #converts start date to string
dd2     = date_end.strftime('%Y%m%d')      #converts end date to string

mydates = pd.date_range(dd1, dd2).tolist() #lists all days from start date to end date 

### Definitions
pars_class, bin_edges = pc.pars_class()

print (mydates)
print('--- all things defined ---')



# In[2]:
#########################################
### --- START THE CODING OF THE  --- ####
#########################################

DateSetFinal = xr.Dataset()
for ii in range(0, len(mydates)):
    print('current working dir:', os.getcwd)
    if os.getcwd != home_cwd :
        os.chdir(home_cwd)
        print('new current working dir:', os.getcwd)
    ###############################
    # Diagnostic plots block 1/5:
    if enable_diagnostic_plots:
        cancellation_file.write(str(mydates[ii])+"\n")
        output_path_diagnostic_plots = "./diagnostic_plots/"+site+"/"
        output_path_diagnostic_plots = output_path_diagnostic_plots+str(mydates[ii].strftime('%Y%m%d'))+"/"
        try:
            os.mkdir(output_path_diagnostic_plots)
        except:
            print("Output path for diagnostic plots already exists. Plots there will be overwritten - no worries ;)")
    else:
        output_path_diagnostic_plots = "sss"
    ####################
    
    print('0')
    ###
    ### set diferent time stamps to read data - time stamps as str ############
    yyyymmdd = mydates[ii].strftime('%Y%m%d')   #writes date string as yyyymmdd
    yyyy     = mydates[ii].strftime('%Y')       #writes date string as yyyy
    yymm     = mydates[ii].strftime('%y%m')     #writes date string as yymm
    mm       = mydates[ii].strftime('%m')       #writes date string as mm
    dd       = mydates[ii].strftime('%d')       #writes date string as dd
    date     = mydates[ii].strftime('%Y-%m-%d') #writes date string as yyyy-mm-dd (for plotting)
    ### set start and end time of the time frame you wnat to analyze ##########
    ###  > is still present if one would like to analyze a single case instead 
    ###    of a longer time series
    start = pd.to_datetime(yyyymmdd +' 00:00')
    end   = start + timedelta(days=1)
    ###
    ### compleat the data input pathes (radar and disdrometer) based on #######
    ### selected date
    if args.site == 'jue':
        path_disdrometer = path_inputput_disdo+'/'+yyyy+'/'+mm+'/'+dd+'/sups_joy_dm00_l1_any_v00_'+yyyymmdd+'.nc'
    elif args.site == 'nor':
        path_disdrometer = path_inputput_disdo+'/'+yymm+'/parsivel'+yyyymmdd+'.nc'
    elif args.site == 'inoe':
    ##############################
        #Disdrometer path still needs to be set according to file structure:
        #path_disdrometer = '/data/obs/site/jue/parsivel/l1/'+yyyy+'/'+mm+'/'+dd+'/sups_joy_dm00_l1_any_v00_'+yyyymmdd+'.nc'
        pass
    ##############################
    path_radar       = path_inputput_radar+'/'+yyyy+'/'+mm+'/'+dd+'/*_v2.nc'
    if not glob.glob(path_radar):
        path_radar       = path_inputput_radar+'/'+yyyy+'/'+mm+'/'+dd+'/*_compact.nc' #if you don't find v2-files search for alternatives...
    ### make list of all files within the defined derectories 
    if glob.glob(path_disdrometer) and glob.glob(path_radar):
        print(yyyymmdd)
        disdrofiles = glob.glob(path_disdrometer)
        radarfiles  = glob.glob(path_radar)
    else :
        print('exit')
        #############
        if enable_diagnostic_plots:
            cancellation_file.write("Input files either for radar or for disdro not found!\n")
        ##############
        continue
    print('    ') 
    print('*** processing of '+date+' *** *** *** *** *** ***')
    ###
    ##############################
    ### Reading the radar data ###
    ##############################
    print('1 >>> read Radar data')    
    ###
    radarfiles = np.sort(radarfiles)
    ###################
    #apschera 2021-12-08:
    RadarList  = [getVarTimeRange(getRadarVar_v2(i, '2001.01.01. 00:00:00.', 'ze'), 
                                  height_bot, height_top, start, end) for i in radarfiles]          
    ######################
    ### combine the selected data along the time axis
    RadarData = xr.concat(RadarList, 'time')
    ### flatten the radar reflectivity between the selected height bins ### 
    av1b = np.min(np.where(height_av1bot <= RadarData.range.values))
    av1t = np.min(np.where(height_av1top <= RadarData.range.values))
    av2b = np.min(np.where(height_av2bot <= RadarData.range.values))
    av2t = np.min(np.where(height_av2top <= RadarData.range.values))
    ### read out the data as a time series for the selected height regions
    Ze_radar_120_150 = 10.0*np.log10(np.average(RadarData[:,av1b:av1t].values,1).flatten())  #reflectivity in dBZ
    Ze_radar_250_300 = 10.0*np.log10(np.average(RadarData[:,av2b:av2t].values,1).flatten())  #reflectivity in dBZ
    ###
    ### radar time array    
    time_radar = RadarData.time.values
    ###
    ### Temperature mesured by the met-station attached to the Radar
    ###  > needed later to define get the right scattering table
    ####################
    #apschera 2021-12-08:
    RadarList = [getRadarVar_v2(i, '2001.01.01. 00:00:00', 'ta') for i in radarfiles]
    temp_radar = xr.concat(RadarList, 'time')  
    #################
    ####################################
    ### Reading the disdrometer data ###
    ####################################    
    print('2 >>> read Disdrometer data')    
    ### 
    print("path_disdrometer: ",path_disdrometer)
    
    ###############################################################################
    
    parsNC   = nc.Dataset(path_disdrometer, 'r')
    DisdroDataFrame2 = [readPars00(i, var_Ze='Ze', 
                                   var_time='time', 
                                   var_vmean='vmean', 
                                   var_dmean='N', 
                                   var_rr='rr', 
                                   var_M='M',
                                   var_T_sensor='T_sensor') for i in disdrofiles] 
    ### write out the data into arrays ---      
    pDF             = pd.concat([i[0] for i in DisdroDataFrame2]).sort_index() # Ze, time and rain rate
    nDF             = pd.concat([i[1] for i in DisdroDataFrame2]).sort_index() # log10(d mean)
    vDF             = pd.concat([i[2] for i in DisdroDataFrame2]).sort_index() # mean v   
    disdro_NMmatrix = pd.concat([i[3] for i in DisdroDataFrame2]).sort_index() # data matrix M
    disdro_VMmatrix = pd.concat([i[4] for i in DisdroDataFrame2]).sort_index() # data matrix V
    disdro_Temp     = pd.concat([i[5] for i in DisdroDataFrame2]).sort_index() # data matrix Temp
    
    ### select only data between the selcted start and end times:
    disdro_Temp     = (disdro_Temp[start:end]).values
    pDF             = pDF[start:end]
    nDF             = nDF[start:end]
    vDF             = vDF[start:end]
    disdro_NMmatrix = disdro_NMmatrix[start:end]
    disdro_VMmatrix = disdro_VMmatrix[start:end]
    rr_disdro       = pDF['rainrate']   
    Ze_disdro       = pDF['Ze']
    dclasses = parsNC.variables['dclasses'][:]
    vclasses = parsNC.variables['vclasses'][:]
    n_disdro   = np.nansum(disdro_NMmatrix.values, axis=1)
    ### disdrometer time array
    time_disdro     = pDF.index.values # time array disdrometer
    #######################
    #apschera 2021-12-15:
    if site == 'nor':
         time_disdro = nor_LOC2UTC(time_disdro)
    nDF, vDF = check_consistency_of_droplet_diameter(nDF, vDF, dclasses, enable_diagnostic_plots, output_path_diagnostic_plots)
    ##########################
    print('  >>> reading of the data completed')     
    ###
    ##################################################
    ### Preperation of the data for the calculation ###
    ##################################################    
    print('3 >>> regrid radar data to disdrometer time grid')  
    ### prepare the arrays that they can be up and down scaled
    disdro_Temp = pd.Series(np.squeeze(disdro_Temp), index=time_disdro)
    disdro_Temp = xr.DataArray(disdro_Temp, dims=('time_d'), coords=[time_disdro])
    disdro_Temp.attrs['units'] = 'K'
    rr_disdro = xr.DataArray(rr_disdro, 
                             dims=('time_d'), coords=[time_disdro])
    rr_disdro.attrs['units'] = 'mm h-1'  
    n_disdro = xr.DataArray(n_disdro, 
                            dims=('time_d'), coords=[time_disdro])
    n_disdro.attrs['units'] = 'no. of particles m-1'
    
    ################################################
    
    Ze_radar_120_150 = xr.DataArray(Ze_radar_120_150, 
                                    dims=('time_r'), coords=[time_radar])
    Ze_radar_120_150.attrs['units'] = 'mm6 m-3'
    Ze_radar_250_300 = xr.DataArray(Ze_radar_250_300, 
                                    dims=('time_r'), coords=[time_radar])
    Ze_radar_250_300.attrs['units'] = 'mm6 m-3'
    temp_radar = xr.DataArray(temp_radar, 
                              dims=('time_r'), coords=[time_radar])
    temp_radar.attrs['units'] = 'K'
    ###
    ### Downscale radar data to the disdrometer time for the data filtering
    
    #################################
    # Diagnostic plots block 2/5:
    if enable_diagnostic_plots:  
        blubfile = open(output_path_diagnostic_plots+"data_arrays_and_shapes_etc.txt", "w")
        blubfile.write("disdrofiles: \n"+str(disdrofiles)+"\n")
        blubfile.write("radarfiles: \n"+str(radarfiles)+"\n")
        blubfile.write("shape(nDF): \n"+str(nDF.shape)+"\n")
        blubfile.write("type(nDF): \n"+str(type(nDF))+"\n")
        blubfile.write("nDF[0:2]: \n"+str(nDF[0:2])+"\n")
        blubfile.write("nDF[0:2]: \n"+str(nDF[-2:-1])+"\n")
        blubfile.write("\n\n")
        blubfile.write("shape(pDF): \n"+str(pDF.shape)+"\n")
        blubfile.write("type(pDF): \n"+str(type(pDF))+"\n")
        blubfile.write("pDF[0:2]: \n"+str(pDF[0:2])+"\n")
        blubfile.write("pDF[0:2]: \n"+str(pDF[-2:-1])+"\n")
        blubfile.write("\n\n")

        refl_bins = np.arange(-60, 20, 2)
        blubfile.write("arrays going into processing - before: \n\n")
        blubfile.write("shape(temp_radar): \n"+str(temp_radar.shape)+"\n")
        blubfile.write("type(temp_radar): \n"+str(type(temp_radar))+"\n")
        blubfile.write("temp_radar[0:2]: \n"+str(temp_radar[0:2])+"\n")
        blubfile.write("temp_radar[0:2]: \n"+str(temp_radar[-2:-1])+"\n")
        blubfile.write("\n\n")
        blubfile.write("shape(Ze_radar_120_150): \n"+str(Ze_radar_120_150.shape)+"\n")
        blubfile.write("type(Ze_radar_120_150): \n"+str(type(Ze_radar_120_150))+"\n")
        blubfile.write("Ze_radar_120_150[0:2]: \n"+str(Ze_radar_120_150[0:2])+"\n")
        blubfile.write("Ze_radar_120_150[0:2]: \n"+str(Ze_radar_120_150[-2:-1])+"\n")
        blubfile.write("\n\n")
        blubfile.write("shape(Ze_radar_250_300): \n"+str(Ze_radar_250_300.shape)+"\n")
        blubfile.write("type(Ze_radar_250_300): \n"+str(type(Ze_radar_250_300))+"\n")
        blubfile.write("Ze_radar_250_300[0:2]: \n"+str(Ze_radar_250_300[0:2])+"\n")
        blubfile.write("Ze_radar_250_300[0:2]: \n"+str(Ze_radar_250_300[-2:-1])+"\n")
        blubfile.write("\n\n")
        blubfile.write("shape(rr_disdro): \n"+str(rr_disdro.shape)+"\n")
        blubfile.write("type(rr_disdro): \n"+str(type(rr_disdro))+"\n")
        blubfile.write("rr_disdro[0:2]: \n"+str(rr_disdro[0:2])+"\n")
        blubfile.write("rr_disdro[0:2]: \n"+str(rr_disdro[-2:-1])+"\n")
        blubfile.write("\n\n")
        blubfile.write("shape(n_disdro): \n"+str(n_disdro.shape)+"\n")
        blubfile.write("type(n_disdro): \n"+str(type(n_disdro))+"\n")
        blubfile.write("n_disdro[0:2]: \n"+str(n_disdro[0:2])+"\n")
        blubfile.write("n_disdro[0:2]: \n"+str(n_disdro[-2:-1])+"\n")
        blubfile.write("\n\n")
        blubfile.write("shape(time_radar): \n"+str(time_radar.shape)+"\n")
        blubfile.write("type(time_radar): \n"+str(type(time_radar))+"\n")
        blubfile.write("time_radar[0:2]: \n"+str(time_radar[0:2])+"\n")
        blubfile.write("time_radar[0:2]: \n"+str(time_radar[-2:-1])+"\n")
        blubfile.write("\n\n")
        blubfile.write("shape(time_disdro): \n"+str(time_disdro.shape)+"\n")
        blubfile.write("type(time_disdro): \n"+str(type(time_disdro))+"\n")
        blubfile.write("time_disdro[0:2]: \n"+str(time_disdro[0:2])+"\n")
        blubfile.write("time_disdro[0:2]: \n"+str(time_disdro[-2:-1])+"\n")
        blubfile.write("\n\n")
        blubfile.write("shape(Ze_disdro): \n"+str(Ze_disdro.shape)+"\n")
        blubfile.write("type(Ze_disdro): \n"+str(type(Ze_disdro))+"\n")
        blubfile.write("Ze_disdro[0:2]: \n"+str(Ze_disdro[0:2])+"\n")
        blubfile.write("Ze_disdro[0:2]: \n"+str(Ze_disdro[-2:-1])+"\n")
        blubfile.write("\n\n")
        blubfile.write("shape(disdro_Temp): \n"+str(disdro_Temp.shape)+"\n")
        blubfile.write("type(disdro_Temp): \n"+str(type(disdro_Temp))+"\n")
        blubfile.write("disdro_Temp[0:2]: \n"+str(disdro_Temp[0:2])+"\n")
        blubfile.write("disdro_Temp[0:2]: \n"+str(disdro_Temp[-2:-1])+"\n")
        blubfile.write("\n\n")
        plt.title("Ze_disdro")
        plt.plot(time_disdro, Ze_disdro)
        plt.xlabel("time_disdro")
        plt.xlim(time_disdro[0],time_disdro[-1])
        plt.ylim(-60,60)
        plt.savefig(output_path_diagnostic_plots+"Ze_disdro.png")
        plt.close()
        plt.title("temperature timelines temp_radar and disdro_Temp")
        plt.plot(time_radar, temp_radar, label="temp_radar")
        plt.plot(time_disdro, disdro_Temp, label="disdro_Temp")
        plt.xlabel("time_disdro")
        plt.xlim(time_radar[0],time_radar[-1])
        plt.legend()
        plt.savefig(output_path_diagnostic_plots+"temperature_timeplots_before.png")
        plt.close()
        plt.title("Ze_radar_120_150")
        plt.plot(time_radar, Ze_radar_120_150)
        plt.xlabel("time_radar")
        plt.xlim(time_radar[0],time_radar[-1])
        plt.ylim(-60,60)
        plt.savefig(output_path_diagnostic_plots+"120Ze_radar_120_150.png")
        plt.close()
        plt.title("Ze_radar_250_300")
        plt.plot(time_radar, Ze_radar_250_300)
        plt.xlabel("time_radar")
        plt.xlim(time_radar[0],time_radar[-1])
        plt.ylim(-60,60)
        plt.savefig(output_path_diagnostic_plots+"250Ze_radar_250_300.png")
        plt.close()
        plt.title("rr_disdro")
        plt.plot(time_disdro, rr_disdro)
        plt.xlim(time_radar[0],time_radar[-1])
        plt.ylim(0, np.max(rr_disdro))
        plt.xlabel("time_disdro")
        plt.savefig(output_path_diagnostic_plots+"rr_disdro.png")
        plt.close()
        plt.title("n_disdro")
        plt.plot(time_disdro, n_disdro)
        plt.ylim(0, np.max(n_disdro))
        plt.xlim(time_radar[0],time_radar[-1])
        plt.xlabel("time_disdro")
        plt.savefig(output_path_diagnostic_plots+"n_disdro.png")
        plt.close()
        #histograms:
        plt.title("histogram Ze_radar_120_150")
        plt.hist(Ze_radar_120_150, bins=refl_bins)
        plt.ylabel("number of occurences in "+str(len(Ze_radar_120_150))+" bins")
        plt.xlabel("reflectivity")
        plt.savefig(output_path_diagnostic_plots+"Ze_radar_120_150_histogram.png")
        plt.close()
        plt.title("histogram Ze_radar_250_300")
        plt.hist(Ze_radar_250_300, bins=refl_bins)
        plt.ylabel("number of occurences in "+str(len(Ze_radar_250_300))+" bins")
        plt.xlabel("reflectivity")
        plt.savefig(output_path_diagnostic_plots+"Ze_radar_250_300_histogram.png")
        plt.close()
        plt.title("histogram temperatures temp_radar and disdro_Temp")
        plt.hist(temp_radar, label="temp_radar")
        plt.hist(disdro_Temp, label="disdro_Temp")
        plt.ylabel("number of occurences in "+str(len(temp_radar))+" bins")
        plt.xlabel("temperature in Kelvin")
        plt.legend()
        plt.savefig(output_path_diagnostic_plots+"temperatures_histogram.png")
        plt.close()
    #################################
    
    temp1_radar = temp_radar.interp(time_r=time_disdro, method='linear')
    temp1_radar = xr.DataArray(temp1_radar, dims=('time_d'), coords=[time_disdro])
    Ze1_radar_120_150 = Ze_radar_120_150.interp(time_r=time_disdro, method='linear')
    Ze1_radar_120_150 = xr.DataArray(Ze1_radar_120_150, dims=('time_d'), coords=[time_disdro])
    Ze1_radar_250_300 = Ze_radar_250_300.interp(time_r=time_disdro, method='linear')
    Ze1_radar_250_300 = xr.DataArray(Ze1_radar_250_300,dims=('time_d'), coords=[time_disdro])
    ###
    ### upscale Disdrometer data to radar time for the data filtering
    rr0_disdro = rr_disdro.interp(time_d=time_radar, method='linear')
    rr0_disdro = xr.DataArray(rr0_disdro, dims=('time_r'), coords=[time_radar])
    n0_disdro  = n_disdro.interp(time_d=time_radar, method='linear')                
    n0_disdro  = xr.DataArray(n0_disdro, dims=('time_r'), coords=[time_radar])
    disdro_Temp0 = disdro_Temp.interp(time_d=time_radar, method='linear')
    disdro_Temp0 = xr.DataArray(disdro_Temp0, dims=('time_r'), coords=[time_radar]) 

    #################################
    # Diagnostic plots block 3/5:
    if enable_diagnostic_plots: 
        blubfile.write("arrays going out of interpolation - after: \n\n")
        blubfile.write("shape(temp1_radar): \n"+str(temp1_radar.shape)+"\n")
        blubfile.write("type(temp1_radar): \n"+str(type(temp1_radar))+"\n")
        blubfile.write("temp1_radar[0:2]: \n"+str(temp1_radar[0:2])+"\n")
        blubfile.write("temp1_radar[0:2]: \n"+str(temp1_radar[-2:-1])+"\n")
        blubfile.write("\n\n")
        blubfile.write("shape(Ze1_radar_120_150): \n"+str(Ze1_radar_120_150.shape)+"\n")
        blubfile.write("type(Ze1_radar_120_150): \n"+str(type(Ze1_radar_120_150))+"\n")
        blubfile.write("Ze1_radar_120_150[0:2]: \n"+str(Ze1_radar_120_150[0:2])+"\n")
        blubfile.write("Ze1_radar_120_150[0:2]: \n"+str(Ze1_radar_120_150[-2:-1])+"\n")
        blubfile.write("\n\n")
        blubfile.write("shape(Ze1_radar_250_300): \n"+str(Ze1_radar_250_300.shape)+"\n")
        blubfile.write("type(Ze1_radar_250_300): \n"+str(type(Ze1_radar_250_300))+"\n")
        blubfile.write("Ze1_radar_250_300[0:2]: \n"+str(Ze1_radar_250_300[0:2])+"\n")
        blubfile.write("Ze1_radar_250_300[0:2]: \n"+str(Ze1_radar_250_300[-2:-1])+"\n")
        blubfile.write("\n\n")
        blubfile.write("shape(rr0_disdro): \n"+str(rr0_disdro.shape)+"\n")
        blubfile.write("type(rr0_disdro): \n"+str(type(rr0_disdro))+"\n")
        blubfile.write("rr0_disdro[0:2]: \n"+str(rr0_disdro[0:2])+"\n")
        blubfile.write("rr0_disdro[0:2]: \n"+str(rr0_disdro[-2:-1])+"\n")
        blubfile.write("\n\n")
        blubfile.write("shape(n0_disdro): \n"+str(n0_disdro.shape)+"\n")
        blubfile.write("type(n0_disdro): \n"+str(type(n0_disdro))+"\n")
        blubfile.write("n0_disdro[0:2]: \n"+str(n0_disdro[0:2])+"\n")
        blubfile.write("n0_disdro[0:2]: \n"+str(n0_disdro[-2:-1])+"\n")
        blubfile.write("\n\n")
        blubfile.write("shapedisdro_Temp0): \n"+str(disdro_Temp0.shape)+"\n")
        blubfile.write("type(disdro_Temp0): \n"+str(type(disdro_Temp0))+"\n")
        blubfile.write("disdro_Temp0[0:2]: \n"+str(disdro_Temp0[0:2])+"\n")
        blubfile.write("disdro_Temp0[0:2]: \n"+str(disdro_Temp0[-2:-1])+"\n")
        blubfile.write("\n\n")
        plt.title("temperature timelines")
        plt.plot(time_disdro, temp1_radar, label="temp1_radar")
        plt.plot(time_radar, disdro_Temp0, label="disdro_Temp0")
        plt.xlabel("time_disdro")
        plt.xlim(time_radar[0],time_radar[-1])
        plt.legend()
        plt.savefig(output_path_diagnostic_plots+"temperature_timeplots.png")
        plt.close()
        plt.title("temp1_radar")
        plt.plot(time_disdro, temp1_radar)
        plt.xlabel("time_disdro")
        plt.xlim(time_radar[0],time_radar[-1])
        plt.savefig(output_path_diagnostic_plots+"temp1_radar.png")
        plt.close()
        plt.title("Ze1_radar_120_150")
        plt.plot(time_disdro, Ze1_radar_120_150)
        plt.xlabel("time_disdro")
        plt.xlim(time_radar[0],time_radar[-1])
        plt.ylim(-60,60)
        plt.savefig(output_path_diagnostic_plots+"120Ze1_radar_120_150.png")
        plt.close()
        plt.title("Ze1_radar_250_300")
        plt.plot(time_disdro, Ze1_radar_250_300)
        plt.xlabel("time_disdro")
        plt.xlim(time_radar[0],time_radar[-1])
        plt.ylim(-60,60)
        plt.savefig(output_path_diagnostic_plots+"250Ze1_radar_250_300.png")
        plt.close()
        plt.title("rr0_disdro")
        plt.plot(time_radar, rr0_disdro)
        plt.xlim(time_radar[0],time_radar[-1])
        plt.xlabel("time_radar")
        plt.ylim(0, np.max(rr_disdro))
        plt.savefig(output_path_diagnostic_plots+"rr0_disdro.png")
        plt.close()
        plt.title("n0_disdro")
        plt.plot(time_radar, n0_disdro)
        plt.ylim(0, np.max(n_disdro))
        plt.xlim(time_radar[0],time_radar[-1])
        plt.xlabel("time_radar")
        plt.savefig(output_path_diagnostic_plots+"n0_disdro.png")
        plt.close()
        #histograms:
        plt.title("histogram Ze1_radar_250_300")
        plt.hist(Ze1_radar_250_300, bins=refl_bins)
        plt.ylabel("number of occurences in "+str(len(Ze1_radar_250_300))+" bins")
        plt.xlabel("reflectivity")
        plt.savefig(output_path_diagnostic_plots+"Ze1_radar_250_300_histogram.png")
        plt.close()
        plt.title("histogram Ze1_radar_120_150")
        plt.hist(Ze1_radar_120_150, bins=refl_bins)
        plt.ylabel("number of occurences in "+str(len(Ze1_radar_120_150))+" bins")
        plt.xlabel("reflectivity")
        plt.savefig(output_path_diagnostic_plots+"Ze1_radar_120_150_histogram.png")
        plt.close()
    ############################################

    ###########################################
    ### Filter out Data that are not wanted ###
    ###########################################  
    ### filter out data based on the defined creteria
    ### Radar       : Ze lower 17.5 dBZ
    ### Disdrometer : rr larger 0.1 mm/h per minute
    ### Disdrometer : 60 particles or more detected per 1 min
    print('4 >>> filter data based on thresholds')  
    print('      Ze_R <',ZeR_upth)  
    print('      rr >',rr_lowth)      
    print('      particles detected >',nD_lowth)
    ### make a mast to filter out the data
    ### upscale the Disdrometer data to original resolution

    Filter  = ((Ze_radar_120_150 > ZeR_upth) | 
               (Ze_radar_250_300 > ZeR_upth) |
               (rr0_disdro < rr_lowth) |
               (n0_disdro < nD_lowth))
    Filter1 = ((Ze1_radar_120_150 > ZeR_upth) | 
               (Ze1_radar_250_300 > ZeR_upth) |
               (rr_disdro < rr_lowth) |
               (n_disdro < nD_lowth))
    ### Fill the arrays with the data befor filtering the data
    Ze_radar_120_150_F  = np.copy(Ze_radar_120_150)
    Ze_radar_250_300_F  = np.copy(Ze_radar_250_300)
    Ze1_radar_120_150_F = np.copy(Ze1_radar_120_150)
    Ze1_radar_250_300_F = np.copy(Ze1_radar_250_300)
    Ze_disdro_F         = np.copy(Ze_disdro)
    rr_disdro_F         = np.copy(rr_disdro)
    temp1_radar_F       = np.copy(temp1_radar)
    temp_radar_F       = np.copy(temp_radar)
    disdro_Temp0_F       = np.copy(disdro_Temp0)
    disdro_Temp_F       = np.copy(disdro_Temp)
    ### apply the filter to the data - data filtered out are set to NaN  
    Ze_radar_120_150_F[Filter] = np.nan
    Ze_radar_250_300_F[Filter] = np.nan  
    ###
    Ze1_radar_120_150_F[Filter1] = np.nan
    Ze1_radar_250_300_F[Filter1] = np.nan
    Ze_disdro_F[Filter1]         = np.nan
    rr_disdro_F[Filter1]         = 0.0
    temp1_radar_F[Filter1] = np.nan
    disdro_Temp0_F[Filter] = np.nan
    temp_radar_F[Filter]   = np.nan
    disdro_Temp_F[Filter1] = np.nan
    ##
    dummy_case1 = np.zeros(shape=Filter1.shape) #Neuer Filter mit Nullen
    #dummy_case1 = np.where(rr_disdro < 0.1, dummy_case1, 1) #wird 0 für rr_disdro<0.1
    dummy_case1 = np.where(rr_disdro_F < 0.1, dummy_case1, 1) #wird 0 für rr_disdro<0.1
    
    ##########################################
    # Diagnostic plots block 4/5:
    if enable_diagnostic_plots: 
        blubfile.write("\nZe_radar_120_150:\n"+str(Ze_radar_120_150.values))
        blubfile.write("\nZe_radar_250_300:\n"+str(Ze_radar_250_300.values))
        blubfile.write("\nrr_disdro:\n"+str(rr_disdro))
        blubfile.write("\nn_disdro:\n"+str(n_disdro))
        blubfile.write("\nFilter:\n"+str(Filter))
        blubfile.write("\nFilter1:\n"+str(Filter1))
        blubfile.write("\nrr0_disdro[Filter]:\n"+str(rr0_disdro[Filter]))
        blubfile.write("\nn0_disdro[Filter]:\n"+str(n0_disdro[Filter]))
        blubfile.write("\nZe_radar_120_150_F:\n"+str(Ze_radar_120_150_F))
        blubfile.write("\nrr_disdro_F:\n"+str(rr_disdro_F))
        blubfile.write("\nn_disdro[Filter1]:\n"+str(n_disdro[Filter1]))
        blubfile.write("\ntemp1_radar:\n"+str(temp1_radar))
        blubfile.write("\ntemp1_radar_F:\n"+str(temp1_radar_F))
        blubfile.write("\nmeanTenv:\n"+str(np.nanmean(temp1_radar_F)))
        
        blubfile.close()
        
        plt.title("Ze1_radar_120_150_F")
        plt.plot(time_disdro, Ze1_radar_120_150_F)
        plt.xlim(time_radar[0],time_radar[-1])
        plt.ylim(-60,60)
        plt.xlabel("time_disdro")
        plt.savefig(output_path_diagnostic_plots+"Ze1_radar_120_150_F.png")
        plt.close()
        #histograms:
        plt.title("histogram Ze1_radar_120_150_F")
        plt.hist(Ze1_radar_120_150_F, bins=refl_bins)
        plt.ylabel("number of occurences in "+str(len(Ze1_radar_120_150_F))+" bins")
        plt.xlabel("reflectivity")
        plt.savefig(output_path_diagnostic_plots+"Ze1_radar_120_150_F_histogram.png")
        plt.close()
        plt.title("Ze1_radar_250_300_F")
        plt.plot(time_disdro, Ze1_radar_250_300_F)
        plt.ylim(-60,60)
        plt.xlim(time_radar[0],time_radar[-1])
        plt.xlabel("time_disdro")
        plt.savefig(output_path_diagnostic_plots+"Ze1_radar_250_300_F.png")
        plt.close()
        #histograms:
        plt.title("histogram Ze1_radar_250_300_F")
        plt.hist(Ze1_radar_250_300_F, bins=refl_bins)
        plt.ylabel("number of occurences in "+str(len(Ze1_radar_250_300_F))+" bins")
        plt.xlabel("reflectivity")
        plt.savefig(output_path_diagnostic_plots+"Ze1_radar_250_300_F_histogram.png")
        plt.close()
    ########################################
    
    ###
    #################################################
    ### Forward simulation of the Disdrometer DSD ###
    #################################################   
    print('5 >>> forward calculation of the Disdrometer size disdribution')
 
    meanTenv = derive_mean_temperature(temp1_radar_F, disdro_Temp_F, cancellation_file)
    #print("meanTenv: ", meanTenv)

    if (277.15 > meanTenv) :
        print(' >>> mean temperature below 4 C -> no forward calculation')
        ZeR_250_300_disdro      = np.empty((time_disdro.shape))
        ZeR_250_300_disdro[:]   = np.NaN       
        ZeR_250_300_disdro_F    = np.empty((time_disdro.shape))
        ZeR_250_300_disdro_F[:] = np.NaN      
        ZeR_250_300_disdro      = np.empty((time_disdro.shape))
        ZeR_250_300_disdro[:]   = np.NaN     
        ZeR_250_300_disdro_F    = np.empty((time_disdro.shape))
        ZeR_250_300_disdro_F[:] = np.NaN
        #############
        if enable_diagnostic_plots:
            cancellation_file.write("Temperature was determined, but is still below 4°C.\n")
        ##############
        continue
                
    else:
        filename_LUT = t_matrix_LUT(meanTenv)
        if filename_LUT=="no_rain":
            #############
            if enable_diagnostic_plots:
                cancellation_file.write("no rain\n")
            ##############
            continue
            
        ##################
        ### Simulation ###
        ##################
        if 'filename_LUT' in locals() :
            ### FORWARD SIMULATION OF THE DISDROMETER DSD TO ZE! :
            fwd_DF = FWD_sim(filename_LUT, pDF.index, nDF.values.T, bin_edges)
            
            ##############################
            # Diagnostic plots block 5/5:
            if enable_diagnostic_plots: 
                ZeR_120_150_disdro = fwd_DF['Ze_tmm'] - fwd_DF['A']*0.0005*(height_av1top+height_av1bot)
                plt.title("ZeR_120_150_disdro forward simulated by FWD_sim()")
                plt.plot(time_disdro,ZeR_120_150_disdro)
                plt.ylabel("ZeR_120_150_disdro forward simulated")
                plt.xlim(time_disdro[0],time_disdro[-1])
                plt.ylim(-60,60)
                plt.xlabel("time_disdro")
                plt.savefig(output_path_diagnostic_plots+"ZeR_120_150_disdro_forward_simulated.png")
                plt.close()
                plt.title("histogram ZeR_120_150_disdro")
                plt.hist(ZeR_120_150_disdro, bins=refl_bins)
                plt.ylabel("number of occurences in "+str(len(ZeR_120_150_disdro))+" bins")
                plt.xlabel("reflectivity")
                plt.savefig(output_path_diagnostic_plots+"ZeR_120_150_disdro_histogram.png")
                plt.close()
            ##############################
            
            # ATTENUATION CORRECTION! :
            ZeR_120_150_disdro            = fwd_DF['Ze_tmm'] - fwd_DF['A']*0.0005*(height_av1top+height_av1bot)
            ZeR_120_150_disdro_F          = np.copy(ZeR_120_150_disdro)
            ZeR_120_150_disdro_F[Filter1] = np.nan
            ZeR_250_300_disdro            = fwd_DF['Ze_tmm'] - fwd_DF['A']*0.0005*(height_av2top+height_av2bot)
            ZeR_250_300_disdro_F          = np.copy(ZeR_250_300_disdro)
            ZeR_250_300_disdro_F[Filter1] = np.nan
        else:
            ZeR_250_300_disdro          = np.zeros((time_disdro.shape))
            ZeR_250_300_disdro_F        = np.zeros((time_disdro.shape))
            ZeR_250_300_disdro          = np.zeros((time_disdro.shape))
            ZeR_250_300_disdro_F        = np.zeros((time_disdro.shape))

    ###
    ####################################
    ### Time filtering of the cases: ###
    ####################################   
    ### filter out data based on the duration of the rain. Within 1h not more 
    ### then 10 minutes rain gap should exist to be counted as an event.
    ### Start the count when the first minute of Disdrometer data fullfill 
    ### the criteria of the thresholds set above (see "(4)"):
    ### Disdrometer : rr larger 0.1 mm/h per minute
    ### Disdrometer : 60 particles or more detected per 1 min
    print('6 >>> filter data based on duration of the event')  
    print('      not more then 10 minutes rain gap within 1h')  
    ### identify the start index of the events
    start_case1 = np.diff(dummy_case1)    
    start_case1[start_case1 == -1] = 0
    index_case1 = np.array(np.nonzero(start_case1))
    index_case1 = index_case1[0,0::]
    
    case_flag    = np.zeros(shape=dummy_case1.shape)
    for tt in range(len(index_case1)-1):
        i1 = index_case1[tt]
        i2 = index_case1[tt+1]  
        case_check05 = np.sum(dummy_case1[i1+1:i1+30])
        case_check   = np.sum(dummy_case1[i1+1:i1+60])
        case_check15 = np.sum(dummy_case1[i1+1:i1+180])       
        case_check2  = np.sum(dummy_case1[i1+1:i1+120])
        case_check3  = np.sum(dummy_case1[i1+1:i1+180])
        
        if i1 >= len(dummy_case1)-29 :
            i2 = len(dummy_case1)-1   
            if 50/60 < np.sum(dummy_case1[i1::])/len(dummy_case1[i1::]) :
                case_check05 = 26
            else:
                case_check05 = 0   
        if i1 >= len(dummy_case1)-59 :
            i2 = len(dummy_case1)-1   
            if 50/60 < np.sum(dummy_case1[i1::])/len(dummy_case1[i1::]) :
                case_check = 51
            else:
                case_check = 0                     
        if i1 >= len(dummy_case1)-89 :
            i2 = len(dummy_case1)-1   
            if 50/60 < np.sum(dummy_case1[i1::])/len(dummy_case1[i1::]) :
                case_check15 = 76
            else:
                case_check15 = 0 
        if i2 >= len(dummy_case1)-119 :
            i2 = len(dummy_case1)-1
            if 50/60 < np.sum(dummy_case1[i1::])/len(dummy_case1[i1::]) :
                case_check2 = 101
            else:
                case_check2 = 0
        if case_check05 >= 26 :
            case_flag[i1+1:i1+30] = 1  
            print(tt, case_check05)
        if case_check >= 50 :
            case_flag[i1+1:i1+60] = 1  
            print(tt, case_check)
        if case_check15 >= 76 :
            case_flag[i1+1:i1+90] = 1            
        if case_check2 >= 100 :
            case_flag[i1:i1+119] = 1   
        if case_check3 >= 150 :
            case_flag[i1:i1+179] = 1
            
    ### TEST PLOT #############################
    fig = plt.figure(figsize=(10.0, 2.5))
    plt.plot(time_disdro, case_flag-1, 'or', label="case duration flag")
    plt.plot(time_disdro, rr_disdro_F, '.b', label="rr")
    plt.plot(time_disdro, dummy_case1, 'c',  label="case identification flag")
    plt.legend(loc="center left", bbox_to_anchor=(0.85, 0.85), 
               ncol=1, framealpha=1., fontsize = 8)
    plt.grid(True, which="both")  
    plt.xlim([time_disdro[0], time_disdro[-1]])
    plt.ylim([0, np.nanmax(rr_disdro_F)+2])
    ###########################################
    
    
    ###################################################################
    ### Identification of the start and the end index of the cases: ###
    ###################################################################   
    ### find the start index of the cases indentified for the Disdrometer
    print('7 >>> filter data based on the indentified cases')  
    diff_flag1  = np.diff(case_flag)
    start_flag1 = diff_flag1
    start_flag1[start_flag1 == -1] = 0
    indexS_flag1 = np.array(np.nonzero(start_flag1)) 
    indexS_flag1 = indexS_flag1[0,:]
    print(indexS_flag1)
    ### find the end index of the cases indentified for the Disdrometer
    diff_flag1  = np.diff(case_flag)
    end_flag1 = diff_flag1
    end_flag1[end_flag1 == 1] = 0
    indexE_flag1 = np.array(np.nonzero(end_flag1)) 
    indexE_flag1 = indexE_flag1[0,:]
    print(indexE_flag1)
    if len(indexS_flag1) - len(indexE_flag1) == 1:
        value_to_add = int(len(end_flag1))
        indexE_flag1 = np.append(indexE_flag1, value_to_add)
        print(indexS_flag1, indexE_flag1)

    ### Find the indexes of the cases indentified for the Radar
    indexS_flag0 = np.zeros(shape=indexS_flag1.shape)
    indexE_flag0 = np.zeros(shape=indexS_flag1.shape)
    time = np.sort(time_radar)
    for ii in range(0,len(indexS_flag1)) :
        print (ii)
        IndS             = indexS_flag1[ii]
        IndE             = indexE_flag1[ii]
        if (time_disdro[IndS] < time[-1] and time_disdro[IndE] < time[-1]):
            indexS_flag0[ii] = np.min(np.where(time_disdro[IndS] < time))
            indexE_flag0[ii] = np.min(np.where(time_disdro[IndE] < time))
        elif ((IndE == len(end_flag1)) and (time_disdro[IndE-1] < time[-1])):
            indexS_flag0[ii] = np.min(np.where(time_disdro[IndS] < time))
            indexE_flag0[ii] = time[-1::]            
        else:
            indexS_flag0[ii] = 0
            indexE_flag0[ii] = 1            

    ##############################
    ### Calculate the off-sets ###
    ##############################   
    print('8 >>> calculate offsets if any sufficent data/case identified') 
    ### initealizing arrays
    off_set0               = np.zeros(shape=time_radar.shape)
    off_set0[:]            = np.nan
    Ze_radar_120_150_F2    = np.zeros(shape=time_radar.shape)
    Ze_radar_250_300_F2    = np.zeros(shape=time_radar.shape)
    Ze_radar_120_150_F2[:] = np.nan
    Ze_radar_250_300_F2[:] = np.nan
    offset_day_l1          = np.zeros(shape=time_radar.shape)
    offset_day_l2          = np.zeros(shape=time_radar.shape)
    ###
    off_set1                = np.zeros(shape=time_disdro.shape)
    off_set1[:]             = np.nan
    Ze_disdro_120_150_F2    = np.zeros(shape=time_disdro.shape)
    Ze_disdro_120_150_F2[:] = np.nan
    Ze_disdro_250_300_F2    = np.zeros(shape=time_disdro.shape)
    Ze_disdro_250_300_F2[:] = np.nan    
    Ze1_radar_120_150_F2    = np.zeros(shape=time_disdro.shape)
    Ze1_radar_250_300_F2    = np.zeros(shape=time_disdro.shape)
    Ze1_radar_120_150_F2[:] = np.nan
    Ze1_radar_250_300_F2[:] = np.nan
    if any(indexS_flag1) : ### if a case is identified then do
        print('      sufficent data/case identified = ',int(len(indexS_flag1)))
        #################################################
        ### Calculate the off-set per identified case ###
        #################################################   
        for ii in range(0, len(indexS_flag1)) : # loop over the number of events 
            IndS0 = int(indexS_flag0[ii])
            IndE0 = int(indexE_flag0[ii])
            IndS1 = indexS_flag1[ii]
            IndE1 = indexE_flag1[ii]    
            row_Radar_120_150 = Ze_radar_120_150_F[IndS0:IndE0][~np.isnan(Ze_radar_120_150_F[IndS0:IndE0])] # NaN-valuse are filtered out 
            row_Radar_250_300 = Ze_radar_250_300_F[IndS0:IndE0][~np.isnan(Ze_radar_250_300_F[IndS0:IndE0])] # NaN-valuse are filtered out 
            row_Disd_120_150  = ZeR_120_150_disdro_F[IndS1:IndE1][~np.isnan(ZeR_120_150_disdro_F[IndS1:IndE1])]   # NaN-valuse are filtered out 
            row_Disd_250_300  = ZeR_250_300_disdro_F[IndS1:IndE1][~np.isnan(ZeR_250_300_disdro_F[IndS1:IndE1])]   # NaN-valuse are filtered out 
            ### Setting the number of bins in the histogram
            min_Ze = -10 # lowerest bin of the histogram: no lower valuse expected due to the filtering
            max_Ze = 25  # highesr bin of the histogram: no higher valuse expected due to the filtering
            bins  = 140   # number of bins in the histogram
        
            ### here you choose which interval to pick, the longest the better
            input_D1 = row_Disd_120_150
            input_R1 = row_Radar_120_150
            
            ##############################
            #plots by apschera 2021-12-06:
            """
            plt.title("disdro ref before calculation: row_Disd_120_150")
            plt.plot(time_disdro,row_Disd_120_150)
            #ergab Fehler für jue 20210202 ValueError: x and y must have same first dimension, but have shapes (1440,) and (137,)
            plt.ylabel("reflectivity")
            plt.xlim(time_disdro[0],time_disdro[-1])
            plt.ylim(-60,60)
            plt.xlabel("time_disdro")
            plt.savefig(output_path_diagnostic_plots+"row_Disd_120_150_before_calc.png")
            plt.close()
            plt.title("histogram row_Disd_120_150")
            plt.hist(row_Disd_120_150, bins=refl_bins)
            plt.ylabel("number of occurences in "+str(len(row_Disd_120_150))+" bins")
            plt.xlabel("reflectivity")
            plt.savefig(output_path_diagnostic_plots+"row_Disd_120_150_histogram.png")
            plt.close()
            plt.title("Radar before calculation: row_Radar_120_150")
            plt.plot(time_disdro,row_Radar_120_150)
            plt.ylabel("reflectivity")
            plt.xlim(time_disdro[0],time_disdro[-1])
            plt.ylim(-60,60)
            plt.xlabel("time_disdro")
            plt.savefig(output_path_diagnostic_plots+"row_Radar_120_150_before_calc.png")
            plt.close()
            plt.title("histogram row_Radar_120_150")
            plt.hist(row_Radar_120_150, bins=refl_bins)
            plt.ylabel("number of occurences in "+str(len(row_Radar_120_150))+" bins")
            plt.xlabel("reflectivity")
            plt.savefig(output_path_diagnostic_plots+"row_Radar_120_150_histogram.png")
            plt.close()
            """
            ##################################
            
            if (len(input_D1) < 49. or len(input_R1) < 49.) :
                continue 
            ### Calculate the offset taking all mesurements of the event
            offset_l1 = calculate_offset(input_R1, input_D1, method='all',
                                         binsize=bins, range_val=[min_Ze, max_Ze], 
                                         shiftrange=20, shiftstep=0.1)
            ### Calculate the offset taking all mesurements of the event    
            input_D2 = row_Disd_250_300            
            input_R2 = row_Radar_250_300
            ### Calculate the offset taking all mesurements of the event
            offset_l2 = calculate_offset(input_R2, input_D2, method='all',
                                         binsize=bins, range_val=[min_Ze, max_Ze], 
                                         shiftrange=20, shiftstep=0.1)
            print(offset_l1["offset_calc_median"])
            print(offset_l2["offset_calc_median"])
            off_set0[IndS0:IndE0] = offset_l2["offset_calc_median"]    
            off_set1[IndS1:IndE1] = offset_l1["offset_calc_median"]    
            #print('compare len() : ', jj, len(row_Radar), len(row_Disd), len(row_RR))
            Ze_radar_120_150_F2[IndS0:IndE0] = Ze_radar_120_150_F[IndS0:IndE0]
            Ze_radar_250_300_F2[IndS0:IndE0] = Ze_radar_250_300_F[IndS0:IndE0]
            Ze_disdro_120_150_F2[IndS1:IndE1] = ZeR_120_150_disdro_F[IndS1:IndE1]
            Ze_disdro_250_300_F2[IndS1:IndE1] = ZeR_250_300_disdro_F[IndS1:IndE1]            
            Ze1_radar_120_150_F2[IndS1:IndE1] = Ze1_radar_120_150_F[IndS1:IndE1]
            Ze1_radar_250_300_F2[IndS1:IndE1] = Ze1_radar_250_300_F[IndS1:IndE1]
        ### TEST PLOT #############################
        fig = plt.figure(figsize=(10.0, 2.5))
        plt.plot(time_disdro, -25+case_flag*25, 'or', label="case duration flag")
        plt.plot(time_disdro, Ze_disdro_120_150_F2, '.k',  label="ZeD")        
        plt.plot(time_disdro, Ze1_radar_120_150_F2, '.c', label="ZeR_1 0.12-0.15 km")
        plt.plot(time_radar,  Ze_radar_250_300_F2,  '.b', label="ZeR_0 0.12-0.15 km")
        plt.legend(loc="center left", bbox_to_anchor=(0.85, 0.85), 
               ncol=1, framealpha=1., fontsize = 8)
        plt.grid(True, which="both")
        plt.ylim([-15, 45.])
        plt.xlim([time_disdro[0], time_disdro[-1]])
        ###########################################
        
        ##############################################
        ### Calculate the mean off-set per day     ### 
        ### mean of all data from identified cases ###
        ##############################################    
        print('      >>> calculate mean off set per day! ')
        row_120_150 = Ze_radar_120_150_F2[:][~np.isnan(Ze_radar_120_150_F2[:])] # NaN-valuse are filtered out 
        row_250_300 = Ze_radar_250_300_F2[:][~np.isnan(Ze_radar_250_300_F2[:])] # NaN-valuse are filtered out 
        dis_120_150 = Ze_disdro_120_150_F2[:][~np.isnan(Ze_disdro_120_150_F2[:])] # NaN-valuse are filtered out 
        dis_250_300 = Ze_disdro_250_300_F2[:][~np.isnan(Ze_disdro_250_300_F2[:])]
        ### Setting the number of bins in the histogram
        min_Ze = -10 # lowerest bin of the histogram: no lower valuse expected due to the filtering
        max_Ze = 25  # highesr bin of the histogram: no higher valuse expected due to the filtering
        bins  = 140   # number of bins in the histogram
        ### here you choose which interval to pick, the longest the better
        input_dis1 = dis_120_150
        input_rad1 = row_120_150
        if (len(input_dis1) < 49. or len(input_rad1) < 49.) :
            continue        
        ### Calculate the offset taking all mesurements of the event
        offset_low = calculate_offset(input_rad1, input_dis1, method='all',
                                     binsize=bins, range_val=[min_Ze, max_Ze], 
                                     shiftrange=20, shiftstep=0.1)
        ### Calculate the offset taking all mesurements of the event    
        input_dis2 = dis_250_300
        input_rad2 = row_250_300        
        ### Calculate the offset taking all mesurements of the event
        offset_high = calculate_offset(input_rad2, input_dis2, method='all',
                                     binsize=bins, range_val=[min_Ze, max_Ze], 
                                     shiftrange=20, shiftstep=0.1)
        print(offset_low["offset_calc_median"], offset_high["offset_calc_median"])
        offset_day_l1[:] = offset_low["offset_calc_median"]  
        offset_day_l2[:] = offset_high["offset_calc_median"]

    ### TEST PLOT #############################
    ### -------------------------------------------------------------------
    timeformat = mdates.DateFormatter('%H:%M') # time format used in the plots
    fs         = 14.
    ms = 3 # font size of the labeling
    fig, axes = plt.subplots(figsize=[12.0, 8.0], nrows=2, sharex=False, 
                             squeeze=True, gridspec_kw = {'height_ratios':[1.,1.]})
    ### ---------------------------------------------------------------------
    axes[0].plot(time_disdro, -25+case_flag*25, 'or', label="case duration flag")
    axes[0].plot(time_disdro, Ze_disdro_120_150_F2, '.k',  label="ZeD")        
    axes[0].plot(time_disdro, Ze1_radar_120_150_F2, '.c', label="ZeR_1 0.12-0.15 km")
    axes[0].plot(time_radar,  Ze_radar_250_300_F2,  '.b', label="ZeR_0 0.25-0.30 km")
    axes[0].legend(loc='center left', bbox_to_anchor=(0.83, 0.82), 
                   ncol=1, framealpha=1., fontsize = 8)
    axes[0].set_ylabel('Ze [dBZ]', fontsize=fs)
    axes[0].text(0.01, 0.87, "a) Comparisson radar disdrometer "+date+" , "+location,
                 transform=axes[0].transAxes, ha="left", fontsize=fs+1)
    axes[0].set_xlim(start, end)  # limets of the x-axes
    axes[0].xaxis.set_major_formatter(timeformat) # change time format                    
    axes[0].set_ylim([-15, 45.])  # limets of the x-axes 
    axes[0].grid(True, which="both")
    ### -------------------------------------------------------------------
    axes[1].plot(time_radar, offset_day_l1, 'c', label="daily mean 0.12-0.15 km")
    axes[1].plot(time_radar, offset_day_l2, 'b', label="daily mean 0.25-0.30 km")
    axes[1].plot(time_radar, off_set0, '.m', label="offset 0.12-0.15 km" )
    axes[1].plot(time_disdro, off_set1, '.r', label="offset 0.25-0.30 km")   
    axes[1].legend(loc='center left', bbox_to_anchor=(0.83, 0.82), 
                   ncol=1, framealpha=1., fontsize = 8)
    axes[1].set_ylabel('Ze [dBZ]', fontsize=fs)
    axes[1].text(0.01, 0.87, "b) "+date+" Ze-offset Radar/Disdrometer, "+location,
                 transform=axes[1].transAxes, ha="left", fontsize=fs+1)
    axes[1].set_xlim(start, end)  # limets of the x-axes
    axes[1].xaxis.set_major_formatter(timeformat) # change time format                    
    axes[1].set_ylim([0, 17.5])  # limets of the x-axes 
    axes[1].grid(True, which="both")
    ### -------------------------------------------------------------------
    OutPutDirectoryCeck(path_output_plots, yyyy, mm)      
    plt.savefig(path_output_plots+'/'+yyyy+'/'+mm+'/'+yyyymmdd+'_'+site+'_ze_offset_radar_and_disdrometer.png')     
    ###########################################


    ############################
    ### PLOT DAILY QUICKLOOK ###
    ############################
    timeformat = mdates.DateFormatter('%H:%M') # time format used in the plots
    fs         = 14.
    ms = 3 # font size of the labeling
    dataRR          = rr_disdro
    RadarZe         = 10*np.log10(RadarData.values)
    dataDZe         = Ze_disdro.values       # Disdrometer Ze forward modeled to Radar
    dataRZe_120_150 = 10*np.log10(Ze1_radar_120_150_F) # Radar Ze between 120-150m
    dataRZe_250_300 = 10*np.log10(Ze1_radar_120_150_F) # Radar Ze between 250-300m 
    dataNM          = disdro_NMmatrix
    dataVM          = disdro_VMmatrix  
#    dataNM          = nDF*20
#    dataVM          = vDF*20
        # colors in total
    colors2 = plt.cm.jet_r(np.linspace(1, 0, 33))
    colors1 = plt.cm.Pastel2(np.linspace(1, 0, 1))
    colors3 = plt.cm.gray_r(np.linspace(1, 0, 1))
    # combine them and build a new colormap
    colors      = np.vstack((colors1, colors2, colors3))
    Ze_Colormap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    ### -------------------------------------------------------------------
    fig, axes = plt.subplots(figsize=[12.0, 17.0], nrows=5, sharex=False, 
                             squeeze=True, gridspec_kw = {'height_ratios':[1.,1.,1.,0.75,0.75]})
    ### ---------------------------------------------------------------------
    im = axes[0].pcolormesh(time_radar, RadarData.range/1000., RadarZe.T, 
                       vmin=-15, vmax=20.0, cmap=Ze_Colormap)
    axes[0].axhline(y=height_av1top/1000., color='grey', linestyle='-', linewidth = 3, alpha=1.) 
    axes[0].axhline(y=height_av1bot/1000., color='grey', linestyle='-', linewidth = 3, alpha=1.) 
    axes[0].axhline(y=height_av2top/1000., color='c', linestyle='-', linewidth = 3, alpha=1.) 
    axes[0].axhline(y=height_av2bot/1000., color='c', linestyle='-', linewidth = 3, alpha=1.)     
    axes[0].set_ylabel('range [km]', fontsize=fs)
    axes[0].text(0.01, 0.87, "a) "+date+" Radar reflectivity, "+location,
                 transform=axes[0].transAxes, ha="left", fontsize=fs+1)
    axes[0].xaxis.set_major_formatter(timeformat) # change time format                
    ### Add the colorbar outside...
    box = axes[0].get_position()
    pad, width = 0.01, 0.02
    cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
    axes[0].set_ylim(0., 1.)  # limets of the x-axes)  # limets of the x-axes 
    fig.colorbar(im, cax=cax, extend='both')
    plt.ylabel('Ze [dBZ]', fontsize=fs)
    axes[0].grid(True, which="both")  
    ### ---------------------------------------------------------------------
    im = axes[1].pcolormesh(time_disdro, dclasses, dataNM.T, 
                            norm=matplotlib.colors.LogNorm(), vmin=0.9, vmax=500, cmap='inferno')
    axes[1].grid(True, which="both")
    axes[1].set_ylabel('diameter [mm] ', fontsize=fs)
    axes[1].text(0.01, 0.90, "b) "+date+" Disdrometer, mean volumer diameter, "+location,
                 transform=axes[1].transAxes, ha="left", fontsize=fs+1)
    axes[1].set_xlim(start, end)  # limets of the x-axes
    axes[1].xaxis.set_major_formatter(timeformat) # change time format                
    ### Add the colorbar outside...
    box = axes[1].get_position()
    pad, width = 0.01, 0.02
    cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
    axes[1].set_ylim(0., 5.0)  # limets of the x-axes)  # limets of the x-axes 
    fig.colorbar(im, cax=cax, extend='max')
    plt.ylabel('counts', fontsize=fs)
    ### ---------------------------------------------------------------------
    im = axes[2].pcolormesh(time_disdro, dclasses, dataVM.T, 
                            norm=matplotlib.colors.LogNorm(), vmin=1, vmax=500, cmap='inferno')
    axes[2].grid(True, which="both")
    axes[2].set_ylabel('fall speed [m s$^{-1}$] ', fontsize=fs)
    axes[2].text(0.01, 0.90, "c) "+date+" Disdrometer, mean particle fall speed, "+location,
                 transform=axes[2].transAxes, ha="left", fontsize=fs+1)
    axes[2].set_xlim(start, end)  # limets of the x-axes
    axes[2].xaxis.set_major_formatter(timeformat) # change time format                
    ### Add the colorbar outside...
    box = axes[2].get_position()
    pad, width = 0.01, 0.02
    cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
    axes[2].set_ylim(0., 10.)  # limets of the x-axes)  # limets of the x-axes 
    fig.colorbar(im, cax=cax, extend='max')
    plt.ylabel('counts', fontsize=fs)
    ### --------------------------------------------------------------------
    axes[3].plot(time_disdro, dataRR,
                 'o-', markersize=3., color='b', label="rr")
    axes[3].axhline(y=0.01, color='grey', linestyle='-', linewidth = 8, alpha=0.75) 
    axes[3].set_ylabel('Rain rate [mm h$^{-1}$]', fontsize=fs)
    axes[3].text(0.01, 0.87, "d) "+date+" Disdrometer rain rate, "+location,
                 transform=axes[3].transAxes, ha="left", fontsize=fs+1)
    axes[3].set_xlim(start, end)  # limets of the x-axes
    axes[3].xaxis.set_major_formatter(timeformat) # change time format                    
    axes[3].set_ylim(-0.05, np.max(dataRR)+0.5)  # limets of the x-axes 
    axes[3].grid(True, which="both")
    axes[3].legend(loc='center left', bbox_to_anchor=(0.92, 0.88), framealpha=1.)
    ### ---------------------------------------------------------------------
    axes[4].plot(time_disdro, n_disdro,
                 'o-', markersize=3., color='m', label="No. of drops")
    axes[4].set_ylabel('counts per min', fontsize=fs)
    axes[4].set_xlabel('UTC', fontsize=fs)
    axes[4].text(0.01, 0.87, "e) "+date+" Counted drops per minute, "+location,
                 transform=axes[4].transAxes, ha="left", fontsize=fs+1)
    axes[4].set_xlim(start, end)  # limets of the x-axes
    axes[4].xaxis.set_major_formatter(timeformat) # change time format                
    axes[4].set_yscale('log')
    axes[4].set_ylim(10, 5000)  # limets of the x-axes)  # limets of the x-axes 
    axes[4].grid(True, )  
    axes[4].legend(loc='center left', bbox_to_anchor=(0.84, 0.88), framealpha=1.)
    ### -------------------------------------------------------------------
    OutPutDirectoryCeck(path_output_plots, yyyy, mm)      
    plt.savefig(path_output_plots+'/'+yyyy+'/'+mm+'/'+yyyymmdd+'_'+site+'_input_radar_and_disdrometer.png')     
    plt.show()    
    ###

## In[]

    ###############################
    ### Create Output Data File ###
    ###############################     
    ###
    ###################
    ### DISDROMETER ###   
    ###################
    ### 1) Disdrometer Rain Rate ---
    xr_rr_disdro = xr.DataArray(
                     rr_disdro,
                     dims=('time_d'),
                     coords=[time_disdro] )
    xr_rr_disdro.attrs['GEOMS_name']     = 'DISDROMETER.RAIN.RATE'
    xr_rr_disdro.attrs['long_name']      = 'rain_rate'
    xr_rr_disdro.attrs['standatrd_name'] = 'rr'    
    xr_rr_disdro.attrs['long_name']      = 'rain rate monitored by the Parsivel2'
    xr_rr_disdro.attrs['units']          = 'mm h-1'  
    xr_rr_disdro.attrs['fill_value']     = 'NaN' 
    ### 2) Disdrometer number of detected particles ---
    xr_n_disdro = xr.DataArray(
                     n_disdro,
                     dims=('time_d'),
                     coords=[time_disdro] )
    xr_n_disdro.attrs['GEOMS_name']     = 'DISDROMETER.PARTICLES'
    xr_n_disdro.attrs['long_name']      = 'number_of_particle'    
    xr_n_disdro.attrs['standatrd_name'] = 'n_particles'
    xr_n_disdro.attrs['long_name']      = 'number of particles counted by the Parsivel2'
    xr_n_disdro.attrs['units']          = 'no.'  
    xr_n_disdro.attrs['fill_value']     = 'NaN' 
    #### 3) Disdrometer Ze ---  
    xr_ze_disdro = xr.DataArray(
                     10.**(Ze_disdro/10.),
                     dims=('time_d'),
                     coords=[time_disdro] )
    xr_ze_disdro.attrs['GEOMS_name']     = 'DISDROMETER.REFLECTIVITY.FACTOR'
    xr_ze_disdro.attrs['long_name']      = 'disdrometer_equivalent_reflectivity_factor'     
    xr_ze_disdro.attrs['standatrd_name'] = 'ze_d'
    xr_ze_disdro.attrs['long_name']      = 'Equivalent reflectivity retrieved from Parsival2 drop size disdribution'
    xr_ze_disdro.attrs['units']          = 'mm6 m-3' 
    xr_ze_disdro.attrs['fill_value']     = 'NaN'
    ### 4) Disdrometer Ze forward modeled for height 1 ---
    xr_ze_disdro_l1 = xr.DataArray(
                     10.**(ZeR_120_150_disdro/10.),
                     dims=('time_d'),
                     coords=[time_disdro] )
    xr_ze_disdro_l1.attrs['GEOMS_name']     = 'DISDROMETER.SIMULATED.REFLECTIVITY.FACTOR1'
    xr_ze_disdro_l1.attrs['long_name']      = 'simulated_equivalent_reflectivity_factor_height_1'        
    xr_ze_disdro_l1.attrs['standatrd_name'] = 'ze_d_siml1'
    xr_ze_disdro_l1.attrs['long_name']      = 'Forward simulated equivalent equivalent reflectivity from Parsival2 drop size disdribution at height level 1'
    xr_ze_disdro_l1.attrs['units']          = 'mm6 m-3' 
    xr_ze_disdro_l1.attrs['fill_value']     = 'NaN'
    xr_ze_disdro_l1.attrs['comment']        = ['Forward simulated equivalent reflectivity '
                                               'from Parsival2 drop size disdribution '+
                                               'using T-matrix. The forwars simulation '+ 
                                               'performed with PyTMatrix by Jusi Leinonen '+ 
                                               '(https://github.com/jleinonen/pytmatrix)'
                                                 ]     
    ### 5) Disdrometer Ze forward modeled for height 2 ---
    xr_ze_disdro_l2 = xr.DataArray(
                     10.**(ZeR_250_300_disdro/10.),
                     dims=('time_d'),
                     coords=[time_disdro] )
    xr_ze_disdro_l2.attrs['GEOMS_name']     = 'DISDROMETER.SIMULATED.REFLECTIVITY.FACTOR2'
    xr_ze_disdro_l2.attrs['long_name']      = 'simulated_equivalent_reflectivity_factor_height_2'            
    xr_ze_disdro_l2.attrs['standatrd_name'] = 'ze_d_siml2'
    xr_ze_disdro_l2.attrs['long_name']      = 'forward simulated equivalent reflectivity from Parsival2 drop size disdribution at height level 2'
    xr_ze_disdro_l2.attrs['units']          = 'mm6 m-3' 
    xr_ze_disdro_l2.attrs['fill_value']     = 'NaN' 
    xr_ze_disdro_l2.attrs['comment']        = ['Forward simulated equivalent reflectivity '
                                               'from Parsival2 drop size disdribution '+
                                               'using T-matrix. The forwars simulation '+ 
                                               'performed with PyTMatrix by Jusi Leinonen '+ 
                                               '(https://github.com/jleinonen/pytmatrix). ' + 
                                               'No filtering is applyed.'
                                                 ]    
    ############################################
    ### final filtered Disdrometer data sets ###
    ############################################
    ### 6) Disdrometer Ze forward modeled and finally filtered for height 1 ---
    xr_ze_F_disdro_l1 = xr.DataArray(
                     10.**(Ze_disdro_120_150_F2/10.),
                     dims=('time_d'),
                     coords=[time_disdro] )
    xr_ze_F_disdro_l1.attrs['GEOMS_name']     = 'DISDROMETER.REFLECTIVITY.FACTOR.1.SIMULATED.FILTERED'
    xr_ze_F_disdro_l1.attrs['long_name']      = 'simulated_filtered_equivalent_reflectivity_factor_height_1'        
    xr_ze_F_disdro_l1.attrs['standatrd_name'] = 'ze_d_l1'
    xr_ze_F_disdro_l1.attrs['long_name']      = 'forward simulated and filtered reflectivity from Parsival2 drop size disdribution at height level 1'
    xr_ze_F_disdro_l1.attrs['units']          = 'mm6 m-3' 
    xr_ze_F_disdro_l1.attrs['fill_value']     = 'NaN' 
    xr_ze_F_disdro_l1.attrs['comment']        = ['Data filtered according to following creteria: '+
                                                 'Radar: Ze lower 17.5 dBZ - '+ 
                                                 'Disdrometer: rr larger 0.1 mm/h per minute - '+ 
                                                 'Disdrometer: 30 particles or more detected per 1 min; '+
                                                 'Criteria have to be fullfilled at least for 50 '+
                                                 'minutes witin 1h, 100 for 2 hours, and so on. '
                                                 ]
    ### 7) Disdrometer Ze forward modeled and finally filtered for height 2 ---
    xr_ze_F_disdro_l2 = xr.DataArray(
                     10.**(Ze_disdro_250_300_F2/10.),
                     dims=('time_d'),
                     coords=[time_disdro] )
    xr_ze_F_disdro_l2.attrs['GEOMS_name']     = 'DISDROMETER.REFLECTIVITY.FACTOR.2.SIMULATED.FILTERED'
    xr_ze_F_disdro_l2.attrs['long_name']      = 'simulated_filtered_equivalent_reflectivity_factor_height_2'        
    xr_ze_F_disdro_l2.attrs['standatrd_name'] = 'ze_d_l2'
    xr_ze_F_disdro_l2.attrs['long_name']      = 'forward simulated and filtered reflectivity from Parsival2 drop size disdribution at height level 2'
    xr_ze_F_disdro_l2.attrs['units']          = 'mm6 m-3' 
    xr_ze_F_disdro_l2.attrs['fill_value']     = 'NaN'    
    xr_ze_F_disdro_l2.attrs['comment']        = ['Data filtered according to following creteria: '+
                                                 'Radar: Ze lower 17.5 dBZ - '+ 
                                                 'Disdrometer: rr larger 0.1 mm/h per minute - '+ 
                                                 'Disdrometer: 30 particles or more detected per 1 min; '+
                                                 'Criteria have to be fullfilled at least for 50 '+
                                                 'minutes witin 1h, 100 for 2 hours, and so on. '
                                                 ]
    ###############################################
    ### integers to be filled into the date set ###
    ###############################################
    xr_l1_bot = xr.DataArray(height_av1bot)
    xr_l1_bot.attrs['GEOMS_name']     = 'HEIGHT.LEVEL.BOTTOM1'
    xr_l1_bot.attrs['long_name']      = 'bottom_height_1'        
    xr_l1_bot.attrs['standatrd_name'] = 'bot_l1'
    xr_l1_bot.attrs['long_name']      = 'bottom of height level 1'
    xr_l1_bot.attrs['units']          = 'm'    
    ###
    xr_l1_top = xr.DataArray(height_av1top)
    xr_l1_top.attrs['GEOMS_name']     = 'HEIGHT.LEVEL.TOP1'
    xr_l1_top.attrs['long_name']      = 'top_height_1'      
    xr_l1_top.attrs['standatrd_name'] = 'top_l1'
    xr_l1_top.attrs['long_name']      = 'top of height level 1'
    xr_l1_top.attrs['units']          = 'm'    
    ###
    xr_l2_bot = xr.DataArray(height_av2bot)
    xr_l1_bot.attrs['GEOMS_name']     = 'HEIGHT.LEVEL.BOTTOM2'
    xr_l1_bot.attrs['long_name']      = 'bottom_height_2'      
    xr_l2_bot.attrs['standatrd_name'] = 'bot_l2'
    xr_l2_bot.attrs['long_name']      = 'bottom of height level 12'
    xr_l2_bot.attrs['units']          = 'm'    
    ###
    xr_l2_top = xr.DataArray(height_av2top)
    xr_l2_top.attrs['GEOMS_name']     = 'HEIGHT.LEVEL.TOP2'
    xr_l2_top.attrs['long_name']      = 'top_height_2'      
    xr_l2_top.attrs['standatrd_name'] = 'top_l2'
    xr_l2_top.attrs['long_name']      = 'top of height level 1'
    xr_l2_top.attrs['units']          = 'm'    
    #############
    ### RADAR ###   
    #############
    ### 1) Radar reflectivity filed the first 1km ---
    xr_ze = xr.DataArray(
                     RadarData,
                     dims=('time_r','range'),
                     coords=[time_radar, RadarData.range] )
    ### 2) Radar reflectivity time series at for height 1 ---
    xr_ze_l1 = xr.DataArray(
                     10.**(Ze_radar_120_150/10.),
                     dims=('time_r'),
                     coords=[time_radar] )
    xr_ze_l1.attrs['GEOMS_name']     = 'RADAR.REFLECTIVITY.FACTOR.HEIGHT1'
    xr_ze_l1.attrs['long_name']      = 'equivalent_reflectivity_factor_time_series_heigh_1' 
    xr_ze_l1.attrs['standatrd_name'] = 'ze_l1'
    xr_ze_l1.attrs['long_name']      = 'Equivalent radar reflectivity factor at vertical polarisation at height level 1'
    xr_ze_l1.attrs['units']          = 'mm6 m-3' 
    xr_ze_l1.attrs['valid_range']    =  [str(np.nanmin(10.**(Ze_radar_120_150/10.))), str(np.nanmax(10.**(Ze_radar_120_150/10.)))]  
    xr_ze_l1.attrs['fill_value']     = 'NaN' 
    #### 3) Radar reflectivity time series at for height 2 ---  
    xr_ze_l2 = xr.DataArray(
                     10.**(Ze_radar_250_300/10.),
                     dims=('time_r'),
                     coords=[time_radar] )
    xr_ze_l2.attrs['GEOMS_name']     = 'RADAR.REFLECTIVITY.FACTOR.HEIGHT2'
    xr_ze_l2.attrs['long_name']      = 'equivalent_reflectivity_factor_time_series_heigh_2' 
    xr_ze_l2.attrs['standatrd_name'] = 'ze_l2'
    xr_ze_l2.attrs['long_name']      = 'Equivalent radar reflectivity factor at vertical polarisation at height level 1'
    xr_ze_l2.attrs['units']          = 'mm6 m-3' 
    xr_ze_l2.attrs['valid_range']    =  [str(np.nanmin(10.**(Ze_radar_250_300/10.))), str(np.nanmax(10.**(Ze_radar_250_300/10.)))]  
    xr_ze_l2.attrs['fill_value']     = 'NaN' 
    ### 4) Temperature time series as measued from the Met sattion at the Radar ---
    xr_temp = xr.DataArray(
                     temp_radar,
                     dims=('time_r'),
                     coords=[time_radar] )
    ############################################
    ### final filtered Radar data sets ###
    ############################################
    ### 6) Disdrometer Ze forward modeled and finally filtered for height 1 ---
    xr_ze_F_l1 = xr.DataArray(
                     10.**(Ze_radar_120_150_F2/10.),
                     dims=('time_r'),
                     coords=[time_radar] )
    xr_ze_F_l1.attrs['GEOMS_name']     = 'RADAR.REFLECTIVITY.FACTOR.1.FILTERED'
    xr_ze_F_l1.attrs['long_name']      = 'equivalent_reflectivity_factor_height_1_filtered'        
    xr_ze_F_l1.attrs['standatrd_name'] = 'ze_l1'
    xr_ze_F_l1.attrs['long_name']      = 'forward simulated and filtered reflectivity from Parsival2 drop size disdribution at height level 1'
    xr_ze_F_l1.attrs['units']          = 'mm6 m-3' 
    xr_ze_F_l1.attrs['fill_value']     = 'NaN' 
    xr_ze_F_l1.attrs['comment']        = ['Data filtered according to following creteria: '+
                                          'Radar: Ze lower 17.5 dBZ - '+ 
                                          'Disdrometer: rr larger 0.1 mm/h per minute - '+ 
                                          'Disdrometer: 30 particles or more detected per 1 min; '+
                                          'Criteria have to be fullfilled at least for 50 '+
                                          'minutes witin 1h, 100 for 2 hours, and so on. '
                                          ]
    ### 7) Disdrometer Ze forward modeled and finally filtered for height 2 ---
    xr_ze_F_l2 = xr.DataArray(
                     10.**(Ze_radar_250_300_F2/10.),
                     dims=('time_r'),
                     coords=[time_radar] )
    xr_ze_F_l2.attrs['GEOMS_name']     = 'RADAR.REFLECTIVITY.FACTOR.2.FILTERED'
    xr_ze_F_l2.attrs['long_name']      = 'equivalent_reflectivity_factor_height_2_filtered'        
    xr_ze_F_l2.attrs['standatrd_name'] = 'ze_l2'
    xr_ze_F_l2.attrs['long_name']      = 'forward simulated and filtered reflectivity from Parsival2 drop size disdribution at height level 2'
    xr_ze_F_l2.attrs['units']          = 'mm6 m-3' 
    xr_ze_F_l2.attrs['fill_value']     = 'NaN'    
    xr_ze_F_l2.attrs['comment']        = ['Data filtered according to following creteria: '+
                                         'Radar: Ze lower 17.5 dBZ - '+ 
                                         'Disdrometer: rr larger 0.1 mm/h per minute - '+ 
                                         'Disdrometer: 30 particles or more detected per 1 min; '+
                                         'Criteria have to be fullfilled at least for 50 '+
                                         'minutes witin 1h, 100 for 2 hours, and so on. '
                                         ]
    ###
    DateSet_Day = xr.Dataset({
                            'rr':xr_rr_disdro,
                            'n_particles':xr_n_disdro,
                            'ze_d':xr_ze_disdro,
                            'ze_d_l1':xr_ze_disdro_l1,
                            'ze_d_l2':xr_ze_disdro_l2,
                            'ze_d_l1_F':xr_ze_F_disdro_l1,
                            'ze_d_l2_F':xr_ze_F_disdro_l2,
                            'ze':xr_ze,   
                            'ze_l1':xr_ze_l1,
                            'ze_l2':xr_ze_l2,
                            'ze_l1_F':xr_ze_F_l1,
                            'ze_l2_F':xr_ze_F_l2,
                            'ta':xr_temp,
                            'lbot_l1':xr_l1_bot,
                            'top_l1':xr_l1_top,
                            'top_l2':xr_l2_bot,
                            'bot_l2':xr_l2_top                            
                            })    
    ###
    
    ## AFFILIATION of the PI
    DateSet_Day.attrs['PI_NAME']        = 'Pfitzenmaier;Lukas'
    DateSet_Day.attrs['PI_AFFILIATION'] = 'University of Cologne,Institute for Geophysics and Meteorology (UoC);Germany'
    DateSet_Day.attrs['PI_ADDRESS']     = 'University of Cologne,Institute for Geophysics and Meteorology;Albergus-Magnus-Platz;50923;Cologne;Germany'
    DateSet_Day.attrs['PI_MAIL']        = 'l.pfitzenmaier@uni-koeln.de'
    ## Name of the organisation responsible for quality controll of the data
    DateSet_Day.attrs['DO_NAME']        = 'Pfitzenmaier;Lukas;' 
    DateSet_Day.attrs['DO_AFFILIATION'] = 'University of Cologne,Institute for Geophysics and Meteorology (UoC);Germany'
    DateSet_Day.attrs['DO_ADDRESS']     = 'University of Cologne,Institute for Geophysics and Meteorology;Albergus-Magnus-Platz;50923;Cologne;Germany'
    DateSet_Day.attrs['DO_MAIL']        = 'info@joyce.cloud'
    ## Name of the responible data submitter
    DateSet_Day.attrs['DS_NAME_RADAR']        = 'Pfitzenmaier;Lukas' 
    DateSet_Day.attrs['DS_AFFILIATION_RADAR'] = 'University of Cologne,Institute for Geophysics and Meteorology (UoC);Germany'
    DateSet_Day.attrs['DS_ADDRESS_RADAR']     = 'University of Cologne,Institute for Geophysics and Meteorology;Albergus-Magnus-Platz;50923;Cologne;Germany'
    DateSet_Day.attrs['DS_MAIL_RADAR']        = 'info@joyce.cloud'
    ## Name of the responible data submitter
    DateSet_Day.attrs['DS_NAME_DISDROMETER']        = 'Schnitt.Sabrina' 
    DateSet_Day.attrs['DS_AFFILIATION_DISDROMETER'] = 'University of Cologne,Institute for Geophysics and Meteorology (UoC);Germany'
    DateSet_Day.attrs['DS_ADDRESS_DISDROMETER']     = 'University of Cologne,Institute for Geophysics and Meteorology;Albergus-Magnus-Platz;50923;Cologne;Germany'
    DateSet_Day.attrs['DS_MAIL_DISDROMETER']        = 'info@joyce.cloud'
    ## Optional – brief description of the file’s containing data
    DateSet_Day.attrs['DATA_DESCRIPTION_RADAR']  = 'daily; containes profiles of w-band radar reflectifiy at the research station JOYCE, Germany'
    DateSet_Day.attrs['DATA_DESCRIPTION_DISDROMETER'] = 'daily; containes time series of the Parsical2 Disdrometer at the research station JOYCE, Germany'
    ## Describes field of research to which the data belongs and the data acquisition method
    DateSet_Day.attrs['DATA_DISCIPLINE_RADAR']    = 'Atmospheric.Physics;Remote.Sensing;Radar.Profiler';
    DateSet_Day.attrs['DATA_DISCIPLINE_DISDROMETER']    = 'Atmospheric.Physics;In.Situ;Disdrometer';
    ## Specifies the origin of the data (EXPERIMENTAL, MODAL, or both) and the 
    ## spatial characteristic of the data set.Spatial dimensions are: 
    ## 0D = SCALAR; 1D = PROFILE; 2D or more = FIELD. 
    ## STATIONARY for fixed locations and MOVING for moving platforms
    DateSet_Day.attrs['DATA_GROUP_RADAR']  = 'Experimantal;Profile;Stationary'
    DateSet_Day.attrs['DATA_GROUP_DISDROMETER'] = 'Experimantal;Stationary'    
    ## Contains the identification of the location of the reported geophysical quantities
    DateSet_Day.attrs['DATA_LOCATION_RADAR']  = 'Juelich Observatory for Cloud Evolution (JOYCE), Huelich, Germany'
    DateSet_Day.attrs['DATA_LOCATION_DISDROMETER'] = 'Juelich Observatory for Cloud Evolution (JOYCE), Huelich, Germany'    
    ## Consist of two information, first, the instrument type used for 
    ## measurements, second, the acronym of the operating institution. 
    ## It may differ from the PIs or DOs affiliation
    DateSet_Day.attrs['DATA_SOURCE_RADAR']  = 'Radar.Standard.Moments.Ldr_;JoyRad10;operated by UoC'
    DateSet_Day.attrs['DATA_SOURCE_DISDROMETER'] = 'Disdrometer.Standard.Measurements;Parsival2;operated by UoC'
    ## Wich script is used for the processing and also where to find it
    DateSet_Day.attrs['DATA_PROCESSING_RADAR']  = 'Processing done by the UoC software (https://github.com/igmk/w-radar)'
    DateSet_Day.attrs['DATA_PROCESSING_DISDROMETER'] = 'Processing done by OTT software inside the Disdrometer' 
    DateSet_Day.attrs['INSTRUMENT_MODEL_RADAR']  = 'RadiometerPhysics GmbH; RPG 94GHz Radar; 94GHz FMCW radar profiler'
    DateSet_Day.attrs['INSTRUMENT_MODEL_DISDROMETER'] = 'OTT; Parsival2; Optical droplet counter'
    DateSet_Day.attrs['FILL_VALUE'] = 'NaN'
    # plotting the results of the antenna misspointing --------------------------

    OutPutDirectoryCeck(path_output_data, yyyy, mm)      
    OutPutPath = (path_output_data+'/'+yyyy+'/'+mm+'/sups_'+site+'_'+radar
                  +'_cr_l4_rpm_v00_'+yyyymmdd+'.nc')
#    path_day = (path_output_data+'/'+yyyy+'/'+mm+'/'+yyyymmdd+'_'+site+'_'+radar+'_disdrometer_Ze_offset.nc')      
    DateSet_Day.to_netcdf(OutPutPath, mode='w')     
    print('--- part 7 ---')
    print(' data are saved in: '+OutPutPath)
#     DateSetFinal = xr.merge([DateSetFinal, DateSet_Day])  

##################
if enable_diagnostic_plots:
    cancellation_file.close()
###################
