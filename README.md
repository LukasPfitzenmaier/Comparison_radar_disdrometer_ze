# Comparison_radar_disdrometer_ze

This Python3-Code compares the reflectivities of a ground-based cloud-radar and a Disdrometer (optical precipitation counter) and calculated the reflectivity-off-set between the instruments for quality control.

The code is optimized for ground based radar profiler from Radiometer Pysics (RPG) operating in w-band and the OTT Parsival2 Disdrometer. The data of the RPG radar are processed by the Matlab-code given in (https://github.com/igmk/w-radar) and given in GEOMS data format. The Parsival2 data converted to HD(CP)2 data format without any further processing. Both instruments should have a time stamp in UTC

## Documentation Content: ##
1. General stucture and short description
2. How to call the program
3. Setting data processing options
4. Input and output files

X. References

## 1. General stucture and short description ##

1. Read the data sets - radar and Disdrometer data
2. Check the data
- use only data for comparisson if the temperature is above 4C. The temperature information are comming from the Radar and the Disdrometer. If the two measurements differ for more that 2K, no temeratrue or constat temerature values measured no farter processing is done.
- check it time staps agree. For some of the station in the original network the Disdrometer data where not recored in UTC-time
- filter Disdrometer data for outlayers. The filtering is done using relationship between dropsize and terminal fall velocity from Gunn-Kinzer (1949). All singals beeing less then one std away from the relation ship are filtered out. At the lower end the lower thresholds are interpolated for the first 8 bins.
4. Forward simulate the by the Disdrometer measured droplet size disdribution to reflectivity
- Forward simulation is done using T-Matrix by J. Leinonen (https://github.com/jleinonen/pytmatrix) 
6. Identification of time periodes to calculate the reflectivity off-sets
- A period for comparisson is identified if 
    - if it last minimum of 10 minutes
    - if the mean rain rate is not lower than 0.1 mm/h 
    - if more that 60 drops pert minute have been counted by the disdrometer.
    - if the measured reflectivity of the radar is not higher then 15 dBZ
    - if within a 30 minute time slot the rain measured by the disdrometer shows a gap of 4 minutes or less.
7. Calculation of the reflectivity off-sets based on the difference of the mean of the radar and the Disdrometer reflectivity PDFs.
8. Finally the results are plotted and the results are saved in a NetCDF file.

## 2. How to call the program ##

The program can be called using the following comands:
./20220420_Comparison_radar_disdrometer_ze.py -s site -y year -m month -d day 
One can also call the program in an editor and set the site and the defult option to your preferences. Processing several days is also possible. To do so set the "-db" option to the anount of days you want to calculate the offsets back

Short name of the site (usally 3 leters long), start date (end date of the processing) and number of days to process (going backwards).
- "site" : observation site for Radar and Disdrometer: e.g. jue, nor, mag
- "year" : year of the observation, usually 4 digits, e.g. yyyy
- "month": month of the observation, usually 2 digits, e.g. mm
- "day"  : day of the observation, usually 2 digits, e.g. dd
- "-db"  : How many days before chosen date shall be processed
- "-op", : --output_path, default="/work/disdro_output/"
- "-edp", "--enable_diagnostic_plots, "Additional plots will be saved for insights into processing.", default=False

## 3. Setting data processing options ##

- You can process data for several days, month and years back in time - See section 2.
- it is possible to get also some diagnostic plots. These are telated to time correction and to the filtering of the Disdrometer data.

## 4. Input and output files ##

The INOUT DATA of the RPG w-band radar have been processed by a Matlab-code developed at the University of Cologne, Germany (https://github.com/igmk/w-radar) before used in this code here. The output format of the data are given in GEOMS format. These GEOMS data format is the input data formate fore the radar data for this Code.

The INPUT DATA from the Parsival2 haven't been processed farther after the measurement. Only thing done is that the data are transfered into NetCDF used in the HD(CP)2 project. 

Both of INPUT DATA sets of the instruments should have a time stamp in UTC. This avoied additional coding to match the time-stamps.

The OUTPUT file is a NetCDF file containing the following variables: 
- 'HEIGHT.LEVEL.BOTTOM1'
- 'HEIGHT.LEVEL.TOP1'
- 'HEIGHT.LEVEL.BOTTOM2'
- 'HEIGHT.LEVEL.TOP2'
- 'DISDROMETER.RAIN.RATE'
- 'DISDROMETER.PARTICLES'
- 'DISDROMETER.REFLECTIVITY.FACTOR'
- 'DISDROMETER.SIMULATED.REFLECTIVITY.FACTOR1'
- 'DISDROMETER.SIMULATED.REFLECTIVITY.FACTOR2'
- 'DISDROMETER.REFLECTIVITY.FACTOR.1.SIMULATED.FILTERED'
- 'DISDROMETER.REFLECTIVITY.FACTOR.2.SIMULATED.FILTERED'
- 'RADAR.REFLECTIVITY.FACTOR.HEIGHT1'
- 'RADAR.REFLECTIVITY.FACTOR.HEIGHT2'
- 'RADAR.REFLECTIVITY.FACTOR.1.FILTERED'
- 'RADAR.REFLECTIVITY.FACTOR.2.FILTERED'
    
X. References ##
- RPG w-band radar processing code developed by the University of Cologne, Germany (https://github.com/igmk/w-radar)
- T-Matrix code in python developed by J. Leinonen (https://github.com/jleinonen/pytmatrix) 
