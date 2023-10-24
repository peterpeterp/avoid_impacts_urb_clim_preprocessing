import os,sys,glob,argparse
import pandas as pd
import numpy as np
import xarray as xr
import scipy
from scipy import stats
from scipy.special import erf
import matplotlib.pyplot as plt
import rioxarray

from urb_object import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--city", type=str)
parser.add_argument("--indicator", type=str)
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()

print(args)

city = args.city
indicator = args.indicator
overwrite = args.overwrite

os.system('mkdir ../../validation_plots/%s' %(city))
os.system('mkdir ../../logs/%s' %(city))
os.system('mkdir ../../data_avoid_impacts_urbclim/%s' %(city))

scenarios = ['CurPol','GS','SP']

study_locations = list(pd.read_table(data_path + 'daily_time_series/Lissabon/WBGT_timeseries_CurPol_enspctl95_2100.csv', sep=',').columns[1:])
dummy = rioxarray.open_rasterio(data_path + 'Iconic_cities/%s/Geotiffs/%s/%s_%s_%s_%s_EPSG4326.tif' \
                                                   %(city, 'CurPol', 'HWMI', '2021_2030', 'CurPol', 'ensmean'))
study_location_dict = {}
for l in study_locations:
    t,q,lat,lon = l.split('_')
    study_location_dict['%s_%s' %(t,q)] = {
        'y' : np.argmin(np.abs(dummy.y.values - float(lat))), 
        'x' : np.argmin(np.abs(dummy.x.values - float(lon)))}

study_locations = list(study_location_dict.keys())

approximation_dict = {'norm':0, 'skewnorm':1, 'fail':np.nan}

out_file = '../../data_avoid_impacts_urbclim/%s/%s_%s.nc' %(city,city,indicator)
if os.path.isfile(out_file) and overwrite == False:
    nc = xr.load_dataset(out_file)
    out = nc['data']
    approximation = nc['approximation']
    nc.close()
else:
    out = xr.DataArray(dims=['study_location','scenario','quantile', 'year'], 
                       coords=dict(study_location=study_locations, 
                                   scenario=scenarios,
                                   quantile=[50,66,90],
                                   year=np.arange(2025,2100,10,'int')))
    approximation = xr.DataArray(dims=['study_location','scenario'], 
                           coords=dict(study_location=study_locations, 
                                   scenario=scenarios))
    approximation.attrs['explanation'] = str(approximation_dict)

for study_location in study_locations:
    for scenario in scenarios:
        print(study_location,scenario)
        if np.any(np.isnan(out.loc[study_location, scenario])):
            urb = urb_object(
                city=city, 
                study_location=study_location,
                indicator=indicator,
                scenario=scenario)
            urb.load_data(coords=study_location_dict[study_location])
            urb.estimate_norm()
            urb.estimate_skewnorm()
            urb.plot()
            urb.evaluate_results_and_store_percentiles()
            urb.print_log()
            approximation.loc[study_location, scenario] = approximation_dict[urb._approximation]
            out.loc[study_location, scenario] = urb._out.values

            xr.Dataset({'data':out, 'approximation':approximation}).to_netcdf(out_file)