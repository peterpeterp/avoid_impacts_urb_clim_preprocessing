import os,sys,glob
import pandas as pd
import numpy as np
import scipy
from scipy import stats
from scipy.special import erf
import matplotlib.pyplot as plt
import rioxarray
from PIL import Image


possible_data_paths = [
    '../../../reversal_of_the_impact_chain/data/urbclim_for_dashboard/',
    # '/mnt/PROVIDE/urb_clim/'
    '/mnt/ftp_vito_urb_clim/',
]

for dp in possible_data_paths:
    try:
        if os.path.isdir(dp):
            data_path = dp
            break
    except:
        pass

class urb_object():
    def __init__(self, 
                 city, 
                 study_location,
                 indicator,
                 scenario,
                ):
        self._city = city
        self._study_location = study_location
        self._indicator = indicator
        self._scenario = scenario
        self._log=''
        
        
    def load_data(self, coords=None):
        if self._study_location == 'city_average':
            self._data = pd.read_table(data_path + '/%s/%s/%s_%s.csv' %(self._city,self._scenario,self._indicator,self._scenario), sep=',', index_col=0).iloc[:-1,1:]
            
        else:
            self._data = pd.DataFrame(columns=np.arange(2025,2100,10,'int'))
            for aggregation in ['ensmean','enspctl05','enspctl95']:
                l = []
                for year in np.arange(2025,2100,10,'int'):
                    year_range = '%s_%s' %(year-4,year+5)
                    fl = data_path + 'Iconic_cities/%s/Geotiffs/%s/%s_%s_%s_%s_EPSG4326.tif' \
                                                   %(self._city, self._scenario, self._indicator, year_range, self._scenario, aggregation)
                    l.append(np.array(Image.open(fl))[coords['y'],coords['x']])
                self._data.loc[aggregation] = l
            
        self._data.index = ['orig_mean','orig_05','orig_95']
            
            
            
    def estimate_norm(self):
        
        def check_quality_norm():
            if np.abs(stats.norm.ppf(0.95, loc, scale) - upper) > \
               np.abs(stats.norm.ppf(0.95, loc, scale) - stats.norm.ppf(0.85, loc, scale)):
                self._log += '\n%s high' %(year)
                return False
            if np.abs(stats.norm.ppf(0.05, loc, scale) - lower) > \
               np.abs(stats.norm.ppf(0.05, loc, scale) - stats.norm.ppf(0.15, loc, scale)):
                self._log += '\n%s low' %(year)
                return False        
            if np.abs(stats.norm.mean(loc, scale) - mean) > \
               np.abs(stats.norm.ppf(0.55, loc, scale) - stats.norm.ppf(0.45, loc, scale)):
                self._log += '\n%s mid' %(year)
                return False

            return True
        
        def error_norm(params):
            loc,scale = params
            return (upper - stats.norm.ppf(0.95, loc, scale)) ** 2 +\
                   (mean - loc) ** 2 +\
                   (lower - stats.norm.ppf(0.05, loc, scale)) ** 2
        
        self._norm_success = True
        for year in self._data.columns.values:
            mean = self._data.loc['orig_mean',year]
            upper = self._data.loc['orig_95',year]
            lower = self._data.loc['orig_05',year]
            
            # check if normal is appropriate
            loc, scale = scipy.optimize.minimize(error_norm, [mean, 2]).x

            self._data.loc['norm_05',year] = stats.norm.ppf(0.05, loc, scale)    
            self._data.loc['norm_50',year] = stats.norm.ppf(0.50, loc, scale)
            self._data.loc['norm_66',year] = stats.norm.ppf(0.66, loc, scale)
            self._data.loc['norm_90',year] = stats.norm.ppf(0.90, loc, scale)    
            self._data.loc['norm_95',year] = stats.norm.ppf(0.95, loc, scale)
            
            if check_quality_norm() == False:
                self._norm_success = False

            
    def estimate_skewnorm(self):
        def check_quality_skewnorm():
            if np.abs(stats.skewnorm.ppf(0.95, shape, loc, scale) - upper) > \
               np.abs(stats.skewnorm.ppf(0.95, shape, loc, scale) - stats.skewnorm.ppf(0.85, shape, loc, scale)):
                self._log += '\n%s high' %(year)
                return False
            if np.abs(stats.skewnorm.ppf(0.05, shape, loc, scale) - lower) > \
               np.abs(stats.skewnorm.ppf(0.05, shape, loc, scale) - stats.skewnorm.ppf(0.15, shape, loc, scale)):
                self._log += '\n%s low' %(year)
                return False        
            if np.abs(stats.skewnorm.mean(shape, loc, scale) - mean) > \
               np.abs(stats.skewnorm.ppf(0.55, shape, loc, scale) - stats.skewnorm.ppf(0.45, shape, loc, scale)):
                self._log += '\n%s mid' %(year)
                return False

            return True
        
        def error_skewnorm(params):
            shape,loc,scale = params
            return (mean - stats.skewnorm.mean(shape, loc, scale)) ** 2 +\
                   (lower - stats.skewnorm.ppf(0.05, shape, loc, scale)) ** 2 +\
                   (upper - stats.skewnorm.ppf(0.95, shape, loc, scale)) ** 2
        
        self._skewnorm_success = True
        for year in self._data.columns.values:
            mean = self._data.loc['orig_mean',year]
            upper = self._data.loc['orig_95',year]
            lower = self._data.loc['orig_05',year]
            
            shape, loc, scale = scipy.optimize.minimize(error_skewnorm, [0, mean, 2]).x
            
            self._data.loc['skewnorm_05',year] = stats.skewnorm.ppf(0.05, shape, loc, scale)    
            self._data.loc['skewnorm_50',year] = stats.skewnorm.ppf(0.50, shape, loc, scale)
            self._data.loc['skewnorm_66',year] = stats.skewnorm.ppf(0.66, shape, loc, scale)
            self._data.loc['skewnorm_90',year] = stats.skewnorm.ppf(0.90, shape, loc, scale)    
            self._data.loc['skewnorm_95',year] = stats.skewnorm.ppf(0.95, shape, loc, scale)
            
            if check_quality_skewnorm() == False:
                self._skewnorm_success = False
                
    def plot(self):
        fig,ax = plt.subplots(nrows=1)
        ax.plot(self._data.columns.values, self._data.loc['orig_mean'].values, color='k', linestyle='-', label='mean')
        for stat,color in zip(['05','95'],['b','darkcyan']):
            ax.plot(self._data.columns.values, self._data.loc['orig_%s' %(stat)].values, color=color, linestyle='-', label='%s' %(stat))
        for stat,color in zip(['05','50','66','90','95'], ['b','r','orange','darkmagenta','darkcyan']):
            for v, lsty in zip(['norm', 'skewnorm'], ['--',':']):
                ax.plot(self._data.columns.values, self._data.loc['%s_%s' %(v,stat)].values, color=color, linestyle=lsty, label='%s %s' %(v, stat))
        ax.set_ylabel(self._indicator)
        failure_dict = {True:'ok', False:'fail'}
        ax.annotate('norm: %s\nskewnorm: %s' %(failure_dict[self._norm_success], failure_dict[self._skewnorm_success]), xy=(0.05,0.95),
                    xycoords='axes fraction', va='top')
        ax.set_title('%s' %(self._scenario))
        
        ax.legend(bbox_to_anchor=(1.05,0.5), loc='center left')
        plt.savefig('../../validation_plots/%s/%s_%s.pdf' %(self._city, self._indicator, self._scenario), bbox_inches='tight')
        plt.clf()
    
    def evaluate_results_and_store_percentiles(self):
        
        self._out = pd.DataFrame(columns=self._data.columns)
        if self._norm_success:
            self._approximation = 'norm'
            for stat in ['50','66','90']:
                self._out.loc[stat] = self._data.loc['norm_%s' %(stat)]
        elif self._skewnorm_success:
            self._approximation = 'skewnorm'
            for stat in ['50','66','90']:
                self._out.loc[stat] = self._data.loc['skewnorm_%s' %(stat)]    
        else:
            self._approximation = 'fail'
            for stat in ['50','66','90']:
                self._out.loc[stat] = self._data.loc['skewnorm_%s' %(stat)]    
                
        self._log += '\n selected: %s' %(self._approximation)
    
        with open('../../logs/%s/%s_%s_%s.csv' %(self._city, self._study_location, self._indicator, self._scenario), 'w') as l:
            l.write(self._log)
    
    def write_csv_file(self):
        asdasd
        out_file = '../data/urbclim_for_dashboard/%s/estimated_quantiles/%s_%s_%s.csv' \
                    %(self._city, self._location, self._indicator, self._scenario)
        with open(out_file, mode='w', encoding='utf-8') as f:
            f.write('city: %s\nlocation: %s\nindicator: %s\nscenario: %s\napproximation: %s\n' \
                    %(self._city, self._location, self._indicator, self._scenario, self._approximation))
        self._out.to_csv(out_file, mode='a', header=True, sep=',')
        
    def print_log(self):
        print('-' * 20)
        print(self._city, self._study_location, self._indicator, self._scenario)
        print(self._log)
