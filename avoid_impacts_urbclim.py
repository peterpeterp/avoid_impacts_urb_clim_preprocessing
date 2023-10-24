import os,sys,glob
import pandas as pd
import numpy as np
import xarray as xr

gmt_tier1 = pd.read_table('../data_avoid_impacts_urbclim/until-2100tier1_temperature_summary.csv', sep=',')
gmt_CurPol = xr.DataArray(gmt_tier1.loc[gmt_tier1.scenario == 'CurPol','1850':].values.squeeze(), 
                          dims=['year'], coords=dict(year=np.arange(1850,2101,1,'int')))
gmt_CurPol_50 = gmt_CurPol.loc[2020:2100]


class avoid_impacts_urbclim():
    def __init__(self, 
                 indicator,
                 level_of_impact,
                 certainty_level,
                 city,
                 study_location):
        
        quantile_dict = {
            'likely' : 66,
            'very likely' : 90,
            'as likely as not' : 50
        }
        self._quantile = quantile_dict[certainty_level]
        
        self._data = xr.load_dataset('../data_avoid_impacts_urbclim/%s/%s_%s.nc' %(city,city,indicator))['data'].loc[:,:,self._quantile]
        
        self._study_location = study_location
        self._level_of_impact = level_of_impact
        
    def prepare_response(self):
        self._out = {
            'range_of_interest' : list(np.nanpercentile(self._data, [0,100])),
            'emissions' : -9999,
            'scenarios' : []
        }
        
        self.estimate_GMT_level()
        
        for scenario in self._data.scenario.values:
            l = []
            for study_location in self._data.study_location.values:
                l.append(self.avoided_until(study_location, scenario))
            self._out['scenarios'].append({"uid": scenario, "study_locations": l})
        
    def avoided_until(self, study_location, scenario):
        years = self._data.year[self._data.loc[study_location, scenario] > self._level_of_impact].values
        
        if len(years) == 0:
            return {
                      "uid": study_location,
                      "year": None,
                      "likely": False,
                      "avoidable": True
                    }
        
        elif len(years) == self._data.shape[-1]:
            return {
                      "uid": study_location,
                      "year": None,
                      "likely": True,
                      "avoidable": False
                    }
        
        else:
            return {
                      "uid": study_location,
                      "year": int(years[0] - 5),
                      "likely": True,
                      "avoidable": True
                    }
        
    def estimate_GMT_level(self):
        year = self.avoided_until(study_location=self._study_location, scenario='CurPol')['year']
        if year is not None:
            self._out['global_mean_temperature'] = float(gmt_CurPol_50.loc[year].values.round(1))
        else:
            self._out['global_mean_temperature'] = None
            
            
if __name__ == '__main__':
    u = avoid_impacts_urbclim(indicator='T2M_dayover30',
                          level_of_impact=70,
                          certainty_level='likely',
                          location='Lissabon',
                          study_location='city_average')
    u.prepare_response()
    print(u._out)
