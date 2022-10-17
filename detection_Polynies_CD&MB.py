#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 13:24:06 2022

@author: uvc35ld
"""

# In[Chargement bibliothèque] : Chargement bibliothèque
import numpy as np
#import netCDF4 as nc
import xarray as xr
# import seaborn as sn
# import rioxarray as rxr
import matplotlib.pyplot as plt
from scipy.stats import linregress # ,pearsonr
import cartopy.crs as ccrs
# import matplotlib.path as mpath
# from cartopy.util import add_cyclic_point
# from matplotlib import cm
import pandas as pd
# import glob
import ToolsPolynias as tp
import MapDrawing as md
# from scipy.stats import pearsonr

import matplotlib as mpl

# In[Chargement des données]

data_polynies_2010_2020 = tp.dataYearsMonths('polynie',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
data_ci_2010_2020 = tp.dataYearsMonths('ci',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley').mean('time')

# In[surfaces test]
surfaces_polynies_2010_2020 = tp.calculSurfacePolynies(data_polynies_2010_2020)

surfaces_polynies_2010_2020.sel(time='2010-08').plot()
plt.show()

md.plotMap_CapeDarnley(data_polynies_2010_2020.sel(time='2010-08-20'))

surfaces_polynies_2010_2020.sel(time='2010').plot()
plt.show()

# In[test date]
surfaces_polynies_2010_2020.sel(time='2010-09').plot()
plt.show()

for x in range(27,30):
    if x < 10:
        date = '2010-09-0'+str(x)
    else :
        date = '2010-09-'+str(x)
    md.plotMap_CapeDarnley(data_polynies_2010_2020.sel(time=date),color='binary',color_bar=False,titre=date)
    print('Surface de : ',surfaces_polynies_2010_2020.sel(time=date).values[0],',le ',date)



date_limite = []
Limite_Surface = 45000

for x in data_polynies_2010_2020.time.values :
    if surfaces_polynies_2010_2020.sel(time=x) > Limite_Surface:
        date_limite.append(x)

# data_polynie_surface = 

polynies_2010_2020_lim = data_polynies_2010_2020.sel(time=np.array(date_limite))

# In[Vérification]

time_lim = date_limite

date_limite_finale = []
for x in time_lim :
    if data_polynies_2010_2020.sel(time=x).sel(latitude=-68).sel(longitude=69.75) == 1 or data_polynies_2010_2020.sel(time=x).sel(latitude=-68).sel(longitude=70) == 1:
        date_limite_finale.append(x)

# In[Calcul des surfaces de polynies]

time_polynie = tp.initialisationTime(data_polynies_2010_2020)
polynie_adapted = data_polynies_2010_2020.sel(time=time_polynie)

surface_polynie_All_0 = tp.calculSurfacePolynies(polynie_adapted,area='All')


surface_polynie_CD = tp.calculSurfacePolynies(polynie_adapted,area='CDP')
surface_polynie_MB = tp.calculSurfacePolynies(polynie_adapted,area='MBP')

surface_polynie_All = surface_polynie_CD + surface_polynie_MB

# In[Tracé des times series]

plt.figure(figsize=(1.5*6.4, 1.5*4.8))

surface_polynie_All.sel(time='2010').plot(label='All')
surface_polynie_CD.sel(time='2010').plot(label='CD')
surface_polynie_MB.sel(time='2010').plot(label='MB')

plt.ylim(0,55000)
plt.tick_params(axis = 'both', labelsize = 20)
plt.xlabel('Time',size=20)
plt.ylabel(r"Surface ($km^2$)",size=20)

plt.legend(fontsize=20,loc='best')



