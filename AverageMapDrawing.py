#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 10:19:19 2022

@author: uvc35ld
"""

# In[Chargement bibliothèques]
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import pandas as pd
from cartopy.util import add_cyclic_point
# import pandas as pd
# from scipy import stats

import MapDrawing as md

import ToolsPolynias as tp


# In[Tracé de t2m]

t2m_2010_2020_Win_CD_around = tp.dataYearsMonths('t2m',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
md.plotMap_CapeDarnley_around(t2m_2010_2020_Win_CD_around.mean('time'), np.linspace(-15,15,31),dimension='$m/s$')


# In[Tracé de u10]

u10_2010_2020_Win_CD_around = tp.dataYearsMonths('u10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley Around')
md.plotMap_CapeDarnley_around(u10_2010_2020_Win_CD_around.mean('time'), np.linspace(-15,15,31),dimension='$m/s$')

# In[Tracé de v10]

v10_2010_2020_Win_CD_around = tp.dataYearsMonths('v10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley Around')
md.plotMap_CapeDarnley_around(v10_2010_2020_Win_CD_around.mean('time'), np.linspace(-15,15,31),dimension='$m/s$')

# In[Tracé Streamplot]

# U10_2010_2020_Win_CD_around = np.sqrt(u10_2010_2020_Win_CD_around*u10_2010_2020_Win_CD_around + v10_2010_2020_Win_CD_around*v10_2010_2020_Win_CD_around)
md.plotStreamplot(u10_2010_2020_Win_CD_around.mean('time'), v10_2010_2020_Win_CD_around.mean('time'),dimension='$m/s$')

# In[Tracé de vorticity]

vo_2010_2020_Win_CD_around = tp.dataYearsMonths('vo',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley Around').mean('level')
md.plotMap_CapeDarnley_around(vo_2010_2020_Win_CD_around.mean('time'), np.linspace(-0.0003,0.0003,16),dimension='$s^{-1}$')

# In[Tracé de e]

e_2010_2020_Win_CD_around = tp.dataYearsMonths('e',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley Around')
md.plotMap_CapeDarnley_around(e_2010_2020_Win_CD_around.mean('time'), np.linspace(-3e-5,0,16),dimension='$kg/m^{2}/s$')

# In[Tracé de humi]

humi_2010_2020_Win_CD_around = tp.dataYearsMonths('humi',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley Around')
md.plotMap_CapeDarnley_around(humi_2010_2020_Win_CD_around.mean('time'), np.linspace(0,0.003,16),dimension='$kg_{water}/kg_{air}$')

# In[Tracé de d]

d_2010_2020_Win_CD_around = tp.dataYearsMonths('d',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley Around').mean('level')
md.plotMap_CapeDarnley_around(d_2010_2020_Win_CD_around.mean('time'), np.linspace(-0.0003,0.0003,16),dimension='$s^{-1}$')

# In[Tracé de blh]

blh_2010_2020_Win_CD_around = tp.dataYearsMonths('blh',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley Around')
md.plotMap_CapeDarnley_around(blh_2010_2020_Win_CD_around.mean('time'), np.linspace(0,0.3,16),dimension='$m$')

# =============================================================================
#  Tracé région de Cape Darnley
# =============================================================================

# In[Tracé de u10]

u10_2010_2020_Win_CD = tp.dataYearsMonths('u10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
md.plotMap_CapeDarnley_around(u10_2010_2020_Win_CD.mean('time'), np.linspace(-15,15,31),dimension='$m/s$')

# In[Tracé de v10]

v10_2010_2020_Win_CD = tp.dataYearsMonths('v10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
md.plotMap_CapeDarnley_around(v10_2010_2020_Win_CD.mean('time'), np.linspace(-15,15,31),dimension='$m/s$')

# In[Tracé Streamplot]

# U10_2010_2020_Win_CD = np.sqrt(u10_2010_2020_Win_CD*u10_2010_2020_Win_CD + v10_2010_2020_Win_CD*v10_2010_2020_Win_CD)
md.plotStreamplot(u10_2010_2020_Win_CD.mean('time'), v10_2010_2020_Win_CD.mean('time'),dimension='$m/s$')

# In[Tracé de vorticity]

vo_2010_2020_Win_CD = tp.dataYearsMonths('vo',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley').mean('level')
md.plotMap_CapeDarnley_around(vo_2010_2020_Win_CD.mean('time'), np.linspace(-0.0003,0.0003,16),dimension='$s^{-1}$')

# In[Tracé de e]

e_2010_2020_Win_CD = tp.dataYearsMonths('e',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
md.plotMap_CapeDarnley_around(e_2010_2020_Win_CD.mean('time'), np.linspace(-3e-5,0,16),dimension='$kg/m^{2}/s$',color='Blues')

# In[Tracé de humi]

humi_2010_2020_Win_CD = tp.dataYearsMonths('humi',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
md.plotMap_CapeDarnley_around(humi_2010_2020_Win_CD.mean('time'), np.linspace(0,0.003,16),dimension='$kg_{water}/kg_{air}$')

# In[Tracé de d]

d_2010_2020_Win_CD = tp.dataYearsMonths('d',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley').mean('level')
md.plotMap_CapeDarnley_around(d_2010_2020_Win_CD.mean('time'), np.linspace(-0.0003,0.0003,16),dimension='$s^{-1}$')

# In[Tracé de blh]

blh_2010_2020_Win_CD_around = tp.dataYearsMonths('blh',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
md.plotMap_CapeDarnley_around(blh_2010_2020_Win_CD_around.mean('time'), np.linspace(0,0.3,16),dimension='$m$')
