#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:03:19 2022

@author: uvc35ld
"""


# In[Chargement bibliothèques]
import numpy as np
import matplotlib.pyplot as plt
#import cartopy.crs as ccrs
# import xarray as xr
# import datetime
# import pandas as pd
# from cartopy.util import add_cyclic_point
# import pandas as pd
# from scipy import stats
# import scipy
# import xrft

# from fitter import Fitter, get_common_distributions, get_distributions

# import scipy.stats as stats  

import MapDrawing as md
import ToolsPolynias as tp

# In[Polynie ]

polynie_CD_Win_2010_2020_0_50 = tp.dataYearsMonths('polynie',List_Years=[2010,2011,2012,2013,2014,2015,2016,2018,2019,2020],List_Months=range(5,10),area='Cape Darnley',tol_polynie='50%')
polynie_CD_Win_2010_2020_0_60 = tp.dataYearsMonths('polynie',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley',tol_polynie='60%')

polynie_CD_Win_2010_2020_0_70 = tp.dataYearsMonths('polynie',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley',tol_polynie='70%')
polynie_CD_Win_2010_2020_0_80 = tp.dataYearsMonths('polynie',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley',tol_polynie='80%')

polynie_CD_Win_2010_2020_0_90 = tp.dataYearsMonths('polynie',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley',tol_polynie='90%')

Surface_polynie_CD_Win_2010_2020_50 = tp.calculSurfacePolynies(polynie_CD_Win_2010_2020_0_50,area='CDP')
Surface_polynie_CD_Win_2010_2020_60 = tp.calculSurfacePolynies(polynie_CD_Win_2010_2020_0_60,area='CDP')
Surface_polynie_CD_Win_2010_2020_70 = tp.calculSurfacePolynies(polynie_CD_Win_2010_2020_0_70,area='CDP')
Surface_polynie_CD_Win_2010_2020_80 = tp.calculSurfacePolynies(polynie_CD_Win_2010_2020_0_80,area='CDP')
Surface_polynie_CD_Win_2010_2020_90 = tp.calculSurfacePolynies(polynie_CD_Win_2010_2020_0_90,area='CDP')

# In[Tracé polynie]
md.plotMap_CapeDarnley(polynie_CD_Win_2010_2020_0_70.sel(time='2010-08-27'),color_bar=False)
md.plotMap_CapeDarnley(polynie_CD_Win_2010_2020_0_80.sel(time='2010-08-27'),color_bar=False)


# In[Tracé figure]

size_figure = [2,2]
taille_ecriture = 20

time_2010 = Surface_polynie_CD_Win_2010_2020_50.sel(time='2010').time.values

plt.figure(figsize=(size_figure[0]*6.4, size_figure[1]*4.8))
Surface_polynie_CD_Win_2010_2020_50.sel(time='2010').plot(label='50%')
print('Tol 50% : Surface moyenne = ',Surface_polynie_CD_Win_2010_2020_50.sel(time='2010').mean('time'))

Surface_polynie_CD_Win_2010_2020_60.sel(time='2010').plot(label='60%')
print('Tol 60% : Surface moyenne = ',Surface_polynie_CD_Win_2010_2020_60.sel(time='2010').mean('time'))

Surface_polynie_CD_Win_2010_2020_70.sel(time='2010').plot(label='70%')#,color='k')
print('Tol 70% : Surface moyenne = ',Surface_polynie_CD_Win_2010_2020_70.sel(time='2010').mean('time'))

Surface_polynie_CD_Win_2010_2020_80.sel(time='2010').plot(label='80%')
print('Tol 80% : Surface moyenne = ',Surface_polynie_CD_Win_2010_2020_80.sel(time='2010').mean('time'))

Surface_polynie_CD_Win_2010_2020_90.sel(time='2010').plot(label='90%')
print('Tol 90% : Surface moyenne = ',Surface_polynie_CD_Win_2010_2020_90.sel(time='2010').mean('time'))

plt.tick_params(axis = 'both', labelsize = taille_ecriture)
plt.xlim(time_2010[0],time_2010[-1])
plt.ylim(0,140000)
plt.grid()

plt.xlabel('Time',fontsize=taille_ecriture)
plt.ylabel(r"Surface ($km^2$)",fontsize=taille_ecriture)

#plt.set_label(r"$km^2$",size=taille_ecriture,rotation=90,labelpad=1)
plt.tick_params(labelsize=taille_ecriture)

plt.legend(fontsize= taille_ecriture)

# In[Tracé figure '2010-07-29']

size_figure = [2,2]
taille_ecriture = 20

time = Surface_polynie_CD_Win_2010_2020_50.sel(time='2010-07').time.values

plt.figure(figsize=(size_figure[0]*6.4, size_figure[1]*4.8))
Surface_polynie_CD_Win_2010_2020_50.sel(time=time).plot(label='50%,'+r"$S_{moy}=$"+
                                                        str(Surface_polynie_CD_Win_2010_2020_50
                                                            .sel(time=time).mean('time')))
Surface_polynie_CD_Win_2010_2020_60.sel(time=time).plot(label='60%')
#Surface_polynie_CD_Win_2010_2020_70.sel(time=time).plot(label='70%')#,color='k')

Surface_polynie_CD_Win_2010_2020_80.sel(time=time).plot(label='80%')

Surface_polynie_CD_Win_2010_2020_90.sel(time=time).plot(label='90%')

plt.tick_params(axis = 'both', labelsize = taille_ecriture)
plt.xlim(time[0],time[-1])
plt.ylim(0,140000)

plt.xlabel('Time',fontsize=taille_ecriture)
plt.ylabel(r"Surface ($km^2$)",fontsize=taille_ecriture)

#plt.set_label(r"$km^2$",size=taille_ecriture,rotation=90,labelpad=1)
plt.tick_params(labelsize=taille_ecriture)

plt.legend(fontsize= taille_ecriture)

# # In[Importation données ci]
# ci_CD_Win_2010 = tp.dataYearsMonths('ci',List_Years=[2010],List_Months=[7],area='Cape Darnley')

# ci_CD_Win_jour = ci_CD_Win_2010.sel(time='2010-07-29')

# md.plotMap_CapeDarnley(ci_CD_Win_jour,color='Reds',Liste_Vmin_Vmax=[0.6,1])

# # In[Série temporelle de la polynie CD/MB - tolérance 70%]

# polynie_CD_Win_2010_2020_0_70 = tp.dataYearsMonths('polynie',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
# polynie_CD_Win_2010_2020_70 = tp.selection_polynie_CD(polynie_CD_Win_2010_2020_0_70)
# Surface_polynie_CD_Win_2010_2020_70 = tp.calculSurfacePolynies(polynie_CD_Win_2010_2020_70,area='CD')

# polynie_MB_Win_2010_2020_0_70 = tp.dataYearsMonths('polynie',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
# polynie_MB_Win_2010_2020_70 = tp.selection_polynie_CD(polynie_MB_Win_2010_2020_0_70)
# Surface_polynie_MB_Win_2010_2020_70 = tp.calculSurfacePolynies(polynie_MB_Win_2010_2020_70,area='MB')

# # In[Série temporelle de la polynie CD/MB - tolérance 90%]

# polynie_CD_Win_2010_2020_0_90 = tp.dataYearsMonths('polynie',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley',tol_polynie='90%')
# polynie_CD_Win_2010_2020_90 = tp.selection_polynie_CD(polynie_CD_Win_2010_2020_0_90)

# Surface_polynie_CD_Win_2010_2020_90 = tp.calculSurfacePolynies(polynie_CD_Win_2010_2020_90,area='CD')


# polynie_MB_Win_2010_2020_90 = tp.selection_polynie_CD(polynie_CD_Win_2010_2020_0_90)
# Surface_polynie_MB_Win_2010_2020_90 = tp.calculSurfacePolynies(polynie_MB_Win_2010_2020_90,area='MB')

# plt.figure()
# Surface_polynie_CD_Win_2010_2020_90.sel(time='2010').plot()


# # In[50% - MB]
# plt.figure()
# Surface_polynie_CD_Win_2010_2020_50.sel(time='2010').plot()

# plt.figure()
# Surface_polynie_CD_Win_2010_2020_50.sel(time=slice('2010-05-01','2010-05-07')).plot()
# x = '2010-05-04'
# md.plotMap_CapeDarnley(polynie_CD_Win_2010_2020_0_50.sel(time=x),
#                        # clevels=np.linspace(0,1,10),
#                        Liste_Vmin_Vmax=[0,1],
#                        color_bar=False,
#                        color='binary',
#                        titre='Polynie '+str(x)[:10])

# plt.figure()
# Surface_polynie_CD_Win_2010_2020_50.sel(time=slice('2010-06-09','2010-06-13')).plot()
# x = '2010-06-11'
# md.plotMap_CapeDarnley(polynie_CD_Win_2010_2020_0_50.sel(time=x),
#                        # clevels=np.linspace(0,1,10),
#                        Liste_Vmin_Vmax=[0,1],
#                        color_bar=False,
#                        color='binary',
#                        titre='Polynie '+str(x)[:10])

# plt.figure()
# Surface_polynie_CD_Win_2010_2020_50.sel(time=slice('2010-07-15','2010-07-25')).plot()
# x = '2010-07-20'
# md.plotMap_CapeDarnley(polynie_CD_Win_2010_2020_0_50.sel(time=x),
#                        # clevels=np.linspace(0,1,10),
#                        Liste_Vmin_Vmax=[0,1],
#                        color_bar=False,
#                        color='binary',
#                        titre='Polynie '+str(x)[:10])


# time_2010_06 = Surface_polynie_CD_Win_2010_2020_50.sel(time=slice('2010-06-09','2010-06-13')).time.values
# for x in time_2010_06:
#     md.plotMap_CapeDarnley(polynie_CD_Win_2010_2020_0_50.sel(time=x),
#                            # clevels=np.linspace(0,1,10),
#                            Liste_Vmin_Vmax=[0,1],
#                            color_bar=False,
#                            color='binary',
#                            titre='Polynie '+str(x)[:10])
