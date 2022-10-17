#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:45:16 2022

@author: uvc35ld
"""


# In[Chargement bibliothèques]
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import pandas as pd
from cartopy.util import add_cyclic_point
from matplotlib import animation
from IPython.display import HTML, display
# import pandas as pd
# from scipy import stats

import MapDrawing as md
import ToolsPolynias as tp



# In[Test Chargement de la polynie - Cape Darnley - 2010-2020]

polynie_MB_Win_2010_2020 = tp.dataYearsMonths('polynie',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
surface_polynie_MB_Win_2010_2020 = tp.calculSurfacePolynies(polynie_MB_Win_2010_2020)

delta_surface_polynie_MB_Win_2010_2020 = tp.calculSurfacePolynies_delta_H(List_Years=range(2010,2021),List_Months=range(5,10),polynie='MBP')


Latitudes = polynie_MB_Win_2010_2020.latitude.values
Longitudes = polynie_MB_Win_2010_2020.longitude.values

# print('#############################')
# ##### Chargement des données différences entre 21h et 23h #####
# delta_polynie_MB_Win_2010_2020 = tp.dataYearsMonths_delta_H('polynie',List_Years=range(2010,2011),List_Months=range(9,10),area='Cape Darnley')
# print('Fin chargement polynie')

CI_moyen = tp.CIMoyenne()
CI_moyen_MB = tp.sel_CapeDarnley(CI_moyen)

# In[Chargement des données de Vitesse]

print('#############################')
##### Chargement des données différences entre 21h et 23h #####
delta_U10_MB_Win_2010_2020 = tp.dataYearsMonths_delta_H('U10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
print('Fin chargement U10')

print('#############################')
##### Chargement des données différences entre 21h et 23h #####
delta_u10_MB_Win_2010_2020 = tp.dataYearsMonths_delta_H('u10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
print('Fin chargement u10')

print('#############################')
delta_v10_MB_Win_2010_2020 = tp.dataYearsMonths_delta_H('v10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
print('Fin chargement v10')

print('#############################')
delta_blh_MB_Win_2010_2020 = tp.dataYearsMonths_delta_H('blh',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
print('Fin chargement blh')

# print('#############################')
# delta_vo_MB_Win_2010_2020 = tp.dataYearsMonths_delta_H('vo',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
# print('Fin chargement vo')

# print('#############################')
# delta_d_MB_Win_2010_2020 = tp.dataYearsMonths_delta_H('d',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
# print('Fin chargement d')

# In[Calcul de la regression de données par rapport à la surface - u10 ]

##### Vitesse a 10m #####
print("Calcul regression delta u10 = F(Surface Polynie)")
slopes_delta_u10_S_MB,intercept_delta_u10_S_MB,r_delta_u10_S_MB = tp.regressionLineaireDataSurfacepolynie(delta_u10_MB_Win_2010_2020, 
                                                                                                    surface_polynie_MB_Win_2010_2020,
                                                                                                    titre='Correlation delta u10 et Surface de la polynie de Mackenzie Bay - Winter 2010-2020')


md.plotMap_CapeDarnley(slopes_delta_u10_S_MB,Liste_Vmin_Vmax=[-0.25,0.25],dimension=r"$m.s^{-1}/m^2$",
                       CI_contour=CI_moyen_MB,
                       level_CI_contour=np.linspace(0.2,0.9,15))

md.plotMap_CapeDarnley(r_delta_u10_S_MB,Liste_Vmin_Vmax=[-0.15,0.15],dimension=r"Sans unité (entre -1 et 1)",
                       CI_contour=CI_moyen_MB,
                       level_CI_contour=np.linspace(0.2,0.9,15))

# In[Calcul de la regression de données par rapport à la surface - v10 ]

##### Vitesse a 10m #####
print("Calcul regression delta v10 = F(Surface Polynie)")
slopes_delta_v10_S_MB,intercept_delta_v10_S_MB,r_delta_v10_S_MB = tp.regressionLineaireDataSurfacepolynie(delta_v10_MB_Win_2010_2020, 
                                                                                                    surface_polynie_MB_Win_2010_2020,
                                                                                                    titre='Correlation delta v10 et Surface de la polynie de Mackenzie Bay - Winter 2010-2020')

md.plotMap_CapeDarnley(slopes_delta_v10_S_MB,
                       Liste_Vmin_Vmax=[-0.25,0.25],
                       dimension=r"$m.s^{-1}/m^2$",
                       CI_contour=CI_moyen_MB,
                       level_CI_contour=np.linspace(0.2,0.9,15))
md.plotMap_CapeDarnley(r_delta_v10_S_MB,
                       Liste_Vmin_Vmax=[-0.15,0.15],
                       dimension=r"Sans unité (entre -1 et 1)",
                       CI_contour=CI_moyen_MB,
                       level_CI_contour=np.linspace(0.2,0.9,15))

# In[Calcul de la regression de données par rapport à la surface - U10 ]

##### Vitesse a 10m #####
print("Calcul regression delta U10 = F(Surface Polynie)")
slopes_delta_U10_S_MB,intercept_delta_U10_S_MB,r_delta_U10_S_MB = tp.regressionLineaireDataSurfacepolynie(delta_U10_MB_Win_2010_2020, 
                                                                                                    surface_polynie_MB_Win_2010_2020,
                                                                                                    titre='Correlation delta U10 et Surface de la polynie de Mackenzie Bay - Winter 2010-2020')


md.plotMap_CapeDarnley(slopes_delta_U10_S_MB,
                       Liste_Vmin_Vmax=[-0.2,0.2],
                       dimension=r"$m.s^{-1}/m^2$",
                       CI_contour=CI_moyen_MB,
                       level_CI_contour=np.linspace(0.2,0.9,15))

md.plotMap_CapeDarnley(r_delta_U10_S_MB,
                       Liste_Vmin_Vmax=[-0.2,0.2],
                       dimension=r"Sans unité (entre -1 et 1)",
                       CI_contour=CI_moyen_MB,
                       level_CI_contour=np.linspace(0.2,0.9,15))

# In[Calcul de la regression de données par rapport à la surface - blh ]

##### Vitesse a 10m #####
print("Calcul regression blh = F(Surface Polynie)")
slopes_delta_blh_S_MB,intercept_delta_blh_S_MB,r_delta_blh_S_MB = tp.regressionLineaireDataSurfacepolynie(delta_blh_MB_Win_2010_2020, 
                                                                                                    surface_polynie_MB_Win_2010_2020,
                                                                                                    titre='Correlation delta blh et Surface de la polynie de Mackenzie Bay - Winter 2010-2020')

md.plotMap_CapeDarnley(slopes_delta_blh_S_MB,
                       # Liste_Vmin_Vmax=[-0.25,0.25],
                       dimension=r"$m.s^{-1}/m^2$",
                       CI_contour=CI_moyen_MB,
                       level_CI_contour=np.linspace(0.2,0.9,15))
md.plotMap_CapeDarnley(r_delta_blh_S_MB,
                       # Liste_Vmin_Vmax=[-0.15,0.15],
                       dimension=r"Sans unité (entre -1 et 1)",
                       CI_contour=CI_moyen_MB,
                       level_CI_contour=np.linspace(0.2,0.9,15))

# # In[Calcul de la regression de données par rapport à la surface - vo ]

# ##### Vorticité #####
# print("Calcul regression F(vo) = Surface Polynie")
# slopes_delta_vo_S_MB,intercept_delta_vo_S_MB,r_delta_vo_S_MB = tp.regressionLineaireDataSurfacepolynie(delta_vo_MB_Win_2010_2020, 
#                                                                                                     surface_polynie_MB_Win_2010_2020,
#                                                                                                     titre='Correlation delta vo et Surface de la polynie de Cape Darnley - Winter 2010-2020')

# md.plotMap_CapeDarnley(slopes_delta_vo_S_MB,Liste_Vmin_Vmax=[-0.25,0.25],dimension=r"$m.s^{-1}/m^2$")
# md.plotMap_CapeDarnley(r_delta_vo_S_MB,Liste_Vmin_Vmax=[-0.15,0.15],dimension=r"Sans unité (entre -1 et 1)")

# # In[Calcul de la regression de données par rapport à la surface - d ]

# ##### drticité #####
# print("Calcul regression F(d) = Surface Polynie")
# slopes_delta_d_S_MB,intercept_delta_d_S_MB,r_delta_d_S_MB = tp.regressionLineaireDataSurfacepolynie(delta_d_MB_Win_2010_2020, 
#                                                                                                     surface_polynie_MB_Win_2010_2020,
#                                                                                                     titre='Correlation delta d et Surface de la polynie de Cape Darnley - Winter 2010-2020')

# md.plotMap_CapeDarnley(slopes_delta_d_S_MB,Liste_Vmin_Vmax=[-0.25,0.25],dimension=r"$m.s^{-1}/m^2$")
# md.plotMap_CapeDarnley(r_delta_d_S_MB,Liste_Vmin_Vmax=[-0.15,0.15],dimension=r"Sans unité (entre -1 et 1)")

# In[Calcul de la regression de Delta data par rapport à Delta S]

# In[Calcul de la regression de données par rapport à la surface - u10 ]

##### Vitesse a 10m #####
print("Calcul regression Delta u10 = F(Delta Surface Polynie)")
slopes_delta_u10_delta_S_MB,intercept_delta_u10_delta_S_MB,r_delta_u10_delta_S_MB = tp.regressionLineaireDataSurfacepolynie(delta_u10_MB_Win_2010_2020, 
                                                                                                    delta_surface_polynie_MB_Win_2010_2020,
                                                                                                    titre='Correlation delta u10 et delta Surface de la polynie de Mackenzie Bay - Winter 2010-2020')

md.plotMap_CapeDarnley(slopes_delta_u10_delta_S_MB,
                       Liste_Vmin_Vmax=[-0.2,0.2],
                       dimension=r"$m.s^{-1}/m^2$",
                       CI_contour=CI_moyen_MB,
                       level_CI_contour=np.linspace(0.2,0.9,15))
md.plotMap_CapeDarnley(r_delta_u10_delta_S_MB,
                       Liste_Vmin_Vmax=[-0.2,0.2],
                       dimension=r"Sans unité (entre -1 et 1)",
                       CI_contour=CI_moyen_MB,
                       level_CI_contour=np.linspace(0.2,0.9,15))

# In[Calcul de la regression de données par rapport à la surface - v10 ]

##### Vitesse a 10m #####
print("Calcul regression F(Delta v10) = Delta Surface Polynie")
slopes_delta_v10_delta_S_MB,intercept_delta_v10_delta_S_MB,r_delta_v10_delta_S_MB = tp.regressionLineaireDataSurfacepolynie(delta_v10_MB_Win_2010_2020, 
                                                                                                    delta_surface_polynie_MB_Win_2010_2020,
                                                                                                    titre='Correlation delta v10 et delta Surface de la polynie de Mackenzie Bay - Winter 2010-2020')

md.plotMap_CapeDarnley(slopes_delta_v10_delta_S_MB,
                       Liste_Vmin_Vmax=[-0.30,0.30],
                       dimension=r"$m.s^{-1}/m^2$",
                       CI_contour=CI_moyen_MB,
                       level_CI_contour=np.linspace(0.2,0.9,15))
md.plotMap_CapeDarnley(r_delta_v10_delta_S_MB,
                       Liste_Vmin_Vmax=[-0.25,0.25],
                       dimension=r"Sans unité (entre -1 et 1)",
                       CI_contour=CI_moyen_MB,
                       level_CI_contour=np.linspace(0.2,0.9,15))

# md.plotMap_CapeDarnley(np.sqrt(slopes_delta_v10_delta_S_MB*slopes_delta_v10_delta_S_MB + slopes_delta_u10_delta_S_MB*slopes_delta_u10_delta_S_MB),
#                        Liste_Vmin_Vmax=[-0.30,0.30],
#                        dimension=r"$m.s^{-1}/m^2$",
#                        CI_contour=CI_moyen_MB,
#                        level_CI_contour=np.linspace(0.2,0.9,15))

# In[Calcul de la regression de données par rapport à la surface - U10 ]

##### Vitesse a 10m #####
print("Calcul regression F(Delta U10) = Delta Surface Polynie")
slopes_delta_U10_delta_S_MB,intercept_delta_U10_delta_S_MB,r_delta_U10_delta_S_MB = tp.regressionLineaireDataSurfacepolynie(delta_U10_MB_Win_2010_2020, 
                                                                                                    delta_surface_polynie_MB_Win_2010_2020,
                                                                                                    titre='Correlation delta U10 et delta Surface de la polynie de Mackenzie Bay - Winter 2010-2020')

md.plotMap_CapeDarnley(slopes_delta_U10_delta_S_MB,
                       Liste_Vmin_Vmax=[-0.3,0.3],
                       dimension=r"$m.s^{-1}/m^2$",
                       CI_contour=CI_moyen_MB,
                       level_CI_contour=np.linspace(0.2,0.9,15))
md.plotMap_CapeDarnley(r_delta_U10_delta_S_MB,
                       Liste_Vmin_Vmax=[-0.25,0.25],
                       dimension=r"Sans unité (entre -1 et 1)",
                       CI_contour=CI_moyen_MB,
                       level_CI_contour=np.linspace(0.2,0.9,15))

# In[Calcul de la regression de données par rapport à la surface - blh ]

##### Vitesse a 10m #####
print("Calcul regression F(Delta blh) = Delta Surface Polynie")
slopes_delta_blh_delta_S_MB,intercept_delta_blh_delta_S_MB,r_delta_blh_delta_S_MB = tp.regressionLineaireDataSurfacepolynie(delta_blh_MB_Win_2010_2020, 
                                                                                                    delta_surface_polynie_MB_Win_2010_2020,
                                                                                                    titre='Correlation delta blh et delta Surface de la polynie de Mackenzie Bay - Winter 2010-2020')

md.plotMap_CapeDarnley(slopes_delta_blh_delta_S_MB,
                       #Liste_Vmin_Vmax=[-0.3,0.3],
                       dimension=r"$m.s^{-1}/m^2$",
                       CI_contour=CI_moyen_MB,
                       level_CI_contour=np.linspace(0.2,0.9,15))
md.plotMap_CapeDarnley(r_delta_blh_delta_S_MB,
                       #Liste_Vmin_Vmax=[-0.25,0.25],
                       dimension=r"Sans unité (entre -1 et 1)",
                       CI_contour=CI_moyen_MB,
                       level_CI_contour=np.linspace(0.2,0.9,15))


# In[Calcul de la regression de data par rapport à Delta S]

# In[Calcul de la regression de données par rapport à la surface - u10 ]

u10_MB_Win_2010_2020 = tp.dataYearsMonths('u10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')

##### Vitesse a 10m #####
print("Calcul regression u10 = F(Delta Surface Polynie)")
slopes_u10_delta_S_MB,intercept_u10_delta_S_MB,r_u10_delta_S_MB = tp.regressionLineaireDataSurfacepolynie(u10_MB_Win_2010_2020, 
                                                                                                    delta_surface_polynie_MB_Win_2010_2020,
                                                                                                    titre='Correlation u10 et delta Surface de la polynie de Mackenzie Bay - Winter 2010-2020')

md.plotMap_CapeDarnley(slopes_u10_delta_S_MB,
                       Liste_Vmin_Vmax=[-0.4,0.4],
                       dimension=r"$m.s^{-1}/m^2$",
                       CI_contour=CI_moyen_MB,
                       level_CI_contour=np.linspace(0.2,0.9,15))
md.plotMap_CapeDarnley(r_u10_delta_S_MB,
                       Liste_Vmin_Vmax=[-0.1,0.1],
                       dimension=r"Sans unité (entre -1 et 1)",
                       CI_contour=CI_moyen_MB,
                       level_CI_contour=np.linspace(0.2,0.9,15))

# In[Calcul de la regression de données par rapport à la surface - v10 ]

v10_MB_Win_2010_2020 = tp.dataYearsMonths('v10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')

##### Vitesse a 10m #####
print("Calcul regression v10 =  F(Delta  Surface Polynie)")
slopes_v10_delta_S_MB,intercept_v10_delta_S_MB,r_v10_delta_S_MB = tp.regressionLineaireDataSurfacepolynie(v10_MB_Win_2010_2020, 
                                                                                                    delta_surface_polynie_MB_Win_2010_2020,
                                                                                                    titre='Correlation v10 et delta Surface de la polynie de Mackenzie Bay - Winter 2010-2020')

md.plotMap_CapeDarnley(slopes_v10_delta_S_MB,
                       Liste_Vmin_Vmax=[-0.30,0.30],
                       dimension=r"$m.s^{-1}/m^2$",
                       CI_contour=CI_moyen_MB,
                       level_CI_contour=np.linspace(0.2,0.9,15))
md.plotMap_CapeDarnley(r_v10_delta_S_MB,
                       Liste_Vmin_Vmax=[-0.1,0.1],
                       dimension=r"Sans unité (entre -1 et 1)",
                       CI_contour=CI_moyen_MB,
                       level_CI_contour=np.linspace(0.2,0.9,15))

# In[Calcul de la regression de données par rapport à la surface - U10 ]

u10_MB_Win_2010_2020 = tp.dataYearsMonths('u10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
v10_MB_Win_2010_2020 = tp.dataYearsMonths('v10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')

U10_MB_Win_2010_2020 = np.sqrt(u10_MB_Win_2010_2020*u10_MB_Win_2010_2020 + v10_MB_Win_2010_2020*v10_MB_Win_2010_2020)

##### Vitesse a 10m #####
print("Calcul regression u10 = F(Delta Surface Polynie)")
slopes_U10_delta_S_MB,intercept_U10_delta_S_MB,r_U10_delta_S_MB = tp.regressionLineaireDataSurfacepolynie(U10_MB_Win_2010_2020, 
                                                                                                    delta_surface_polynie_MB_Win_2010_2020,
                                                                                                    titre='Correlation u10 et delta Surface de la polynie de Mackenzie Bay - Winter 2010-2020')

md.plotMap_CapeDarnley(slopes_U10_delta_S_MB,
                       Liste_Vmin_Vmax=[-0.3,0.3],
                       dimension=r"$m.s^{-1}/m^2$",
                       CI_contour=CI_moyen_MB,
                       level_CI_contour=np.linspace(0.2,0.9,15))
md.plotMap_CapeDarnley(r_U10_delta_S_MB,
                       Liste_Vmin_Vmax=[-0.1,0.1],
                       dimension=r"Sans unité (entre -1 et 1)",
                       CI_contour=CI_moyen_MB,
                       level_CI_contour=np.linspace(0.2,0.9,15))

# In[Calcul de la regression de données par rapport à la surface - blh ]

# blh_MB_Win_2010_2020 = tp.dataYearsMonths('blh',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')

# ##### Vitesse a 10m #####
# print("Calcul regression blh = F(Delta Surface Polynie)")
# slopes_delta_blh_delta_S_MB,intercept_delta_blh_delta_S_MB,r_delta_blh_delta_S_MB = tp.regressionLineaireDataSurfacepolynie(blh_MB_Win_2010_2020, 
#                                                                                                     delta_surface_polynie_MB_Win_2010_2020,
#                                                                                                     titre='Correlation blh et delta Surface de la polynie de Cape Darnley - Winter 2010-2020')

# md.plotMap_CapeDarnley(slopes_delta_blh_delta_S_MB,
#                        # Liste_Vmin_Vmax=[-1,1],
#                        dimension=r"$m.s^{-1}/m^2$",
#                        CI_contour=CI_moyen_MB,
#                        level_CI_contour=np.linspace(0.2,0.9,15))

# md.plotMap_CapeDarnley(r_delta_blh_delta_S_MB,
#                        # Liste_Vmin_Vmax=[-0.2,0.2],
#                        dimension=r"Sans unité (entre -1 et 1)",
#                        CI_contour=CI_moyen_MB,
#                        level_CI_contour=np.linspace(0.2,0.9,15))


# In[Selection de l'ouverture de la polynie]

# time_polynie = surface_polynie_MB_Win_2010_2020.time.values

# time_polynya = []
# delta_surface_polynya = []

# for i in range(2010,2021,1):
#     surface_an = surface_polynie_MB_Win_2010_2020.sel(time=str(i)).values
#     time_an = surface_polynie_MB_Win_2010_2020.sel(time=str(i)).time.values
    
#     for j in range(len(time_an)-1):
#         time_polynya.append(str(time_an[j])[:10])
#         delta_surface_polynya.append(surface_an[j+1] - surface_an[j])

# # In[Sélection de données ouverture ]

# List_u10_MB_Win_2010_2020 = []
# List_v10_MB_Win_2010_2020 = []

# List_blh_MB_Win_2010_2020 = []

# List_surface_polynie_MB_Win_2010_2020 = []

# for x in time_polynya :
#     List_u10_MB_Win_2010_2020.append(delta_u10_MB_Win_2010_2020.sel(time=x).values)
#     List_v10_MB_Win_2010_2020.append(delta_v10_MB_Win_2010_2020.sel(time=x).values)
    
#     List_blh_MB_Win_2010_2020.append(delta_blh_MB_Win_2010_2020.sel(time=x).values)
    
#     List_surface_polynie_MB_Win_2010_2020.append(surface_polynie_MB_Win_2010_2020.sel(time=x).values)
    
# print(np.mean(List_u10_MB_Win_2010_2020))
# print(np.mean(List_v10_MB_Win_2010_2020))

# plt.figure(constrained_layout=True, figsize=(1.5*6.4, 1.5*4.8))
# plt.scatter(delta_surface_polynya,List_u10_MB_Win_2010_2020,marker='+',color='r')
# # plt.scatter(np.mean(delta_surface_polynya),np.mean(u10_MB_Win_2010_2020),marker='o',color='b')
# # plt.title('u10')

# plt.xlabel(r"Time derivative of surface ($\frac{km^2}{h}$)",size=20)
# plt.ylabel(r"Time derivative of speed($\frac{m.s^{-1}}{h}$)",size=20)

# plt.tick_params(axis = 'both', labelsize = 15)
# plt.grid()
# plt.show()

# plt.figure(constrained_layout=True, figsize=(1.5*6.4, 1.5*4.8))
# plt.scatter(delta_surface_polynya,List_v10_MB_Win_2010_2020,marker='+',color='r')
# # plt.scatter(np.mean(delta_surface_polynya),np.mean(List_v10_MB_Win_2010_2020),marker='o',color='b')
# # plt.title('v10')

# plt.xlabel(r"Time derivative of surface ($\frac{km^2}{h}$)",size=20)
# plt.ylabel(r"Time derivative of speed ($\frac{m.s^{-1}}{h})$",size=20)
# plt.tick_params(axis = 'both', labelsize = 15)
# plt.grid()
# plt.show()

# plt.figure(constrained_layout=True, figsize=(1.5*6.4, 1.5*4.8))
# plt.scatter(delta_surface_polynya,List_blh_MB_Win_2010_2020,marker='+',color='r')
# # plt.scatter(np.mean(delta_surface_polynya),np.mean(List_blh_MB_Win_2010_2020),marker='o',color='b')
# #plt.title('blh',fontsize=20)

# plt.xlabel(r"Time derivative of surface ($\frac{km^2}{h}$)",size=20)
# plt.ylabel(r"Time derivative of boundary layer height ($\frac{m.s^{-1}}{h})$",size=20)
# plt.tick_params(axis = 'both', labelsize = 15)
# plt.grid()
# plt.show()

# # In[Tracé de nuage de points]

# plt.figure(constrained_layout=True, figsize=(1*6.4, 2*4.8))
# plt.scatter(surface_polynie_MB_Win_2010_2020,delta_u10_MB_Win_2010_2020,marker='+',color='r')
# plt.show()

# plt.figure(constrained_layout=True, figsize=(1*6.4, 2*4.8))
# plt.scatter(surface_polynie_MB_Win_2010_2020,delta_v10_MB_Win_2010_2020,marker='+',color='r')
# plt.show()