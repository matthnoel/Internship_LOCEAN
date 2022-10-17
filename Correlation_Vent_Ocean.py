#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 15:12:05 2022

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
import matplotlib as mpl

import MapDrawing as md
import ToolsPolynias as tp


# In[Test Chargement de la polynie - Cape Darnley - 2010-2020]

polynie_CD_Win_2010_2020 = tp.dataYearsMonths('polynie',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
surface_polynie_CD_Win_2010_2020 = tp.calculSurfacePolynies(polynie_CD_Win_2010_2020)

# delta_surface_polynie_CD_Win_2010_2020 = tp.calculSurfacePolynies_delta_H(List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')


Latitudes = polynie_CD_Win_2010_2020.latitude.values
Longitudes = polynie_CD_Win_2010_2020.longitude.values

# print('#############################')
# ##### Chargement des données différences entre 21h et 23h #####
# delta_polynie_CD_Win_2010_2020 = tp.dataYearsMonths_delta_H('polynie',List_Years=range(2010,2011),List_Months=range(9,10),area='Cape Darnley')
# print('Fin chargement polynie')

CI_moyen = tp.CIMoyenne()
CI_moyen_CD = tp.sel_CapeDarnley(CI_moyen)

occurence_polynie_CD = tp.OccurencePolynieMoyenne_CD()

# In[Calcul de la regression de données par rapport à la surface - v10 ]

# print('#############################')
# delta_v10_CD_Win_2010_2020 = tp.dataYearsMonths_delta_H('v10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
# print('Fin chargement v10')


# ##### Vitesse a 10m #####
# print("Calcul regression F(Delta v10) = Delta Surface Polynie")
# slopes_delta_v10_delta_S_CD,intercept_delta_v10_delta_S_CD,r_delta_v10_delta_S_CD = tp.regressionLineaireDataSurfacepolynie(delta_v10_CD_Win_2010_2020, 
#                                                                                                     delta_surface_polynie_CD_Win_2010_2020,
#                                                                                                     titre='Correlation delta v10 et delta Surface de la polynie de Cape Darnley - Winter 2010-2020')

# md.plotMap_CapeDarnley(slopes_delta_v10_delta_S_CD,
#                        Liste_Vmin_Vmax=[-0.30,0.30],
#                        dimension=r"$m.s^{-1}/m^2$",
#                        CI_contour=occurence_polynie_CD,
#                        level_CI_contour=np.linspace(0.2,0.6,3))
# md.plotMap_CapeDarnley(r_delta_v10_delta_S_CD,
#                        Liste_Vmin_Vmax=[-0.25,0.25],
#                        dimension=r"Sans unité (entre -1 et 1)",
#                        CI_contour=occurence_polynie_CD,
#                        level_CI_contour=np.linspace(0.2,0.6,3))


# In[Creation Zone Ocean]

# u10_CD_Win_2010_2020 = tp.dataYearsMonths('u10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')

data = polynie_CD_Win_2010_2020 #u10_CD_Win_2010_2020.mean('time')

Latitudes = data.latitude.values
Longitudes = data.longitude.values

zone_ocean = np.zeros((Latitudes.shape[0],Longitudes.shape[0]))
for i in range(len(Latitudes)):
    for j in range(len(Longitudes)):
        if (Latitudes[i] == -68 or Latitudes[i] == -68 )and Longitudes[j] >= 69.75 and Longitudes[j] <= 70.25 :
            zone_ocean[i,j] = 1

data_zone_ocean = tp.creationDataArray(zone_ocean, Latitudes, Longitudes)
print(data_zone_ocean.shape)

md.plotMap_CapeDarnley(data_zone_ocean,color='binary', color_bar=False,
                       CI_contour=occurence_polynie_CD,
                       level_CI_contour=np.linspace(0.2,0.6,3)
                       )

# In[Chargement des données u10]

# u10_CD_Win_2010_2020 = tp.dataYearsMonths('u10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
u10_CD_Win_2010_2020 = tp.dataYearsMonths('u10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')

print('Fin chargement de u10')

### Tracé Cape Darnley Around ###
# md.plotMap_CapeDarnley_around(u10_CD_around_Win_2010_2020, dimension='$m.s^{-1}$',CI_contour=occurence_polynie_CD,level_CI_contour=np.linspace(0.2,0.9,8))

data = u10_CD_Win_2010_2020.mean('time')

Latitudes = data.latitude.values
Longitudes = data.longitude.values

time_series = u10_CD_Win_2010_2020.sel(latitude=-68).sel(longitude=slice(69.75,70.25)).mean('longitude')

##### Vitesse a 10m #####
print("Calcul regression Surface Polynie = F(v10)")
slopes_u10_S_CD_aro,intercept_u10_S_CD_aro,r_u10_S_CD_aro = tp.regressionLineaireDataSurfacepolynie(u10_CD_Win_2010_2020, 
                                                                                                    time_series,
                                                                                                    titre='Correlation v10 et Surface de la polynie de Cape Darnley - Winter 2010-2020')

# md.plotMap_CapeDarnley(slopes_u10_S_CD_aro,dimension=r"$m.s^{-1}/m^{2}$",# titre='Regression de u10',
#                        Liste_Vmin_Vmax=[-7,7],CI_contour=occurence_polynie_CD,
#                        level_CI_contour=np.linspace(0.2,0.6,3))


# md.plotMap_CapeDarnley(r_u10_S_CD_aro,dimension=r"Sans unité",# titre='Regression de v10',
#                        Liste_Vmin_Vmax=[-1,1],CI_contour=occurence_polynie_CD,
#                        level_CI_contour=np.linspace(0.2,0.6,3))

md.plotMap_Regression(slopes_u10_S_CD_aro, r_u10_S_CD_aro,dimension= r"$m.s^{-1}/(m.s^{-1})$",
                      Liste_Rmin_Rmax=[-6,6],
                      Liste_Corrmin_Corrmax=[0.5,1],
                      CI_contour=occurence_polynie_CD,
                      level_CI_contour=np.linspace(0.2,0.6,3))



# In[Chargement des données v10]

# v10_CD_Win_2010_2020 = tp.dataYearsMonths('v10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
v10_CD_Win_2010_2020 = tp.dataYearsMonths('v10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')

print('Fin chargement de v10')

### Tracé Cape Darnley Around ###
# md.plotMap_CapeDarnley_around(v10_CD_around_Win_2010_2020, dimension='$m.s^{-1}$',CI_contour=occurence_polynie_CD,level_CI_contour=np.linspace(0.2,0.9,8))

data = v10_CD_Win_2010_2020.mean('time')

Latitudes = data.latitude.values
Longitudes = data.longitude.values

time_series = v10_CD_Win_2010_2020.sel(latitude=-68).sel(longitude=slice(69.75,70.25)).mean('longitude')

##### Vitesse a 10m #####
print("Calcul regression v10 = F(v10_Ocean)")
slopes_v10_S_CD_aro,intercept_v10_S_CD_aro,r_v10_S_CD_aro = tp.regressionLineaireDataSurfacepolynie(v10_CD_Win_2010_2020, 
                                                                                                    time_series,
                                                                                                    titre='Correlation v10 et Surface de la polynie de Cape Darnley - Winter 2010-2020')

# md.plotMap_CapeDarnley(slopes_v10_S_CD_aro,dimension=r"$m.s^{-1}/m^{2}$",# titre='Regression de v10',
#                        Liste_Vmin_Vmax=[-7,7],CI_contour=occurence_polynie_CD,
#                        level_CI_contour=np.linspace(0.2,0.6,3))


# md.plotMap_CapeDarnley(r_v10_S_CD_aro,dimension=r"Sans unité",# titre='Regression de v10',
#                        Liste_Vmin_Vmax=[-1,1],CI_contour=occurence_polynie_CD,
#                        level_CI_contour=np.linspace(0.2,0.6,3))

md.plotMap_Regression(slopes_v10_S_CD_aro, r_v10_S_CD_aro,dimension= r"$m.s^{-1}/(m.s^{-1})$",
                      Liste_Rmin_Rmax=[0,6],
                      Liste_Corrmin_Corrmax=[0.5,1],
                      CI_contour=occurence_polynie_CD,
                      level_CI_contour=np.linspace(0.2,0.6,3))


# In[Chargement des données U10]

# v10_CD_Win_2010_2020 = tp.dataYearsMonths('v10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
U10_CD_Win_2010_2020 = np.sqrt(u10_CD_Win_2010_2020*u10_CD_Win_2010_2020 + v10_CD_Win_2010_2020*v10_CD_Win_2010_2020)

print('Fin chargement de U10')

### Tracé Cape Darnley Around ###
# md.plotMap_CapeDarnley_around(v10_CD_around_Win_2010_2020, dimension='$m.s^{-1}$',CI_contour=occurence_polynie_CD,level_CI_contour=np.linspace(0.2,0.9,8))

data = U10_CD_Win_2010_2020.mean('time')

Latitudes = data.latitude.values
Longitudes = data.longitude.values

time_series = U10_CD_Win_2010_2020.sel(latitude=-68).sel(longitude=slice(69.75,70.25)).mean('longitude')

##### Vitesse a 10m #####
print("Calcul regression U10 = F(U10_Ocean)")
slopes_U10_S_CD_aro,intercept_U10_S_CD_aro,r_U10_S_CD_aro = tp.regressionLineaireDataSurfacepolynie(U10_CD_Win_2010_2020, 
                                                                                                    time_series,
                                                                                                    titre='Correlation v10 et Surface de la polynie de Cape Darnley - Winter 2010-2020')

# md.plotMap_CapeDarnley(slopes_U10_S_CD_aro,dimension=r"$m.s^{-1}/m^{2}$",# titre='Regression de v10',
#                        Liste_Vmin_Vmax=[-7,7],CI_contour=occurence_polynie_CD,
#                        level_CI_contour=np.linspace(0.2,0.6,3))


# md.plotMap_CapeDarnley(r_U10_S_CD_aro,dimension=r"Sans unité",# titre='Regression de v10',
#                        Liste_Vmin_Vmax=[-1,1],CI_contour=occurence_polynie_CD,
#                        level_CI_contour=np.linspace(0.2,0.6,3))

md.plotMap_Regression(slopes_U10_S_CD_aro, r_U10_S_CD_aro,dimension= r"$m.s^{-1}/(m.s^{-1})$",
                      #Liste_Rmin_Rmax=[0,5],
                      Liste_Corrmin_Corrmax=[-1,1],
                      CI_contour=occurence_polynie_CD,
                      level_CI_contour=np.linspace(0.2,0.6,3))


# In[Tracé test]

data = polynie_CD_Win_2010_2020.mean('time')

color = 'RdBu_r'
dimension = r"$m.s^{-1}$"
Liste_Vmin_Vmax = [-0.3,0.3]
titre = 'test'

plt.figure(figsize=(12,10), frameon=True)


ax = plt.axes(projection=ccrs.SouthPolarStereo())#Orthographic(-65,-75))(-65,-75))

# Ajout des lignes de côtes / grid :
ax.coastlines(resolution="10m",linewidth=3)
gl = ax.gridlines(linestyle='--',color='black',
              draw_labels=True)
gl.top_labels = False
gl.right_labels = False
''
# Tracé du contour de la donnée choisie :


# # Tracé du contour de la donnée choisie :

# im = data.plot(ax=ax,
#                     transform=ccrs.PlateCarree(),
#                     cmap=color,#plt.cm.gray,
#                     add_colorbar=False
#                     # add_colorbar=False,
#                     # norm = mpl.colors.Normalize(vmin=Liste_Vmin_Vmax[0], vmax=Liste_Vmin_Vmax[1])
#                     );
im2 = data_zone_ocean.plot(ax=ax,
                    transform=ccrs.PlateCarree(),
                    cmap='binary',#plt.cm.gray,
                    add_colorbar=False
                    # add_colorbar=False,
                    # norm = mpl.colors.Normalize(vmin=Liste_Vmin_Vmax[0], vmax=Liste_Vmin_Vmax[1])
                    );


cont = occurence_polynie_CD.plot.contour(ax=ax,
                        transform=ccrs.PlateCarree(),
                        # levels=level_CI_contour,
                        cmap='black',#plt.cm.gray,
                        add_colorbar=False
                        );
ax.clabel(cont, inline=True, fontsize=15)




# cmap = color
# norm = mpl.colors.Normalize(vmin=Liste_Vmin_Vmax[0], vmax=Liste_Vmin_Vmax[1])

# cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),
#     orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
# cb.set_label(dimension,size=18,rotation=90,labelpad=1)
# cb.ax.tick_params(labelsize=18)


# # Création de la barre de couleur (Bonne orientation et bonne taille...) :
# cb = plt.colorbar(im, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
# cb.set_label(dimension,size=18,rotation=90,labelpad=1)
# cb.ax.tick_params(labelsize=18)

# Création du titre :
# ax.set_title(titre,fontsize=20)
# ax.set_xlabel("")

plt.show()

# In[test2]

data = u10_CD_Win_2010_2020.mean('time')

color = 'RdBu_r'
dimension = r"$m.s^{-1}$"
Liste_Vmin_Vmax = [-15,15]
titre = 'test'

plt.figure(figsize=(12,10), frameon=True)


ax = plt.axes(projection=ccrs.SouthPolarStereo())#Orthographic(-65,-75))(-65,-75))

# Ajout des lignes de côtes / grid :
ax.coastlines(resolution="10m",linewidth=3)
gl = ax.gridlines(linestyle='--',color='black',
              draw_labels=True)
gl.top_labels = False
gl.right_labels = False
''
# Tracé du contour de la donnée choisie :


# Tracé du contour de la donnée choisie :

im = data.plot(ax=ax,
                    transform=ccrs.PlateCarree(),
                    cmap=color,#plt.cm.gray,
                    add_colorbar=False
                    # add_colorbar=False,
                    # norm = mpl.colors.Normalize(vmin=Liste_Vmin_Vmax[0], vmax=Liste_Vmin_Vmax[1])
                    );

im2 = data_zone_ocean.plot(ax=ax,
                    transform=ccrs.PlateCarree(),
                    cmap='binary',#plt.cm.gray,
                    add_colorbar=False
                    # add_colorbar=False,
                    # norm = mpl.colors.Normalize(vmin=Liste_Vmin_Vmax[0], vmax=Liste_Vmin_Vmax[1])
                    );


cont = occurence_polynie_CD.plot.contour(ax=ax,
                                         levels=np.linspace(0.2,0.6,3),
                        transform=ccrs.PlateCarree(),
                        # levels=level_CI_contour,
                        cmap='black',#plt.cm.gray,
                        add_colorbar=False
                        );
ax.clabel(cont, inline=True, fontsize=15)




# cmap = color
# norm = mpl.colors.Normalize(vmin=Liste_Vmin_Vmax[0], vmax=Liste_Vmin_Vmax[1])

# cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),
#     orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
# cb.set_label(dimension,size=18,rotation=90,labelpad=1)
# cb.ax.tick_params(labelsize=18)


# # Création de la barre de couleur (Bonne orientation et bonne taille...) :
# cb = plt.colorbar(im, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
# cb.set_label(dimension,size=18,rotation=90,labelpad=1)
# cb.ax.tick_params(labelsize=18)

# Création du titre :
# ax.set_title(titre,fontsize=20)
#ax.set_xlabel("")

plt.show()


