#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:04:34 2022

@author: uvc35ld
"""


# In[Chargement bibliothèques]
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
# import pandas as pd
import matplotlib as mpl

import MapDrawing as md
import ToolsPolynias as tp

# In[Chargement des données Polynies ]

    
##### polynie pour Wintembre :
polynie_CD_around_Win_2010_2020_0 = tp.dataYearsMonths('polynie',List_Months=range(5,10),List_Years=range(2010,2021),area='Cape Darnley Around')
polynie_CD_Win_2010_2020_0 = tp.sel_CapeDarnley(polynie_CD_around_Win_2010_2020_0)

polynie_CD_Win_2010_2020 = tp.selection_polynie_CD(polynie_CD_Win_2010_2020_0)

time_polynies_2010_2020 = polynie_CD_Win_2010_2020.time.values
surfaces_polynies_2010_2020 = tp.calculSurfacePolynies(polynie_CD_Win_2010_2020,area='CDP')

print('Fin de chargement de la surface de la polynie')

CI_moyen = tp.CIMoyenne()
CI_moyen_CD = tp.sel_CapeDarnley(CI_moyen)
print('Fin de chargement de CI')

occurence_polynie_CD = tp.OccurencePolynieMoyenne_CD()

# In[Creation Zone Terre]

u10_CD_Win_2010_2020 = tp.dataYearsMonths('u10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')

data = u10_CD_Win_2010_2020.mean('time')

Latitudes = data.latitude.values
Longitudes = data.longitude.values

zone_terre = np.zeros(data.shape)
for i in range(len(Latitudes)):
    for j in range(len(Latitudes)):
        if Latitudes[i] == -68.25 and Longitudes[j] >= 66.75 and Longitudes[j] <= 70.25 :
            zone_terre[i,j] = 1

data_zone_terre = tp.creationDataArray(zone_terre, Latitudes, Longitudes)

# In[Chargement des données u10]

# u10_CD_Win_2010_2020 = tp.dataYearsMonths('u10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
u10_CD_Win_2010_2020 = tp.dataYearsMonths('u10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')

print('Fin chargement de u10')

### Tracé Cape Darnley Around ###
# md.plotMap_CapeDarnley_around(u10_CD_around_Win_2010_2020, dimension='$m.s^{-1}$',CI_contour=occurence_polynie_CD,level_CI_contour=np.linspace(0.2,0.9,8))

data = u10_CD_Win_2010_2020.mean('time')

Latitudes = data.latitude.values
Longitudes = data.longitude.values

time_series = u10_CD_Win_2010_2020.sel(latitude=-68.25).sel(longitude=slice(66.75,70.25)).mean('longitude')

##### Vitesse a 10m #####
print("Calcul regression Surface Polynie = F(v10)")
slopes_u10_S_CD_aro,intercept_u10_S_CD_aro,r_u10_S_CD_aro = tp.regressionLineaireDataSurfacepolynie(u10_CD_Win_2010_2020, 
                                                                                                    time_series,
                                                                                                    titre='Correlation v10 et Surface de la polynie de Cape Darnley - Winter 2010-2020')

# md.plotMap_CapeDarnley(slopes_u10_S_CD_aro,dimension=r"$m.s^{-1}/m^{2}$",# titre='Regression de u10',
#                        Liste_Vmin_Vmax=[-5,5],CI_contour=occurence_polynie_CD,
#                        level_CI_contour=np.linspace(0.2,0.6,3))


# md.plotMap_CapeDarnley(r_u10_S_CD_aro,dimension=r"Sans unité",# titre='Regression de v10',
#                        Liste_Vmin_Vmax=[-1,1],CI_contour=occurence_polynie_CD,
#                        level_CI_contour=np.linspace(0.2,0.6,3))

md.plotMap_Regression(slopes_u10_S_CD_aro, r_u10_S_CD_aro,dimension= r"$m.s^{-1}/(m.s^{-1})$",
                      Liste_Rmin_Rmax=[0,5],
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

time_series = v10_CD_Win_2010_2020.sel(latitude=-68.25).sel(longitude=slice(66.75,70.25)).mean('longitude')

##### Vitesse a 10m #####
print("Calcul regression Surface Polynie = F(v10)")
slopes_v10_S_CD_aro,intercept_v10_S_CD_aro,r_v10_S_CD_aro = tp.regressionLineaireDataSurfacepolynie(v10_CD_Win_2010_2020, 
                                                                                                    time_series,
                                                                                                    titre='Correlation v10 et Surface de la polynie de Cape Darnley - Winter 2010-2020')

# md.plotMap_CapeDarnley(slopes_v10_S_CD_aro,dimension=r"$m.s^{-1}/m^{2}$",# titre='Regression de v10',
#                        Liste_Vmin_Vmax=[-5,5],CI_contour=occurence_polynie_CD,
#                        level_CI_contour=np.linspace(0.2,0.6,3))


# md.plotMap_CapeDarnley(r_v10_S_CD_aro,dimension=r"Sans unité",# titre='Regression de v10',
#                        Liste_Vmin_Vmax=[-1,1],CI_contour=occurence_polynie_CD,
#                        level_CI_contour=np.linspace(0.2,0.6,3))

md.plotMap_Regression(slopes_v10_S_CD_aro, r_v10_S_CD_aro,#dimension=r"Sans unité",# titre='Regression de v10',
                      dimension= r"$m.s^{-1}/(m.s^{-1})$",
                      Liste_Rmin_Rmax=[0,5],
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

time_series = U10_CD_Win_2010_2020.sel(latitude=-68.25).sel(longitude=slice(66.75,70.25)).mean('longitude')

##### Vitesse a 10m #####
print("Calcul regression Surface Polynie = F(U10)")
slopes_U10_S_CD_aro,intercept_U10_S_CD_aro,r_U10_S_CD_aro = tp.regressionLineaireDataSurfacepolynie(U10_CD_Win_2010_2020, 
                                                                                                    time_series,
                                                                                                    titre='Correlation U10 et Surface de la polynie de Cape Darnley - Winter 2010-2020')

# md.plotMap_CapeDarnley(slopes_U10_S_CD_aro,dimension=r"$m.s^{-1}/km^{2}$",# titre='Regression de v10',
#                        Liste_Vmin_Vmax=[-5,5],CI_contour=occurence_polynie_CD,
#                        level_CI_contour=np.linspace(0.2,0.6,3))


# md.plotMap_CapeDarnley(r_U10_S_CD_aro,dimension=r"Sans unité",# titre='Regression de v10',
#                        Liste_Vmin_Vmax=[-1,1],CI_contour=occurence_polynie_CD,
#                        level_CI_contour=np.linspace(0.2,0.6,3))

md.plotMap_Regression(slopes_U10_S_CD_aro, r_U10_S_CD_aro,#dimension=r"Sans unité",# titre='Regression de v10',
                      dimension= r"$m.s^{-1}/km^{2}$",
                      Liste_Rmin_Rmax=[0,5],
                      Liste_Corrmin_Corrmax=[-1,1],
                      CI_contour=occurence_polynie_CD,
                      level_CI_contour=np.linspace(0.2,0.6,3))


# In[Tracé test]

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
im2 = data_zone_terre.plot(ax=ax,
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
