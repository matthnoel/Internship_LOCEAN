#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:45:27 2022

@author: uvc35ld
"""

# In[Chargement bibliothèques]
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import matplotlib as mpl
# import xarray as xr
# import datetime
# import pandas as pd
# from cartopy.util import add_cyclic_point
# import pandas as pd
# from scipy import stats
# import scipy
# import xrft

import MapDrawing as md
import ToolsPolynias as tp

from eofs.xarray import Eof as eof

# In[Chargement de la polynie - Cape Darnley - 2010]

polynie_CD_Win_2010 = tp.dataYearsMonths('polynie',List_Years=[2010],List_Months=range(5,10),area='Cape Darnley')

Latitudes = polynie_CD_Win_2010.latitude.values
Longitudes = polynie_CD_Win_2010.longitude.values

n_time = polynie_CD_Win_2010.shape[0]

# for i in range(10,100,10):
#     md.plotMap_Ant(polynie_CD_Win_2010.isel(time=i), np.linspace(0,1,3),color='binary',color_bar=False )
    
occurence_polynies_Win_2010 = np.zeros((Latitudes.shape[0],Longitudes.shape[0]))

for i in range(Latitudes.shape[0]):
    for j in range(Longitudes.shape[0]):
        compteur = 0
        for k in range(polynie_CD_Win_2010.shape[0]):
            if polynie_CD_Win_2010[k,i,j] == 1 :
                compteur += 1
        occurence_polynies_Win_2010[i,j] = compteur/n_time

data_occurence_polynies_Win_2010 = tp.creationDataArray(occurence_polynies_Win_2010, Latitudes, Longitudes)

md.plotMap_CapeDarnley(data_occurence_polynies_Win_2010, np.linspace(0,0.75,16),color='Reds',titre='Occurences polynies 2010')

# In[EOF Base Polynie]

solver = eof(polynie_CD_Win_2010)

solver.varianceFraction(neigs=10).plot.step(where='mid')


eofs = solver.eofsAsCovariance(neofs=2)      # beware of syntaxe: A C
pcs = solver.pcs(npcs=2, pcscaling=1)
varfrac = solver.varianceFraction(neigs=2)   # beware of syntaxe: F

eofs = solver.eofsAsCovariance(neofs=2)      # beware of syntaxe: A C
pcs = solver.pcs(npcs=2, pcscaling=1)
varfrac = solver.varianceFraction(neigs=2)   # beware of syntaxe: F
# create a 4 pannels figure
fig, axes = plt.subplots(2,2,figsize=(10, 5),constrained_layout=True)


# plot EOF1 and PC1
eofs.sel(mode=0).plot(ax=axes[0,0], cbar_kwargs={'label': '°C for 1 std of PC'})#,transform=ccrs.PlateCarree())

axes[0,0].set_title('EOF 1: '+str(int(varfrac.values[0]*100))+'%') 
pcs.sel(mode=0).plot(ax=axes[1,0])
axes[1,0].set_title('PC 1')
# plot EOF2 and PC2
eofs.sel(mode=1).plot(ax=axes[0,1], cbar_kwargs={'label': '°C for 1 std of PC'})#,transform=ccrs.PlateCarree())

axes[0,1].set_title('EOF 2: '+str(int(varfrac.values[1]*100))+'%')
pcs.sel(mode=1).plot(ax=axes[1,1])
axes[1,1].set_title('PC 2')


# In[EOF Polynies]

solver = eof(polynie_CD_Win_2010)

solver.varianceFraction(neigs=10).plot.step(where='mid')
plt.grid()
plt.xlim(0,9)
plt.ylim(0,0.35)

eofs = solver.eofsAsCovariance(neofs=2)      # beware of syntaxe: A C
pcs = solver.pcs(npcs=2, pcscaling=1)
varfrac = solver.varianceFraction(neigs=2)   # beware of syntaxe: F
# create a 4 pannels figure
#fig, axes = plt.subplots(2,2,figsize=(10, 5),constrained_layout=True)

fig = plt.figure(figsize=(2*6.4, 2*4.8))

# plot EOF1 and PC1

####### EOF1 #######
ax=fig.add_subplot(2,2,1, projection=ccrs.SouthPolarStereo()) 
eofs.sel(mode=0).plot(ax=ax, cbar_kwargs={'label': '°C for 1 std of PC'},transform=ccrs.PlateCarree())#,projection=ccrs.SouthPolarStereo())
ax.set_title('EOF 1: '+str(int(varfrac.values[0]*100))+'%') 
ax.coastlines(resolution="10m",linewidth=2)
gl = ax.gridlines(linestyle='--',color='black',
              draw_labels=True)

####### PC1 #######
ax=fig.add_subplot(2,2,3)#, projection=ccrs.SouthPolarStereo()) 
pcs.sel(mode=0).plot(ax=ax)
ax.set_title('PC 1')


# plot EOF2 and PC2

####### EOF2 #######
ax=fig.add_subplot(2,2,2, projection=ccrs.SouthPolarStereo()) 
eofs.sel(mode=1).plot(ax=ax, cbar_kwargs={'label': '°C for 1 std of PC'},transform=ccrs.PlateCarree())#,projection=ccrs.SouthPolarStereo())
ax.set_title('EOF 2: '+str(int(varfrac.values[1]*100))+'%') 
ax.coastlines(resolution="10m",linewidth=2)
gl = ax.gridlines(linestyle='--',color='black',
              draw_labels=True)

####### PC2 #######
ax=fig.add_subplot(2,2,4)#, projection=ccrs.SouthPolarStereo()) 
pcs.sel(mode=1).plot(ax=ax)
ax.set_title('PC 2')


        
# In[Chargement de la polynie - Cape Darnley - 2010-2020]

polynie_CD_Win_2010_2020 = tp.dataYearsMonths('polynie',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')

print('Fin de chargement des polynies')

Latitudes = polynie_CD_Win_2010_2020.latitude.values
Longitudes = polynie_CD_Win_2010_2020.longitude.values

n_time = polynie_CD_Win_2010_2020.shape[0]

# for i in range(10,100,10):
#     md.plotMap_Ant(polynie_CD_Win_2010.isel(time=i), np.linspace(0,1,3),color='binary',color_bar=False )
    
occurence_polynies_Win_2010_2020 = np.zeros((Latitudes.shape[0],Longitudes.shape[0]))

for i in range(Latitudes.shape[0]):
    for j in range(Longitudes.shape[0]):
        compteur = 0
        for k in range(polynie_CD_Win_2010_2020.shape[0]) :
            if polynie_CD_Win_2010_2020[k,i,j] == 1:
                compteur += 1
        occurence_polynies_Win_2010_2020[i,j] = compteur/n_time #/n_time

data_occurence_polynies_Win_2010_2020 = tp.creationDataArray(occurence_polynies_Win_2010_2020, Latitudes, Longitudes)

md.plotMap_CapeDarnley(data_occurence_polynies_Win_2010_2020, np.linspace(0,0.75,16),color='Reds',titre='Occurences polynies 2010-2020')

# In[Définition polynie moyenne]

taux = 0.2

polynie_moyenne = np.zeros(data_occurence_polynies_Win_2010_2020.values.shape)
for i in range(polynie_moyenne.shape[0]):
    for j in range(polynie_moyenne.shape[1]):
        if data_occurence_polynies_Win_2010_2020.values[i,j] >= taux :
            polynie_moyenne[i,j]=1
data_polynie_moyenne = tp.creationDataArray(polynie_moyenne, 
                                            data_occurence_polynies_Win_2010_2020.latitude.values, 
                                            data_occurence_polynies_Win_2010_2020.longitude.values,
                                            nameDataArray='Polynie')
            
md.plotMap_CapeDarnley(data_polynie_moyenne,color='binary',color_bar=False,titre='Limite taux occurence '+str(int(taux*100))+' %')

path_NETCDF_sortie = '/gpfswork/rech/omr/uvc35ld/stage/Polynyas/ERA5/polynies_era5_Mean_Win_2010-2020.nc'

dataset_polynie = data_polynie_moyenne.to_dataset(name='polynie')
netCDF_polynie = dataset_polynie.to_netcdf(path=path_NETCDF_sortie)

# In[Test]

md.plotMap_CapeDarnley(data_occurence_polynies_Win_2010_2020,color='Reds',titre='Occurences polynies 2010-2020')

Liste_Vmin_Vmax = [0,0.75]

data_Cape = tp.sel_CapeDarnley(data_occurence_polynies_Win_2010_2020)
# clevels = np.linspace(0,1,11)
# dimension='sans unité'
# titre = 'test'

plt.figure(figsize=(12,10), frameon=True)

# Projection map :
ax = plt.axes(projection=ccrs.SouthPolarStereo())#Orthographic(-65,-75))

# Ajout des lignes de côtes / grid :
ax.coastlines(resolution="10m",linewidth=1)
gl = ax.gridlines(linestyle='--',color='black',
              draw_labels=True)
gl.top_labels = False
gl.right_labels = False
''
# Tracé du contour de la donnée choisie :
    
# Création de la barre de couleur (Bonne orientation et bonne taille...) :
cmap = 'Reds'
norm = mpl.colors.Normalize(vmin=Liste_Vmin_Vmax[0], vmax=Liste_Vmin_Vmax[1])


im = data_Cape.plot(ax=ax,
                   transform=ccrs.PlateCarree(),
                   add_colorbar=False,
                   cmap= 'Reds',
                   vmin= Liste_Vmin_Vmax[0],
                   vmax= Liste_Vmin_Vmax[1]
                   );

cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),
    orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
cb.set_label('1',size=18,rotation=90,labelpad=1)
cb.ax.tick_params(labelsize=18)


List_EO = np.linspace(67.5,71,8)
List_NS = np.linspace(-69,-67.5,8)

# for i in range(len(List_EO)):
#     plt.plot(List_EO[i], List_NS[i],
#               color='k', linewidth=10, marker='o',
#               transform=ccrs.PlateCarree(),
#               )

plt.plot(69.75, -68,
              color='k', linewidth=10, marker='o',
              transform=ccrs.PlateCarree(),
              )
plt.plot(70, -68,
              color='k', linewidth=10, marker='o',
              transform=ccrs.PlateCarree(),
              )

# Création du titre :
ax.set_title('Test',fontsize=20)
#ax.set_xlabel("")

plt.show()




# In[Tracés test]

# md.plotMap_CapeDarnley(data_occurence_polynies_Win_2010_2020, np.linspace(0,0.75,16),color='Reds')

# In[EOF Base Polynie]

solver = eof(polynie_CD_Win_2010_2020)

solver.varianceFraction(neigs=10).plot.step(where='mid')
plt.grid()
plt.xlim(0,9)

eofs = solver.eofsAsCovariance(neofs=2)      # beware of syntaxe: A C
pcs = solver.pcs(npcs=2, pcscaling=1)
varfrac = solver.varianceFraction(neigs=2)   # beware of syntaxe: F
# create a 4 pannels figure
fig, axes = plt.subplots(2,2,figsize=(10, 5),constrained_layout=True)


# plot EOF1 and PC1
eofs.sel(mode=0).plot(ax=axes[0,0], cbar_kwargs={'label': '1 for 1 std of PC'})#,transform=ccrs.PlateCarree())

axes[0,0].set_title('EOF 1: '+str(int(varfrac.values[0]*100))+'%') 
pcs.sel(mode=0).plot(ax=axes[1,0])
axes[1,0].set_title('PC 1')

# plot EOF2 and PC2
eofs.sel(mode=1).plot(ax=axes[0,1], cbar_kwargs={'label': '1 for 1 std of PC'})#,transform=ccrs.PlateCarree())

axes[0,1].set_title('EOF 2: '+str(int(varfrac.values[1]*100))+'%')
pcs.sel(mode=1).plot(ax=axes[1,1])
axes[1,1].set_title('PC 2')


# In[EOF Polynies]

solver = eof(polynie_CD_Win_2010_2020)

solver.varianceFraction(neigs=10).plot.step(where='mid')
plt.xlabel('EOF - number of modes')
plt.ylabel('Fractions of variance')
plt.grid()
plt.xlim(0,5)
plt.ylim(0,0.40)

eofs = solver.eofsAsCovariance(neofs=2)      # beware of syntaxe: A C
pcs = solver.pcs(npcs=2, pcscaling=1)
varfrac = solver.varianceFraction(neigs=2)   # beware of syntaxe: F
# create a 4 pannels figure
#fig, axes = plt.subplots(2,2,figsize=(10, 5),constrained_layout=True)

fig = plt.figure(figsize=(2*6.4, 2*4.8))

# plot EOF1 and PC1

####### EOF1 #######
ax=fig.add_subplot(2,2,1, projection=ccrs.SouthPolarStereo()) 
eofs.sel(mode=0).plot(ax=ax, cbar_kwargs={'label': '1 for 1 std of PC'},transform=ccrs.PlateCarree())#,projection=ccrs.SouthPolarStereo())
ax.set_title('EOF 1: '+str(int(varfrac.values[0]*100))+'% - Winter 2010-2020') 
ax.coastlines(resolution="10m",linewidth=2)
gl = ax.gridlines(linestyle='--',color='black',
              draw_labels=True)

####### PC1 #######
ax=fig.add_subplot(2,2,3)#, projection=ccrs.SouthPolarStereo()) 
pcs.sel(mode=0).sel(time='2010').plot(ax=ax)
ax.set_title('PC 1 - Winter 2010')
plt.grid()

# plot EOF2 and PC2

####### EOF2 #######
ax=fig.add_subplot(2,2,2, projection=ccrs.SouthPolarStereo()) 
eofs.sel(mode=1).plot(ax=ax, cbar_kwargs={'label': '1 for 1 std of PC'},transform=ccrs.PlateCarree())#,projection=ccrs.SouthPolarStereo())
ax.set_title('EOF 2: '+str(int(varfrac.values[1]*100))+'% - Winter 2010-2020') 
ax.coastlines(resolution="10m",linewidth=2)
gl = ax.gridlines(linestyle='--',color='black',
              draw_labels=True)

####### PC2 #######
ax=fig.add_subplot(2,2,4)#, projection=ccrs.SouthPolarStereo()) 
pcs.sel(mode=1).sel(time='2010').plot(ax=ax)
ax.set_title('PC 2 - Winter 2010')
plt.grid()



