#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:09:28 2022

@author: uvc35ld
"""


# In[Chargement bibliothèques]
import numpy as np
import matplotlib.pyplot as plt

import xarray as xr
# import pandas as pd

import MapDrawing as md
import ToolsPolynias as tp


# In[Chargement des données Polynies ]

    
##### polynie pour Wintembre :
polynie_CD_around_Win_2010_2020_0 = tp.dataYearsMonths('polynie',List_Months=range(5,10),List_Years=range(2010,2021),area='Cape Darnley Around')
polynie_CD_Win_2010_2020_0 = tp.sel_CapeDarnley(polynie_CD_around_Win_2010_2020_0)

polynie_CD_Win_2010_2020 = tp.selection_polynie_CD(polynie_CD_Win_2010_2020_0)

time_polynies_2010_2020 = polynie_CD_Win_2010_2020.time.values
surfaces_polynies_2010_2020 = tp.calculSurfacePolynies(polynie_CD_Win_2010_2020,area='MBP')

print('Fin de chargement de la surface de la polynie')

occurence_polynie_CD = tp.OccurencePolynieMoyenne_CD()

CI_moyen = tp.CIMoyenne()
CI_moyen_CD = tp.sel_CapeDarnley(CI_moyen)
print('Fin de chargement de CI')


# In[Chargement des données u10]

# u10_CD_Win_2010_2020 = tp.dataYearsMonths('u10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
u10_CD_around_Win_2010_2020 = tp.dataYearsMonths('u10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley Around')

print('Fin chargement de u10')

# In[Chargement des données v10]

# v10_CD_Win_2010_2020 = tp.dataYearsMonths('v10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
v10_CD_around_Win_2010_2020 = tp.dataYearsMonths('v10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley Around')

print('Fin chargement de v10')

# In[Chargement des données vo]

# vo_CD_Win_2010_2020 = tp.dataYearsMonths('vo',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
vo_CD_around_Win_2010_2020 = tp.dataYearsMonths('vo',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley Around').mean('level')

print('Fin chargement de vo')

# In[Chargement des données d]

# d_CD_Win_2010_2020 = tp.dataYearsMonths('d',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
d_CD_around_Win_2010_2020 = tp.dataYearsMonths('d',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley Around')

print('Fin chargement de d')

# # In[Chargement des données blh]

# # blh_CD_Win_2010_2020 = tp.dataYearsMonths('blh',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
# blh_CD_around_Win_2010_2020 = tp.dataYearsMonths('blh',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley Around')

# print('Fin chargement de blh')

# In[Seuil polynie fermée ]
seuil = 0.3

time_Surface_Polynies_Win_CD_seuil = []
time_vo_Surface_Polynies_Win_CD_seuil = []
time_Surface_Polynies_Win_CD_seuil_OPEN = []
compteur_min = 0
compteur_max = 0

Mean_surface_polynie_CD_2010_2020 = surfaces_polynies_2010_2020.mean('time')
time = surfaces_polynies_2010_2020.time.values

### Détermination des instants lorsque la polynie est fermée ou ouverte ###

for i in range(surfaces_polynies_2010_2020.shape[0]) :
    if (surfaces_polynies_2010_2020[i] < seuil*Mean_surface_polynie_CD_2010_2020):
        compteur_min += 1
        time_Surface_Polynies_Win_CD_seuil.append(time[i])
        
    elif (surfaces_polynies_2010_2020[i] > (1 - seuil)*Mean_surface_polynie_CD_2010_2020) :
        time_Surface_Polynies_Win_CD_seuil_OPEN.append(time[i])
        compteur_max += 1

array_time_Surface_Polynies_Win_CD_seuil = np.array(time_Surface_Polynies_Win_CD_seuil)
array_time_Surface_Polynies_Win_CD_seuil_OPEN = np.array(time_Surface_Polynies_Win_CD_seuil_OPEN)
        
print('Nombre de jours en dessous du seuil (~'+str(int(seuil*100))+'%): ',compteur_min,'/',surfaces_polynies_2010_2020.shape[0])
print('Nombre de jours en dessus du seuil (~'+str(int((1-seuil)*100))+'%): ',compteur_max,'/',surfaces_polynies_2010_2020.shape[0])

## Moyenne des surfaces lorsque la polynie est fermée
data_Surface_Polynies_Win_CD_seuil = surfaces_polynies_2010_2020.sel(time=time_Surface_Polynies_Win_CD_seuil)
data_Surface_Polynies_Win_CD_seuil_m = data_Surface_Polynies_Win_CD_seuil.mean('time')

data_Surface_Polynies_Win_CD_seuil_OPEN = surfaces_polynies_2010_2020.sel(time=time_Surface_Polynies_Win_CD_seuil_OPEN)
data_Surface_Polynies_Win_CD_seuil_m_OPEN = data_Surface_Polynies_Win_CD_seuil_OPEN.mean('time')

# In[Adaptation liste time u10]

time_Surface_Polynies_Win_CD_seuil_adapted = [str(x)[:10] for x in time_Surface_Polynies_Win_CD_seuil]
time_Surface_Polynies_Win_CD_seuil_adapted_OPEN = [str(x)[:10] for x in time_Surface_Polynies_Win_CD_seuil_OPEN]


# In[Tracé de u10]

## Adaptation liste time u10 ##

time_Surface_Polynies_Win_CD_seuil_adapted = [str(x)[:10] for x in time_Surface_Polynies_Win_CD_seuil]
time_Surface_Polynies_Win_CD_seuil_adapted_OPEN = [str(x)[:10] for x in time_Surface_Polynies_Win_CD_seuil_OPEN]

## Temps adaptés pour u10 ##
time_u10_CD_around_Win_2010_2020_fermee = [u10_CD_around_Win_2010_2020.sel(time=x).time.values[0] for x in time_Surface_Polynies_Win_CD_seuil_adapted]
time_u10_CD_around_Win_2010_2020_ouverte = [u10_CD_around_Win_2010_2020.sel(time=x).time.values[0] for x in time_Surface_Polynies_Win_CD_seuil_adapted_OPEN]

## Valeur de u10 pour les polynies fermée/ouverte ## 
data_u10_Win_CD_around_seuil_fermee = u10_CD_around_Win_2010_2020.sel(time=time_u10_CD_around_Win_2010_2020_fermee)
data_u10_Win_CD_around_seuil_ouverte = u10_CD_around_Win_2010_2020.sel(time=time_u10_CD_around_Win_2010_2020_ouverte)

## MOYENNE valeur de u10 pour les polynies fermée/ouverte ## 
u10_Win_CD_around_seuil_fermee_m = data_u10_Win_CD_around_seuil_fermee.mean('time')
u10_Win_CD_around_seuil_ouverte_m = data_u10_Win_CD_around_seuil_ouverte.mean('time')

## Valeur de u10 pour les polynies fermée/ouverte - delta ## 
u10_Win_CD_around_seuil_delta = u10_Win_CD_around_seuil_ouverte_m - u10_Win_CD_around_seuil_fermee_m

### Tracé Cape Darnley Around ###
md.plotMap_CapeDarnley_around(u10_Win_CD_around_seuil_fermee_m, dimension='$m.s^{-1}$',CI_contour=CI_moyen,level_CI_contour=np.linspace(0.2,0.9,8))
md.plotMap_CapeDarnley_around(u10_Win_CD_around_seuil_ouverte_m, dimension='$m.s^{-1}$',CI_contour=CI_moyen,level_CI_contour=np.linspace(0.2,0.9,8))
md.plotMap_CapeDarnley_around(u10_Win_CD_around_seuil_delta, dimension='$m.s^{-1}$',CI_contour=CI_moyen,level_CI_contour=np.linspace(0.2,0.9,8))

# md.plotMap_Anomalie(u10_Win_CD_around_seuil_ouverte_m,
#                     Liste_Vmin_Vmax=[-5,5],Liste_Vmin_Vmax_Delta=[-5,5],titre=['Polynie ouverte','Polynie fermee','Anomalie (= ouverte - fermee)'],
#                     CI_contour=CI_moyen,level_CI_contour=np.linspace(0.2,0.9,8))

### Tracé Cape Darnley ###
md.plotMap_CapeDarnley(u10_Win_CD_around_seuil_fermee_m, dimension='$m.s^{-1}$',CI_contour=occurence_polynie_CD,level_CI_contour=np.linspace(0.2,0.6,3))
md.plotMap_CapeDarnley(u10_Win_CD_around_seuil_ouverte_m, dimension='$m.s^{-1}$',CI_contour=occurence_polynie_CD,level_CI_contour=np.linspace(0.2,0.6,3))
md.plotMap_CapeDarnley(u10_Win_CD_around_seuil_delta, dimension='$m.s^{-1}$',CI_contour=occurence_polynie_CD,level_CI_contour=np.linspace(0.2,0.6,3))

md.plotMap_Anomalie(tp.sel_CapeDarnley(u10_Win_CD_around_seuil_ouverte_m), tp.sel_CapeDarnley(u10_Win_CD_around_seuil_fermee_m),
                    Liste_Vmin_Vmax=[-15,15],Liste_Vmin_Vmax_Delta=[-6,6],titre=['Polynie ouverte ('+str(compteur_min)+' j. ouverts / '+str(surfaces_polynies_2010_2020.shape[0])+' j.)',
                                                                                 'Polynie fermee ('+str(compteur_max)+' j. fermés / '+str(surfaces_polynies_2010_2020.shape[0])+' j.)',
                                                                                 'Anomalie (= ouverte - fermee)'],
                    CI_contour=occurence_polynie_CD,level_CI_contour=np.linspace(0.2,0.6,3)#,path_save='/gpfswork/rech/omr/uvc35ld/stage/IMAGES/Anomalies_CD/maps_u10_anomalie_OP_seuil_30_Sep_CD_2010-2020.png'
                    )


# In[Tracé de v10]

## Adaptation liste time v10 ##

time_Surface_Polynies_Win_CD_seuil_adapted = [str(x)[:10] for x in time_Surface_Polynies_Win_CD_seuil]
time_Surface_Polynies_Win_CD_seuil_adapted_OPEN = [str(x)[:10] for x in time_Surface_Polynies_Win_CD_seuil_OPEN]

## Temps adaptés pour v10 ##
time_v10_CD_around_Win_2010_2020_fermee = [v10_CD_around_Win_2010_2020.sel(time=x).time.values[0] for x in time_Surface_Polynies_Win_CD_seuil_adapted]
time_v10_CD_around_Win_2010_2020_ouverte = [v10_CD_around_Win_2010_2020.sel(time=x).time.values[0] for x in time_Surface_Polynies_Win_CD_seuil_adapted_OPEN]

## Valeur de v10 pour les polynies fermée/ouverte ## 
data_v10_Win_CD_around_seuil_fermee = v10_CD_around_Win_2010_2020.sel(time=time_v10_CD_around_Win_2010_2020_fermee)
data_v10_Win_CD_around_seuil_ouverte = v10_CD_around_Win_2010_2020.sel(time=time_v10_CD_around_Win_2010_2020_ouverte)

## MOYENNE valeur de v10 pour les polynies fermée/ouverte ## 
v10_Win_CD_around_seuil_fermee_m = data_v10_Win_CD_around_seuil_fermee.mean('time')
v10_Win_CD_around_seuil_ouverte_m = data_v10_Win_CD_around_seuil_ouverte.mean('time')

## Valeur de v10 pour les polynies fermée/ouverte - delta ## 
v10_Win_CD_around_seuil_delta = v10_Win_CD_around_seuil_ouverte_m - v10_Win_CD_around_seuil_fermee_m

### Tracé Cape Darnley Around ###
md.plotMap_CapeDarnley_around(v10_Win_CD_around_seuil_fermee_m, dimension='$m.s^{-1}$',Liste_Vmin_Vmax=[-15,15],CI_contour=CI_moyen,level_CI_contour=np.linspace(0.2,0.9,8))
md.plotMap_CapeDarnley_around(v10_Win_CD_around_seuil_ouverte_m, dimension='$m.s^{-1}$',Liste_Vmin_Vmax=[-15,15],CI_contour=CI_moyen,level_CI_contour=np.linspace(0.2,0.9,8))
md.plotMap_CapeDarnley_around(v10_Win_CD_around_seuil_delta, dimension='$m.s^{-1}$',Liste_Vmin_Vmax=[-5,5],CI_contour=CI_moyen,level_CI_contour=np.linspace(0.2,0.9,8))

### Tracé Cape Darnley ###
md.plotMap_CapeDarnley(v10_Win_CD_around_seuil_fermee_m, dimension='$m.s^{-1}$',Liste_Vmin_Vmax=[-15,15],CI_contour=occurence_polynie_CD,level_CI_contour=np.linspace(0.2,0.6,3))
md.plotMap_CapeDarnley(v10_Win_CD_around_seuil_ouverte_m, dimension='$m.s^{-1}$',Liste_Vmin_Vmax=[-15,15],CI_contour=occurence_polynie_CD,level_CI_contour=np.linspace(0.2,0.6,3))
md.plotMap_CapeDarnley(v10_Win_CD_around_seuil_delta, dimension='$m.s^{-1}$',Liste_Vmin_Vmax=[-6,6],CI_contour=occurence_polynie_CD,level_CI_contour=np.linspace(0.2,0.6,3))

md.plotMap_Anomalie(tp.sel_CapeDarnley(v10_Win_CD_around_seuil_ouverte_m), tp.sel_CapeDarnley(v10_Win_CD_around_seuil_fermee_m),
                    Liste_Vmin_Vmax=[-15,15],Liste_Vmin_Vmax_Delta=[-6,6],titre=['Polynie ouverte ('+str(compteur_min)+' j. ouverts / '+str(surfaces_polynies_2010_2020.shape[0])+' j.)',
                                                                                 'Polynie fermee ('+str(compteur_max)+' j. fermés / '+str(surfaces_polynies_2010_2020.shape[0])+' j.)',
                                                                                 'Anomalie (= ouverte - fermee)'],
                    CI_contour=occurence_polynie_CD,level_CI_contour=np.linspace(0.2,0.6,3)#,path_save='/gpfswork/rech/omr/uvc35ld/stage/IMAGES/Anomalies_CD/maps_v10_anomalie_OP_seuil_30_Sep_CD_2010-2020.png'
                    )








# In[Tracé de U10]




## Valeur de U10 pour les polynies fermée/ouverte ## 
data_U10_Win_CD_around_seuil_fermee = np.sqrt(data_u10_Win_CD_around_seuil_fermee*data_u10_Win_CD_around_seuil_fermee + data_v10_Win_CD_around_seuil_fermee*data_v10_Win_CD_around_seuil_fermee)
data_U10_Win_CD_around_seuil_ouverte = np.sqrt(data_u10_Win_CD_around_seuil_ouverte*data_u10_Win_CD_around_seuil_ouverte + data_v10_Win_CD_around_seuil_ouverte * data_v10_Win_CD_around_seuil_ouverte)

## MOYENNE valeur de U10 pour les polynies fermée/ouverte ## 
U10_Win_CD_around_seuil_fermee_m = data_U10_Win_CD_around_seuil_fermee.mean('time')
U10_Win_CD_around_seuil_ouverte_m = data_U10_Win_CD_around_seuil_ouverte.mean('time')

## Valeur de U10 pour les polynies fermée/ouverte - delta ## 
U10_Win_CD_around_seuil_delta = U10_Win_CD_around_seuil_ouverte_m - U10_Win_CD_around_seuil_fermee_m

### Tracé Cape Darnley Around ###
md.plotMap_CapeDarnley_around(U10_Win_CD_around_seuil_fermee_m, dimension='$m.s^{-1}$',Liste_Vmin_Vmax=[-15,15],CI_contour=CI_moyen,level_CI_contour=np.linspace(0.2,0.9,8))
md.plotMap_CapeDarnley_around(U10_Win_CD_around_seuil_ouverte_m, dimension='$m.s^{-1}$',Liste_Vmin_Vmax=[-15,15],CI_contour=CI_moyen,level_CI_contour=np.linspace(0.2,0.9,8))
md.plotMap_CapeDarnley_around(U10_Win_CD_around_seuil_delta, dimension='$m.s^{-1}$',Liste_Vmin_Vmax=[-5,5],CI_contour=CI_moyen,level_CI_contour=np.linspace(0.2,0.9,8))

### Tracé Cape Darnley ###
md.plotMap_CapeDarnley(U10_Win_CD_around_seuil_fermee_m, dimension='$m.s^{-1}$',Liste_Vmin_Vmax=[0,15],CI_contour=CI_moyen_CD,level_CI_contour=np.linspace(0.2,0.6,3))
md.plotMap_CapeDarnley(U10_Win_CD_around_seuil_ouverte_m, dimension='$m.s^{-1}$',Liste_Vmin_Vmax=[0,15],CI_contour=CI_moyen_CD,level_CI_contour=np.linspace(0.2,0.6,3))
md.plotMap_CapeDarnley(U10_Win_CD_around_seuil_delta, dimension='$m.s^{-1}$',Liste_Vmin_Vmax=[-4,4],CI_contour=CI_moyen_CD,level_CI_contour=np.linspace(0.2,0.6,3))

md.plotMap_Anomalie(tp.sel_CapeDarnley(U10_Win_CD_around_seuil_ouverte_m), tp.sel_CapeDarnley(U10_Win_CD_around_seuil_fermee_m),
                    Liste_Vmin_Vmax=[0,15],Liste_Vmin_Vmax_Delta=[-3,3],titre=['Polynie ouverte ('+str(compteur_min)+' j. ouverts / '+str(surfaces_polynies_2010_2020.shape[0])+' j.)',
                                                                                 'Polynie fermee ('+str(compteur_max)+' j. fermés / '+str(surfaces_polynies_2010_2020.shape[0])+' j.)',
                                                                                 'Anomalie (= ouverte - fermee)'],
                    CI_contour=occurence_polynie_CD,level_CI_contour=np.linspace(0.2,0.6,3)#,path_save='/gpfswork/rech/omr/uvc35ld/stage/IMAGES/Anomalies_CD/maps_v10_anomalie_OP_seuil_30_Sep_CD_2010-2020.png'
                    )








# In[Tracé de vo]

## Adaptation liste time vo ##

time_Surface_Polynies_Win_CD_seuil_adapted = [str(x)[:10] for x in time_Surface_Polynies_Win_CD_seuil]
time_Surface_Polynies_Win_CD_seuil_adapted_OPEN = [str(x)[:10] for x in time_Surface_Polynies_Win_CD_seuil_OPEN]

## Temps adaptés pour vo ##
time_vo_CD_around_Win_2010_2020_fermee = [vo_CD_around_Win_2010_2020.sel(time=x).time.values[0] for x in time_Surface_Polynies_Win_CD_seuil_adapted]
time_vo_CD_around_Win_2010_2020_ouverte = [vo_CD_around_Win_2010_2020.sel(time=x).time.values[0] for x in time_Surface_Polynies_Win_CD_seuil_adapted_OPEN]

## Valeur de vo pour les polynies fermée/ouverte ## 
data_vo_Win_CD_around_seuil_fermee = vo_CD_around_Win_2010_2020.sel(time=time_vo_CD_around_Win_2010_2020_fermee)
data_vo_Win_CD_around_seuil_ouverte = vo_CD_around_Win_2010_2020.sel(time=time_vo_CD_around_Win_2010_2020_ouverte)

## MOYENNE valeur de vo pour les polynies fermée/ouverte ## 
vo_Win_CD_around_seuil_fermee_m = data_vo_Win_CD_around_seuil_fermee.mean('time')
vo_Win_CD_around_seuil_ouverte_m = data_vo_Win_CD_around_seuil_ouverte.mean('time')

## Valeur de vo pour les polynies fermée/ouverte - delta ## 
vo_Win_CD_around_seuil_delta = vo_Win_CD_around_seuil_ouverte_m - vo_Win_CD_around_seuil_fermee_m

### Tracé Cape Darnley Around ###
md.plotMap_CapeDarnley_around(vo_Win_CD_around_seuil_fermee_m, dimension='$s^{-1}$',Liste_Vmin_Vmax=[-3e-4,3e-4],CI_contour=CI_moyen,level_CI_contour=np.linspace(0.2,0.6,8))
md.plotMap_CapeDarnley_around(vo_Win_CD_around_seuil_ouverte_m, dimension='$s^{-1}$',Liste_Vmin_Vmax=[-3e-4,3e-4],CI_contour=CI_moyen,level_CI_contour=np.linspace(0.2,0.9,8))
md.plotMap_CapeDarnley_around(vo_Win_CD_around_seuil_delta, dimension='$s^{-1}$',Liste_Vmin_Vmax=[-8e-5,8e-5],CI_contour=CI_moyen,level_CI_contour=np.linspace(0.2,0.9,8))

# ,Liste_Vmin_Vmax=[-15,15]

### Tracé Cape Darnley ###
md.plotMap_CapeDarnley(vo_Win_CD_around_seuil_fermee_m, dimension='$s^{-1}$',Liste_Vmin_Vmax=[-3e-4,3e-4],CI_contour=CI_moyen_CD,level_CI_contour=np.linspace(0.2,0.9,15))
md.plotMap_CapeDarnley(vo_Win_CD_around_seuil_ouverte_m, dimension='$s^{-1}$',Liste_Vmin_Vmax=[-3e-4,3e-4],CI_contour=CI_moyen_CD,level_CI_contour=np.linspace(0.2,0.9,15))
md.plotMap_CapeDarnley(vo_Win_CD_around_seuil_delta, dimension='$s^{-1}$',Liste_Vmin_Vmax=[-8e-5,8e-5],CI_contour=CI_moyen_CD,level_CI_contour=np.linspace(0.2,0.9,15))

md.plotMap_Anomalie(tp.sel_CapeDarnley(vo_Win_CD_around_seuil_ouverte_m), tp.sel_CapeDarnley(vo_Win_CD_around_seuil_fermee_m),dimension=r"$s^{-1}$",
                    Liste_Vmin_Vmax=[-3e-4,3e-4],Liste_Vmin_Vmax_Delta=[-8e-5,8e-5],titre=['Polynie ouverte ('+str(compteur_min)+' j. ouverts / '+str(surfaces_polynies_2010_2020.shape[0])+' j.)',
                                                                                 'Polynie fermee ('+str(compteur_max)+' j. fermés / '+str(surfaces_polynies_2010_2020.shape[0])+' j.)',
                                                                                 'Anomalie (= ouverte - fermee)'],
                    CI_contour=occurence_polynie_CD,level_CI_contour=np.linspace(0.2,0.6,3)#,path_save='/gpfswork/rech/omr/uvc35ld/stage/IMAGES/Anomalies_CD/maps_v10_anomalie_OP_seuil_30_Sep_CD_2010-2020.png'
                    )
                            
# In[Tracé de d]

## Adaptation liste time d ##

time_Surface_Polynies_Win_CD_seuil_adapted = [str(x)[:10] for x in time_Surface_Polynies_Win_CD_seuil]
time_Surface_Polynies_Win_CD_seuil_adapted_OPEN = [str(x)[:10] for x in time_Surface_Polynies_Win_CD_seuil_OPEN]

## Temps adaptés pour d ##
time_d_CD_around_Win_2010_2020_fermee = [d_CD_around_Win_2010_2020.sel(time=x).time.values[0] for x in time_Surface_Polynies_Win_CD_seuil_adapted]
time_d_CD_around_Win_2010_2020_ouverte = [d_CD_around_Win_2010_2020.sel(time=x).time.values[0] for x in time_Surface_Polynies_Win_CD_seuil_adapted_OPEN]

## Valeur de d pour les polynies fermée/ouverte ## 
data_d_Win_CD_around_seuil_fermee = d_CD_around_Win_2010_2020.sel(time=time_d_CD_around_Win_2010_2020_fermee)
data_d_Win_CD_around_seuil_ouverte = d_CD_around_Win_2010_2020.sel(time=time_d_CD_around_Win_2010_2020_ouverte)

## MOYENNE valeur de d pour les polynies fermée/ouverte ## 
d_Win_CD_around_seuil_fermee_m = data_d_Win_CD_around_seuil_fermee.mean('time')
d_Win_CD_around_seuil_ouverte_m = data_d_Win_CD_around_seuil_ouverte.mean('time')

## Valeur de d pour les polynies fermée/ouverte - delta ## 
d_Win_CD_around_seuil_delta = d_Win_CD_around_seuil_ouverte_m - d_Win_CD_around_seuil_fermee_m

### Tracé Cape Darnley Around ###
md.plotMap_CapeDarnley_around(d_Win_CD_around_seuil_fermee_m, dimension='$s^{-1}$',Liste_Vmin_Vmax=[-3e-4,3e-4],CI_contour=CI_moyen,level_CI_contour=np.linspace(0.2,0.9,8))
md.plotMap_CapeDarnley_around(d_Win_CD_around_seuil_ouverte_m, dimension='$s^{-1}$',Liste_Vmin_Vmax=[-3e-4,3e-4],CI_contour=CI_moyen,level_CI_contour=np.linspace(0.2,0.9,8))
md.plotMap_CapeDarnley_around(d_Win_CD_around_seuil_delta, dimension='$s^{-1}$',Liste_Vmin_Vmax=[-8e-5,8e-5],CI_contour=CI_moyen,level_CI_contour=np.linspace(0.2,0.9,8))

# ,Liste_Vmin_Vmax=[-15,15]

### Tracé Cape Darnley ###
md.plotMap_CapeDarnley(d_Win_CD_around_seuil_fermee_m, dimension='$s^{-1}$',Liste_Vmin_Vmax=[-3e-4,3e-4],CI_contour=CI_moyen_CD,level_CI_contour=np.linspace(0.2,0.9,15))
md.plotMap_CapeDarnley(d_Win_CD_around_seuil_ouverte_m, dimension='$s^{-1}$',Liste_Vmin_Vmax=[-3e-4,3e-4],CI_contour=CI_moyen_CD,level_CI_contour=np.linspace(0.2,0.9,15))
md.plotMap_CapeDarnley(d_Win_CD_around_seuil_delta, dimension='$s^{-1}$',Liste_Vmin_Vmax=[-8e-5,8e-5],CI_contour=CI_moyen_CD,level_CI_contour=np.linspace(0.2,0.9,15))

md.plotMap_Anomalie(tp.sel_CapeDarnley(d_Win_CD_around_seuil_ouverte_m), tp.sel_CapeDarnley(d_Win_CD_around_seuil_fermee_m),dimension=r"$s^{-1}$",
                    Liste_Vmin_Vmax=[-3e-4,3e-4],Liste_Vmin_Vmax_Delta=[-8e-5,8e-5],titre=['Polynie ouverte ('+str(compteur_min)+' j. ouverts / '+str(surfaces_polynies_2010_2020.shape[0])+' j.)',
                                                                                 'Polynie fermee ('+str(compteur_max)+' j. fermés / '+str(surfaces_polynies_2010_2020.shape[0])+' j.)',
                                                                                 'Anomalie (= ouverte - fermee)'],
                    CI_contour=occurence_polynie_CD,level_CI_contour=np.linspace(0.2,0.6,3)#,path_save='/gpfswork/rech/omr/uvc35ld/stage/IMAGES/Anomalies_CD/maps_v10_anomalie_OP_seuil_30_Sep_CD_2010-2020.png'
                    )

# In[Tracé de blh]

# ## Adaptation liste time blh ##

# time_Surface_Polynies_Win_CD_seuil_adapted = [str(x)[:10] for x in time_Surface_Polynies_Win_CD_seuil]
# time_Surface_Polynies_Win_CD_seuil_adapted_OPEN = [str(x)[:10] for x in time_Surface_Polynies_Win_CD_seuil_OPEN]

# ## Temps adaptés pour blh ##
# time_blh_CD_around_Win_2010_2020_fermee = [blh_CD_around_Win_2010_2020.sel(time=x).time.values[0] for x in time_Surface_Polynies_Win_CD_seuil_adapted]
# time_blh_CD_around_Win_2010_2020_ouverte = [blh_CD_around_Win_2010_2020.sel(time=x).time.values[0] for x in time_Surface_Polynies_Win_CD_seuil_adapted_OPEN]

# ## Valeur de blh pour les polynies fermée/ouverte ## 
# data_blh_Win_CD_around_seuil_fermee = blh_CD_around_Win_2010_2020.sel(time=time_blh_CD_around_Win_2010_2020_fermee)
# data_blh_Win_CD_around_seuil_ouverte = blh_CD_around_Win_2010_2020.sel(time=time_blh_CD_around_Win_2010_2020_ouverte)

# ## MOYENNE valeur de blh pour les polynies fermée/ouverte ## 
# blh_Win_CD_around_seuil_fermee_m = data_blh_Win_CD_around_seuil_fermee.mean('time')
# blh_Win_CD_around_seuil_ouverte_m = data_blh_Win_CD_around_seuil_ouverte.mean('time')

# ## Valeur de blh pour les polynies fermée/ouverte - delta ## 
# blh_Win_CD_around_seuil_delta = blh_Win_CD_around_seuil_ouverte_m - blh_Win_CD_around_seuil_fermee_m

# ### Tracé Cape Darnley Around ###
# md.plotMap_CapeDarnley_around(blh_Win_CD_around_seuil_fermee_m, dimension='$s^{-1}$',Liste_Vmin_Vmax=[-3e-4,3e-4],CI_contour=CI_moyen,level_CI_contour=np.linspace(0.2,0.9,8))
# md.plotMap_CapeDarnley_around(blh_Win_CD_around_seuil_ouverte_m, dimension='$s^{-1}$',Liste_Vmin_Vmax=[-3e-4,3e-4],CI_contour=CI_moyen,level_CI_contour=np.linspace(0.2,0.9,8))
# md.plotMap_CapeDarnley_around(blh_Win_CD_around_seuil_delta, dimension='$s^{-1}$',Liste_Vmin_Vmax=[-8e-5,8e-5],CI_contour=CI_moyen,level_CI_contour=np.linspace(0.2,0.9,8))

# # ,Liste_Vmin_Vmax=[-15,15]

# ### Tracé Cape Darnley ###
# md.plotMap_CapeDarnley(blh_Win_CD_around_seuil_fermee_m, dimension='$s^{-1}$',Liste_Vmin_Vmax=[-3e-4,3e-4],CI_contour=CI_moyen_CD,level_CI_contour=np.linspace(0.2,0.9,15))
# md.plotMap_CapeDarnley(blh_Win_CD_around_seuil_ouverte_m, dimension='$s^{-1}$',Liste_Vmin_Vmax=[-3e-4,3e-4],CI_contour=CI_moyen_CD,level_CI_contour=np.linspace(0.2,0.9,15))
# md.plotMap_CapeDarnley(blh_Win_CD_around_seuil_delta, dimension='$s^{-1}$',Liste_Vmin_Vmax=[-8e-5,8e-5],CI_contour=CI_moyen_CD,level_CI_contour=np.linspace(0.2,0.9,15))

    