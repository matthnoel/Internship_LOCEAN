#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 10:42:21 2022

@author: uvc35ld
"""

# In[Chargement bibliothèques]
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
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

# In[Chargement des données]

### Calcul de Surface Polynie ###
polynie_CD_Win_2010_2020 = tp.dataYearsMonths('polynie',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley Around')
surface_polynie_CD_Win_2010_2020 = tp.calculSurfacePolynies(polynie_CD_Win_2010_2020,area='CDP')

### Importation de données ###
u10_CD_Win_2010_2020 = tp.dataYearsMonths('u10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley Around')
v10_CD_Win_2010_2020 = tp.dataYearsMonths('v10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley Around')

U10_CD_Win_2010_2020 = np.sqrt(u10_CD_Win_2010_2020*u10_CD_Win_2010_2020+v10_CD_Win_2010_2020*v10_CD_Win_2010_2020)

vo_CD_Win_2010_2020 = tp.dataYearsMonths('vo',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley Around').mean('level')
d_CD_Win_2010_2020 = tp.dataYearsMonths('d',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley Around').mean('level')

# blh_CD_Win_2010_2020 = tp.dataYearsMonths('blh',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley Around')
e_CD_Win_2010_2020 = tp.dataYearsMonths('e',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley Around')

occurence_polynie_CD = tp.OccurencePolynieMoyenne_CD()

# In[Calcul de la Corrélation/Regression]

##### Vitesse a 10m #####
print("Calcul regression Surface Polynie = F(v10)")
slopes_v10_S_CD_aro,intercept_v10_S_CD_aro,r_v10_S_CD_aro = tp.regressionLineaireDataSurfacepolynie(v10_CD_Win_2010_2020, 
                                                                                                    surface_polynie_CD_Win_2010_2020,
                                                                                                    titre='Correlation v10 et Surface de la polynie de Cape Darnley - Winter 2010-2020')
print("Regression Surface Polynie = F(u10)")
slopes_u10_S_CD_aro,intercept_u10_S_CD_aro,r_u10_S_CD_aro  = tp.regressionLineaireDataSurfacepolynie(u10_CD_Win_2010_2020, 
                                                                                                     surface_polynie_CD_Win_2010_2020,
                                                                                                     titre='Correlation u10 et Surface de la polynie de Cape Darnley - Winter 2010-2020')

print("Regression Surface Polynie = F(U10)")
slopes_U10_S_CD_aro,intercept_U10_S_CD_aro,r_U10_S_CD_aro  = tp.regressionLineaireDataSurfacepolynie(U10_CD_Win_2010_2020, 
                                                                                                     surface_polynie_CD_Win_2010_2020,
                                                                                                     titre='Correlation U10 et Surface de la polynie de Cape Darnley - Winter 2010-2020')


##### Vorticité #####
print("Regression Surface Polynie = F(vo)")
slopes_vo_S_CD_aro,intercept_vo_S_CD_aro,r_vo_S_CD_aro  = tp.regressionLineaireDataSurfacepolynie(vo_CD_Win_2010_2020, 
                                                                                                     surface_polynie_CD_Win_2010_2020,
                                                                                                     titre='Correlation vo et Surface de la polynie de Cape Darnley - Winter 2010-2020')
##### Divergence #####
print("Regression Surface Polynie = F(d)")
slopes_d_S_CD_aro,intercept_d_S_CD_aro,r_d_S_CD_aro  = tp.regressionLineaireDataSurfacepolynie(d_CD_Win_2010_2020, 
                                                                                                      surface_polynie_CD_Win_2010_2020,
                                                                                                      titre='Correlation divergenece et Surface de la polynie de Cape Darnley - Winter 2010-2020')
 
# ##### Hauteur de la couche limite #####
# print("Regression Surface Polynie = F(blh)")
# slopes_blh_S_CD_aro,intercept_blh_S_CD_aro,r_blh_S_CD_aro  = tp.regressionLineaireDataSurfacepolynie(blh_CD_Win_2010_2020, 
#                                                                                                      surface_polynie_CD_Win_2010_2020,
                                                                                                     # titre='Correlation blh et Surface de la polynie de Cape Darnley - Winter 2010-2020')

##### Evaporation #####
print("Regression Surface Polynie = F(e)")
slopes_e_S_CD_aro,intercept_e_S_CD_aro,r_e_S_CD_aro  = tp.regressionLineaireDataSurfacepolynie(e_CD_Win_2010_2020, 
                                                                                               surface_polynie_CD_Win_2010_2020,
                                                                                               titre='Correlation e et Surface de la polynie de Cape Darnley - Winter 2010-2020')

# In[Tracé des regressions CD Around]


md.plotMap_CapeDarnley_around(slopes_v10_S_CD_aro,dimension=r"$m.s^{-1}$ for 1 std of the Cape Darnley polynya surface",
                              stations=True)#,titre='Regression of v10 against the polynya surface \n Winter 2010-2020')

md.plotMap_CapeDarnley_around(slopes_u10_S_CD_aro,dimension=r"$m.s^{-1}$ for 1 std of the Cape Darnley polynya surface",stations=True)#,titre='Regression of u10 against the polynya surface \n Winter 2010-2020')

md.plotMap_CapeDarnley_around(slopes_vo_S_CD_aro,dimension=r"$s^{-1}$ for 1 std of the Cape Darnley polynya surface",stations=True)#,titre='Regression of v10 against the polynya surface \n Winter 2010-2020')

md.plotMap_CapeDarnley_around(slopes_d_S_CD_aro,dimension=r"$s^{-1}$ for 1 std of the Cape Darnley polynya surface",stations=True)#,titre='Regression of u10 against the polynya surface \n Winter 2010-2020')

# md.plotMap_CapeDarnley_around(slopes_blh_S_CD_aro,dimension=r"$m.s^{-1}$ for 1 std of the Cape Darnley polynya surface",stations=True)#,titre='Regression of v10 against the polynya surface \n Winter 2010-2020')

md.plotMap_CapeDarnley_around(slopes_e_S_CD_aro,dimension=r"$kg.m^{-2}.s^{-1}$ for 1 std of the Cape Darnley polynya surface",stations=True)#,titre='Regression of u10 against the polynya surface \n Winter 2010-2020')

# In[Tracé des regressions CD]

md.plotMap_Regression(tp.sel_CapeDarnley(slopes_u10_S_CD_aro),tp.sel_CapeDarnley(r_u10_S_CD_aro),Liste_Rmin_Rmax=[-1.3,1.3],Liste_Corrmin_Corrmax=[-0.3,0.3],
                      dimension=r"$m.s^{-1}/km^{2}$",titre=['Régression','Corrélation'],
                      CI_contour=occurence_polynie_CD,level_CI_contour=np.linspace(0.2,0.6,3))#,titre='Regression of u10 against the polynya surface \n Winter 2010-2020')

md.plotMap_Regression(tp.sel_CapeDarnley(slopes_v10_S_CD_aro),tp.sel_CapeDarnley(r_v10_S_CD_aro),Liste_Rmin_Rmax=[-1.3,1.3],Liste_Corrmin_Corrmax=[-0.3,0.3],
                      dimension=r"$m.s^{-1}/km^{2}$",titre=['Régression','Corrélation'],
                      CI_contour=occurence_polynie_CD,level_CI_contour=np.linspace(0.2,0.6,3))#,titre='Regression of v10 against the polynya surface \n Winter 2010-2020')

#slopes_U10_S_CD_aro = np.sqrt(slopes_u10_S_CD_aro*slopes_u10_S_CD_aro + slopes_v10_S_CD_aro*slopes_v10_S_CD_aro)
md.plotMap_Regression(tp.sel_CapeDarnley(slopes_U10_S_CD_aro),tp.sel_CapeDarnley(r_U10_S_CD_aro),Liste_Rmin_Rmax=[-1.3,1.3],Liste_Corrmin_Corrmax=[-0.3,0.3],
                      dimension=r"$m.s^{-1}/km^{2}$",titre=['Régression','Corrélation'],
                      CI_contour=occurence_polynie_CD,level_CI_contour=np.linspace(0.2,0.6,3))#,titre='Regression of U10 against the polynya surface \n Winter 2010-2020')


md.plotMap_Regression(tp.sel_CapeDarnley(slopes_vo_S_CD_aro),tp.sel_CapeDarnley(r_vo_S_CD_aro),Liste_Rmin_Rmax=[-1.5*1e-5,1.5*1e-5],Liste_Corrmin_Corrmax=[-0.25,0.25],
                      dimension=r"$s^{-1}/km^{2}$",titre=['Régression','Corrélation'],
                      CI_contour=occurence_polynie_CD,level_CI_contour=np.linspace(0.2,0.6,3))#,titre='Regression of v10 against the polynya surface \n Winter 2010-2020')

md.plotMap_Regression(tp.sel_CapeDarnley(slopes_d_S_CD_aro),tp.sel_CapeDarnley(r_d_S_CD_aro),#dimension=r"$s^{-1}$ for 1 std of the Cape Darnley polynya surface",
                      dimension=r"$s^{-1}/km^{2}$",titre=['Régression','Corrélation'],
                      CI_contour=occurence_polynie_CD,level_CI_contour=np.linspace(0.2,0.6,3))#,titre='Regression of u10 against the polynya surface \n Winter 2010-2020')

# md.plotMap_Regression(tp.sel_CapeDarnley(slopes_blh_S_CD_aro),tp.sel_CapeDarnley(r_blh_CD_aro),dimension=r"$m.s^{-1}$ for 1 std of the Cape Darnley polynya surface",
#                        CI_contour=occurence_polynie_CD,level_CI_contour=np.linspace(0.2,0.6,3))#,titre='Regression of v10 against the polynya surface \n Winter 2010-2020')

md.plotMap_Regression(tp.sel_CapeDarnley(slopes_e_S_CD_aro),tp.sel_CapeDarnley(r_e_S_CD_aro),Liste_Rmin_Rmax=[-3*1e-9,3*1e-9],Liste_Corrmin_Corrmax=[-0.3,0.3],
                      dimension=r"$kg.m^{-2}.s^{-1}/km^{2}$",titre=['Régression','Corrélation'],
                      CI_contour=occurence_polynie_CD,level_CI_contour=np.linspace(0.2,0.6,3))#,titre='Regression of u10 against the polynya surface \n Winter 2010-2020')
