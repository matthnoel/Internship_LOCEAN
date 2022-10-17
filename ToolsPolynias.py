#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 09:44:37 2022

@author: Matthias NOËL
"""

# In[Chargement bibliothèques]
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
from scipy import stats
import datetime
from math import radians,degrees
import math as m

from skimage.segmentation import flood_fill
from MapDrawing import sel_CapeDarnley,plotMap_Ant,plotMap_CapeDarnley_around,sel_CapeDarnley_around


# In[Chargement de données annuelles]


# def dataYears(data,List_Years=range(2010,2013),path=None,area='Antarctica',reanalysis='ERA'):
    
#     if  reanalysis == 'ERA':
#         if data == 'polynie' :
#             path = '/gpfswork/rech/omr/uvc35ld/stage/Polynyas/ERA5/Polynyas_70%/DAILY/'+str(data)+'s_era5_ant_y'
        
#         else :
#             path = '/gpfsstore/rech/omr/romr008/DATA/ERA5/DAILY/'+str(data)+'_era5_ant_y'
            
#     if reanalysis == 'MERRA':
        
#         if data == 'polynie' :
#             path = '/gpfswork/rech/omr/uvc35ld/stage/Polynyas/MERRA2/Polynyas_70%/DAILY/'+str(data)+'_merra2_ant_y'
        
#         else :
#             path = '/gpfsstore/rech/omr/romr008/DATA/MERRA2/PROCESSED/DAILY/'+str(data)+'_merra2_ant_y'
            
#     if reanalysis == 'JRA':
        
#         if data == 'polynie' :
#             path = '/gpfswork/rech/omr/uvc35ld/stage/Polynyas/JRA55/Polynyas_70%/DAILY/'+str(data)+'_jra55_ant_y'
        
#         else :
#             path = '/gpfsstore/rech/omr/romr008/DATA/JRA55/PROCESSED/DAILY/'+str(data)+'_jra55_ant_y'
        
#     ##### data pour toutes les années (2010 --> 2020):
#     data_an = []
#     times_an = []
#     times = 0
    
#     if reanalysis == 'JRA':
#         for x in List_Years:
#             if area=='Antarctica' :
#                 data_intermediaire = xr.open_dataset(path+ str(x) +'.nc')[data]
#             if area == 'Cape Darnley':
#                 data_intermediaire = sel_CapeDarnley(xr.open_dataset(path+ str(x) +'.nc')[data])
#             if area == 'Cape Darnley Around':
#                 data_intermediaire = sel_CapeDarnley_around(xr.open_dataset(path+ str(x) +'.nc')[data])
#             data_an.append(data_intermediaire)
#             times_an.append(data_intermediaire.time_counter.values)
#             times = times + times_an[-1].shape[0]
            
#     else :
    
#         for x in List_Years:
#             if area=='Antarctica' :
#                 data_intermediaire = xr.open_dataset(path+ str(x) +'.nc')[data]
#             if area == 'Cape Darnley':
#                 data_intermediaire = sel_CapeDarnley(xr.open_dataset(path+ str(x) +'.nc')[data])
#             if area == 'Cape Darnley Around':
#                 data_intermediaire = sel_CapeDarnley_around(xr.open_dataset(path+ str(x) +'.nc')[data])
#             data_an.append(data_intermediaire)
#             times_an.append(data_intermediaire.time.values)
#             times = times + times_an[-1].shape[0]
        
#     data_2010_2020 = xr.merge(data_an)
   
#     return data_2010_2020


# In[Selection Polynies distinctes]

# =============================================================================
# FONCTION : selection_polynie_CD :
#
#   Selection the correct polynyas (which have a separation between Cape Darnley Polynya and Mackenzie Bay Polynya.
#
#  Input :
#    - data_polynie (xarray - (latitude, longitude, time) or (latitude,longitude)) : data of polynyas
#    - tol_polynie (str) : if the loaded data is a polynya --> precision of the tolerabce threshold (by default 70%) must be in ['50%','60%','70%','80%','90%']
#
#  Output :
#     - data (xarray) : data of correct polynyas in Cape Darnley.
#   
# =============================================================================

def selection_polynie_CD(data_polynie,tol_polynie='70%'):
    
    # Selection of polynya in the area 
    data_polynie_CD = sel_CapeDarnley(data_polynie)
    
    # Calculation of the area of the Cape Darnley Polynya
    surfaces_polynies = calculSurfacePolynies(data_polynie_CD)
    
    # Pre-selection of the too large polynyas 
    date_limite = []
    if tol_polynie == '50%' :
        Limite_Surface = 45000 #450000
    
    if tol_polynie == '60%' :
        Limite_Surface = 45000 #450000
    
    if tol_polynie == '70%' :
        Limite_Surface = 45000 #450000
            
    if tol_polynie == '80%' :
        Limite_Surface = 45000 #450000
    
    if tol_polynie == '90%' :
        Limite_Surface = 45000 #450000
        
    for x in data_polynie.time.values :
        if surfaces_polynies.sel(time=x) > Limite_Surface:
            date_limite.append(x)

    # Verification of some points that need to be out of polynyas (landfast-ice) 
    date_limite_finale = []
    for x in date_limite :
        if data_polynie.sel(time=x).sel(latitude=-68).sel(longitude=69.75) == 1 or data_polynie.sel(time=x).sel(latitude=-68).sel(longitude=70) == 1:
            date_limite_finale.append(x)
    
    time_polynie_OK = list(data_polynie.time.values)
    
    # Deletion of days when the map of polynya is not relevant
    for x in date_limite_finale :
        del time_polynie_OK[time_polynie_OK.index(x)]
    
    return data_polynie.sel(time = time_polynie_OK)


# In[Chargement liste de dates adaptées !!]

# =============================================================================
# FONCTION : initialisationTime :
#
#   Initialization of the list of times associated with a data item and corresponding to relevant polynyas
#   
#  Input :
#    - data (str) : Name of the parameter tha we want to study
#    - List_Months (list) : List of months to study
#    - List_Years (list) : List of years to study
#    - reanalysis (str) : reanalysis used in tis study
#
#  Output :
#     - time_data (array) : List of days associated with a data item and corresponding to relevant polynyas
#   
# =============================================================================

def initialisationTime(data,List_Months=range(5,10),List_Years=range(2010,2021),reanalysis='ERA'):
    
    ##### Polynie Hiver #####
    polynie_CD_Win_2010_2020_0 = dataYearsMonths('polynie',List_Months=List_Months,List_Years=List_Years,area='Cape Darnley',reanalysis=reanalysis)
    
    ##### Extraction des Temps des polynies admissibles #####
    time_polynie_CD_Win_2010_2020 = selection_polynie_CD(polynie_CD_Win_2010_2020_0).time.values
    List_time_polynie = [str(x)[:10] for x in time_polynie_CD_Win_2010_2020]
    
    List_time_data = [data.sel(time=x).time.values[0] for x in List_time_polynie]
    time_data = data.sel(time=List_time_data).time.values
    
    return time_data


# In[Chargement des polynies moyennes]


# =============================================================================
# FONCTION : meanPolynya :
#
#   Load the mean polynya stored in '/gpfswork/rech/omr/uvc35ld/stage/Polynyas/ERA5'
#
#  Input :
#    - reanalysis (str) : Name of the name of the reanalysis (by default ERA) must be in ['ERA', 'MERRA', 'JRA']
#    - area (str) : Name of the area or the station (by default Antarctica) must be in ['Antarctica', 'Cape Darnley', 'Cape Darnley Around', 'Davis','Mawson']
#
#  Output :
#     - data (xarray) : xarray of the data (DAILY) (Latitudes - of the area, Longitudes - of the area,time - List of days for each month/year)
#   
# =============================================================================

def meanPolynya(reanalysis='ERA',area='Antarctica'):
    
    path_polynya = '/gpfswork/rech/omr/uvc35ld/stage/Polynyas'
    
    if reanalysis == 'ERA':
        
        # Selection of the area 
        if area=='Antarctica' :
            data = xr.open_dataset(path_polynya + '/ERA5/polynies_era5_Mean_Win_2010-2020.nc').polynie # Selection the correct 'year-month'
        if area == 'Cape Darnley':
            data = sel_CapeDarnley(xr.open_dataset(path_polynya + '/ERA5/polynies_era5_Mean_Win_2010-2020.nc').polynie) # Selection of the correct area 
        if area == 'Cape Darnley Around':
            data = sel_CapeDarnley_around(xr.open_dataset(path_polynya + '/ERA5/polynies_era5_Mean_Win_2010-2020.nc').polynie) # Selection of the correct area 

    return data

# In[Chargement des proba présence polynie]

# =============================================================================
# FONCTION : OccurencePolynieMoyenne_CD :
#
#   Load the occurence rate of the polynyas in the Cape Darnley Region
#
#  Input :
#    - reanalysis (str) : Name of the name of the reanalysis (by default ERA) must be in ['ERA', 'MERRA', 'JRA']
#
#  Output :
#     - data (xarray) : xarray of the data (DAILY) (Latitudes - of the area, Longitudes - of the area,time - List of days for each month/year)
#   
# =============================================================================

def OccurencePolynieMoyenne_CD(reanalysis='ERA'):
    
    # Creation of the path to access the occurence rate 
    path_polynya = '/gpfswork/rech/omr/uvc35ld/stage/Polynyas'
    
    if reanalysis == 'ERA':
        path_reanalysis = '/ERA5/occurence_polynies_era5_CD_Win_2010-2020.nc'
    
    # Loading of the occurence rate (winters 2010-2020)
    occurence_polynie_CD_Win_2010_2020 = xr.open_dataset(path_polynya + path_reanalysis)['Occurences polynie']

    return occurence_polynie_CD_Win_2010_2020


# In[Définir la masque du continent]

# =============================================================================
# FONCTION : masqueContinent :
#
#   Determine the mask of the continent of Antarctica
#
#  Input :
#    - 
#
#  Output :
#     - xr_masque_continent (xarray - data(Latitudes,Longitudes)) : xarray of (0,1) to locate the continent (1 if the point is on the continent, 0 else).
#   
# =============================================================================

def masqueContinent():
    
    # Loading of the data of reference : sea ice coverture (ci)
    ci_ERA_sep_2014 = dataYearsMonths('ci',List_Years=range(2014,2015),List_Months=range(9,10),reanalysis = 'ERA')
    ci_ERA_sep_2014_m = ci_ERA_sep_2014.mean('time')
    
    # Definition of the mask array - fill with Nan if you are on the continent
    masque_continent = np.zeros(ci_ERA_sep_2014_m.shape)
    for i in range(masque_continent[:,0].shape[0]):
        for j in range(masque_continent[0,:].shape[0]):
            if np.isnan(ci_ERA_sep_2014_m.values[i,j]):
                masque_continent[i,j] = m.nan
    
    # Loading of the useful data
    Latitudes = ci_ERA_sep_2014_m.latitude.values
    Longitudes = ci_ERA_sep_2014_m.longitude.values
    
    # Creation of the dataArray 
    xr_masque_continent = creationDataArray(masque_continent, Latitudes, Longitudes)
    
    plotMap_Ant(xr_masque_continent,color='jet',titre='Masque')
    
    return xr_masque_continent


# In[Chargement des CI moyennes]


def CIMoyenne(List_Months=range(5,10),List_Years=range(2010,2021),reanalysis='ERA'):
    
    if reanalysis == 'ERA' :
    
        #### Chargement des dnnées brutes de CI ####
        data_0 = dataYearsMonths('ci',List_Years=List_Years,List_Months=List_Months,area='Cape Darnley Around',reanalysis=reanalysis)
    
    #### Chargement de la liste de temps sans les dates non adaptées ####
    time_data_ci = initialisationTime(data_0,List_Years=List_Years,List_Months=List_Months)
    
    #### Sélection des données de couverture de glace moyenne sur les dates de polynies adaptées ####
    data_moyenne = data_0.sel(time = time_data_ci).mean('time')
    
    return data_moyenne



# In[Chargement des U alpha]

# =============================================================================
# FONCTION : U_alpha_CD_init :
#
#   Load the velocity of wind at 10m high above the surface, projected on the direction alpha
#
#  Input :
#    - 
#
#  Output :
#     - data (xarray) : xarray of the velocity of wind projected on the direction alpha (time,alpha,Latitudes,Longitudes) where alpha is in [-170,-160,...,0,10,...,180]
#   
# =============================================================================

def U_alpha_CD_init():

    path = '/gpfswork/rech/omr/uvc35ld/stage/Polynyas/U_alpha_era5_CD_Win_2010-2020.nc'

    U_alpha_CD_Win_2010_2020 = xr.open_dataset(path)['U alpha']

    return U_alpha_CD_Win_2010_2020



# In[Chargement de données d'années et de mois choisis]

# =============================================================================
# FONCTION : dataYearsMonths :
#
#   Load the data in the NetCDF ()
#
#  Input :
#    - data (str) : Name of the parameter to load
#    - List_Years (list) : List of the years to load
#    - List_Months (list) : List of the months to load
#    - area (str) : Name of the area or the station (by default Antarctica) must be in ['Antarctica', 'Cape Darnley', 'Cape Darnley Around', 'Davis','Mawson']
#    - tol_polynie (str) : if the loaded data is a polynya --> procision of the tolerabce threshold (by default 70%)
#    - reanalysis (str) : Name of the name of the reanalysis (by default ERA) must be in ['ERA', 'MERRA', 'JRA']
#
#  Output :
#     - data (xarray) : xarray of the data (DAILY) (Latitudes - of the area, Longitudes - of the area,time - List of days for each month/year)
#   
# =============================================================================

def dataYearsMonths(data,List_Years=range(2010,2013),List_Months=range(5,10),area='Antarctica',tol_polynie='70%',reanalysis = 'ERA'):
    
    print('#############################')
    
    ### Selection of the path to the data storage space ###
    path_storage_data = '/gpfsstore/rech/omr/romr008/DATA'
    path_storage_polynya = '/gpfswork/rech/omr/uvc35ld/stage/Polynyas'
    
    if  reanalysis == 'ERA':
        
        if data == 'polynie' :
            path = path_storage_polynya + '/ERA5/Polynyas_'+ tol_polynie +'/DAILY/'+str(data)+'s_era5_ant_'+tol_polynie+'_y'
        else :
            path = path_storage_data+'/ERA5/DAILY/'+str(data)+'_era5_ant_y'
        
            
    if reanalysis == 'MERRA':
        
        if data == 'polynie' :
            path = path_storage_polynya + '/MERRA2/Polynyas_70%/DAILY/'+str(data)+'_merra2_ant_y'
        
        else :
            path = path_storage_data+'/MERRA2/PROCESSED/DAILY/'+str(data)+'_merra2_ant_y'
            
    if reanalysis == 'JRA':
        
        if data == 'polynie' :
            path = path_storage_polynya + '/JRA55/Polynyas_70%/DAILY/'+str(data)+'_jra55_ant_y'
        
        else :
            path = path_storage_data+'/JRA55/PROCESSED/DAILY/'+str(data)+'_jra55_ant_y'

    ###########################
    ### Loading of the data ###
    data_an = []
    times_an = []
    times = 0
    
    ### if reanalysis is JRA, parameters names are different (lon,lat,time_counter) ###
    if reanalysis == 'JRA':
        
        ### Varaition in the list of years ###
        for x in List_Years:
            data_mois = []
            times_mois = []
            
            ### Varaition in the list of months ###
            for y in List_Months:
                
                ### Loading of the data for a specific 'Year-Month' ###
                
                date = str(x)+'-'+str(y)
                
                if area=='Antarctica' :
                    data_intermediaire = xr.open_dataset(path+ str(x) +'.nc')[data].sel(time_counter=date) # Selection the correct 'year-month'
                if area == 'Cape Darnley':
                    data_intermediaire = sel_CapeDarnley(xr.open_dataset(path+ str(x) +'.nc')[data]).sel(time_counter=date) # Selection of the correct area 
                if area == 'Cape Darnley Around':
                    data_intermediaire = sel_CapeDarnley_around(xr.open_dataset(path+ str(x) +'.nc')[data]).sel(time_counter=date) # Selection of the correct area 
                if area == 'Davis':
                    data_intermediaire_1 = sel_CapeDarnley_around(xr.open_dataset(path+ str(x) +'.nc')[data]).sel(time_counter=date) 
                    data_intermediaire = data_intermediaire_1.sel(latitude=-68.25).sel(longitude=77.5) # Selection of the correct station (only a point on the grid)
                if area == 'Mawson':
                    data_intermediaire_1 = sel_CapeDarnley_around(xr.open_dataset(path+ str(x) +'.nc')[data]).sel(time_counter=date)
                    data_intermediaire = data_intermediaire_1.sel(latitude=-67.25).sel(longitude=62.5) # Selection of the correct station (only a point on the grid)

                data_mois.append(data_intermediaire)
                times_mois.append(data_intermediaire.time_counter.values)
                #times_2 = times_2 + times_an[-1].shape[0]
            data_intermediaire = xr.merge(data_mois)
                
            data_an.append(data_intermediaire)
            times_an.append(data_intermediaire.time_counter.values)
            times = times + times_an[-1].shape[0]
            
            print('Chargement de '+data+' année '+str(x))
        
    else :
    
        for x in List_Years:
            data_mois = []
            times_mois = []
    
            for y in List_Months:
                
                ### Loading of the data for a specific 'Year-Month' ###
                
                date = str(x)+'-'+str(y)
                
                if area=='Antarctica' :
                    data_intermediaire = xr.open_dataset(path+ str(x) +'.nc')[data].sel(time=date) # Selection the correct 'year-month'
                if area == 'Cape Darnley':
                    data_intermediaire = sel_CapeDarnley(xr.open_dataset(path+ str(x) +'.nc')[data]).sel(time=date) # Selection of the correct area 
                if area == 'Cape Darnley Around':
                    data_intermediaire = sel_CapeDarnley_around(xr.open_dataset(path+ str(x) +'.nc')[data]).sel(time=date) # Selection of the correct area 
                if area == 'Davis':
                    data_intermediaire_1 = sel_CapeDarnley_around(xr.open_dataset(path+ str(x) +'.nc')[data]).sel(time=date)
                    data_intermediaire = data_intermediaire_1.sel(latitude=-68.25).sel(longitude=77.5) # Selection of the correct station (only a point on the grid)
                if area == 'Mawson':
                    data_intermediaire_1 = sel_CapeDarnley_around(xr.open_dataset(path+ str(x) +'.nc')[data]).sel(time=date)
                    data_intermediaire = data_intermediaire_1.sel(latitude=-67.25).sel(longitude=62.5) # Selection of the correct station (only a point on the grid)

                data_mois.append(data_intermediaire)
                times_mois.append(data_intermediaire.time.values)
                #times_2 = times_2 + times_an[-1].shape[0]
            data_intermediaire = xr.merge(data_mois)
                
            data_an.append(data_intermediaire)
            times_an.append(data_intermediaire.time.values)
            times = times + times_an[-1].shape[0]
            
            print('Chargement de '+data+' année '+str(x))

    
    data_2010_2020 = xr.merge(data_an)
    
    
    print('#############################')
    return data_2010_2020[data]


# In[Chargement de données d'années et de mois choisis - 21h]

# =============================================================================
# FONCTION : dataYearsMonths_delta_H :
#
#   Loading and calculation of the variation of the data between 21h and 23h
#
#  Input :
#    - data (str) : Name of the parameter to load
#    - delta_h (int) : variation of hour between 21h and 21 + delta_hh (by default 2)
#    - List_Years (list) : List of the years to load
#    - List_Months (list) : List of the months to load
#    - area (str) : Name of the area or the station (by default Antarctica) must be in ['Antarctica', 'Cape Darnley', 'Cape Darnley Around', 'Davis','Mawson']
#    - tol_polynie (str) : if the loaded data is a polynya --> procision of the tolerabce threshold (by default 70%)
#    - reanalysis (str) : Name of the name of the reanalysis (by default ERA) must be in ['ERA', 'MERRA', 'JRA']
#
#  Output :
#     - data (xarray) : xarray of the delta data (= data[21h+delta_h] - data[21h] (DAILY) (Latitudes - of the area, Longitudes - of the area,time - List of days for each month/year)
#   
# =============================================================================

def dataYearsMonths_delta_H(data,delta_h=2,List_Years=range(2010,2013),List_Months=range(5,10),area='Cape Darnley',reanalysis = 'ERA'):
    
    print('#############################')
    ### Selection of the path to the data storage space ###
    
    path_polynie = '/gpfswork/rech/omr/uvc35ld/stage/Polynyas'
    path_store = '/gpfsstore/rech/omr/romr008/DATA'
    
    if reanalysis == 'ERA':
    
        if data == 'polynie' :
            path = path_polynie + '/ERA5/DAILY/'+str(data)+'s_era5_ant_y'
            
        elif data == 'U10':
            path_u = path_store + '/ERA5/HOURLY/u10_era5_ant_y'
            path_v = path_store + '/ERA5/HOURLY/v10_era5_ant_y'
            
        else :
            path = path_store + '/ERA5/HOURLY/'+str(data)+'_era5_ant_y'
        
    
    H1 = 21
    H2 = H1 + delta_h
    
    ###########################
    ### Loading of the data ###
    data_an = []
    times_an = []
    times = 0
    
    ### Varaition in the list of years ###
    for x in List_Years:
        data_mois = []
        # times_mois = []
        
        ### Varaition in the list of months ###
        for y in List_Months:
            
            ### Loading of the data for a specific 'Year-Month' ###
            date = str(x)+'-'+str(y)
            if data != 'U10':
                if area=='Antarctica' :
                    data_intermediaire = xr.open_dataset(path+ str(x) +'.nc')[data].sel(time=date)
                if area == 'Cape Darnley':
                    data_intermediaire = sel_CapeDarnley(xr.open_dataset(path+ str(x) +'.nc')[data]).sel(time=date)
                if area == 'Cape Darnley Around':
                    data_intermediaire = sel_CapeDarnley_around(xr.open_dataset(path+ str(x) +'.nc')[data]).sel(time=date)
                if area == 'Davis':
                    data_intermediaire_1 = sel_CapeDarnley_around(xr.open_dataset(path+ str(x) +'.nc')[data]).sel(time=date)
                    data_intermediaire = data_intermediaire_1.sel(latitude=-68.25).sel(longitude=77.5)
                if area == 'Mawson':
                    data_intermediaire_1 = sel_CapeDarnley_around(xr.open_dataset(path+ str(x) +'.nc')[data]).sel(time=date)
                    data_intermediaire = data_intermediaire_1.sel(latitude=-67.25).sel(longitude=62.5)
                
                if len(data_intermediaire.shape) == 3:
                    data_mois.append(data_intermediaire)
                    # times_mois.append(data_intermediaire.time.values)
                elif len(data_intermediaire.shape) == 4:
                    data_mois.append(data_intermediaire.mean('level'))
                #times_2 = times_2 + times_an[-1].shape[0]
            else :
                
                ### Loading of the data for u10 and v10 and  calculate the norm U10 ###
                data_intermediaire_u =  xr.open_dataset(path_u+ str(x) +'.nc')['u10']
                data_intermediaire_v =  xr.open_dataset(path_v+ str(x) +'.nc')['v10']
                data_intermediaire_0 = np.sqrt(data_intermediaire_u * data_intermediaire_u + data_intermediaire_v*data_intermediaire_v)

                if area=='Antarctica' :
                    data_intermediaire = np.sqrt(data_intermediaire_0.sel(time=date))
                if area == 'Cape Darnley':
                    data_intermediaire = sel_CapeDarnley(data_intermediaire_0).sel(time=date)
                if area == 'Cape Darnley Around':
                    data_intermediaire = sel_CapeDarnley_around(data_intermediaire_0).sel(time=date)
                if area == 'Davis':
                    data_intermediaire_1 = sel_CapeDarnley_around(data_intermediaire_0).sel(time=date)
                    data_intermediaire = data_intermediaire_1.sel(latitude=-68.25).sel(longitude=77.5)
                if area == 'Mawson':
                    data_intermediaire_1 = sel_CapeDarnley_around(data_intermediaire_0).sel(time=date)
                    data_intermediaire = data_intermediaire_1.sel(latitude=-67.25).sel(longitude=62.5)
                
                if len(data_intermediaire.shape) == 3:
                    data_mois.append(data_intermediaire.to_dataset(name=data))
                    # times_mois.append(data_intermediaire.time.values)
                elif len(data_intermediaire.shape) == 4:
                    data_mois.append(data_intermediaire.mean('level').to_dataset(name=data))
                #times_2 = times_2 + times_an[-1].shape[0]
        # print(data_mois)
                
        data_intermediaire = xr.merge(data_mois)
        

        
        # In the variable 'data_intermediaire', we have the data of all hours over a month.
        # We select those associated with H1 on the one hand and H2 on the other.
        
        # List hour/day :
        
        times_days_hours = data_intermediaire.time.values
        
        ##############################################################################
        ## Selection of the data of H1
        times_days_hours_H1 = []
        List_data_intermediaire_H1 = []
        for z in times_days_hours:
            hour = str(z)[11:13]
            # print(hour)
            if hour == str(H1):
                times_days_hours_H1.append(z) ## Sélection des heures valant H1
                List_data_intermediaire_H1.append(data_intermediaire.sel(time=z)[data].values)
                # List_data_intermediaire_H1.append(data_intermediaire.sel(time=z)[data])
        # print('Nombre de jours : '+str(len(List_data_intermediaire_H1)),', (Latitudes,Longitudes) = '+str(List_data_intermediaire_H1[0].shape))
        
        # data_intermediaire_H1 = xr.merge(List_data_intermediaire_H1)
        
        array_data_intermediaire_H1 = np.zeros((len(List_data_intermediaire_H1),data_intermediaire.latitude.shape[0],data_intermediaire.longitude.shape[0]))
        for i in range(len(List_data_intermediaire_H1)):
            array_data_intermediaire_H1[i,:,:] = List_data_intermediaire_H1[i][:,:]
        
        data_intermediaire_H1 = creationDataArray(array_data_intermediaire_H1, data_intermediaire.latitude.values,data_intermediaire.longitude.values,time = times_days_hours_H1)
        
        # print(data_intermediaire_H1.time)
        
        ##############################################################################
        ## Selection of the data of H2
        times_days_hours_H2 = []
        List_data_intermediaire_H2 = []
        for z in times_days_hours:
            hour = str(z)[11:13]
            # print('Hour = ' + str(hour)+' , z = '+str(z))
            if hour == str(H2):
                times_days_hours_H2.append(z) ## Sélection des heures valant H2
                List_data_intermediaire_H2.append(data_intermediaire.sel(time=z)[data].values)
                # print('OK')
                # print(data_intermediaire.sel(time=z)[data].values.shape)
        # print('Nombre de jours : '+str(len(List_data_intermediaire_H2)),', (Latitudes,Longitudes) = '+str(List_data_intermediaire_H2[0].shape))


        array_data_intermediaire_H2 = np.zeros((len(List_data_intermediaire_H2),data_intermediaire.latitude.shape[0],data_intermediaire.longitude.shape[0]))
        for i in range(len(List_data_intermediaire_H2)):
           array_data_intermediaire_H2[i,:,:] = List_data_intermediaire_H2[i][:,:]
        
        data_intermediaire_H2 = creationDataArray(array_data_intermediaire_H2, data_intermediaire.latitude.values,data_intermediaire.longitude.values,time = times_days_hours_H2)
        
        # print(data_intermediaire_H1.shape)
        # print(data_intermediaire_H2.shape)
        
        ## Calculation of the variation of delta_H
        array_intermediaire_delta_H = np.zeros(data_intermediaire_H2.shape)
        if H2 <=23 :

            array_intermediaire_delta_H = data_intermediaire_H2.values - data_intermediaire_H1.values
        
        data_intermediaire_delta_H = creationDataArray(array_intermediaire_delta_H, data_intermediaire.latitude.values,data_intermediaire.longitude.values,
                                                       time = times_days_hours_H1,
                                                       nameDataArray=data)
        
        
        data_an.append(data_intermediaire_delta_H.to_dataset(name='delta '+data))
        times_an.append(data_intermediaire_delta_H.time.values)
        times = times + times_an[-1].shape[0]
        
        print('Calcul delta '+data+' année '+str(x))
    # print(data_an)
    # print(data_an[0])
    
    # print(data_an)
    
    
    data_2010_2020 = xr.merge(data_an)
    
    print('#############################')
   
    return data_2010_2020['delta '+data]

# In[Calcul dérivée de la surface]


# =============================================================================
# FONCTION : calculSurfacePolynies_delta_H :
#
#   Loading and calculation of the variation of the data between 21h and 23h
#
#  Input :
#    - data (str) : Name of the parameter to load
#    - delta_h (int) : variation of hour between 21h and 21 + delta_hh (by default 2)
#    - List_Years (list) : List of the years to load
#    - List_Months (list) : List of the months to load
#    - area (str) : Name of the area or the station (by default Antarctica) must be in ['Antarctica', 'Cape Darnley', 'Cape Darnley Around', 'Davis','Mawson']
#    - tol_polynie (str) : if the loaded data is a polynya --> procision of the tolerabce threshold (by default 70%)
#    - reanalysis (str) : Name of the name of the reanalysis (by default ERA) must be in ['ERA', 'MERRA', 'JRA']
#
#  Output :
#     - data (xarray) : xarray of the delta data (= data[21h+delta_h] - data[21h] (DAILY) (Latitudes - of the area, Longitudes - of the area,time - List of days for each month/year)
#   
# =============================================================================

def calculSurfacePolynies_delta_H(List_Years=range(2010,2013),List_Months=range(5,10),polynie='CD',reanalysis='ERA'):
    
    print('#############################')
    
    # if (List_Months[0] > 1) and (List_Months[-1] < 12) :
    #     List_Months_new = [List_Months[0]-1] + list(List_Months) + [List_Months[-1]+1]
    
    # Addition of a one-day offset to calculate the area difference
    if (List_Months[-1] < 12) :
        List_Months_new = list(List_Months) + [List_Months[-1]+1]

    
    data_an = []
    
    ###########################
    ### Loading of the data ###
    
    for x in List_Years:
        data_polynie_intermediaire_brut = dataYearsMonths('polynie',List_Years=[x],List_Months=List_Months_new,area='Cape Darnley',reanalysis=reanalysis)
        # print(data_polynie_intermediaire_brut.shape)
        
        time_brut = data_polynie_intermediaire_brut.time
 
    
        # Calculation of the days which limit the range between the fisrt day and the last day of each year
        Years_Months_min = str(x)+'-'+str(List_Months_new[0])
        Years_Months_max = str(x)+'-'+str(List_Months_new[-1])
        
        date_min = time_brut.sel(time=Years_Months_min).values[0]
        date_max = time_brut.sel(time=Years_Months_max).values[0]
        
        # Selection of the days relevant       
        data_polynie_intermediaire = data_polynie_intermediaire_brut.sel(time = slice(date_min,date_max))
        
        # Calculation of the surface for each day
        surface_intermediaire = calculSurfacePolynies(data_polynie_intermediaire,area=polynie)

        # Calculation of the variation of the Capa Darnlay surface variation between two successive days
        List_delta_S_intermediaire = [surface_intermediaire[i+1]-surface_intermediaire[i] for i in range(len(surface_intermediaire)-1)]
        time_delta_S_intermediaire = data_polynie_intermediaire.time.values[:-1]
        
        # Creation of the xarray associated to this varation of days
        data_delta_S_intemediaire = xr.DataArray(
            ## Valeurs du tableau de données        
            data = List_delta_S_intermediaire,
            coords=dict(
                ## Time
                time=(["time"], time_delta_S_intermediaire)
            ),
            attrs=dict(
                description = 'delta S'
                )
        )
        

        data_an.append(data_delta_S_intemediaire.to_dataset(name='delta S polynie'))
    # print(data_an)
    
    data_S = xr.merge(data_an)
    
    return data_S['delta S polynie']
    

# In[Chargement de données à l'heure entre 2 jours]

# def dataHourly_Days(data,days,area='Cape Darnley'):
#     path = '/gpfsstore/rech/omr/romr008/DATA/ERA5/HOURLY/'+str(data)+'_era5_ant_y'
    
#     day1,day2 = days[0],days[1]

#     ##### data pour toutes les années (2010 --> 2020):
#     data_days = []
#     times_days = []
#     # times = 0

#     for x in [day1,day2]:
#         date = str(x)[:10]
        
#         if area=='Antarctica' :
#             data_intermediaire = xr.open_dataset(path+ str(x) +'.nc')[data].sel(time=date)
#         if area == 'Cape Darnley':
#             data_intermediaire = sel_CapeDarnley(xr.open_dataset(path+ date[:4] +'.nc')[data]).sel(time=date)
#         if area == 'Cape Darnley Around':
#             data_intermediaire = sel_CapeDarnley_around(xr.open_dataset(path+ date[:4]  +'.nc')[data]).sel(time=date)
#         if area == 'Davis':
#             data_intermediaire_1 = sel_CapeDarnley_around(xr.open_dataset(path+ date[:4] +'.nc')[data]).sel(time=date)
#             data_intermediaire = data_intermediaire_1.sel(latitude=-68.25).sel(longitude=77.5)
#         if area == 'Mawson':
#             data_intermediaire_1 = sel_CapeDarnley_around(xr.open_dataset(path+ date[:4] +'.nc')[data]).sel(time=date)
#             data_intermediaire = data_intermediaire_1.sel(latitude=-67.25).sel(longitude=62.5)
            
#         data_days.append(data_intermediaire)
#         times_days.append(data_intermediaire.time.values)
#         #times_2 = times_2 + times_an[-1].shape[0]
#     data_final = xr.merge(data_days)
        
#     # data_an.append(data_intermediaire)
#     # times_an.append(data_intermediaire.time.values)
#     # times = times + times_an[-1].shape[0]
        
#     # data_2010_2020 = xr.merge(data_an)
    
#     return data_final


# In[Chargement de données à l'heure]

# def dataHour(data,List_Years=[2015],List_Months=[7],List_Days=range(21,30),area='Cape Darnley'):

#     if data == 'polynie' :
#         path = '/gpfswork/rech/omr/uvc35ld/stage/Polynyas/HOURLY/polynies_era5_CD_HOURLY_2015-07.nc'
#         data_2010_2020 = xr.open_dataset(path)
    
#     else :
#         path = '/gpfsstore/rech/omr/romr008/DATA/ERA5/HOURLY/'+str(data)+'_era5_ant_y'
    

#         ##### data pour toutes les années (2010 --> 2020):
#         data_an = []
#         times_an = []
#         times = 0
    
#         for x in List_Years:
#             data_mois = []
#             times_mois = []
#             for y in List_Months:
#                 date = str(x)+'-'+str(y)
                
#                 if area=='Antarctica' :
#                     data_intermediaire = xr.open_dataset(path+ str(x) +'.nc')[data].sel(time=date)
#                 if area == 'Cape Darnley':
#                     data_intermediaire = sel_CapeDarnley(xr.open_dataset(path+ str(x) +'.nc')[data]).sel(time=date)
#                 if area == 'Cape Darnley Around':
#                     data_intermediaire = sel_CapeDarnley_around(xr.open_dataset(path+ str(x) +'.nc')[data]).sel(time=date)
#                 if area == 'Davis':
#                     data_intermediaire_1 = sel_CapeDarnley_around(xr.open_dataset(path+ str(x) +'.nc')[data]).sel(time=date)
#                     data_intermediaire = data_intermediaire_1.sel(latitude=-68.25).sel(longitude=77.5)
#                 if area == 'Mawson':
#                     data_intermediaire_1 = sel_CapeDarnley_around(xr.open_dataset(path+ str(x) +'.nc')[data]).sel(time=date)
#                     data_intermediaire = data_intermediaire_1.sel(latitude=-67.25).sel(longitude=62.5)
                    
#                 data_mois.append(data_intermediaire)
#                 times_mois.append(data_intermediaire.time.values)
#                 #times_2 = times_2 + times_an[-1].shape[0]
#             data_intermediaire = xr.merge(data_mois)
                
#             data_an.append(data_intermediaire)
#             times_an.append(data_intermediaire.time.values)
#             times = times + times_an[-1].shape[0]
            
#         data_2010_2020 = xr.merge(data_an)
   
#     return data_2010_2020[data]
    

# In[Chargement de données mensuelles]

# def dataByMonth(data,month,area='Antarctica'):
    
#     ##### v10 pour Octobre :
#     L_data = []
#     L_data_i_time = []
#     # L_data_month_mean = []

#     if (data == 'polynie'):
#         if (area == 'Antarctica'):
#             for i in range(10,21):
#                 data_intermediaire = xr.open_dataset('/gpfswork/rech/omr/uvc35ld/stage/Polynyas/DAILY/'+str(data)+'s_era5_ant_y20'+ str(i) +'.nc')[data]
#                 L_data.append(data_intermediaire)
#         if (area == 'CapeDarnley'):
#             for i in range(10,21):
#                 data_intermediaire = xr.open_dataset('/gpfswork/rech/omr/uvc35ld/stage/polynyas/'+str(data)+'s_era5_ant_y20'+ str(i) +'.nc')[data]
#                 L_data.append(sel_CapeDarnley(data_intermediaire))
#         if (area == 'CapeDarnley_around'):
#             for i in range(10,21):
#                 data_intermediaire = xr.open_dataset('/gpfswork/rech/omr/uvc35ld/stage/polynyas/'+str(data)+'s_era5_ant_y20'+ str(i) +'.nc')[data]
#                 L_data.append(sel_CapeDarnley_around(data_intermediaire))
#     else :
    
    
#         if (area == 'Antarctica'):
#             for i in range(10,21):
#                 data_intermediaire = xr.open_dataset('/gpfsstore/rech/omr/romr008/DATA/ERA5/DAILY/'+str(data)+'_era5_ant_y20'+ str(i) +'.nc')[data]
#                 L_data.append(data_intermediaire)
                
#         if (area == 'CapeDarnley'):
#             for i in range(10,21):
#                 data_intermediaire = xr.open_dataset('/gpfsstore/rech/omr/romr008/DATA/ERA5/DAILY/'+str(data)+'_era5_ant_y20'+ str(i) +'.nc')[data]
#                 L_data.append(sel_CapeDarnley(data_intermediaire))
                
#         if (area == 'CapeDarnley_around'):
#             for i in range(10,21):
#                 data_intermediaire = xr.open_dataset('/gpfsstore/rech/omr/romr008/DATA/ERA5/DAILY/'+str(data)+'_era5_ant_y20'+ str(i) +'.nc')[data]
#                 L_data.append(sel_CapeDarnley_around(data_intermediaire))
                

    
#     latitudes = L_data[0].latitude.values
#     longitudes = L_data[0].longitude.values
        
#     ##### u10 pour Septembre :
#     data_month = []
#     data_month_m = []
    
   
#     for i in range(11):
        
#         ## Création d'une liste d'indices d'éléments 
#         #print(L_data[i].time.dt.month)
#         time_an = L_data[i].time.dt.month.values
#         L_month= []
#         for j in range(len(time_an)):
#             if (time_an[j] == month):
#                 L_month.append(j)
        
#         L_month = np.array(L_month)
               
#         L_data_i_time.append(L_month)
#         #
#         data_month_intermediaire = L_data[i].isel(time=L_month)
        
#         #print(data_month_intermediaire)
#         data_month.append(data_month_intermediaire)
        
        
#         data_month_intermediaire_m = data_month_intermediaire.mean('time')
#         data_month_m.append(data_month_intermediaire_m)
    
#     L_data_i_time = np.array(L_data_i_time)
        
#     #### Création d'un dataArray des data de tous les mois de month :

#     array_data_month_tot = np.zeros((L_data_i_time.size,latitudes.shape[0],longitudes.shape[0]))

#     time_data = []
    
#     for i in range(11):
#         size_dim2 = L_data_i_time.shape[1]       
#         L = data_month[i].time.data
        
#         for x in L:
#             time_data.append(x)

#         array_data_month_tot[i*size_dim2:(i+1)*size_dim2,:,:] = data_month[i].values


#     data_month = creationDataArray(array_data_month_tot,
#                                   latitudes,
#                                   longitudes,
#                                   time=time_data,
#                                   nameDataArray= "data each month " + str(int(month)) +" from 2010 to 2020"
#                                   )
#     return data_month

# In[Chargement de données - moyennes mensuelles]

# def dataMonthlyMean(data,month,area='Antarctica',polynie=False):
    
#     ##### v10 pour Octobre :
#     L_data = []
#     L_data_i_time = []
#     # L_data_month_mean = []
    
#     if (not polynie):
    
#         if (area == 'Antarctica'):
#             for i in range(10,21):
#                 data_intermediaire = xr.open_dataset('/gpfsstore/rech/omr/romr008/DATA/ERA5/DAILY/'+str(data)+'_era5_ant_y20'+ str(i) +'.nc')[data]
#                 L_data.append(data_intermediaire)
        
        
#         if (area == 'CapeDarnley'):
#             for i in range(10,21):
#                 data_intermediaire = xr.open_dataset('/gpfsstore/rech/omr/romr008/DATA/ERA5/DAILY/'+str(data)+'_era5_ant_y20'+ str(i) +'.nc')[data]
#                 L_data.append(sel_CapeDarnley(data_intermediaire))
                
#         if (area == 'CapeDarnley_around'):
#             for i in range(10,21):
#                 data_intermediaire = xr.open_dataset('/gpfsstore/rech/omr/romr008/DATA/ERA5/DAILY/'+str(data)+'_era5_ant_y20'+ str(i) +'.nc')[data]
#                 L_data.append(sel_CapeDarnley_around(data_intermediaire))
                
    
#         latitudes = L_data[0].latitude.values
#         longitudes = L_data[0].longitude.values
            
#         ##### u10 pour Septembre :

#         data_month_m = []
        
       
#         for i in range(11):
            
#             ## Création d'une liste d'indices d'éléments 
#             #print(L_data[i].time.dt.month)
#             time_an = L_data[i].time.dt.month.values
#             L_month= []
#             for j in range(len(time_an)):
#                 if (time_an[j] == month):
#                     L_month.append(j)
            
#             L_month = np.array(L_month)
                   
#             L_data_i_time.append(L_month)
#             #
#             data_month_intermediaire = L_data[i].isel(time=L_month)
#             data_month_intermediaire_m = data_month_intermediaire.mean('time')
#             data_month_m.append(data_month_intermediaire_m)
        
#         L_data_i_time = np.array(L_data_i_time)
            
#         #### Création d'un dataArray des data de tous les mois de month :
    
#         array_data_month_tot_m = np.zeros((11,latitudes.shape[0],longitudes.shape[0]))
    
#         time_data = []
        
        
        
#         str_month = str(month)
        
#         for i in range(11):
    
#             time_data.append(datetime.datetime.strptime('20'+str(10+i)+'-'+str_month, '%Y-%m'))
    
#             array_data_month_tot_m[i,:,:] = data_month_m[i].values
    
    
#         data_month_m = creationDataArray(array_data_month_tot_m,
#                                       latitudes,
#                                       longitudes,
#                                       time=time_data,
#                                       nameDataArray= "monthly mean data " + str(int(month)) +" from 2010 to 2020"
#                                       )
#     else :
#         Months = ['Jan','Fev','Mar','Avr','Mai','Juin','Juil','Aou','Sep','Oct','Nov','Dec']
#         data_month_m = xr.open_dataset('/gpfswork/rech/omr/uvc35ld/cartes_Antarctique/CI/Polynyas/polynies_era5_ant_m_'+Months[month-1]+'.nc')[data]

#     return data_month_m


# In[Moyenne données sur une polynie]

# =============================================================================
# FONCTION : meanPolynya :
#
#   Calculation of the mean of data on the polynya.
#
#  Input :
#    - data (xarray - (latitudes,longitudes) or (latitudes,longitudes,time)) : data to work on 
#    - data (xarray - (latitudes,longitudes) or (latitudes,longitudes,time)) : data of polynya index (same size as data).
#    - reanalysis (str) : Name of the name of the reanalysis (by default ERA) must be in ['ERA', 'MERRA', 'JRA']
#
#  Output :
#     - data (xarray - (latitudes,longitudes) or (latitudes,longitudes,time)) : calculation of surface average of data.
#   
# =============================================================================

def meanPolynya(data,polynie,reanalysis='ERA'):
    # Rayon de la Terre
    R = 6.4 *1e3
    
    # Loading of useful data
    
    if reanalysis == 'ERA' :
        
        Lat = polynie.latitude.values
        Lon = polynie.longitude.values
    
    if reanalysis == 'MERRA' or reanalysis == 'JRA':
        
        Lat = polynie.lat.values
        Lon = polynie.lon.values
        
    Polynies = polynie.values
    Data = data.values
    forme = Polynies.shape
        
    # Test of the dimension of data (only 2d no dimension for the time) : Polynya(Latitudes,Longitudes)
    
    if (len(forme)==2):
        S=0
        M=0
        for i in range(len(Lat)):
            lat = Lat[i]
            for j in range(len(Lon)):
                if Polynies[i,j]==1:
                    
                    # Calculation of the surface for a point on the grid (depends on the latitude of the point)
                    s = abs(R*R*np.radians(0.25)*np.radians(0.25)*np.sin(np.radians(lat)))
                    M += s*Data[i,j]
                    S += s
        if (S == 0):
            M=0
        else:
            M = M/S

    # Test of the dimension of data (only 3d no dimension for the time) : Polynya(time,Latitudes,Longitudes)
    if (len(forme)==3):
        S=[]
        M=[]
        for i in range(forme[0]):
            s=0
            m=0
            for j in range(forme[1]):  
                lat = Lat[j]
                
                for k in range(forme[2]):
                    if Polynies[i,j,k]==1:
                        
                        # Calculation of the surface for a point on the grid (depends on the latitude of the point)
                        s1 = abs(R*R*np.radians(0.25)*np.radians(0.25)*np.sin(np.radians(lat)))
                        s += s1
                        m += s1*Data[i,j,k]
            if (s == 0):
                M.append(0)
            else :
                M.append(m/s)  
        M = np.array(M)
    return M
    

# In[creation DataArray]

# =============================================================================
# FONCTION : creationDataArray :
#
#   Creation of a dataArray from existing array.
#
#  Input :
#    - data (array - with 2 or 3 dimension) : data to work on 
#    - Latitudes (list or array (1 d))) : list of latitudes (size of the fisrt dimension of data).
#    - Longitudes (list or array (1 d))) : list of longitudes (size of the second dimension of data).
#    - time (list or array (1 d))) : list of time (size of the last dimension of data) - None by default.
#
#  Output :
#     - dataArray (xarray) : xarray created from data (Latitudes, Longitudes,time (or not)).
#   
# =============================================================================

def creationDataArray(Data,Latitudes,Longitudes,time = None,nameDataArray="DataArray"):
    #print('time creation DataArray : ',time)
    # Creation d'un xarray de données issue de l'algo
    shape = Data.shape
    n = len(shape)
    
    # Test of the dimension of date (Is there a dimension for time ?)
    if (n == 2):
        # print('time is none')
        
        # Creation of the dataArray from Latitudes and Longitudes.
        data_DataArray = xr.DataArray(
            
            ## Valeurs du tableau de données        
            data = Data
            ,
            coords=dict(
                ## Coordonnées Latitudes
                latitude=(["latitude"], Latitudes)
            ,
                ## Coordonnées Longitudes
                longitude=(["longitude"], Longitudes)
            ),
            attrs=dict(
                description = nameDataArray
                )
        )
    elif (n == 3):
        # Verification if there a parameter for time (else creation by default)
        if (time is None):
            time = range(shape[0])
        
        # Creation of the dataArray
        data_DataArray = xr.DataArray(
            
            ## Valeurs du tableau de données        
            data = Data
            ,
            coords=dict(
                time=(["time"], time)
            ,
                ## Coordonnées Latitudes
                latitude=(["latitude"], Latitudes)
            ,
                ## Coordonnées Longitudes
                longitude=(["longitude"], Longitudes)
            ),
            attrs=dict(
                description = nameDataArray
                )
        )
    # data is not 2d or 3d --> Error and output is None
    else :
        print('Error : data is not 2 or 3 dimension')
        data_DataArray = None
    
    return data_DataArray

# In[Creation of netCDF for polynya index]

# # =============================================================================
# # FONCTION : creationPolyniesNetCDF :
# #
# #   Creation of a netCDF
# #
# #  Input :
# #    - data (array - with 2 or 3 dimension) : data to work on 
# #    - Latitudes (list or array (1 d))) : list of latitudes (size of the fisrt dimension of data).
# #    - Longitudes (list or array (1 d))) : list of longitudes (size of the second dimension of data).
# #    - time (list or array (1 d))) : list of time (size of the last dimension of data) - None by default.
# #
# #  Output :
# #     - dataArray (xarray) : xarray created from data (Latitudes, Longitudes,time (or not)).
# #   
# # =============================================================================

# def creationPolyniesNetCDF(ci,path_NETCDF_sortie,reanalysis = 'ERA'):
    
#     ci_test = ci #.isel(time=slice(243,246))
    
#     # print(ci_test)
#     if reanalysis == 'ERA':
#         latitudes = ci_test.latitude
#         longitudes = ci_test.longitude
    

#     data_polynie = detectionPolynya(ci_test)


#     data_polynie = creationDataArray(np.array(data_polynie),latitudes.data,longitudes.data)
    
#     dataset_polynie = data_polynie.to_dataset(name='polynie')
#     netCDF_polynie = dataset_polynie.to_netcdf(path=path_NETCDF_sortie)
    
#     return netCDF_polynie




# In[Calcul de Surface Polynies ]

# =============================================================================
# FONCTION : calculSurfacePolynies :
#
#   Calculation of the surface of the polynyas
#
#  Input :
#    - dataPolynyas (array - with 2 or 3 dimension) : data of the polynya index.
#    - area (str) :     Latitudes (list or array (1 d))) : list of latitudes (size of the fisrt dimension of data).
#    - Longitudes (list or array (1 d))) : list of longitudes (size of the second dimension of data).
#    - time (list or array (1 d))) : list of time (size of the last dimension of data) - None by default.
#
#  Output :
#     - dataArray (xarray) : xarray created from data (Latitudes, Longitudes,time (or not)).
#   
# =============================================================================

def calculSurfacePolynies(dataPolynies,area='All',reanalysis='ERA'):
    
    # Rayon de la Terre
    R = 6.4 *1e3
    
    # Loading of useful data
    
    if reanalysis == 'ERA' :
        Lat = dataPolynies.latitude.values
        Lon = dataPolynies.longitude.values
    
    if reanalysis == 'MERRA' or reanalysis == 'JRA':
        Lat = dataPolynies.lat.values
        Lon = dataPolynies.lon.values


    # Choice of the area in ['Antarctica','Cape Darnley','Cape Darnley Around','CD','MB']
    
    if area == 'Antarctica' or area == 'Cape Darnley' or area == 'Cape Darnley Around' :
        
        # Loading and selection of the polynya index
        if area == 'Cape Darnley' :
            Polynies = sel_CapeDarnley(dataPolynies).values
        
        if area == 'Cape Darnley Around' :
            Polynies = sel_CapeDarnley_around(dataPolynies).values
        
        forme = Polynies.shape
        
        # Test of the dimension of data (only 2d no dimension for the time) : Polynya(Latitudes,Longitudes)
        if (len(forme)==2):
            S=0
            for i in range(len(Lat)):
                lat = Lat[i]
                for j in range(len(Lon)):
                    
                    # Test if the point is in the polynya
                    if Polynies[i,j]==1:
                        # Calculation of the surface for a point on the grid (depends on the latitude of the point)
                        S +=abs(R*R*np.radians(0.25)*np.radians(0.25)*np.sin(np.radians(lat)))
    
        # Test of the dimension of data (only 3d no dimension for the time) : Polynya(time,Latitudes,Longitudes)
        if (len(forme)==3):
            S=[]
            Time = dataPolynies.time.values
            for i in range(forme[0]):
                s=0
                for j in range(forme[1]):  
                    lat = Lat[j]
                    
                    for k in range(forme[2]):
                        # Test if the point is in the polynya
                        if Polynies[i,j,k]==1:
                            # Calculation of the surface for a point on the grid (depends on the latitude of the point)
                            s +=abs(R*R*np.radians(0.25)*np.radians(0.25)*np.sin(np.radians(lat)))
                S.append(s)  
            
            # Creation of the dataArray of surfaces : Surfaces(Time)
            S = np.array(S)
            data_S = xr.DataArray(data = S,coords=dict(time=(["time"],Time)),attrs=dict(description = 'Surface polynie'))
            S = data_S
    
    # Choice of the polynya in ['CDP','MBP'] 

    if area == 'CDP':
        
        # Loading and selection of the polynya index
        Polynies = sel_CapeDarnley(dataPolynies).values
        forme = Polynies.shape
        
        
        # Sélection of th polynya from a limit between CDP and MBP.
        Liste_limite_polynies_latitude = []
        List_NS = list(np.linspace(-69,-67.5,15))
        
        compteur = 0
        for i in range(len(Lon)):
            # Limit on the West of the landfast ice tongue
            if Lon[i] < 67.5 :
                Liste_limite_polynies_latitude.append(-69)
                compteur += 1
                
            # Limit on the East of the landfast ice tongue
            elif Lon[i] > 71 :
                Liste_limite_polynies_latitude.append(-67.5)
                
            # Limit on the landfast ice tongue between CDP and MBP
            else :
                Liste_limite_polynies_latitude.append(List_NS[i-compteur])

        # Test of the dimension of data (only 2d no dimension for the time) : Polynya(Latitudes,Longitudes)
        if (len(forme)==2):
            S=0
            for i in range(len(Lat)):
                lat = Lat[i]
                for j in range(len(Lon)):
                    if Polynies[i,j]==1 and lat > Liste_limite_polynies_latitude[j]:
                        # Calculation of the surface for a point on the grid (depends on the latitude of the point)
                        S +=abs(R*R*np.radians(0.25)*np.radians(0.25)*np.sin(np.radians(lat)))
    
        # Test of the dimension of data (only 3d no dimension for the time) : Polynya(time,Latitudes,Longitudes)
        if (len(forme)==3):
            S=[]
            Time = dataPolynies.time.values
            for i in range(forme[0]):
                s=0
                for j in range(forme[1]):  
                    lat = Lat[j]
                    
                    for k in range(forme[2]):
                        if Polynies[i,j,k]==1 and lat > Liste_limite_polynies_latitude[k] :
                            # Calculation of the surface for a point on the grid (depends on the latitude of the point)
                            s +=abs(R*R*np.radians(0.25)*np.radians(0.25)*np.sin(np.radians(lat)))
                S.append(s)  
            S = np.array(S)
            # Creation of the dataArray of surfaces : Surfaces(Time)
            data_S = xr.DataArray(data = S,coords=dict(time=(["time"],Time)),attrs=dict(description = 'Surface polynie'))
            S = data_S

    if area == 'MBP':
        
        Polynies = sel_CapeDarnley(dataPolynies).values
        forme = Polynies.shape
        
        
        # Sélection of th polynya from a limit between CDP and MBP.
        Liste_limite_polynies_latitude = []
        List_NS = list(np.linspace(-69,-67.5,15))
        
        for i in range(len(Lon)):
            # Limit on the West of the landfast ice tongue
            if Lon[i] < 67.5 :
                Liste_limite_polynies_latitude.append(-69)
            
            # Limit on the East of the landfast ice tongue
            elif Lon[i] > 71 :
                Liste_limite_polynies_latitude.append(-67.5)
                
            # Limit on the landfast ice tongue between CDP and MBP
            else :
                Liste_limite_polynies_latitude = list(Liste_limite_polynies_latitude) + list(List_NS)
        
        # Test of the dimension of data (only 2d no dimension for the time) : Polynya(Latitudes,Longitudes)
        if (len(forme)==2):
            S=0
            for i in range(len(Lat)):
                lat = Lat[i]
                for j in range(len(Lon)):
                    if Polynies[i,j]==1 and lat < Liste_limite_polynies_latitude[j]:
                        # Calculation of the surface for a point on the grid (depends on the latitude of the point)
                        S +=abs(R*R*np.radians(0.25)*np.radians(0.25)*np.sin(np.radians(lat)))
    
        # Si forme de taille de 3 : Polynies(time,Latitudes,Longitudes)
        if (len(forme)==3):
            S=[]
            Time = dataPolynies.time.values
            for i in range(forme[0]):
                s=0
                for j in range(forme[1]):  
                    lat = Lat[j]
                    
                    for k in range(forme[2]):
                        if Polynies[i,j,k]==1 and lat < Liste_limite_polynies_latitude[k] :
                            # Calculation of the surface for a point on the grid (depends on the latitude of the point)
                            s +=abs(R*R*np.radians(0.25)*np.radians(0.25)*np.sin(np.radians(lat)))
                S.append(s)  
            S = np.array(S)
            data_S = xr.DataArray(data = S,coords=dict(time=(["time"],Time)),attrs=dict(description = 'Surface polynie'))
            S = data_S
        

    return S





# In[Calcul moyenne surfacique]


# =============================================================================
# FONCTION : meanSurface :
#
#   Calculation of the surface of the polynyas
#
#  Input :
#    - data (array - with 2 or 3 dimension) : data of interest.
#    - dataPolynyas (array - with 2 or 3 dimension) : data of the polynya index.
#    - area (str) :     Latitudes (list or array (1 d))) : list of latitudes (size of the fisrt dimension of data).
#    - Longitudes (list or array (1 d))) : list of longitudes (size of the second dimension of data).
#    - time (list or array (1 d))) : list of time (size of the last dimension of data) - None by default.
#
#  Output :
#     - dataArray (xarray) : xarray created from data (Latitudes, Longitudes,time (or not)).
#   
# =============================================================================

def meanSurface(data,dataPolynies=None):
    
    # Rayon de la Terre
    R = 6.4 *1e3
    
    # Grandeurs des coordonnées/valeurs des polynies
    Lat = data.latitude.values
    Lon = data.longitude.values
    arrayData = data.values
    forme = data.shape
    
    if (dataPolynies is None) :
        
        if (len(forme)==2):
            S=0
            M=0
            for i in range(len(Lat)):
                lat = Lat[i]
                for j in range(len(Lon)):
                    s = abs(R*R*np.radians(0.25)*np.radians(0.25)*np.sin(np.radians(lat)))
                    M += s * arrayData[i,j]
                    S += s
            M = M/S
        
        # Si forme de taille de 3 : Polynies(time,Latitudes,Longitudes)
        if (len(forme)==3):
            S=[]
            M=[]
            Time = data.time.values
            for i in range(forme[0]):
                s=0
                m=0
                for j in range(forme[1]):  
                    lat = Lat[j]
                    for k in range(forme[2]):
                        s1 =abs(R*R*np.radians(0.25)*np.radians(0.25)*np.sin(np.radians(lat)))
                        s += s1
                        m += s1 * arrayData[i,j,k]
                m = m/s
                M.append(m)
                S.append(s)  
            
            M = np.array(M)
            data_M = xr.DataArray(
                data = M,
                coords=dict(
                    ## Temps
                    time=(["time"],Time)),
                attrs=dict(description = 'Moyenne Surfacique sur la polynie')
                )
            M = data_M

        
    else :

        Polynies = dataPolynies().values
        if (len(forme)==2):
            S=0
            M=0
            for i in range(len(Lat)):
                lat = Lat[i]
                for j in range(len(Lon)):
                    if Polynies[i,j]==1:
                        s = abs(R*R*np.radians(0.25)*np.radians(0.25)*np.sin(np.radians(lat)))
                        S += s
                        M += s * arrayData[i,j]
            M = M/S
            
        # Si forme de taille de 3 : Polynies(time,Latitudes,Longitudes)
        if (len(forme)==3):
            S=[]
            M=[]
            Time = dataPolynies.time.values
            for i in range(forme[0]):
                s=0
                m=0
                for j in range(forme[1]):  
                    lat = Lat[j]
                    
                    for k in range(forme[2]):
                        if Polynies[i,j,k]==1:
                            s1 = abs(R*R*np.radians(0.25)*np.radians(0.25)*np.sin(np.radians(lat)))
                            s += s1
                m = m/s
                M.append(m)
                S.append(s)  
            M = np.array(M)
            data_M = xr.DataArray(     
                data = M,
                coords=dict(
                    ## Temps
                    time=(["time"],Time)),
                attrs=dict(description = 'Moyenne Surfacique sur la polynie')
                )
            S = data_M

    return M
    
    
    
    
    
    

# In[Calcul de corrélation]

# =============================================================================
# FONCTION : correlationDataSurfacePolynieCapeDarnley :
#
#   Calculation of the surface of the polynyas
#
#  Input :
#    - data (xarray - 3 dimensions) - data(time,Latitudes,Longitudes) : data of interest.
#    - surfacePolynya (array - 1 dimension) : data of interest.
#    - area (str) :     Latitudes (list or array (1 d))) : list of latitudes (size of the fisrt dimension of data).
#    - Longitudes (list or array (1 d))) : list of longitudes (size of the second dimension of data).
#    - time (list or array (1 d))) : list of time (size of the last dimension of data) - None by default.
#
#  Output :
#     - corrData (xarray) - corrData(Latitudes,Longitudes) : xarray 
#   
# =============================================================================


def correlationDataSurfacePolynie(data,surfacePolynie,titre = None):
    
    # Creation of the dataArray of surfaces : Surfaces(Time)
    dataSurfacePolynie = xr.DataArray(        
        ## Valeurs du tableau de données        
        data = surfacePolynie
        ,
        coords=dict(
            time=(["time"], data.time.values)
        ),
        attrs=dict(
            description = 'Surface Polynya'
            )
    )
    
    # Calculation of the correlation between data and the surfaces on each point of the grid.
    corrData = xr.corr(dataSurfacePolynie,data,dim='time')
    
    print(np.min(corrData.values),np.max(corrData.values))

    return corrData

# def correlationDataSurfacePolynieCapeDarnley_around(data,surfacePolynie,titre = None,dimension=None,levels=np.linspace(-1,1,21),stations=False):
    
#     dataSurfacePolynie = xr.DataArray(
        
#         ## Valeurs du tableau de données        
#         data = surfacePolynie
#         ,
#         coords=dict(
#             time=(["time"], data.time.values)
#         ),
#         attrs=dict(
#             description = 'Surface Polynie de Cape Darnley'
#             )
#     )


#     corrData = xr.corr(dataSurfacePolynie,data,dim='time')
#     # print('Encadrement des corrélation : [',np.min(corrData.values),',',np.max(corrData.values),'] ')
    
#     #plotMap_CapeDarnley_around(corrData, levels,titre=titre,dimension=dimension,stations=stations)
#     return corrData

# In[Calcul de RMSE]

def RMSE(data,data_mesure):
    array_data = data.values
    forme = array_data.shape
    array_rmse = np.zeros((forme[1],forme[2]))
    Latitudes = data.latitude
    Longitudes = data.longitude
    for i in range(forme[1]):
        for j in range(forme[2]):
            y_predicted = array_data[:,i,j]
            y_actual = np.array(data_mesure)
            array_rmse[i,j] = np.sqrt(np.subtract(np.square(y_actual),np.square(y_predicted))).mean()
    
    print(array_rmse)
    print(Latitudes.shape)
    print(Longitudes.shape)
    data_rmse = creationDataArray(array_rmse, Latitudes, Longitudes)
    return data_rmse

def RMSE_station(data_ERA_station,data_mesure_station):
    array_data_ERA_station = data_ERA_station.values
    array_data_mesure_station = data_mesure_station.values
    
    rmse = 0
    
    for i in range(array_data_ERA_station.shape[0]):
        rmse += (array_data_ERA_station[i] - array_data_mesure_station[i])**2
    
    return np.sqrt(rmse/array_data_ERA_station.shape[0])

# In[Calcul de régression]

def regressionLineaireDataSurfacepolynie(data,surfacePolynie,titre= None):
    Latitudes = data.latitude.values
    Longitudes = data.longitude.values
    #Time = data.time.values 
    
    Slopes = np.zeros((Latitudes.shape[0],Longitudes.shape[0]))
    R_values = np.zeros((Latitudes.shape[0],Longitudes.shape[0]))
    Intercepts = np.zeros((Latitudes.shape[0],Longitudes.shape[0]))
    for i in range(Latitudes.shape[0]):
        for j in range(Longitudes.shape[0]):
            # data_array_lat_lon = ((data.isel(latitude=i).isel(longitude=j)- data.isel(latitude=i).isel(longitude=j).mean('time'))/np.std(data.isel(latitude=i).isel(longitude=j))).values
            data_surfacePolynie= (surfacePolynie - np.mean(surfacePolynie))/np.std(surfacePolynie)
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(data_surfacePolynie,data.isel(latitude=i,longitude=j).values)
            Slopes[i,j] = slope
            Intercepts[i,j] = intercept
            R_values[i,j] = r_value
    
    data_slopes = creationDataArray(Slopes, Latitudes, Longitudes)
    data_intercept = creationDataArray(Intercepts, Latitudes, Longitudes)
    data_r = creationDataArray(R_values, Latitudes, Longitudes)
    
    
    return data_slopes,data_intercept,data_r



def predict(x,slope,intercept):
   return slope * x + intercept



# In[Détection polynie de Cape Darnley]

# def detectionPolynyaCapeDarnley(data_ci,lon_dep=69.0,lat_dep=-67.25,trace = False,CIlevels = None,CIlevels2 = np.linspace(0,1,11),niveau=70):
    
#     print('Start : Detection the Cape Darnley s polynya')
    
#     ## On sélectionne les données sur la polynie de Cape Darnley
#     ci_CD = sel_CapeDarnley(data_ci*100)
#     #plotMap_CapeDarnley(ci_CD, CIlevels, '%')
#     # CIlevels2 = np.linspace(0,1,2)
    
    
#     ## Liste des longitudes et des latitudes
#     Lat_CD = ci_CD.latitude.values
#     Lon_CD = ci_CD.longitude.values
    
#     ## On crée un array d'entier
#     forme = ci_CD.shape
#     ci_CD_int = np.zeros(forme)
    
#     if (len(forme) == 2):
#         for i in range(len(ci_CD_int)):
#             for j in range(len(ci_CD_int[0])):
#                 valeur = np.array(ci_CD)[i,j]
#                 if (not np.isnan(valeur)):
#                     ci_CD_int[i,j] = int(valeur)
#                 else :
#                     ci_CD_int[i,j] = 100
#         # filled_ci_CD = flood_fill(ci_CD_int, (i_lon,i_lat),100, tolerance = 30)*0.01
    
        
#         # Filtrage des données de CI --> conservation des CI correspondant à la polynie
#         # soit ci comprise entre 0 et 0.7
        
#         Surface_polynie = np.zeros(ci_CD_int.shape)
#         for i in range(forme[0]):
#             for j in range(forme[1]):
#                 if ((ci_CD_int[i,j] < niveau) and (ci_CD_int[i,j] >= 0)):
#                     Surface_polynie[i,j] = 1
#         # print('Nombre de points dans la polynie : ',np.sum(Surface_polynie))
#         data_Polynie = creationDataArray(Surface_polynie,Lat_CD,Lon_CD,nameDataArray='Data Polynie de Cape Darnley')
#         #print(data_Polynie)
                    
#     if (len(forme) == 3):
#         print('Arrangement du jeu de données CI')
#         for i in range(ci_CD_int.shape[0]):
#             for j in range(ci_CD_int.shape[1]):
#                 for k in range(ci_CD_int.shape[2]):
#                     valeur = np.array(ci_CD)[i,j,k]
#                     if (not np.isnan(valeur)):
#                         ci_CD_int[i,j,k] = int(valeur)
#                     else :
#                         ci_CD_int[i,j,k] = 100
                        
#         Surface_polynie = np.zeros(forme)
#         print('Nombre de pas de temps : ' + str(forme[0]))
#         for i in range(forme[0]):
#             print('Filtrage CI : ',i,'/',forme[0]-1)

#             #filled_ci_CD[i,:,:] = ci_CD_int[i,:,:]*0.01
#             # Filtrage des données de CI --> conservation des CI correspondant à la polynie
#             # soit ci comprise entre 0 et 0.7
#             for j in range(forme[1]):
#                 for k in range(forme[2]):
#                     if ((ci_CD_int[i,j,k] <= niveau) and (ci_CD_int[i,j,k] >= 0)):
#                         Surface_polynie[i,j,k] = 1
#         data_Polynie = creationDataArray(Surface_polynie,Lat_CD,Lon_CD,time=range(forme[0]),nameDataArray='Data Polynie de Cape Darnley')

#     print('Start : flood fill algorithm')
#     ## Ci remplie par l'algorithme de fill flood
#     if (len(forme) == 2):
#         # filled_ci_CD = flood_fill(ci_CD_int, (i_lon,i_lat),100, tolerance = 30)*0.01
    
#         filled_ci_CD = ci_CD_int*0.01
        
#         # Filtrage des données de CI --> conservation des CI correspondant à la polynie
#         # soit ci comprise entre 0 et 0.7
        
#         Surface_polynie = np.zeros(filled_ci_CD.shape)
#         for i in range(forme[0]):
#             for j in range(forme[1]):
#                 if ((filled_ci_CD[i,j] < niveau*0.01) and (filled_ci_CD[i,j] >= 0)):
#                     Surface_polynie[i,j] = 1
#         # print('Nombre de points dans la polynie : ',np.sum(Surface_polynie))
#         data_Polynie = creationDataArray(Surface_polynie,Lat_CD,Lon_CD,nameDataArray='Data Polynie de Cape Darnley')
#         print(data_Polynie)
#     if (len(forme) == 3):
#         filled_ci_CD = np.zeros(forme)
#         Surface_polynie = np.zeros(forme)
#         print(forme)
#         print('Nombre de pas de temps : ' + str(forme[0]))
#         for i in range(forme[0]):
#             print('Filtrage CI : ',i,'/',forme[0]-1)

#             filled_ci_CD[i,:,:] = ci_CD_int[i,:,:]*0.01
#             # Filtrage des données de CI --> conservation des CI correspondant à la polynie
#             # soit ci comprise entre 0 et 0.7
#             for j in range(forme[1]):
#                 for k in range(forme[2]):
#                     if ((filled_ci_CD[i,j,k] < niveau*0.01) and (filled_ci_CD[i,j,k] >= 0)):
#                         Surface_polynie[i,j,k] = 1
#         data_Polynie = creationDataArray(Surface_polynie,Lat_CD,Lon_CD,time=range(forme[0]),nameDataArray='Data Polynie de Cape Darnley')

#     if trace :
        
#         #plotMap_CapeDarnley(data_Surface_Polynie,CIlevels)
        
#         data_filled_ci_CD = creationDataArray(filled_ci_CD,Lat_CD,Lon_CD,nameDataArray='Data Polynie de Cape Darnley')
        
#         #plotMap_CapeDarnley(data_filled_ci_CD,CIlevels2,titre='Polynie de Cape Darnley')
        
#         if (data_filled_ci_CD.shape ==2 ):
#             plt.figure(figsize=(10,10), frameon=True)
        
#             # Projection map :
#             ax = plt.axes(projection=ccrs.Orthographic(-65,-75))
        
#             # Ajout des lignes de côtes / grid :
#             ax.coastlines(resolution="110m",linewidth=1)
#             gl = ax.gridlines(linestyle='--',color='black',
#                           draw_labels=True)
#             gl.top_labels = False
#             gl.right_labels = False
#             ''
            
#             data_filled_ci_CD.plot.contourf(ax=ax,
#                         transform=ccrs.PlateCarree(),
#                         levels=np.linspace(0,1,11),
#                         cmap='jet',
#                         add_colorbar=False
#                         );
        
#             plt.plot([lon_dep], [lat_dep],
#                       color='k', linewidth=100,
#                       marker='o',transform=ccrs.PlateCarree())
        
#             # # Création de la barre de couleur (Bonne orientation et bonne taille...) :
#             # cb = plt.colorbar(im1, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
#             # cb.set_label('dimension',size=18,rotation=90,labelpad=1)
#             # cb.ax.tick_params(labelsize=18)
        
#             # Création du titre :
#             ax.set_title('Polynie de Cape Darnley',fontsize=20)
#             #ax.set_xlabel("")
        
#             plt.show()
#         elif (data_filled_ci_CD.shape ==3):
#             plt.figure(figsize=(10,10), frameon=True)
        
#             # Projection map :
#             ax = plt.axes(projection=ccrs.Orthographic(-65,-75))
        
#             # Ajout des lignes de côtes / grid :
#             ax.coastlines(resolution="110m",linewidth=1)
#             gl = ax.gridlines(linestyle='--',color='black',
#                           draw_labels=True)
#             gl.top_labels = False
#             gl.right_labels = False
#             ''
            
#             data_filled_ci_CD[0,:,:].plot.contourf(ax=ax,
#                         transform=ccrs.PlateCarree(),
#                         levels=np.linspace(0,1,11),
#                         cmap='jet',
#                         add_colorbar=False
#                         );
        
#             plt.plot([lon_dep], [lat_dep],
#                       color='k', linewidth=100,
#                       marker='o',transform=ccrs.PlateCarree())
        
#             # # Création de la barre de couleur (Bonne orientation et bonne taille...) :
#             # cb = plt.colorbar(im1, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
#             # cb.set_label('dimension',size=18,rotation=90,labelpad=1)
#             # cb.ax.tick_params(labelsize=18)
        
#             # Création du titre :
#             ax.set_title('Polynie de Cape Darnley',fontsize=20)
#             #ax.set_xlabel("")
        
#             plt.show()
    
#     return data_Polynie

####### Selection des données sur la polynie de Cape Darnley

# def selectionPolynyaCP(data_surface_polynie,data):
#     Latitudes = data_surface_polynie.latitude.values
#     Longitudes = data_surface_polynie.longitude.values
    
#     data_selection = np.zeros(data.shape)
#     forme = data.shape
#     print(forme[0])
#     if (len(forme) == 2):
#         print('OK dans la boucle taille 2')
#         print(np.sum(data_surface_polynie.values))
#         for i in range(forme[0]):
#             for j in range(forme[1]):
#                 valeur = data.values[i,j]
#                 if (data_surface_polynie.values[i,j] == 1):
#                     data_selection[i,j] = valeur
#                     print('test egalité OK')
        
#         data_polynie = creationDataArray(data_selection, Latitudes, Longitudes,
#                                          nameDataArray="Données selectionnées (dataArray).")

#     if (len(forme) == 3):
#         for i in range(forme[0]):
#             for j in range(forme[1]):
#                 for k in range(forme[2]):
#                     valeur = data.values[i,j,k]
#                     if (data_surface_polynie.values[i,j,k] == 1):
#                         data_selection[i,j] = valeur
#         temps = data.time.values
        
#         data_polynie = creationDataArray(data_selection, Latitudes, Longitudes,
#                                          time=temps,
#                                          nameDataArray="Données selectionnées (dataArray).")
#     return data_polynie
    

# In[Algorithme FloodFill]

# =============================================================================
# FONCTION : floodFillPolynya :
#
#   Calculation of the surface of the polynyas
#
#  Input :
#    - data_ci (xarray - 2 dimensions) - data_ci(Latitudes,Longitudes) : sea ice couverture around Antarctica for 1 day.
#    - lon_dep (float) : longitude of the point of beginning of injection.
#    - lat_dep (float) : latitude of the point of beginning of injection
#    - continent (bool) : parameter to describe if you are on the continent or not (True by default).
#    - new_value (float) : new value of sea ice couverture on each point.
#    - tolerance (float) : gap of reference for the difference of value between the beginning point and the other one.
#    - trace (bool) : parameter to describe if you draw maps or not during the process.
#    - reanalysis (str) : type of reanalysis for the data.
#
#  Output :
#     - data_filled_ci (xarray) - data_filled_ci(Latitudes,Longitudes) : xarray of sea ice couverture after using the flood fill algorithm
#   
# =============================================================================

def floodFillPolynya(data_ci,lon_dep,lat_dep,ocean = True,new_value= 100,tolerance = 30 ,trace = False,CIlevels = None,CIlevels2 = None,reanalysis='ERA'):
    
    # Multiplication of the parameter ci by 100 to work with int (%)
    ci = data_ci*100
    
    # Loading the useful data 
    if reanalysis == 'ERA':
        Lat = ci.latitude.values
        Lon = ci.longitude.values
        
    if reanalysis == 'MERRA':
        Lat = ci.lat.values
        Lon = ci.lon.values
    
    # Creation of the output array
    ci_int = np.zeros(ci.shape)
    
    # Test if you are on the continent or not (NB : in ERA5, on the continent ci = Nan).
    for i in range(len(ci_int)):
        for j in range(len(ci_int[0])):
            valeur = np.array(ci)[i,j]
            
            if (not np.isnan(valeur)):
                ci_int[i,j] = int(valeur)
            else :
                if ocean :
                    ci_int[i,j] = valeur
                else :
                    ci_int[i,j] = 0
    
    # Dtermination of the coordinates of the starting point
    i_lon = 0
    i_lat = 0
    for i in range(len(Lat)):
        if (Lat[i] == lat_dep):
            i_lat = i
    for i in range(len(Lon)):
        if (Lon[i] == lon_dep):
            i_lon = i
    
    # We apply the daughter flood algorithm with the following starting point (latitude,longitude)
    # we set a new value and a tolerance
    filled_ci = flood_fill(ci_int,(i_lat,i_lon),new_value, tolerance = tolerance)*0.01

    # Creation of a xarray of the data
    data_filled_ci = creationDataArray(filled_ci,Lat,Lon,nameDataArray="Filled ci.")

    if trace :
        plotMap_Ant(data_filled_ci,CIlevels)
    
    return data_filled_ci 
    

# In[Detections Polynies - Antarctique]

# =============================================================================
# FONCTION : detectionPolynya :
#
#   Detection of the polynyas in Antractica from a map of sea ice couverture (ci)
#
#  Input :
#    - ci_antarctique (xarray - 2 dimensions) - ci_antarctique(Latitudes,Longitudes) : sea ice couverture around Antarctica for 1 day.
#    - ci_limit (float in [0,1]) : limit concentration of sea ice below which we consider there is polynyas
#    - trace (bool) : boolean allowing to draw the maps of the algorithm progress
#    - reanalysis (str) : precision on the type of reanalysis used (difference of the name of parameters).
#
#  Output :
#     - final_polynyas (xarray) - final_polynyas(Latitudes,Longitudes) : xarray of polynya index (1 if you are in a polynya, 0 else)
#   
# =============================================================================

def detectionPolynya(ci_antarctique, ci_limit = 0.7, trace=False, reanalysis='ERA'):
    Levels = np.linspace(0,100,21)
    CIlevels = np.linspace(0,1,11)
    # CIlevels2 = np.linspace(0,1,21)
    
    # Loading of the 
    if reanalysis == 'ERA':
        
        Latitudes = ci_antarctique.latitude.values
        Longitudes = ci_antarctique.longitude.values
        
    if reanalysis == 'MERRA':
        
        Latitudes = ci_antarctique.lat.values
        Longitudes = ci_antarctique.lon.values
    
    # Draw the originla map of the sea ice cover
    if trace :
        plotMap_Ant(ci_antarctique,dimension="%",color='jet',titre='Carte entrée')#,titre='Map CI Antarctica - couverture normale')

    # Application of the floodFillPolynya to the original map of sea ice couverture, 
    # from the starting point : (longitude,latitude) = (240,-60) (in a n area of open water even if in winter)
    # We fix the tolerance at 100*ci_limit
    filled_ci_2014_Sep_m1 = floodFillPolynya(ci_antarctique,
                                             240,
                                             -60,
                                             trace=False,
                                             tolerance=ci_limit*100,
                                             CIlevels= Levels,
                                             reanalysis=reanalysis)

    # Draw the originla map of the sea ice cover
    if trace :
        plotMap_Ant(filled_ci_2014_Sep_m1, dimension="%",color='jet')#,titre='Map CI Antarctica - couverture glace en mer')
    
    # Application of the filter (depends on ci_limit) to determine the polynyas (coastal and open water)
    Polinies = np.zeros(filled_ci_2014_Sep_m1.values.shape)
    for i in range(Polinies.shape[0]):
        for j in range(Polinies.shape[1]):
            
            # if the sea ice couverture is less than the limit --> It is a polynya
            if ((filled_ci_2014_Sep_m1.values[i,j] <ci_limit) and (filled_ci_2014_Sep_m1.values[i,j] >= 0)):
                Polinies[i,j] = 1
    
    # Creation of a dataArray of the polynyas detected (coastal and OWP)
    data_polinies = creationDataArray(Polinies, Latitudes, Longitudes, nameDataArray="Filled ci.")
    
    if trace :
        plotMap_Ant(data_polinies,CIlevels,color='binary',color_bar=False)#,titre='Map CI Antarctica - Polynies et OWC')
    
    # Application of the floodFillPolynya to the original map of sea ice couverture, 
    # from the starting point : (longitude,latitude) = (90,-89) (on the continent)
    # Here, we fill the continent and the coastl polynya with sea ice
    filled_ci_2014_Sep_m2 = floodFillPolynya(filled_ci_2014_Sep_m1,
                                             90,
                                             -89,
                                             new_value=100,
                                             ocean = False,
                                             trace=False,
                                             tolerance=ci_limit*100,
                                             CIlevels2= CIlevels)
    
    # Application of the filter (depends on ci_limit) to determine the Open Water Polynyas
    OWP = np.zeros(filled_ci_2014_Sep_m1.values.shape)
    for i in range(OWP.shape[0]):
        for j in range(OWP.shape[1]):
            if ((filled_ci_2014_Sep_m2.values[i,j] < ci_limit) and (filled_ci_2014_Sep_m2.values[i,j] >= 0)):
                OWP[i,j] = 1
    
    # Creation of a dataArray of the polynyas detected (only OWP)
    data_OWP = creationDataArray(OWP, Latitudes, Longitudes,nameDataArray="Filled ci.")
    
    # Difference of the data of all polynya and of the Open Water Polynya to determine the coastal polynyas only
    final_polynyas = data_polinies - data_OWP
    
    if trace :
        plotMap_Ant(final_polynyas,CIlevels,color='binary',color_bar=False)#,titre='Map CI Antarctica - polynies seules')

    print('Surface des polynies = ',calculSurfacePolynies(sel_CapeDarnley(final_polynyas),area='CD'), ' km2')
    
    # Return the index of costal polynyas --> give the positions of polynyas around the Antarctica
    return final_polynyas




# In[Detection trait de Côte]

# def cote(List_latitudes):
#     L_test = list(np.isnan(List_latitudes))
#     for i in range(len(L_test)-1):
#         if L_test[i] != L_test[i+1]:
#             return i

# def detectionCote(data):
#     latitudes = data.latitude.values
#     longitudes = data.longitude.values
    
#     if len(data.shape) == 3:
#         array_data = data.isel(time=0).mean('time').values
#     else :
#         array_data = data.values
#     Latitudes_cotes = []
#     for i in range(len(longitudes)):
#         i_lat_cote = cote(list(array_data[:,i]))
#         Latitudes_cotes.append(latitudes[i_lat_cote])
    
#     return Latitudes_cotes, longitudes

# def angleTopographieCote(List_latitudes,longitudes):
#     R = 6.4 *1e3
#     N = len(longitudes)
#     angles =[]
#     for i in range(N-1):
#         latitude = List_latitudes[i]
#         r = R*np.sin(latitude)
#         dx = 2*np.pi*r/N
#         dy = radians(abs(latitude-List_latitudes[i+1])) * R
#         angle_i = degrees(np.arctan(dy/dx))
#         angles.append(angle_i)
    
#     latitude = List_latitudes[N-1]
#     r = R*np.sin(latitude)
#     dx = 2*np.pi*r/N
#     dy = radians(abs(latitude-List_latitudes[0])) * R
#     angle_i = degrees(np.arctan(dy/dx))
#     angles.append(angle_i)
    
#     return angles

# In[Fonction autre]

# # def test_format_liste_dates(liste):
# #     liste_sortie = []
# #     for x in liste :
# #         liste_sortie.append(np.datetime64(test_format_date(x)))
# #     return sortie


# def test_format_date(entier):
#     sortie = ''
#     if entier < 10 :
#         sortie = '0'+str(entier)
#     else :
#         sortie = str(entier)
#     return sortie

# def dataArray_station(Data,time,nameDataArray='Data Array'):
    
#     data_DataArray = xr.DataArray(
        
#         ## Valeurs du tableau de données        
#         data = Data
#         ,
#         coords=dict(
#             time=(["time"], time)
#         ),
#         attrs=dict(
#             description = nameDataArray
#             )
#     )
#     return data_DataArray

# In[Selection données à partir d'une liste de temps]

# def selection_data_ListTime(data,List_time):
#     sel_data = []
#     for x in List_time:
#         sel_data.append(data.sel(time=x))#.values)
#     return sel_data #xr.merge(sel_data)
