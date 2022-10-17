#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:10:18 2022

@author: Matthias NOËL
"""

# In[Chargement bibliothèque] : Chargement bibliothèque
import numpy as np
#import netCDF4 as nc
import xarray as xr
# import seaborn as sn
# import rioxarray as rxr
import matplotlib.pyplot as plt
from scipy.stats import linregress # ,pearsonr

import math as m

# import cartopy.crs as ccrs
# import matplotlib.path as mpath
# from cartopy.util import add_cyclic_point
# from matplotlib import cm
import pandas as pd
# import glob
import ToolsPolynias as tp
import MapDrawing as md
# from scipy.stats import pearsonr

# In[Données de Mawson]
aws_dir = '/gpfsstore/rech/omr/romr008/DATA/AWS/AUS_stations'
database = 'Mawson_1980_2016.txt'
stations = 'Mawson'

aws_stats = {} 

# Get files
database_folder = aws_dir+"/"+database
# datafiles = glob.glob(database_folder+"/*Data*")
# stationnumbers = glob.glob(database_folder+"/*StnDet*")

data_Mawson = pd.read_csv(database_folder, sep='\s+', names=["id", "number", "date", "direction",
                                                             "vitesse", "GUS", "CLG", "SKC", "L",
                               "M", "H",  "VSB", "MW1", "MW2", "MW3", "MW4", "AW1", "AW2", "AW3", "AW4",
                               "W","TEMP", "DEWP","SLP","ALT", "STP", "MAX", "MIN", "PCP01", "PCP06",
                               "PCP24", "PCPXX", "SD"])


##### Extraction des vitesses de la station MAWSON #####
SPD_maw = list(data_Mawson.vitesse)
np.array(SPD_maw.pop(0))
for i in range(len(SPD_maw)):
    speed = SPD_maw[i]
    if (speed == '***'):
        speed = -1
        
    else :
        speed = float(SPD_maw[i])/3.6#/0.621371 *1/3.6
    SPD_maw[i] = speed

date_maw = list(data_Mawson.date)
np.array(date_maw.pop(0))
for i in range(len(date_maw)):
    date = str(date_maw[i])
    year,month,day,hour,minute = int(date[:4]),int(date[4:6]),int(date[6:8]),int(date[8:10]),int(date[10:])
    
    mois = tp.test_format_date(month)
    jour = tp.test_format_date(day)
    heure = tp.test_format_date(hour)
    minu = tp.test_format_date(minute)
    
    date_maw[i] = np.datetime64(str(year)+'-'+mois+'-'+jour+'T'+heure+':' + minu)
    
data_U_maw = xr.DataArray(
    
    ## Valeurs du tableau de données        
    data = SPD_maw
    ,
    coords=dict(
        time=(["time"], date_maw)
    ),
    attrs=dict(
        description = 'Data mesurée Mawson'
        )
)

# In[Données de Davis]
aws_dir = '/gpfsstore/rech/omr/romr008/DATA/AWS/AUS_stations'
database = 'Davis_1980_2016.txt'
stations = 'Davis'

aws_stats = {} 

# Get files
database_folder = aws_dir+"/"+database
# datafiles = glob.glob(database_folder+"/*Data*")
# stationnumbers = glob.glob(database_folder+"/*StnDet*")

data_Davis = pd.read_csv(database_folder, sep='\s+', names=["id", "number", "date", "direction",
                                                             "vitesse", "GUS", "CLG", "SKC", "L",
                               "M", "H",  "VSB", "MW1", "MW2", "MW3", "MW4", "AW1", "AW2", "AW3", "AW4",
                               "W","TEMP", "DEWP","SLP","ALT", "STP", "MAX", "MIN", "PCP01", "PCP06",
                               "PCP24", "PCPXX", "SD"])

##### Extraction des vitesses de la station DAVIS #####

SPD_dav = list(data_Davis.vitesse)
np.array(SPD_dav.pop(0))
for i in range(len(SPD_dav)):
    speed = SPD_dav[i]
    if (speed == '***'):
        speed = -1
    else :
        speed = float(SPD_dav[i])/0.621371 *1/3.6
    SPD_dav[i] = speed
    
date_dav = list(data_Davis.date)
np.array(date_dav.pop(0))
for i in range(len(date_dav)):
    date = str(date_dav[i])
    #print(type(date))
    year,month,day,hour,minute = int(date[:4]),int(date[4:6]),int(date[6:8]),int(date[8:10]),int(date[10:])
    
    mois = tp.test_format_date(month)
    jour = tp.test_format_date(day)
    heure = tp.test_format_date(hour)
    minu = tp.test_format_date(minute)
    
    date_dav[i] = np.datetime64(str(year)+'-'+mois+'-'+jour+'T'+heure+':' + minu)

# In[Moyenne jour à jour]

# date_min_maw = 20295
# date_min_dav = 20280

################ DAVIS  ################
Date_dav = [np.datetime64(str(date_dav[0])[:10])]
compteur = 1 
mean = SPD_dav[0]
U_dav = []
for i in range(1,len(date_dav)) :
    date = np.datetime64(str(date_dav[i])[:10])

    if date != Date_dav[-1]:
        Date_dav.append(date)
        U_dav.append(mean/compteur)
        
        if SPD_dav[i] != -1 :
            mean = SPD_dav[i]
        else :
            mean = 0
            
        compteur = 1
    else :
        if SPD_dav[i] != -1 :
            mean += SPD_dav[i]
            compteur += 1
        
U_dav.append(mean/compteur)   
data_U_dav = tp.dataArray_station(U_dav,Date_dav).sel(time=slice('2010-01-01','2017-01-01'))

plt.figure()
data_U_dav.plot()

################  MAWSON  ################
Date_maw = [np.datetime64(str(date_maw[0])[:10])]
compteur = 1 
mean = SPD_maw[0]
U_maw = []
for i in range(1,len(date_maw)) :
    date = np.datetime64(str(date_maw[i])[:10])
    
    if date != Date_maw[-1]:
        Date_maw.append(date)
        U_maw.append(mean/compteur)
        
        mean = SPD_maw[i]
        compteur = 1
    else :
        mean += SPD_maw[i]
        compteur += 1
        
U_maw.append(mean/compteur) 
    
data_U_maw = tp.dataArray_station(U_maw,Date_maw).sel(time=slice('2010-01-01','2017-01-01'))

plt.figure(figsize=(1.5*6.4,1.5*4.8))

data_U_maw.sel(time=slice('2010-01-01','2017-01-01')).plot()

# In[Importation des données de ERA5 ]

u10_ERA_maw = tp.dataYearsMonths('u10',List_Years=range(2010,2021),List_Months=range(1,13),area='Mawson')
v10_ERA_maw = tp.dataYearsMonths('v10',List_Years=range(2010,2021),List_Months=range(1,13),area='Mawson')
U10_ERA_maw = np.sqrt(u10_ERA_maw*u10_ERA_maw + v10_ERA_maw*v10_ERA_maw).sel(time=slice('2010-01-01','2017-01-01'))

u10_ERA_dav = tp.dataYearsMonths('u10',List_Years=range(2010,2021),List_Months=range(1,13),area='Davis')
v10_ERA_dav = tp.dataYearsMonths('v10',List_Years=range(2010,2021),List_Months=range(1,13),area='Davis')
U10_ERA_dav = np.sqrt(u10_ERA_dav*u10_ERA_dav + v10_ERA_dav*v10_ERA_dav).sel(time=slice('2010-01-01','2017-01-01'))

u10_ERA_CD_around = tp.dataYearsMonths('u10',List_Years=range(2010,2021),List_Months=range(1,13),area='Cape Darnley Around')
v10_ERA_CD_around = tp.dataYearsMonths('v10',List_Years=range(2010,2021),List_Months=range(1,13),area='Cape Darnley Around')
U10_ERA_CD_around = np.sqrt(u10_ERA_CD_around*u10_ERA_CD_around + v10_ERA_CD_around*v10_ERA_CD_around).sel(time=slice('2010-01-01','2017-01-01'))

# In[Tracés ERA/AWS]
plt.figure()
data_U_maw.sel(time=slice('2010-05-01','2010-10-01')).plot()
U10_ERA_maw.sel(time=slice('2010-05-01','2010-10-01')).plot()

time = U10_ERA_maw.sel(time=slice('2010-05-01','2010-10-01')).time.data

# r = data_U_maw.sel(time=slice('2010-05-01','2010-10-01')).data/U10_ERA_maw.sel(time=slice('2010-05-01','2010-10-01')).data
# plt.plot(time,r)

# In[Tracés ERA/AWS DAVIS]

time = U10_ERA_dav.sel(time=slice('2010-05-01','2010-10-01')).time.data
time_U_data = [str(x)[:10] for x in time]

plt.figure(figsize=(1.5*6.4,1.5*4.8))

plt.plot(time,data_U_dav.sel(time=time_U_data),'b--',label='Mesures')
plt.plot(time,U10_ERA_dav.sel(time=time),'k',label='ERA5')

plt.legend(fontsize=20)
plt.xlim(time[0],time[-1])
plt.ylim(0,24)


plt.xlabel(r"time",size=20)
plt.ylabel(r"$m.s^{-1}$",size=20)

plt.tick_params(labelsize=18)

plt.title('DAVIS',size=20)

plt.grid()

plt.show()



# In[Tracés ERA/AWS MAWSON]

time = U10_ERA_maw.sel(time=slice('2010-05-01','2010-10-01')).time.data
time_U_data = [str(x)[:10] for x in time]

plt.figure(figsize=(1.5*6.4,1.5*4.8))

plt.plot(time,data_U_maw.sel(time=time_U_data),'b--',label='Mesures')
plt.plot(time,U10_ERA_maw.sel(time=time),'k',label='ERA5')

plt.legend(fontsize=20)
plt.xlim(time[0],time[-1])
plt.ylim(0,24)


plt.xlabel(r"time",size=20)
plt.ylabel(r"$m.s^{-1}$",size=20)

plt.tick_params(labelsize=18)

plt.title('MAWSON',size=20)

plt.grid()

plt.show()

# In[Tracés ERA/AWS]

fig = plt.figure(constrained_layout=True, figsize=(2*1.5*6.4, 1*1.5*4.8))
axs = fig.subplots(2, 1, sharex=True)



axs[0].plot(time,data_U_maw.sel(time=time_U_data),'k',label='Mesures')
axs[0].plot(time,U10_ERA_maw.sel(time=time),'b--',label='ERA5')

axs[0].legend(fontsize=20)
axs[0].set_xlim(time[0],time[-1])
axs[0].set_ylim(0,24)


# axs[0].set_xlabel(r"time",size=20)
axs[0].set_ylabel(r"$m.s^{-1}$",size=20)

axs[0].tick_params(labelsize=18)

axs[0].set_title('MAWSON',size=20)

axs[0].grid()

##### DAVIS #####

axs[1].plot(time,data_U_dav.sel(time=time_U_data),'k',label='Mesures')
axs[1].plot(time,U10_ERA_dav.sel(time=time),'b--',label='ERA5')

axs[1].legend(fontsize=20)
axs[1].set_xlim(time[0],time[-1])
axs[1].set_ylim(0,24)


axs[1].set_xlabel(r"time",size=21)
axs[1].set_ylabel(r"$m.s^{-1}$",size=21)

axs[1].tick_params(labelsize=18)

axs[1].set_title('DAVIS',size=21)

axs[1].grid()

plt.show()


# r = data_U_dav.sel(time=slice('2010-05-01','2010-10-01')).data/U10_ERA_dav.sel(time=slice('2010-05-01','2010-10-01')).data
# plt.plot(time,r)

# In[Travaux sur les données MAWSON]

# dates = []
# data_U10 = []

# times = U10_ERA_CD_around.time.values
# Latitudes = U10_ERA_CD_around.latitude.values
# Longitudes = U10_ERA_CD_around.longitude.values

# times_maw = data_U_maw.time.values
# array_U10_ERA_maw = np.zeros((len(times_maw),len(Latitudes),len(Longitudes)))

# times_modif_maw = []

# for i in range(len(times_maw)):
#     array_U10_ERA_maw[i,:,:] = U10_ERA_CD_around.sel(time = str(times_maw[i])[:10]).mean('time')
#     times_modif_maw.append(times_maw[i])


# U10_ERA_CD_around_1 = tp.creationDataArray(np.array(array_U10_ERA_maw), Latitudes, Longitudes,time=times_modif_maw)

# corr_maw_U10 = tp.correlationDataSurfacePolynieCapeDarnley_around(U10_ERA_CD_around_1, 1.12 * data_U_maw,titre='Corrélation entre vitesse mesurée à Mawson et ERA5',stations=True)

# array_rmse_maw_U10 = np.zeros((len(Latitudes),len(Longitudes)))

# for i in range(len(Latitudes)):
#     for j in range(len(Longitudes)):
#         y_predicted = list(U10_ERA_CD_around_1.values[:,i,j])
#         y_actual = list(1.12 * data_U_maw.values)
        
#         somme = 0
#         n = len(y_actual)
#         compteur = 0
#         for k in range(n):
#             terme_calcul = (y_actual[k] - y_predicted[k])**2
#             if not(m.isnan(terme_calcul)):
#                 somme += terme_calcul
#                 compteur += 1

        
#         array_rmse_maw_U10[i,j] = np.sqrt(somme/n)


# data_RMSE_maw_U10 = tp.creationDataArray(array_rmse_maw_U10, Latitudes, Longitudes)

# print(data_RMSE_maw_U10.sel(latitude=-67.5).sel(longitude=63))

# md.plotMap_CapeDarnley_around(data_RMSE_maw_U10, titre='RMSE observation / ERA5 MAWSON',stations=True)



# ##############################

dates = []
data_T = []

times = U10_ERA_CD_around.time.values
Latitudes = U10_ERA_CD_around.latitude.values
Longitudes = U10_ERA_CD_around.longitude.values

times_maw = data_U_maw.time.values
array_T_ERA_maw = np.zeros((len(times_maw),len(Latitudes),len(Longitudes)))

times_modif_maw = []

for i in range(len(times_maw)):
    array_T_ERA_maw[i,:,:] = U10_ERA_CD_around.sel(time = str(times_maw[i])[:10]).mean('time') 
    times_modif_maw.append(times_maw[i])


U10_ERA_CD_around_1 = tp.creationDataArray(np.array(array_T_ERA_maw), Latitudes, Longitudes,time=times_modif_maw)

corr_maw_U = tp.correlationDataSurfacePolynieCapeDarnley_around(U10_ERA_CD_around_1, data_U_maw,titre='Corrélation entre temperature mesurée à Mawson et ERA5',stations=True)

array_rmse_maw_U = np.zeros((len(Latitudes),len(Longitudes)))



for i in range(len(Latitudes)):
    for j in range(len(Longitudes)):
        y_predicted = list(U10_ERA_CD_around_1.values[:,i,j])
        y_actual = list(data_U_maw.values)
        
        
        somme = 0
        n = len(y_actual)
        compteur = 0
        for k in range(n):
            terme_calcul = (y_actual[k] - y_predicted[k])**2
            if not(m.isnan(terme_calcul)):
                somme += terme_calcul
                compteur += 1

        array_rmse_maw_U[i,j] = np.sqrt(somme/n)


data_RMSE_maw_U = tp.creationDataArray(array_rmse_maw_U, Latitudes, Longitudes)

print(data_RMSE_maw_U.sel(latitude=-67.5).sel(longitude=63))



# In[Travaux sur les données DAVIS]

# dates = []
# data_U10 = []

# times = U10_ERA_CD_around.time.values
# Latitudes = U10_ERA_CD_around.latitude.values
# Longitudes = U10_ERA_CD_around.longitude.values

# times_dav = data_U_dav.time.values
# array_U10_ERA_dav = np.zeros((len(times_dav),len(Latitudes),len(Longitudes)))

# times_modif_dav = []


# for i in range(len(times_dav)):
#     array_U10_ERA_dav[i,:,:] = U10_ERA_CD_around.sel(time = str(times_dav[i])[:10]).mean('time')
#     times_modif_dav.append(times_dav[i])


# U10_ERA_CD_around_1 = tp.creationDataArray(np.array(array_U10_ERA_dav), Latitudes, Longitudes,time=times_modif_dav)

# corr_dav_U10 = tp.correlationDataSurfacePolynieCapeDarnley_around(U10_ERA_CD_around_1, 1.12 * data_U_dav,titre='Corrélation entre vitesse mesurée à DAVIS et ERA5',stations=True,levels=np.linspace(-1,1,11))

# array_rmse_dav_U10 = np.zeros((len(Latitudes),len(Longitudes)))

# for i in range(len(Latitudes)):
#     for j in range(len(Longitudes)):
#         y_predicted = U10_ERA_CD_around_1.values[:,i,j]
#         y_actual = 1.12 * data_U_dav.values
        
#         somme = 0
#         n = len(y_actual)
#         compteur = 0
#         for k in range(n):
#             terme_calcul = (y_actual[k] - y_predicted[k])**2
#             if not(m.isnan(terme_calcul)):
#                 somme += terme_calcul
#                 compteur += 1

        
#         array_rmse_dav_U10[i,j] = np.sqrt(somme/n)


# data_RMSE_dav_U10 = tp.creationDataArray(array_rmse_dav_U10, Latitudes, Longitudes)

# md.plotMap_CapeDarnley_around(data_RMSE_dav_U10, titre='RMSE observation / ERA5 DAVIS',stations=True)

# print(data_RMSE_dav_U10.sel(latitude=-68.5).sel(longitude=78))


#############################

dates = []
data_U = []

times = U10_ERA_CD_around.time.values
Latitudes = U10_ERA_CD_around.latitude.values
Longitudes = U10_ERA_CD_around.longitude.values

times_dav = data_U_dav.time.values
array_T_ERA_dav = np.zeros((len(times_dav),len(Latitudes),len(Longitudes)))

times_modif_dav = []


for i in range(len(times_dav)):
    array_T_ERA_dav[i,:,:] = U10_ERA_CD_around.sel(time = str(times_dav[i])[:10]).mean('time')
    times_modif_dav.append(times_dav[i])


U10_ERA_CD_around_1 = tp.creationDataArray(np.array(array_T_ERA_dav), Latitudes, Longitudes,time=times_modif_dav)

corr_dav_U = tp.correlationDataSurfacePolynieCapeDarnley_around(U10_ERA_CD_around_1, data_U_dav,titre='Corrélation entre température mesurée à DAVIS et ERA5',stations=True,levels=np.linspace(-1,1,11))

array_rmse_dav_U = np.zeros((len(Latitudes),len(Longitudes)))

for i in range(len(Latitudes)):
    for j in range(len(Longitudes)):
        y_predicted = U10_ERA_CD_around_1.values[:,i,j]
        y_actual = data_U_dav.values
        
        somme = 0
        n = len(y_actual)
        compteur = 0
        for k in range(n):
            terme_calcul = (y_actual[k] - y_predicted[k])**2
            if not(m.isnan(terme_calcul)):
                somme += terme_calcul
                compteur += 1

        array_rmse_dav_U[i,j] = np.sqrt(somme/n)

data_RMSE_dav_U = tp.creationDataArray(array_rmse_dav_U, Latitudes, Longitudes)

# md.plotMap_CapeDarnley_around(data_RMSE_dav_T,titre='RMSE observation / ERA5 DAVIS')

print(data_RMSE_dav_U.sel(latitude=-68.5).sel(longitude=78))



# In[Arrangement et choix de données Vitesse à Davis]

dav_lat ,dav_lon = -68.5, 78
data_U_dav_ERA5 = U10_ERA_CD_around.sel(latitude=dav_lat).sel(longitude=dav_lon)

time_U_dav_ERA5 = np.array([str(x)[:10] for x in data_U_dav_ERA5.time.values])
time_U_dav = np.array([str(x)[:10] for x in data_U_dav.time.values])

time_all = []
for x in time_U_dav_ERA5 :
    if x in time_U_dav :
        time_all.append(str(x))
time_all = np.array(time_all)

array_U_dav_ERA5 = np.zeros(time_all.shape)
for i in range(time_all.shape[0]):
    x = time_all[i]
    array_U_dav_ERA5[i] = data_U_dav_ERA5.sel(time=x)
    
array_U_dav = np.zeros(time_all.shape)
for i in range(time_all.shape[0]):
    x = time_all[i]
    array_U_dav[i] = data_U_dav.sel(time=x)

# In[Calcul de la régression de la température à Davis]
regress_dav =  linregress(array_U_dav_ERA5,array_U_dav)
fit = np.polyfit(array_U_dav,array_U_dav_ERA5,1)

# fit = np.polyfit(1.12 *data_U_maw.values,data_U_maw_ERA5,1)

data_U_dav_fit = [fit[0]*x + fit[1] for x in data_U_dav]

plt.figure(figsize=(1.5*6.4, 1.5*4.8), frameon=True)
taille_ecriture = 20

plt.scatter(array_U_dav,array_U_dav_ERA5,marker='+',color='r')
plt.plot(data_U_dav.values,data_U_dav_fit,color='k',linewidth=4,label='Linear regression')
plt.plot(data_U_dav.values,data_U_dav.values,'b:',linewidth=4,label=r"$y=x$")

plt.xlim(np.min(array_U_dav),np.max(array_U_dav))
plt.ylim(np.min(data_U_dav.values),np.max(data_U_dav.values))

plt.xlabel(r"Observation ($m.s^{-1}$)",size=taille_ecriture)
plt.ylabel(r"ERA5 ($m.s^{-1}$)",size=taille_ecriture)

r_dav = str(1e-3 * int(1000 * regress_dav.rvalue))
RMSE_dav = str(1e-3 * int(1000 * (data_RMSE_dav_U.sel(latitude=dav_lat).sel(longitude=dav_lon).values)))

plt.text(16, 7, "$r=$"+str(r_dav)+" \n$RMSE=$"+str(RMSE_dav)+r" $m.s^{-1}$",
          horizontalalignment='left',
          fontsize =taille_ecriture,
          color = 'k',
          
          
         bbox=dict(facecolor='white'))


plt.tick_params(axis = 'both', labelsize = taille_ecriture)
plt.grid()
plt.legend(fontsize=taille_ecriture)
plt.show()

print('pente = ',regress_dav.slope)
print('ordonnée à l origine =', regress_dav.intercept)
print('Corrélation = ',regress_dav.rvalue)
print('$R^2 =$ ',regress_dav.rvalue**2)



# In[Arrangement et choix de données température à Mawson]

maw_lat ,maw_lon = -67.5, 63


data_U_maw_ERA5 = U10_ERA_CD_around.sel(latitude=maw_lat).sel(longitude=maw_lon)

time_U_maw_ERA5 = np.array([str(x)[:10] for x in data_U_maw_ERA5.time.values])
time_U_maw = np.array([str(x)[:10] for x in data_U_maw.time.values])

time_all = []
for x in time_U_maw_ERA5 :
    if x in time_U_maw :
        time_all.append(str(x))
time_all = np.array(time_all)

array_U_maw_ERA5 = np.zeros(time_all.shape)
for i in range(time_all.shape[0]):
    x = time_all[i]
    array_U_maw_ERA5[i] = data_U_maw_ERA5.sel(time=x)
    
array_U_maw = np.zeros(time_all.shape)
for i in range(time_all.shape[0]):
    x = time_all[i]
    array_U_maw[i] = data_U_maw.sel(time=x)

# In[Calcul de la régression de la température à Mawson]
regress_maw =  linregress(array_U_maw_ERA5,array_U_maw)
fit = np.polyfit(array_U_maw,array_U_maw_ERA5,1)

# fit = np.polyfit(1.12 *data_U_maw.values,data_U_maw_ERA5,1)

data_U_maw_fit = [fit[0]*x + fit[1] for x in data_U_maw]

plt.figure(figsize=(1.5*6.4, 1.5*4.8), frameon=True)
taille_ecriture = 20

plt.scatter(array_U_maw,array_U_maw_ERA5,marker='+',color='r')
plt.plot(data_U_maw.values,data_U_maw_fit,color='k',linewidth=4,label='Linear regression')
plt.plot(data_U_maw.values,data_U_maw.values,'b:',linewidth=4,label=r"$y=x$")

plt.xlim(np.min(array_U_maw),np.max(array_U_maw))
plt.ylim(np.min(data_U_maw.values),np.max(data_U_maw.values))

plt.xlabel(r"Observation ($m.s^{-1}$)",size=taille_ecriture)
plt.ylabel(r"ERA5 ($m.s^{-1}$)",size=taille_ecriture)

r_maw = str(1e-3 * int(1000 * regress_maw.rvalue))

RMSE_maw = str(1e-3 * int(1000 * (data_RMSE_maw_U.sel(latitude=maw_lat).sel(longitude=maw_lon).values)))

plt.text(22, 6, "$r=$"+str(r_maw)+" \n$RMSE="+str(RMSE_maw)+r"m.s^{-1}$",
          horizontalalignment='left',
          fontsize =taille_ecriture,
          color = 'k',
          bbox=dict(facecolor='white'))
# plt.text(-5, -30, '$RMSE=$'+str(2.097),
#           horizontalalignment='left',
#           fontsize =taille_ecriture,
#           color = 'k',
#          bbox=dict(facecolor='white'))

plt.tick_params(axis = 'both', labelsize = taille_ecriture)
plt.grid()
plt.legend(fontsize=taille_ecriture)
plt.show()

print('pente = ',regress_maw.slope)
print('ordonnée à l origine =', regress_maw.intercept)
print('Corrélation = ',regress_maw.rvalue)
print('$R^2 =$ ',regress_maw.rvalue**2)


# In[Tracé des deux données de stations]



taille_ecriture = 20
taille_formule = 15

# plt.figure(figsize=(1*6.4, 4*4.8), frameon=True)

fig = plt.figure(constrained_layout=True, figsize=(1*1.5*6.4, 2*1.5*4.8))
axs = fig.subplots(2, 1, sharex=True)

###  DAVIS  ###

axs[0].set_title('DAVIS',fontsize = taille_ecriture)

axs[0].scatter(array_U_dav,array_U_dav_ERA5,marker='+',color='r')
axs[0].plot(data_U_dav.values,data_U_dav_fit,color='k',linewidth=4,label='Linear regression')
axs[0].plot(data_U_dav.values,data_U_dav.values,'b:',linewidth=4,label=r"$y=x$")

axs[0].legend(fontsize=taille_ecriture)

axs[0].set_xlim(np.min(array_U_dav),np.max(array_U_dav))
axs[0].set_ylim(np.min(data_U_dav.values),np.max(data_U_dav.values))

axs[0].set_xlabel(r"Observation ($m.s^{-1}$)",size=taille_ecriture)
axs[0].set_ylabel(r"ERA5 ($m.s^{-1}$)",size=taille_ecriture)

r_dav = str(1e-3 * int(1000 * regress_dav.rvalue))
RMSE_dav = str(1e-3 * int(1000 * (data_RMSE_dav_U.sel(latitude=dav_lat).sel(longitude=dav_lon).values)))


axs[0].text(14, 5, "$r=$"+str(r_dav)+" \n$RMSE=$"+str(RMSE_dav)+r" $m.s^{-1}$",
          horizontalalignment='left',
          fontsize =taille_ecriture,
          color = 'k',
         bbox=dict(facecolor='white'))

axs[0].grid()
axs[0].tick_params(axis = 'both', labelsize = taille_ecriture)

### MAWSON ###

axs[1].set_title('MAWSON',fontsize = taille_ecriture)

axs[1].scatter(array_U_maw,array_U_maw_ERA5,marker='+',color='r')
axs[1].plot(data_U_maw.values,data_U_maw_fit,color='k',linewidth=4,label='Linear regression')
axs[1].plot(data_U_maw.values,data_U_maw.values,'b:',linewidth=4,label=r"$y=x$")

axs[1].legend(fontsize=taille_ecriture)

axs[1].set_xlim(np.min(array_U_maw),np.max(array_U_maw))
axs[1].set_ylim(np.min(data_U_maw.values),np.max(data_U_maw.values))

axs[1].set_xlabel(r"Observation ($m.s^{-1}$)",size=taille_ecriture)
axs[1].set_ylabel(r"ERA5 ($m.s^{-1}$)",size=taille_ecriture)

r_maw = str(1e-3 * int(1000 * regress_maw.rvalue))

RMSE_maw = str(1e-3 * int(1000 * (data_RMSE_maw_U.sel(latitude=maw_lat).sel(longitude=maw_lon).values)))

axs[1].text(14, 5, "$r=$"+str(r_maw)+" \nRMSE="+str(RMSE_maw)+r"$m.s^{-1}$",
          horizontalalignment='left',
          fontsize =taille_ecriture,
          color = 'k',
          bbox=dict(facecolor='white'))

axs[1].grid()
axs[1].tick_params(axis = 'both', labelsize = taille_ecriture)


fig.show()

