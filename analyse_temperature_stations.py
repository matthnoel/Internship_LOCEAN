#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 09:51:07 2022

@author: uvc35ld
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
Temp_maw = list(data_Mawson.TEMP)
np.array(Temp_maw.pop(0))
for i in range(len(Temp_maw)):
    temperature = Temp_maw[i]
    if (temperature == '****'):
        temperature = -1
        
    else :
        temperature = (float(Temp_maw[i])-32)/1.8
    Temp_maw[i] = temperature

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
    data = Temp_maw
    ,
    coords=dict(
        time=(["time"], date_maw)
    ),
    attrs=dict(
        description = 'Data mesurée Mawson'
        )
).sel(time=slice('2010-01-01','2017-01-01'))

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

Temp_dav = list(data_Davis.TEMP)
np.array(Temp_dav.pop(0))
for i in range(len(Temp_dav)):
    temperature = Temp_dav[i]
    if (temperature == '****'):
        temperature = -1
    else :
        temperature = (float(Temp_dav[i])-32)/1.8
    Temp_dav[i] = temperature
    
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

################ DAVIS  ################
Date_dav = [np.datetime64(str(date_dav[0])[:10])]
compteur = 1 
mean = Temp_dav[0]
T_dav = []
for i in range(1,len(date_dav)) :
    date = np.datetime64(str(date_dav[i])[:10])

    if date != Date_dav[-1]:
        Date_dav.append(date)
        T_dav.append(mean/compteur)
        
        if Temp_dav[i] != -1 :
            mean = Temp_dav[i]
        else :
            mean = 0
            
        compteur = 1
    else :
        if Temp_dav[i] != -1 :
            mean += Temp_dav[i]
            compteur += 1
        
T_dav.append(mean/compteur)   
data_T_dav = tp.dataArray_station(T_dav,Date_dav).sel(time=slice('2010-01-01','2017-01-01'))

plt.figure()
data_T_dav.plot()

################  MAWSON  ################
Date_maw = [np.datetime64(str(date_maw[0])[:10])]
compteur = 1 
mean = Temp_dav[0]
T_maw = []
for i in range(1,len(date_maw)) :
    date = np.datetime64(str(date_maw[i])[:10])
    
    if date != Date_maw[-1]:
        Date_maw.append(date)
        T_maw.append(mean/compteur)
        
        mean = Temp_maw[i]
        compteur = 1
    else :
        mean += Temp_maw[i]
        compteur += 1
        
T_maw.append(mean/compteur) 
    
data_T_maw = tp.dataArray_station(T_maw,Date_maw).sel(time=slice('2010-01-01','2017-01-01'))

plt.figure()
data_T_maw.plot()

# In[Importation des données]

t2m_ERA_maw = tp.dataYearsMonths('t2m',List_Years=range(2010,2021),List_Months=range(1,13),area='Mawson').sel(time=slice('2010-01-01','2017-01-01'))

t2m_ERA_dav = tp.dataYearsMonths('t2m',List_Years=range(2010,2021),List_Months=range(1,13),area='Davis').sel(time=slice('2010-01-01','2017-01-01'))

t2m_ERA_CD_around = tp.dataYearsMonths('t2m',List_Years=range(2010,2021),List_Months=range(1,13),area='Cape Darnley Around').sel(time=slice('2010-01-01','2017-01-01'))

# In[Tracés ERA/AWS]

t2m_ERA_maw = tp.dataYearsMonths('t2m',List_Years=range(2010,2021),List_Months=range(1,13),area='Mawson').sel(time=slice('2010-01-01','2017-01-01'))

t2m_ERA_maw.data  = t2m_ERA_maw.data - 273.15

# plt.figure()
# data_T_maw.sel(time=slice('2010-05-01','2010-10-01')).plot()
# t2m_ERA_maw.sel(time=slice('2010-05-01','2010-10-01')).plot()

# time = t2m_ERA_maw.sel(time=slice('2010-05-01','2010-10-01')).time.data 

# r = data_T_maw.sel(time=slice('2010-05-01','2010-10-01')).data/t2m_ERA_maw.sel(time=slice('2010-05-01','2010-10-01')).data
# plt.plot(time,r)


t2m_ERA_dav = tp.dataYearsMonths('t2m',List_Years=range(2010,2021),List_Months=range(1,13),area='Davis').sel(time=slice('2010-01-01','2017-01-01'))

t2m_ERA_dav.data  = t2m_ERA_dav.data - 273.15


# In[Tracés ERA/AWS DAVIS]

time_dav = t2m_ERA_dav.sel(time=slice('2010-05-01','2010-10-01')).time.data
time_T_data_dav = [str(x)[:10] for x in time_dav]

array_data_T_dav_Win_2010 = np.zeros(t2m_ERA_dav.sel(time=time_dav).shape)
for i in range(t2m_ERA_dav.sel(time=time_dav).shape[0]):
    array_data_T_dav_Win_2010[i] = t2m_ERA_dav.sel(time=time_dav)[i]

plt.figure(figsize=(1.5*6.4,1.5*4.8))

plt.plot(time_dav,data_T_dav.sel(time=time_T_data_dav),'b--',label='Mesures')
plt.plot(time_dav,array_data_T_dav_Win_2010,'k',label='ERA5')

plt.legend(fontsize=20)
plt.xlim(time_dav[0],time_dav[-1])
# plt.ylim(0,24)


plt.xlabel(r"time",size=20)
plt.ylabel(r"$°C$",size=20)

plt.tick_params(labelsize=18)

plt.title('DAVIS',size=20)

plt.grid()

plt.show()



# In[Tracés ERA/AWS MAWSON]

time_maw = t2m_ERA_maw.sel(time=slice('2010-05-01','2010-10-01')).time.data
time_T_data_maw = [str(x)[:10] for x in time_maw]

array_data_T_maw_Win_2010 = np.zeros(t2m_ERA_maw.sel(time=time_maw).shape)
for i in range(t2m_ERA_maw.sel(time=time_maw).shape[0]):
    array_data_T_maw_Win_2010[i] = t2m_ERA_maw.sel(time=time_maw)[i]

plt.figure(figsize=(1.5*6.4,1.5*4.8))

plt.plot(time_maw,data_T_maw.sel(time=time_T_data_maw),'b--',label='Mesures')
plt.plot(time_maw,array_data_T_maw_Win_2010,'k',label='ERA5')

plt.legend(fontsize=20)
plt.xlim(time_maw[0],time_maw[-1])
# plt.ylim(0,24)


plt.xlabel(r"time",size=20)
plt.ylabel(r"$°C$",size=20)

plt.tick_params(labelsize=18)

plt.title('MAWSON',size=20)

plt.grid()

plt.show()

# In[Tracés ERA/AWS]

fig = plt.figure(constrained_layout=True, figsize=(2*1.5*6.4, 1*1.5*4.8))
axs = fig.subplots(2, 1, sharex=True)



axs[0].plot(time_maw,data_T_maw.sel(time=time_T_data_maw),'k',label='Mesures')
axs[0].plot(time_maw,t2m_ERA_maw.sel(time=time_maw),'b--',label='ERA5')

axs[0].legend(fontsize=20)
axs[0].set_xlim(time_maw[0],time_maw[-1])
axs[0].set_ylim(-35,0)


# axs[0].set_xlabel(r"time",size=20)
axs[0].set_ylabel(r"$m.s^{-1}$",size=20)

axs[0].tick_params(labelsize=18)

axs[0].set_title('MAWSON',size=20)

axs[0].grid()

##### DAVIS #####

axs[1].plot(time_dav,data_T_dav.sel(time=time_T_data_dav),'k',label='Mesures')
axs[1].plot(time_dav,t2m_ERA_dav.sel(time=time_dav),'b--',label='ERA5')

axs[1].legend(fontsize=20)
axs[1].set_xlim(time_dav[0],time_dav[-1])
axs[1].set_ylim(-35,0)


axs[1].set_xlabel(r"time",size=21)
axs[1].set_ylabel(r"$m.s^{-1}$",size=21)

axs[1].tick_params(labelsize=18)

axs[1].set_title('DAVIS',size=21)

axs[1].grid()

plt.show()










# In[Travaux sur les données MAWSON]

dates = []
data_T = []

times = t2m_ERA_CD_around.time.values
Latitudes = t2m_ERA_CD_around.latitude.values
Longitudes = t2m_ERA_CD_around.longitude.values

times_maw = data_T_maw.time.values
array_T_ERA_maw = np.zeros((len(times_maw),len(Latitudes),len(Longitudes)))

times_modif_maw = []

for i in range(len(times_maw)):
    array_T_ERA_maw[i,:,:] = t2m_ERA_CD_around.sel(time = str(times_maw[i])[:10]).mean('time') - 273.15
    times_modif_maw.append(times_maw[i])


t2m_ERA_CD_around_1 = tp.creationDataArray(np.array(array_T_ERA_maw), Latitudes, Longitudes,time=times_modif_maw)

corr_maw_T = tp.correlationDataSurfacePolynieCapeDarnley_around(t2m_ERA_CD_around_1, data_T_maw,titre='Corrélation entre temperature mesurée à Mawson et ERA5',stations=True)

array_rmse_maw_T = np.zeros((len(Latitudes),len(Longitudes)))



for i in range(len(Latitudes)):
    for j in range(len(Longitudes)):
        y_predicted = list(t2m_ERA_CD_around_1.values[:,i,j])
        y_actual = list(data_T_maw.values)
        
        
        somme = 0
        n = len(y_actual)
        compteur = 0
        for k in range(n):
            terme_calcul = (y_actual[k] - y_predicted[k])**2
            if not(m.isnan(terme_calcul)):
                somme += terme_calcul
                compteur += 1

        array_rmse_maw_T[i,j] = np.sqrt(somme/n)


data_RMSE_maw_T = tp.creationDataArray(array_rmse_maw_T, Latitudes, Longitudes)

print(data_RMSE_maw_T.sel(latitude=-67.5).sel(longitude=63))

# md.plotMap_CapeDarnley_around(data_RMSE_maw_T, titre='RMSE observation / ERA5 MAWSON',stations=True)

# In[Travaux sur les données DAVIS]

dates = []
data_T = []

times = t2m_ERA_CD_around.time.values
Latitudes = t2m_ERA_CD_around.latitude.values
Longitudes = t2m_ERA_CD_around.longitude.values

times_dav = data_T_dav.time.values
array_T_ERA_dav = np.zeros((len(times_dav),len(Latitudes),len(Longitudes)))

times_modif_dav = []


for i in range(len(times_dav)):
    array_T_ERA_dav[i,:,:] = t2m_ERA_CD_around.sel(time = str(times_dav[i])[:10]).mean('time') - 273.15
    times_modif_dav.append(times_dav[i])


t2m_ERA_CD_around_1 = tp.creationDataArray(np.array(array_T_ERA_dav), Latitudes, Longitudes,time=times_modif_dav)

corr_dav_T = tp.correlationDataSurfacePolynieCapeDarnley_around(t2m_ERA_CD_around_1, data_T_dav,titre='Corrélation entre température mesurée à DAVIS et ERA5',stations=True,levels=np.linspace(-1,1,11))

array_rmse_dav_T = np.zeros((len(Latitudes),len(Longitudes)))

for i in range(len(Latitudes)):
    for j in range(len(Longitudes)):
        y_predicted = t2m_ERA_CD_around_1.values[:,i,j]
        y_actual = data_T_dav.values
        
        somme = 0
        n = len(y_actual)
        compteur = 0
        for k in range(n):
            terme_calcul = (y_actual[k] - y_predicted[k])**2
            if not(m.isnan(terme_calcul)):
                somme += terme_calcul
                compteur += 1

        array_rmse_dav_T[i,j] = np.sqrt(somme/n)

data_RMSE_dav_T = tp.creationDataArray(array_rmse_dav_T, Latitudes, Longitudes)

# md.plotMap_CapeDarnley_around(data_RMSE_dav_T,titre='RMSE observation / ERA5 DAVIS')

print(data_RMSE_dav_T.sel(latitude=-68.5).sel(longitude=78))

# In[Arrangement et choix de données température à Davis]

dav_lat ,dav_lon = -68.5, 78
data_T_dav_ERA5 = t2m_ERA_CD_around.sel(latitude=dav_lat).sel(longitude=dav_lon)

time_T_dav_ERA5 = np.array([str(x)[:10] for x in data_T_dav_ERA5.time.values])
time_T_dav = np.array([str(x)[:10] for x in data_T_dav.time.values])

time_all = []
for x in time_T_dav_ERA5 :
    if x in time_T_dav :
        time_all.append(str(x))
time_all = np.array(time_all)

array_T_dav_ERA5 = np.zeros(time_all.shape)
for i in range(time_all.shape[0]):
    x = time_all[i]
    array_T_dav_ERA5[i] = data_T_dav_ERA5.sel(time=x) - 273.15
    
array_T_dav = np.zeros(time_all.shape)
for i in range(time_all.shape[0]):
    x = time_all[i]
    array_T_dav[i] = data_T_dav.sel(time=x)

# In[Calcul de la régression de la température à Davis]
regress_dav =  linregress(array_T_dav_ERA5,array_T_dav)
fit = np.polyfit(array_T_dav,array_T_dav_ERA5,1)

# fit = np.polyfit(1.12 *data_U_maw.values,data_U_maw_ERA5,1)

data_T_dav_fit = [fit[0]*x + fit[1] for x in data_T_dav]

plt.figure(figsize=(1.5*6.4, 1.5*4.8), frameon=True)
taille_ecriture = 20

plt.scatter(array_T_dav,array_T_dav_ERA5,marker='+',color='r')
plt.plot(data_T_dav.values,data_T_dav_fit,color='k',linewidth=4,label='Linear regression')
plt.plot(data_T_dav.values,data_T_dav.values,'b:',linewidth=4,label=r"$y=x$")

plt.xlim(np.min(array_T_dav),np.max(array_T_dav))
plt.ylim(np.min(data_T_dav.values),np.max(data_T_dav.values))

plt.xlabel('Observation ($°C$)',size=taille_ecriture)
plt.ylabel('ERA5 ($°C$)',size=taille_ecriture)

r_dav = str(1e-3 * int(1000 * regress_dav.rvalue))
RMSE_dav = str(1e-3 * int(1000 * (data_RMSE_dav_T.sel(latitude=dav_lat).sel(longitude=dav_lon).values)))

plt.text(-7, -25, "$r=$"+str(r_dav)+" \n$RMSE="+str(RMSE_dav)+"°C$",
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


data_T_maw_ERA5 = t2m_ERA_CD_around.sel(latitude=maw_lat).sel(longitude=maw_lon)

time_T_maw_ERA5 = np.array([str(x)[:10] for x in data_T_maw_ERA5.time.values])
time_T_maw = np.array([str(x)[:10] for x in data_T_maw.time.values])

time_all = []
for x in time_T_maw_ERA5 :
    if x in time_T_maw :
        time_all.append(str(x))
time_all = np.array(time_all)

array_T_maw_ERA5 = np.zeros(time_all.shape)
for i in range(time_all.shape[0]):
    x = time_all[i]
    array_T_maw_ERA5[i] = data_T_maw_ERA5.sel(time=x) - 273.15
    
array_T_maw = np.zeros(time_all.shape)
for i in range(time_all.shape[0]):
    x = time_all[i]
    array_T_maw[i] = data_T_maw.sel(time=x)

# In[Calcul de la régression de la température à Mawson]
regress_maw =  linregress(array_T_maw_ERA5,array_T_maw)
fit = np.polyfit(array_T_maw,array_T_maw_ERA5,1)

# fit = np.polyfit(1.12 *data_U_maw.values,data_U_maw_ERA5,1)

data_T_maw_fit = [fit[0]*x + fit[1] for x in data_T_maw]

plt.figure(figsize=(1.5*6.4, 1.5*4.8), frameon=True)
taille_ecriture = 20

plt.scatter(array_T_maw,array_T_maw_ERA5,marker='+',color='r')
plt.plot(data_T_maw.values,data_T_maw_fit,color='k',linewidth=4,label='Linear regression')
plt.plot(data_T_maw.values,data_T_maw.values,'b:',linewidth=4,label=r"$y=x$")

plt.xlim(np.min(array_T_maw),np.max(array_T_maw))
plt.ylim(np.min(data_T_maw.values),np.max(data_T_maw.values))

plt.xlabel('Observation ($°C$)',size=taille_ecriture)
plt.ylabel('ERA5 ($°C$)',size=taille_ecriture)

r_maw = str(1e-3 * int(1000 * regress_maw.rvalue))

RMSE_maw = str(1e-3 * int(1000 * (data_RMSE_maw_T.sel(latitude=maw_lat).sel(longitude=maw_lon).values)))

plt.text(-7, -25, "$r=$"+str(r_maw)+" \n$RMSE="+str(RMSE_maw)+"°C$",
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

axs[0].scatter(array_T_dav,array_T_dav_ERA5,marker='+',color='r')
axs[0].plot(data_T_dav.values,data_T_dav_fit,color='k',linewidth=4,label='Linear regression')
axs[0].plot(data_T_dav.values,data_T_dav.values,'b:',linewidth=4,label=r"$y=x$")

axs[0].legend(fontsize=taille_ecriture)

axs[0].set_xlim(np.min(array_T_dav),np.max(array_T_dav))
axs[0].set_ylim(np.min(data_T_dav.values),np.max(data_T_dav.values))

axs[0].set_xlabel('Observation ($°C$)',size=taille_ecriture)
axs[0].set_ylabel('ERA5 ($°C$)',size=taille_ecriture)

r_dav = str(1e-3 * int(1000 * regress_dav.rvalue))
RMSE_dav = str(1e-3 * int(1000 * (data_RMSE_dav_T.sel(latitude=dav_lat).sel(longitude=dav_lon).values)))

axs[0].text(-7, -25, "$r=$"+str(r_dav)+" \n$RMSE="+str(RMSE_dav)+"°C$",
          horizontalalignment='left',
          fontsize =taille_ecriture,
          color = 'k',
         bbox=dict(facecolor='white'))

axs[0].grid()
axs[0].tick_params(axis = 'both', labelsize = taille_ecriture)

### MAWSON ###

axs[1].set_title('MAWSON',fontsize = taille_ecriture)

axs[1].scatter(array_T_maw,array_T_maw_ERA5,marker='+',color='r')
axs[1].plot(data_T_maw.values,data_T_maw_fit,color='k',linewidth=4,label='Linear regression')
axs[1].plot(data_T_maw.values,data_T_maw.values,'b:',linewidth=4,label=r"$y=x$")

axs[1].legend(fontsize=taille_ecriture)

axs[1].set_xlim(np.min(array_T_maw),np.max(array_T_maw))
axs[1].set_ylim(np.min(data_T_maw.values),np.max(data_T_maw.values))

axs[1].set_xlabel('Observation ($°C$)',size=taille_ecriture)
axs[1].set_ylabel('ERA5 ($°C$)',size=taille_ecriture)

r_maw = str(1e-3 * int(1000 * regress_maw.rvalue))

RMSE_maw = str(1e-3 * int(1000 * (data_RMSE_maw_T.sel(latitude=maw_lat).sel(longitude=maw_lon).values)))

axs[1].text(-7, -25, "$r=$"+str(r_maw)+" \n$RMSE="+str(RMSE_maw)+"°C$",
          horizontalalignment='left',
          fontsize =taille_ecriture,
          color = 'k',
         bbox=dict(facecolor='white'))

axs[1].grid()
axs[1].tick_params(axis = 'both', labelsize = taille_ecriture)


fig.show()
