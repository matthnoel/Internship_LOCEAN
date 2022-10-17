#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 14:47:03 2022

@author: Matthias NOEL
"""


# In[Chargement bibliothèques]
import numpy as np
import matplotlib.pyplot as plt
#import cartopy.crs as ccrs
import xarray as xr
import datetime
import pandas as pd
# from cartopy.util import add_cyclic_point
# import pandas as pd
# from scipy import stats


import xrft

import MapDrawing as md

import ToolsPolynias as tp

Months = ['Jan','Fev','Mar','Avr','Mai','Juin','Juil','Aou','Sep','Oct','Nov','Dec']

# In[Chargement des données polynie]
#xr.open_dataset('/gpfswork/rech/omr/uvc35ld/cartes_Antarctique/CI/Polynias/'+str(data)+'s_era5_ant_y20'+ str(i) +'.nc')[data]

# polynie_Mar = tp.dataByMonth('polynie', 3)
# polynie_Avr = tp.dataByMonth('polynie', 4)
# polynie_Mai = tp.dataByMonth('polynie', 5)
# polynie_Juin = tp.dataByMonth('polynie', 6)
# polynie_Juil = tp.dataByMonth('polynie', 7)
# polynie_Aou = tp.dataByMonth('polynie', 8)
# polynie_Sep = tp.dataByMonth('polynie', 9)
# polynie_Oct = tp.dataByMonth('polynie', 10)

# Surface_polynie_Mar_CD = tp.calculSurfacePolynies(tp.sel_CapeDarnley(polynie_Mar))
# Surface_polynie_Avr_CD = tp.calculSurfacePolynies(tp.sel_CapeDarnley(polynie_Avr))
# Surface_polynie_Mai_CD = tp.calculSurfacePolynies(tp.sel_CapeDarnley(polynie_Mai))
# Surface_polynie_Juin_CD = tp.calculSurfacePolynies(tp.sel_CapeDarnley(polynie_Juin))
# Surface_polynie_Juil_CD = tp.calculSurfacePolynies(tp.sel_CapeDarnley(polynie_Juil))
# Surface_polynie_Aou_CD = tp.calculSurfacePolynies(tp.sel_CapeDarnley(polynie_Aou))
# Surface_polynie_Sep_CD = tp.calculSurfacePolynies(tp.sel_CapeDarnley(polynie_Sep))
# Surface_polynie_Oct_CD = tp.calculSurfacePolynies(tp.sel_CapeDarnley(polynie_Oct))

# DS_polynie_Mars_CD = xr.Dataset(dict(polynie=Surface_polynie_Mar_CD))
# DS_polynie_Avr_CD = xr.Dataset(dict(polynie=Surface_polynie_Avr_CD))
# DS_polynie_Mai_CD = xr.Dataset(dict(polynie=Surface_polynie_Mai_CD))
# DS_polynie_Juin_CD = xr.Dataset(dict(polynie=Surface_polynie_Juin_CD))
# DS_polynie_Juil_CD = xr.Dataset(dict(polynie=Surface_polynie_Juil_CD))
# DS_polynie_Aou_CD = xr.Dataset(dict(polynie=Surface_polynie_Aou_CD))
# DS_polynie_Sep_CD = xr.Dataset(dict(polynie=Surface_polynie_Sep_CD))
# DS_polynie_Oct_CD = xr.Dataset(dict(polynie=Surface_polynie_Oct_CD))

polynie_Win_2010_2020_CD = tp.dataYearsMonths('polynie',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')

surface_AMJJASO_CD= tp.calculSurfacePolynies(polynie_Win_2010_2020_CD,area='CDP')

surface_AMJJASO_CD.sel(time='2010').plot()

# In[Chargement des données de vitesses]

u10_2010_2020 = tp.dataYearsMonths('u10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')
v10_2010_2020 = tp.dataYearsMonths('v10',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')

polynie_2010_2020 = tp.dataYearsMonths('polynie',List_Years=range(2010,2021),List_Months=range(5,10),area='Cape Darnley')

surface_polynie_2010_2020 = tp.calculSurfacePolynies(polynie_2010_2020,area='CDP')

U10_2010_2020 = np.sqrt(u10_2010_2020*u10_2010_2020 + v10_2010_2020*v10_2010_2020)

U10_2010_CD = tp.meanSurface(U10_2010_2020)

surface_polynie_2010_2020.sel(time='2010').sel(time=slice('2010-05-01','2010-09-30')).plot()

# In[Moyenne glissante]

Pd_surface_polynie_2010_2020 = pd.Series(surface_polynie_2010_2020.isel(time=slice(0,1000)))

Moyennes_glissees_Surface_Polynies_CD = []
Moyennes_glissees_Surface_Polynies_CD.append(Pd_surface_polynie_2010_2020)

for i in range(1,7):
    Moyennes_glissees_Surface_Polynies_CD.append(Pd_surface_polynie_2010_2020.rolling(5*i).mean())
    
plt.figure(figsize=(10,8))

for i in range(7):
    plt.plot(range(surface_polynie_2010_2020.isel(time=slice(0,1000)).shape[0]),Moyennes_glissees_Surface_Polynies_CD[i],label='Moyenne glissée sur '+str(5*i)+' jours')

plt.xlabel('Time (day)')
plt.ylabel('Surface Polynies de Cape Darnley ($km^2$)')
#plt.axhline(y=0.2*data_Surface_Polynies_Sep_CD.mean('time'),color='black')
plt.xlim(0,1000)
plt.ylim(0,130000)
plt.legend()
plt.grid()
plt.show()




plt.figure(figsize=(10,8))

i=6
plt.plot(range(surface_polynie_2010_2020.isel(time=slice(0,1000)).shape[0]),Moyennes_glissees_Surface_Polynies_CD[i],label='Moyenne glissée sur '+str(5*i)+' jours')

plt.xlabel('Time (day)')
plt.ylabel('Surface Polynies de Cape Darnley ($km^2$)')
#plt.axhline(y=0.2*data_Surface_Polynies_Sep_CD.mean('time'),color='black')
plt.xlim(0,1000)
#plt.ylim(0,130000)
plt.legend()
plt.grid()
plt.show()



# In[Tracé]
surface_AMJJASO_CD.sel(time='2010').plot()
plt.show()
U10_2010_CD.plot()

# In[Analyse Spectrale]
data_used = U10_2010_CD

time = []
for x in data_used.time.values :
    date,hour = str(x).split('T')
    year,month,day = date.split('-')
    time.append(datetime.datetime(int(year),int(month),int(day)))
time = np.array(time)

data_U10 = xr.DataArray(
    ## Valeurs du tableau de données        
    data = data_used
    ,
    coords=dict(
        ## Coordonnées Latitudes
        time=(["time"], time)
    ),
    attrs=dict(
        description = 'data spectre'
        )
)



# donnee = xr.DataArray( data = mydata.variable[t, :, :].data, dims = ["y","x"], coords={"x": ( ["x"], xx_ubs ),"y": ( ["y"], yy_ubs )}, )
spectre_U10 = 2. * xrft.fft(data_U10.sel(time='2010'),dim="time", window='tukey', truncate=True).compute()

plt.figure(figsize=(10,7))
plt.plot(1/(spectre_U10.freq_time*(3600*24)), spectre_U10)
plt.title('Spectre Vitesse')
plt.xscale("log")
plt.yscale("log")
plt.xlabel('jour')
plt.ylim(1e2,2e3)
plt.grid()



# In[Spectres annuelles surface polynie]


# surface_polynie_2010_2020.sel(time=slice('2010-05-01','2010-09-30')).plot()

print(surface_polynie_2010_2020.sel(time=slice('2010-05-01','2010-09-30')).shape)


L_spectre = []


surface_0 = surface_polynie_2010_2020.sel(time=slice('2010-05-01','2010-09-30'))

Spectre_final = 2. * xrft.fft(surface_0,dim="time", window='tukey', truncate=True).compute()

plt.figure(figsize=(10,7))
plt.plot(Spectre_final.freq_time, Spectre_final)
plt.title('Spectre Surface polynie de Cape Darnley - entre Mai et Septembre ' + str(2010))
plt.xscale("log")
plt.yscale("log")
# plt.xlabel('jour')
#plt.ylim(1e6,2e7)
plt.grid()

for i in range(2011,2021):
    year = str(i)
    surface_i = surface_polynie_2010_2020.sel(time=slice(year+'-05-01',year+'-09-30'))
    
    spectre_i = 2. * ( abs( xrft.fft(surface_i,dim="time", window='tukey', truncate=True).compute()))**2
    L_spectre.append(spectre_i)
    Spectre_final = Spectre_final + spectre_i
    
    plt.figure(figsize=(10,7))
    plt.plot(spectre_i.freq_time, spectre_i)
    plt.title('Spectre Surface Polynie de Cape Darnley - entre Mai et Septembre ' + str(i))
    plt.xscale("log")
    plt.yscale("log")
    # plt.xlabel('jour')
    #plt.ylim(1e6,2e7)
    plt.grid()

Spectre_final = Spectre_final/11

# In[Tracé de la moyenne des spectres]
taille_ecriture = 20

plt.figure(figsize=(1.5*6.4, 1.5*4.8))
#plt.plot(1/(Spectre_final.freq_time*(3600*24)), Spectre_final)
plt.plot(Spectre_final.freq_time*3600*24,Spectre_final)
# plt.title('Spectre Surface Polynie de Cape Darnley \n Moyenne des spectres de 05-01 à 09-30 entre 2010 et 2020'
#           ,fontsize = taille_ecriture)
plt.xscale("log")
plt.yscale("log")
#plt.xlabel('jour')
#plt.ylim(1e6,2e7)
plt.grid()
plt.tick_params(axis = 'both', labelsize = taille_ecriture)

plt.ylabel('Power ($(km^2)^2$)',size=taille_ecriture)
plt.xlabel('Frequency ($day^{-1}$)',size=taille_ecriture)

plt.show()

# In[Spectres de la vitesse du vent moyenneé sur la surface polynie]

## U10_2010_CD --> Moyenne surfacique de la vitesse du vent sur la polynie.

print(U10_2010_CD.sel(time=slice('2010-05-01','2010-09-30')).shape)


L_spectre = []


surface_0 = U10_2010_CD.sel(time=slice('2010-05-01','2010-09-30'))

Spectre_final = 2. * xrft.fft(surface_0,dim="time", window='tukey', truncate=True).compute()

for i in range(2011,2021):
    year = str(i)
    surface_i = U10_2010_CD.sel(time=slice(year+'-05-01',year+'-09-30'))
    
    spectre_i = 2. * (abs(xrft.fft(surface_i,dim="time", window='tukey', truncate=True).compute())**2)
    L_spectre.append(spectre_i)
    Spectre_final = Spectre_final + spectre_i
    
    plt.figure(figsize=(10,7))
    plt.plot(spectre_i.freq_time, spectre_i)
    plt.title('Spectre U10 - Moyenne des spectres de 05-01 à 09-30 entre 2010 et 2020')
    plt.xscale("log")
    plt.yscale("log")
    # plt.xlabel('jour')
    #plt.ylim(1e6,2e7)
    plt.grid()
    

Spectre_final = Spectre_final/11

    

plt.figure(figsize=(10,7))
plt.plot(Spectre_final.freq_time, Spectre_final)
plt.title('Spectre U10 - Moyenne des spectres de 05-01 à 09-30 entre 2010 et 2020')
plt.xscale("log")
plt.yscale("log")
# plt.xlabel('jour')
#plt.ylim(1e6,2e7)
plt.grid()


