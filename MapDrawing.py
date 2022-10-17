#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 14:27:45 2022

@author: Matthias NOEL
"""

# In[Chargement bibliothèque]
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib as mpl

from cartopy.util import add_cyclic_point

import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature


# In[Fonction carte Antarctique] :
    
def plotMap_Ant(data,clevels=None,color = 'RdBu_r',dimension=None,titre=None,name_save=None,color_bar=True):
    
    ########## ENTREE ##########
    # data - xarray de dimension 2 (latitude,longitude...) : données d'intérêt
    # clevels - array (dim 1) de taille nombre de niveaux de couleurs (colorbar) et
    #                         de valeur échelle de représentation
    # dimension - str : unité de la donnée représentée.
    # titre - str : titre du graphe.
    
    ########### SORTIE ##########
    # Ne retourne rien MAIS TRACE de DATA au POLE SUD
    
    # Création de la figure :
    plt.figure(figsize=(10,10), frameon=True)
    
    # Projection map :
    ax = plt.axes(projection=ccrs.SouthPolarStereo())
    
    # Ajout des lignes de côtes / grid :
    ax.coastlines(resolution="10m",linewidth=1)
    gl = ax.gridlines(linestyle='--',color='black',
                  draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    
    # Tracé du contour de la donnée choisie :
    if clevels is None :
        im = data.plot(ax=ax,
                           transform=ccrs.PlateCarree(),
                           cmap=color,
                           add_colorbar=False
                           );
    else :
        im = data.plot(ax=ax,
                           transform=ccrs.PlateCarree(),
                           levels=clevels,
                           cmap=color,
                           add_colorbar=False
                           );
        
    # im = data.plot.contourf(ax=ax,
    #                        transform=ccrs.PlateCarree(),
    #                        levels=clevels,
    #                        cmap=color,
    #                        add_colorbar=False
    #                        );
    
    if color_bar :
        # Création de la barre de couleur (Bonne orientation et bonne taille...) :
        cb = plt.colorbar(im, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
        cb.set_label(dimension,size=18,rotation=90,labelpad=1)
        cb.ax.tick_params(labelsize=18)
            
    # Création du titre :
    ax.set_title(titre,fontsize=20)
    #ax.set_xlabel("")
    
    plt.show()
    # plt.savefig('/linkhome/rech/genloc01/reee217/matthias/cartes_Antarctique/'+name_save)

# In[Streamplot - ]

def plotStreamplot(data_u,data_v,U = None,Levels=None,title=None,stations=False,Liste_Vmin_Vmax=None, color="RdBu_r",dimension='$m/s$',CI_contour=None,level_CI_contour=None):
    
    lat = data_v.latitude.values
    lon = data_v.longitude.values

    v = data_v#.mean('time').values
    u = data_u#.mean('time').values
    
    if U is None :    
        U = np.sqrt(u*u+v*v)
    data_U10 = creationDataArray(U, lat, lon)
    

    plt.figure(figsize=(10,10), frameon=True)

    # Projection map :
    ax = plt.axes(projection=ccrs.SouthPolarStereo())#Orthographic(-65,-75))

    # Ajout des lignes de côtes / grid :
    ax.coastlines(resolution="10m",linewidth=3)
    gl = ax.gridlines(linestyle='--',color='black',
                  draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    # Tracé du contour de la donnée choisie :

    if Liste_Vmin_Vmax is not None:
        im = data_U10.plot(ax=ax,
                            transform=ccrs.PlateCarree(),
                            levels=Levels,
                            cmap=color,
                            add_colorbar=False,
                            norm = mpl.colors.Normalize(vmin=Liste_Vmin_Vmax[0], vmax=Liste_Vmin_Vmax[1])
                            );
    else :
            if Levels is None:
                im = data_U10.plot(ax=ax,
                                        transform=ccrs.PlateCarree(),
                                        cmap=color,
                                        add_colorbar=False
                                        );
            else :
                im = data_U10.plot(ax=ax,
                                    transform=ccrs.PlateCarree(),
                                    levels=Levels,
                                    cmap=color,
                                    add_colorbar=False
                                    );

        
    # im = data_U10.plot.contourf(ax=ax,
    #                         transform=ccrs.PlateCarree(),
    #                         levels=Levels,
    #                         cmap=color,
    #                         add_colorbar=False
    #                         );

    # plot velocity field
    u, lonu = add_cyclic_point(u, coord = lon)
    v, lonv = add_cyclic_point(v, coord = lon)

    lonu = np.where(lonu>=180.,lonu-360.,lonu)
    X, Y = np.meshgrid(lonu, lat)


    #print('Preparation OK')

    #plt.streamplot(X,Y, u, v)

    ax.streamplot(X,Y, u, v,
                   linewidth=1,
                   arrowsize = 2,
                   density=[1.5,1.5],
                   color ='black',
                   transform=ccrs.PlateCarree())
    if (CI_contour is not None):
        polynie_CD = sel_CapeDarnley(CI_contour)
        if (level_CI_contour is not None):
            cont = polynie_CD.plot.contour(ax=ax,
                                   transform=ccrs.PlateCarree(),
                                   levels=level_CI_contour,
                                   cmap='black',#plt.cm.gray,
                                   add_colorbar=False
                                   );
        else :
            cont = polynie_CD.plot.contour(ax=ax,
                                   transform=ccrs.PlateCarree(),
                                   cmap='black',#plt.cm.gray,
                                   add_colorbar=False
                                   );
        ax.clabel(cont, inline=True, fontsize=15)
    
    if stations :
        
        ## Station Davis :
        dav_lat ,dav_lon = -68.25, 78.75
        plt.plot([dav_lon], [dav_lat],
                  linewidth=10, marker='o',
                  transform=ccrs.PlateCarree(),
                  color = 'm'
                  )
        plt.text(dav_lon - 3, dav_lat - 1, 'Davis',
                  horizontalalignment='right',
                  fontsize =20,
                  transform=ccrs.PlateCarree(),
                  color = 'w')
        
        ## Station Mawson :
        maw_lat ,maw_lon = -67.5, 63.75
        plt.plot([maw_lon], [maw_lat],
                  linewidth=10, marker='o',
                  transform=ccrs.PlateCarree(),
                  color = 'm'
                  )
        plt.text(maw_lon - 3, maw_lat - 1, 'Mawson',
                  horizontalalignment='right',
                  fontsize =20,
                  transform=ccrs.PlateCarree(),
                  color = 'w'
                  )

    #im = data.plot.streamplot('latitude','longitude')#,ax=ax)
    # Création de la barre de couleur (Bonne orientation et bonne taille...) :
    cb = plt.colorbar(im, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
    cb.set_label(dimension,size=18,rotation=90,labelpad=1)
    cb.ax.tick_params(labelsize=18)

    # Création du titre :
    ax.set_title(title,fontsize=20)
    ax.set_xlabel("")

    plt.show()
    
    

# In[Sélection région de Cape Darnley]
# On se place autour de la polynie de Cape Darnley :
# Latitude entre : 65° S et 70.5°S
# Longitude entre : 63.5°E et 74°E

def sel_CapeDarnley(data):
    data_sortie = data.sel(latitude=slice(-65,-70.5),longitude=slice(63.5,74))
    return data_sortie

def sel_CapeDarnley_jour(data,jour):
    data_sortie = data.sel(latitude=slice(-65,-70.5),longitude=slice(63.5,74))
    return data_sortie.isel(time = jour)

def sel_CapeDarnley_around(data):
    data_sortie = data.sel(latitude=slice(-50,-90),longitude=slice(20,120))
    return data_sortie

# In[Tracé 1D]

def plot_1D(X,Y,xlabel=None,ylabel=None,label=None,taille_ecriture = 20,size_figure=[1.5,1.5],
            title=None,log_scale = False,grid=True,xlim=None,ylim=None):
    

    plt.figure(figsize=(size_figure[0]*6.4, size_figure[1]*4.8))
    
    plt.plot(X,Y,label=label)
    
    plt.title(title,fontsize = taille_ecriture)
    plt.tick_params(axis = 'both', labelsize = taille_ecriture)
    
    if log_scale :
        plt.xscale("log")
        plt.yscale("log")
        
    if grid :
        plt.grid()
    
    if xlim is not None:
        plt.xlim(xlim[0],xlim[1])
    if ylim is not None :
        plt.ylim(ylim[0],ylim[1])
        
    if label is not None:
        plt.legend(fontsize=taille_ecriture, loc='upper right')
    
    if xlabel is not None :
        plt.xlabel(xlabel,size=taille_ecriture)
    if ylabel is not None :
        plt.ylabel(ylabel,size=taille_ecriture)

    plt.show()


# In[Tracé autours de Cape Darnley]

def plotMap_CapeDarnley(data,clevels=None,color = 'RdBu_r',dimension=None,titre=None,name_save=None,polynie=None,
                        color_bar=True,ax=None,Liste_Vmin_Vmax=None,CI_contour=None,level_CI_contour=None):
    
    #data = data_ci.isel(time=slice(242,272)).ci
    data_Cape = sel_CapeDarnley(data)
    #print(data_Cape)
    # clevels = np.linspace(0,1,11)
    # dimension='sans unité'
    # titre = 'test'

    plt.figure(figsize=(12,10), frameon=True)
    
    if ax is None :
        # Projection map :
        ax = plt.axes(projection=ccrs.SouthPolarStereo())#Orthographic(-65,-75))(-65,-75))

    # Ajout des lignes de côtes / grid :
    ax.coastlines(resolution="10m",linewidth=3)
    gl = ax.gridlines(linestyle='--',color='black',
                  draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    ''
    # Tracé du contour de la donnée choisie :
    # im = data_Cape.plot(ax=ax,
    #                    transform=ccrs.PlateCarree(),
    #                    levels=clevels,
    #                    cmap=color,#plt.cm.gray,
    #                    add_colorbar=False
    #                    );
    
    # Tracé du contour de la donnée choisie :
    if clevels is None:
        if  Liste_Vmin_Vmax is not None :
            im = data_Cape.plot(ax=ax,
                                transform=ccrs.PlateCarree(),
                                cmap=color,#plt.cm.gray,
                                add_colorbar=False,
                                norm = mpl.colors.Normalize(vmin=Liste_Vmin_Vmax[0], vmax=Liste_Vmin_Vmax[1])
                                );
        else :
            im = data_Cape.plot(ax=ax,
                                transform=ccrs.PlateCarree(),
                                cmap=color,#plt.cm.gray,
                                add_colorbar=False,
                                );
    else :
        im = data_Cape.plot(ax=ax,
                                transform=ccrs.PlateCarree(),
                                levels=clevels,
                                cmap=color,#plt.cm.gray,
                                add_colorbar=True
                                );
    if CI_contour is not None :
        if level_CI_contour is not None :
            cont = CI_contour.plot.contour(ax=ax,
                                    transform=ccrs.PlateCarree(),
                                    levels=level_CI_contour,
                                    cmap='black',#plt.cm.gray,
                                    add_colorbar=False
                                    );
            ax.clabel(cont, inline=True, fontsize=15)
        else :
            cont = CI_contour.plot.contour(ax=ax,
                        transform=ccrs.PlateCarree(),
                        #levels=level_CI_contour,
                        cmap='black',#plt.cm.gray,
                        add_colorbar=False
                        );
            ax.clabel(cont, inline=True, fontsize=15)

    
    if (polynie is not None):
        polynie_CD = sel_CapeDarnley(polynie)
        
        polynie_CD.plot.contour(ax=ax,
                                transform=ccrs.PlateCarree(),
                                levels=np.linspace(0,1,2),
                                cmap='black',#plt.cm.gray,
                                add_colorbar=False,
                                linewidth=20
                                );
    if color_bar :
        if  Liste_Vmin_Vmax is not None :
            cmap = color
            norm = mpl.colors.Normalize(vmin=Liste_Vmin_Vmax[0], vmax=Liste_Vmin_Vmax[1])
            
            cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),
                orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
            cb.set_label(dimension,size=18,rotation=90,labelpad=1)
            cb.ax.tick_params(labelsize=18)
        else :

            cb = plt.colorbar(im,
                orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
            cb.set_label(dimension,size=18,rotation=90,labelpad=1)
            cb.ax.tick_params(labelsize=18)
    
    # # Création de la barre de couleur (Bonne orientation et bonne taille...) :
    # cb = plt.colorbar(im, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
    # cb.set_label(dimension,size=18,rotation=90,labelpad=1)
    # cb.ax.tick_params(labelsize=18)

    # Création du titre :
    ax.set_title(titre,fontsize=20)
    #ax.set_xlabel("")

    plt.show()
    
    

def plotMap_CapeDarnley_around(data,Liste_Vmin_Vmax=None,clevels=None,color = 'RdBu_r',dimension=None,titre=None,
                               name_save=None,polynie=None,color_bar=True,stations=False,CI_contour=None,level_CI_contour=None):

    #data = data_ci.isel(time=slice(242,272)).ci
    data_Cape = sel_CapeDarnley_around(data)
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
    if clevels is None :
        if  Liste_Vmin_Vmax is not None :
            im = data_Cape.plot(ax=ax,
                                   transform=ccrs.PlateCarree(),
                                   cmap=color,
                                   add_colorbar=False,
                                   norm=mpl.colors.Normalize(vmin=Liste_Vmin_Vmax[0], vmax=Liste_Vmin_Vmax[1])
                                   );
        else :
            im = data_Cape.plot(ax=ax,
                                   transform=ccrs.PlateCarree(),
                                   cmap=color,
                                   add_colorbar=False,
                                   );
    else :
        im = data_Cape.plot.contourf(ax=ax,
                           transform=ccrs.PlateCarree(),
                           levels=clevels,
                           cmap=color,
                           add_colorbar=False
                           );
    if CI_contour is not None :
        if level_CI_contour is not None :
            cont = CI_contour.plot.contour(ax=ax,
                                    transform=ccrs.PlateCarree(),
                                    levels=level_CI_contour,
                                    cmap='black',#plt.cm.gray,
                                    add_colorbar=False
                                    );
            ax.clabel(cont, inline=True, fontsize=15)
        else :
            cont = CI_contour.plot.contour(ax=ax,
                        transform=ccrs.PlateCarree(),
                        #levels=level_contour,
                        cmap='black',#plt.cm.gray,
                        add_colorbar=False
                        );
            ax.clabel(cont, inline=True, fontsize=15)
    
    if (polynie is not None):
        polynie_CD_around = sel_CapeDarnley_around(polynie)
        polynie_CD_around.plot.contour(ax=ax,
                               transform=ccrs.PlateCarree(),
                               levels=np.linspace(0,1,2),
                               cmap='black',#plt.cm.gray,
                               add_colorbar=False
                               );

    # Création de la barre de couleur (Bonne orientation et bonne taille...) :
    if color_bar :
        
        if  Liste_Vmin_Vmax is not None :
            cmap = color
            norm = mpl.colors.Normalize(vmin=Liste_Vmin_Vmax[0], vmax=Liste_Vmin_Vmax[1])
            
            cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),
                orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
            cb.set_label(dimension,size=18,rotation=90,labelpad=1)
            cb.ax.tick_params(labelsize=18)
        else :

            cb = plt.colorbar(im,
                orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
            cb.set_label(dimension,size=18,rotation=90,labelpad=1)
            cb.ax.tick_params(labelsize=18)
        
    if stations :
        dav_lat ,dav_lon = -68.25, 78.75
        plt.plot([dav_lon], [dav_lat],
              color='k', linewidth=10, marker='o',
              transform=ccrs.PlateCarree(),
              )
        plt.text(dav_lon + 4.5, dav_lat - 1, 'Davis',
              horizontalalignment='right',
              fontsize =20,
              transform=ccrs.PlateCarree())
        
        maw_lat ,maw_lon = -67.5, 63.75
        plt.plot([maw_lon], [maw_lat],
              color='k', linewidth=10, marker='o',
              transform=ccrs.PlateCarree(),
              )
        plt.text(maw_lon - 3, maw_lat - 1, 'Mawson',
              horizontalalignment='right',
              fontsize =20,
              transform=ccrs.PlateCarree())
    
    # Création du titre :
    ax.set_title(titre,fontsize=20)
    #ax.set_xlabel("")

    plt.show()

# In[Tracé Anomalie]
def plotMap_Anomalie(data_ouverte_m,data_fermee_m,Liste_Vmin_Vmax=None,titre=None,
                     Liste_Vmin_Vmax_Delta=None,CI_contour=None,level_CI_contour=None,area = 'Cape Darnley',dimension=r"$m.s^{-1}$",color='RdBu_r',path_save=None):
    
    data_delta = data_ouverte_m - data_fermee_m

    fig = plt.figure(constrained_layout=True, figsize=(3*6.4, 4.8))
    
    # fig = plt.figure(figsize=(3*2*6.4, 1*2*4.8),constrained_layout=True)
    # fig = plt.figure()
    ########## Tracé de la polynie Fermée ##########
    
    ax=fig.add_subplot(1,3,1, projection=ccrs.SouthPolarStereo()) 
    
    
    if  Liste_Vmin_Vmax is not None :
        im = data_ouverte_m.plot(ax=ax,
                                transform=ccrs.PlateCarree(),
                                cmap=color,#plt.cm.gray,
                                add_colorbar=False,
                                norm = mpl.colors.Normalize(vmin=Liste_Vmin_Vmax[0], vmax=Liste_Vmin_Vmax[1])
                                );
    else :
        im = data_ouverte_m.plot(ax=ax,
                                transform=ccrs.PlateCarree(),
                                cmap=color,#plt.cm.gray,
                                add_colorbar=False,                                
                                );


    ax.coastlines(resolution="10m",linewidth=3)
    gl = ax.gridlines(linestyle='--',color='black',
                  draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    
    if CI_contour is not None :
        if level_CI_contour is not None :
            cont = CI_contour.plot.contour(ax=ax,
                                    transform=ccrs.PlateCarree(),
                                    levels=level_CI_contour,
                                    cmap='black',#plt.cm.gray,
                                    add_colorbar=False
                                    );
            ax.clabel(cont, inline=True, fontsize=15)
        else :
            cont = CI_contour.plot.contour(ax=ax,
                        transform=ccrs.PlateCarree(),
                        #levels=level_CI_contour,
                        cmap='black',#plt.cm.gray,
                        add_colorbar=False
                        );
            ax.clabel(cont, inline=True, fontsize=15)
    
    if  Liste_Vmin_Vmax_Delta is not None :
        cmap = color
        norm = mpl.colors.Normalize(vmin=Liste_Vmin_Vmax[0], vmax=Liste_Vmin_Vmax[1])
        
        cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap), orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
        cb.set_label(dimension,size=18,rotation=90,labelpad=1)
        cb.ax.tick_params(labelsize=18)
            
    else :
        cb = plt.colorbar(im, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
        cb.set_label(dimension,size=18,rotation=90,labelpad=1)
        cb.ax.tick_params(labelsize=18)
    
    if titre is not None:
        ax.set_title(titre[0],fontsize=20)
    
    
    ########## Tracé de la polynie Ouverte ##########
    
    ax = fig.add_subplot(1,3,2, projection=ccrs.SouthPolarStereo()) 
    
    
    if  Liste_Vmin_Vmax is not None :
        im = data_fermee_m.plot(ax=ax,
                                transform=ccrs.PlateCarree(),
                                cmap=color,#plt.cm.gray,
                                add_colorbar=False,
                                norm = mpl.colors.Normalize(vmin=Liste_Vmin_Vmax[0], vmax=Liste_Vmin_Vmax[1])
                                );
    else :
        im = data_fermee_m.plot(ax=ax,
                                transform=ccrs.PlateCarree(),
                                cmap=color,#plt.cm.gray,
                                add_colorbar=False,
                                );

    ax.coastlines(resolution="10m",linewidth=3)
    gl = ax.gridlines(linestyle='--',color='black',
                  draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    
    

    if CI_contour is not None :
        if level_CI_contour is not None :
            cont = CI_contour.plot.contour(ax=ax,
                                    transform=ccrs.PlateCarree(),
                                    levels=level_CI_contour,
                                    cmap='black',#plt.cm.gray,
                                    add_colorbar=False
                                    );
            ax.clabel(cont, inline=True, fontsize=15)
        else :
            cont = CI_contour.plot.contour(ax=ax,
                        transform=ccrs.PlateCarree(),
                        #levels=level_CI_contour,
                        cmap='black',#plt.cm.gray,
                        add_colorbar=False
                        );
            ax.clabel(cont, inline=True, fontsize=15)
    
    if  Liste_Vmin_Vmax is not None :
        cmap = color
        norm = mpl.colors.Normalize(vmin=Liste_Vmin_Vmax[0], vmax=Liste_Vmin_Vmax[1])
        
        cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap), orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
        cb.set_label(dimension,size=18,rotation=90,labelpad=1)
        cb.ax.tick_params(labelsize=18)
            
    else :
        cb = plt.colorbar(im, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
        cb.set_label(dimension,size=18,rotation=90,labelpad=1)
        cb.ax.tick_params(labelsize=18)
        
    if titre is not None:
        ax.set_title(titre[1],fontsize=20)
    
    
    # if  Liste_Vmin_Vmax is not None :
    #     cmap = color
    #     norm = mpl.colors.Normalize(vmin=Liste_Vmin_Vmax[0], vmax=Liste_Vmin_Vmax[1])
        
    #     cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap), orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
    #     cb.set_label(dimension,size=18,rotation=90,labelpad=1)
    #     cb.ax.tick_params(labelsize=18)
            
    # else :
    #     cb = plt.colorbar(im, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
    #     cb.set_label(dimension,size=18,rotation=90,labelpad=1)
    #     cb.ax.tick_params(labelsize=18)
        
    ########## Tracé de l'Anomalie ##########
    
    ax=fig.add_subplot(1,3,3, projection=ccrs.SouthPolarStereo()) 
    
    if  Liste_Vmin_Vmax is not None :
        im = data_delta.plot(ax=ax,
                                transform=ccrs.PlateCarree(),
                                cmap=color,#plt.cm.gray,
                                add_colorbar=False,
                                norm = mpl.colors.Normalize(vmin=Liste_Vmin_Vmax_Delta[0], vmax=Liste_Vmin_Vmax_Delta[1])
                                );
    else :
        im = data_delta.plot(ax=ax,
                                transform=ccrs.PlateCarree(),
                                cmap=color,#plt.cm.gray,
                                add_colorbar=False
                                );
    
    ax.coastlines(resolution="10m",linewidth=3)
    gl = ax.gridlines(linestyle='--',color='black',
                  draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    
    

    if CI_contour is not None :
        if level_CI_contour is not None :
            cont = CI_contour.plot.contour(ax=ax,
                                    transform=ccrs.PlateCarree(),
                                    levels=level_CI_contour,
                                    cmap='black',#plt.cm.gray,
                                    add_colorbar=False
                                    );
            ax.clabel(cont, inline=True, fontsize=15)
        else :
            cont = CI_contour.plot.contour(ax=ax,
                        transform=ccrs.PlateCarree(),
                        #levels=level_CI_contour,
                        cmap='black',#plt.cm.gray,
                        add_colorbar=False
                        );
            ax.clabel(cont, inline=True, fontsize=15)
    
    if  Liste_Vmin_Vmax_Delta is not None :
        cmap = color
        norm = mpl.colors.Normalize(vmin=Liste_Vmin_Vmax_Delta[0], vmax=Liste_Vmin_Vmax_Delta[1])
        
        cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap), orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
        cb.set_label(dimension,size=18,rotation=90,labelpad=1)
        cb.ax.tick_params(labelsize=18)
            
    else :
        cb = plt.colorbar(im, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
        cb.set_label(dimension,size=18,rotation=90,labelpad=1)
        cb.ax.tick_params(labelsize=18)
    
    if titre is not None:
        ax.set_title(titre[2],fontsize=20)
    
    if path_save is not None :
        fig.savefig(path_save)
        
    plt.show()
    
# In[Tracé Regression]
def plotMap_Regression(data_regression,data_correlation,Liste_Rmin_Rmax=None,titre=None,
                     Liste_Corrmin_Corrmax=None,CI_contour=None,level_CI_contour=None,area = 'Cape Darnley',dimension=r"$m.s^{-1}$",color='RdBu_r',path_save=None):
    

    fig = plt.figure(constrained_layout=True, figsize=(2*6.4, 4.8))
    
    # fig = plt.figure(figsize=(3*2*6.4, 1*2*4.8),constrained_layout=True)
    # fig = plt.figure()
    ########## Tracé de la Regression ##########
    
    ax=fig.add_subplot(1,2,1, projection=ccrs.SouthPolarStereo()) 
    
    
    if  Liste_Rmin_Rmax is not None :
        im = data_regression.plot(ax=ax,
                                transform=ccrs.PlateCarree(),
                                cmap=color,#plt.cm.gray,
                                add_colorbar=False,
                                norm = mpl.colors.Normalize(vmin=Liste_Rmin_Rmax[0], vmax=Liste_Rmin_Rmax[1])
                                );
    else :
        im = data_regression.plot(ax=ax,
                                transform=ccrs.PlateCarree(),
                                cmap=color,#plt.cm.gray,
                                add_colorbar=False,                                
                                );


    ax.coastlines(resolution="10m",linewidth=3)
    gl = ax.gridlines(linestyle='--',color='black',
                  draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    
    if CI_contour is not None :
        if level_CI_contour is not None :
            cont = CI_contour.plot.contour(ax=ax,
                                    transform=ccrs.PlateCarree(),
                                    levels=level_CI_contour,
                                    cmap='black',#plt.cm.gray,
                                    add_colorbar=False
                                    );
            ax.clabel(cont, inline=True, fontsize=15)
        else :
            cont = CI_contour.plot.contour(ax=ax,
                        transform=ccrs.PlateCarree(),
                        #levels=level_CI_contour,
                        cmap='black',#plt.cm.gray,
                        add_colorbar=False
                        );
            ax.clabel(cont, inline=True, fontsize=15)
    
    if  Liste_Rmin_Rmax is not None :
        cmap = color
        norm = mpl.colors.Normalize(vmin=Liste_Rmin_Rmax[0], vmax=Liste_Rmin_Rmax[1])
        
        cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap), orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
        cb.set_label(dimension,size=18,rotation=90,labelpad=1)
        cb.ax.tick_params(labelsize=18)
            
    else :
        cb = plt.colorbar(im, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
        cb.set_label(dimension,size=18,rotation=90,labelpad=1)
        cb.ax.tick_params(labelsize=18)
    
    if titre is not None:
        ax.set_title(titre[0],fontsize=20)
    
    
    ########## Tracé de la Corrélation ##########
    
    ax = fig.add_subplot(1,2,2, projection=ccrs.SouthPolarStereo()) 
    
    
    if  Liste_Corrmin_Corrmax is not None :
        im = data_correlation.plot(ax=ax,
                                transform=ccrs.PlateCarree(),
                                cmap=color,#plt.cm.gray,
                                add_colorbar=False,
                                norm = mpl.colors.Normalize(vmin=Liste_Corrmin_Corrmax[0], vmax=Liste_Corrmin_Corrmax[1])
                                );
    else :
        im = data_correlation.plot(ax=ax,
                                transform=ccrs.PlateCarree(),
                                cmap=color,#plt.cm.gray,
                                add_colorbar=False,
                                );

    ax.coastlines(resolution="10m",linewidth=3)
    gl = ax.gridlines(linestyle='--',color='black',
                  draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    
    

    if CI_contour is not None :
        if level_CI_contour is not None :
            cont = CI_contour.plot.contour(ax=ax,
                                    transform=ccrs.PlateCarree(),
                                    levels=level_CI_contour,
                                    cmap='black',#plt.cm.gray,
                                    add_colorbar=False
                                    );
            ax.clabel(cont, inline=True, fontsize=15)
        else :
            cont = CI_contour.plot.contour(ax=ax,
                        transform=ccrs.PlateCarree(),
                        #levels=level_CI_contour,
                        cmap='black',#plt.cm.gray,
                        add_colorbar=False
                        );
            ax.clabel(cont, inline=True, fontsize=15)
    
    if  Liste_Corrmin_Corrmax is not None :
        cmap = color
        norm = mpl.colors.Normalize(vmin=Liste_Corrmin_Corrmax[0], vmax=Liste_Corrmin_Corrmax[1])
        
        cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap), orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
        cb.set_label(r"Sans unité (entre -1 et 1)",size=18,rotation=90,labelpad=1)
        cb.ax.tick_params(labelsize=18)
            
    else :
        cb = plt.colorbar(im, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
        cb.set_label(r"Sans unité (entre -1 et 1)",size=18,rotation=90,labelpad=1)
        cb.ax.tick_params(labelsize=18)
        
    if titre is not None:
        ax.set_title(titre[1],fontsize=20)
    
    
    
    if path_save is not None :
        fig.savefig(path_save)
        
    plt.show()
    
    
# In[creation DataArray]

def creationDataArray(Data,Latitudes,Longitudes,time = None,nameDataArray="DataArray"):
    #print('time creation DataArray : ',time)
    # Creation d'un xarray de données issue de l'algo
    shape = Data.shape
    n = len(shape)
    
    if (n == 2):
        # print('time is none')
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
        
        if (time is None):
            time = range(shape[0])
        
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
    else :
        data_DataArray = None
    
    return data_DataArray
