# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 18:12:51 2022

@author: Bastien Piron
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

import Poutre_flexion
import Poutre_traction_compression
import cavite_1D
import Couplage_struct_cavite_1D
import ballotement
import cavite_2D_carre
import Couplage_cavite2D_poutre_flexion
import Couplage_ballotement_poutreflexion
import Couplage_aquarium

def get_frequence(N):

    #------------------------Paramétres---------------------
    ##############
    #Structure
    ##############
    Lx=1.0;#m
    Nx=10#points de discrétisation selon x
    dx=float(Lx/(Nx-1));#pas selon x
    
    bs=0.01;#m
    hs=0.01;#m
    
    Es=205.0e9;
    
    Is=bs*hs*(bs*bs+hs*hs)/12.0;
    
    rho_s=7800.0;
    
    Ss=bs*hs;
    
    
    ######################
    #Cavité & Ballotement
    ######################    
    
    Ly=0.5;#m
    
    Ny=N#points de discrétisation selon y
    
    dy=float(Ly/(Ny-1));#pas selon y    
    
    rho_c=1.2
    rho_b=1000.0
    c=340.0
     
    
    W,V,compteur=Couplage_aquarium.Analyse_modale(Nx,Ny, Lx, Ly, dx, dy, Ss, rho_c, rho_b, rho_s, Es, Is, c)
    # W,V,compteur=Couplage_ballotement_poutreflexion.Analyse_modale(Nx, Ny, dx, dy, Lx, Ly, Es, Is, rho_s,rho_b, Ss )
 
    # W,V,compteur=Couplage_cavite2D_poutre_flexion.Analyse_modale(Nx, Ny, dx, dy, Lx, Ly, Es, Is, rho_c, rho_s,Ss, c)
    # W,V_reel, compteur=cavite_2D_carre.Analyse_modale(dx,dy,c, rho_c, Lx, Ly, Nx,Ny)  
    # W,V_reel,compteur=ballotement.Analyse_modale(dx,dy,c, rho_b, Lx, Ly, Nx,Ny)  
   # W,V=Couplage_struct_cavite_1D.Analyse_modale(Ls, bs, hs, Es, N, rho_s, Ss, Lc, Nc, rho_c, c, Sc)
   # W,V=Poutre_flexion.Analyse_modale(N, dx, L, rho, S, E, I, b, h)
   # W,V=Poutre_traction_compression.Analyse_modale(N, rho, E, S, L)
   # W,V=cavite_1D.Analyse_modale(N, rho, c, L, S)
    return W


def graphflexion(N, Nmax, Num_f, nbr_mode):
    
    fig, ax = plt.subplots(nbr_mode, 1,figsize=(14,10))
    
    for k in range(nbr_mode):
        
       
        W=np.zeros((Nmax+1-N),dtype="float64");
        
        #Tri des fréquences propres dans l'ordre croissant
        for i in range(N,Nmax+1):
           W_n=get_frequence(i)
           check_up_para(W_n, N, Nmax, Num_f)
           W[i-N]=W_n[Num_f]
           
    
    
        #--------------Graphique----------------
        #Paramétres graphiques:
        maxi=W.max()
        mini=W.min()
        
        b_inf=mini-(maxi-mini)*0.5 #coeff à ajuster pour l'affichage
        b_sup=maxi+(maxi-mini)*0.5 #coeff à ajuster pour l'affichage
        
        color=['black','red','orange','yellow','green','blue','purple']
    
        
        #Graphique:
    
       
        
        plt.xlim([b_inf, b_sup]) #limite l'axe des abscisses
        plt.ylim([-2.0, 2.0]) #limite l'axe des ordonnées 
        ax[k].yaxis.set_ticks([0.0])
        ax[k].tick_params(axis='both', which='major', labelsize=18)
        ax[k].tick_params(axis='y', labelsize=0, length = 0)
        # ax[k].get_yaxis().set_visible(False) #retire l'axe des ordonnées
        
        ax[k].grid() #grille
        ax[k].plot([b_inf, b_sup], [0,0], color='black', linewidth=0.2) #Ligne y=0 (axe médian)
        
        for i in range(Nmax+1-N):
            ax[k].plot([W[i],W[i]],[-1.0/(i+1.0)**0.7,1.0/(1.0+i)**0.7],color[i], label='N {}'.format((N+i)*10*2+10),linewidth=5) #Rtrace les fréquences sous foorme de barre
    
           
         
        
        ax[k].legend(loc="right", fontsize=15)
        ax[k].set_ylabel('f n°{}'.format(Num_f+1), fontsize=20)
        
        
        N=N+1
        Nmax=Nmax+1
        
        Num_f=Num_f+1
        
    plt.tight_layout()
    plt.xlabel('Fréquence (Hz)', fontsize=18)
  
    #------------------------------------


def check_up_para(W_n, N, Nmax, Num_f):
    
    if (Num_f+1>len(W_n)):
        
        print('Ordre N=',N,'ne contient pas la fréquence à étudier \n')
        sys.exit()
        
    if((Nmax-N)>6):
        print('Nmax-N=6 non respecté i.e Nmax-N=',Nmax-N,'\n')
        sys.exit()
        
    
"""

Programme principal

Ce programme permet de tracer l'évolution de la valeur d'une fréquence propre (Num_f) 
en fonction de l ordre N. Les valeurs doivent converger vers la fréquence propres exactes avec N->infini

Il est possible de choisir l'ordre à partir du quel l'on veut tracer les fréquences.
Généralement, l'ordre auquel apparait la fréquence, donne une valeur trés grossiére de cette fréquence.
Pour des raisons de lisibilité et de représentation, il est préférable de tracer au moins à l'ordre suivant, si ce n'est plus...


Utilisation:

Remplacer le contenu de la fonction get_frequence par un code calculant les fréquences propres du systéme.
Elles doivent être stockées dans un vecteur (ligne ou Colonne) de la taille du nombre de fréquence.

"""

#-------------------Données----------------
N=7 #premier ordre des fréquences à oberserver 

Num_f=0 #Sélectionne la fréquence à observer (commence à 0)
nbr_mode=4

Nmax=13 #Dernier ordre des fréquences à observer (max=N+6)

#-------------------------------------



    
    
graphflexion(N, Nmax, Num_f,nbr_mode)
    

    

