# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:04:34 2022

@author: Bastien Piron
"""

import numpy as np

import matplotlib.pyplot as plt

import sys



def Construction_matrice(N):
    
    Dim=N+1;
    
    Me=np.zeros((2,2),dtype="float64");
    Ke=np.zeros((2,2),dtype="float64");
    
    M=np.zeros((Dim,Dim),dtype="float64");
    K=np.zeros((Dim,Dim),dtype="float64");

    
    Ke[0,0]=1.0;
    Ke[0,1]=-1.0;
    Ke[1,0]=-1.0;
    Ke[1,1]=1.0;


    Me[0,0]=2.0;
    Me[0,1]=1.0;
    Me[1,0]=1.0;
    Me[1,1]=2.0;



    for k in range(N):
        for i in range(2):
            for j in range(2):
                M[k+i,k+j]= M[k+i,k+j]+Me[i,j];
                K[k+i,k+j]= K[k+i,k+j]+Ke[i,j];
                
    return M,K



def Analyse_modale(Ls, bs, hs, Es, Ns, rho_s, Ss, Lc, Nc, rho_c, c, Sc ):
    
    dxs=Ls/Ns
    dxc=Lc/Nc
    #doit respecter l'hypothése des poutres
    
    Dims=Ns+1
    
    Dimc=Nc+1
    
    ########################################
    #Décalration des variables
    ########################################
    
    Ms=np.zeros((Dims,Dims),dtype="float64")
    Ks=np.zeros((Dims,Dims),dtype="float64")
    
    
    Mc=np.zeros((Dimc,Dimc),dtype="float64")
    Kc=np.zeros((Dimc,Dimc),dtype="float64")
    
    Mcouplage=np.zeros((Dimc,Dims),dtype="float64")
    Kcouplage=np.zeros((Dims, Dimc),dtype="float64")
    
    M=np.zeros((Dims+Dimc,Dimc+Dims),dtype="float64")
    K=np.zeros((Dims+Dimc,Dimc+Dims),dtype="float64")
    
    
    # Mc=np.zeros((Dim2,Dim2),dtype="float64");
    # Kc=np.zeros((Dim2,Dim2),dtype="float64");
    
    
    ###############################################
    #Construction et assemblage des matrices M et K
    ###############################################
    
    #----------Création matrices de la poutre---------------
    Ms,Ks=Construction_matrice(Ns)
    
    Ms=Ms*rho_s*Ss*dxs/6.0
    Ks=Ks*Es*Ss/dxs
    
    #---------Création matrice de la cavité---------------
    Mc,Kc=Construction_matrice(Nc)
    
    Mc=Mc*Sc*dxc/(6.0*c*c)
    Kc=Kc*Sc/dxc
    
    #---------Création matrice de couplage---------------
    
    Mcouplage[0,Dims-1]=1.0
    Mcouplage=Mcouplage*Sc*rho_c
    
    Kcouplage[Dims-1,0]=1.0
    Kcouplage=-Sc*Kcouplage
    
    #----------Assemblage de la matice de raideur K ----------
    
    #Matrice de raideur de la structure:
    K[0:Dims,0:Dims]=K[0:Dims,0:Dims]+Ks[:,:]
    
    #Matrice de raideur de couplage: 
    K[0:Dims,Dims:]=K[0:Dims,Dims:]+Kcouplage[:,:]
    
    #Matrice de raideur de la cavité:
    K[Dims:,Dims:]= K[Dims:,Dims:]+Kc[:,:]  
    
    #----------Assemblage de la matice de masse M ----------
    
    #Matrice de masse de la structure:
    M[0:Dims,0:Dims]=M[0:Dims,0:Dims]+Ms[:,:]
    
    #Matrice de masse de couplage: 
    M[Dims:,0:Dims]=M[Dims:,0:Dims]+ Mcouplage[:,:]
    
    #Matrice de masse de la cavité:
    M[Dims:,Dims:]= M[Dims:,Dims:]+Mc[:,:]  
            
    
    ###################
    #Conditions limites
    ###################
    
    #Condition d'encastrement:
    
    Mf=np.zeros((Dims+Dimc-1,Dims+Dimc-1),dtype="float64");
    Kf=np.zeros((Dims+Dimc-1,Dims+Dimc-1),dtype="float64");
    
    Mf[:,:]= Mf[:,:]+M[1:,1:];
    Kf[:,:]= Kf[:,:]+K[1:,1:];
    
    # for i in range(Dims+Dimc-1):
    #     for j in range(Dims+Dimc-1):
    #         Mf[i,j]= Mf[i,j]+M[1+i,1+j];
    #         Kf[i,j]= Kf[i,j]+K[1+i,1+j];
    
    
    ##########################
    #Résolution analyse modale
    #########################
    
    
    
    W,V=np.linalg.eig(np.linalg.inv(Mf)@Kf);
    
    
    W_tri=np.zeros((Dimc+Dims-1,2),dtype="float64")
    W_tri2=np.zeros((Dims+Dimc-1,1),dtype="float64")
    V_reel=np.zeros((Dims+Dimc,Dims+Dimc-1),dtype="float64")
        
    #Tri des valeurs:
    
    for i in range(Dims+Dimc-1):
        if abs(W[i])<10e-3:
            W[i]=0.0 #filtrage
        W_tri[i,0]=np.sqrt(W[i])/(2.0*np.pi)
        W_tri[i,1]=i
     
    
    idx = np.argsort(W_tri[:,0])
    
    
    for j in range(Dims+Dimc-1):
        W_tri2[j] =W_tri[idx[j],0]
        for i in range(Dims+Dimc-1):
            V_reel[i+1,j]=V[i,idx[j]]  
    
    
    
    print("Fréquences de résonnance du cavité+poutre (TC) : \n ", W_tri2)

    return W_tri2, V_reel

def graph(V_reel, Ns, Ls, Lc, Nc):
    
    #Pour affichage, Aff_min et Aff_max compris entre 0 et Dims+Dimc-1:
    Dims=Ns+1
    
    Dimc=Nc+1

    Aff_min=3
    Aff_max=Aff_min+6


    if (Aff_min>Aff_max):
        
        print('Aff_min>Aff_max \n')
        sys.exit()
        
    if(Aff_max>Dims+Dimc-1):
        print('Aff_max supérieur à Dims+Dimc-1\n')
        sys.exit()


    fig, axs = plt.subplots(Aff_max-Aff_min, 2,sharex='col', figsize=(10,10))
    X_solide=np.linspace(0, Ls,Ns+1)
    X_cavite=np.linspace(0,Lc,Nc+1)





    for i in range(Aff_max-Aff_min):
        
        if(i==Aff_max-Aff_min-1):
            axs[i,0].plot(X_solide,V_reel[0:Ns+1,i+Aff_min],color='red');
            axs[i,0].set(ylabel='M°{}'.format(i+Aff_min))
            axs[i,0].set(xlabel='Ls(m)')
            axs[i,0].set_yticks([0])
            axs[i,0].grid(True)
            axs[i,1].plot(X_cavite,V_reel[Ns+1:Ns+Nc+2,i+Aff_min], color='blue');
            axs[i,1].set(xlabel='Lf(m)')
            axs[i,1].set_yticks([0])
            axs[i,1].grid(True)
            
        axs[i,0].plot(X_solide,V_reel[0:Ns+1,i+Aff_min],color='red');
        axs[i,0].set_yticks([0])
        axs[i,0].set(ylabel='M°{}'.format(i+Aff_min))
        axs[i,0].grid(True)
        #axs[i,0].set(ylabel='u')
        axs[i,1].plot(X_cavite,V_reel[Ns+1:Ns+Nc+2,i+Aff_min], color='blue');
        axs[i,1].set_yticks([0])
        axs[i,1].grid(True)
        #axs[i,1].set(ylabel='p')
       
    axs[0,0].set(title='Déplacements')  
    axs[0,1].set(title='Pressions (relative)')  

    return None
        
        
    
    
"""

PROGRAMME PRINCIPAL

Ce programme permet de réaliser une analyse d une poutre modale en 1D
Les variables modifiables sont présentées ci-dessous.

Variables:
    Poutre:
        Ls: Longueur de la poutre
        bs: épaisseur de la poutre
        hs: hauteur de la poutre
        Es: Module d Young
        Ns: nbr de point de discrétion selon la longueur
        Is: Moment quadratique de la poutre
        rho_s: Densité de la poutre
        Ss: Section de la poutre
        dxs: pas de discrétsation 
    
    Cavité:
        Lc: Longueur de la cavité (1D)
        bc: largeur de la cavité (identique à la poutre)
        hc: hauteur de la cavité (identique à la poutre)
        Nc: nbr de point de discrétion selon la longueur
        rho_c: Densité du fluide dans la cavité
        Sc: Section de la cavité (identique à la poutre)
        dxc: pas de discrétsation de la cavité 
    

"""
#------------------------Paramétres---------------------
##############
#Structure
##############

Ls=1.0#m
bs=0.01#m
hs=0.01#m

Es=205.0e9
Ns=5#nombre d'éléments



rho_s=7800.0;

Ss=bs*hs

dxs=Ls/Ns
##################
#Cavité
##################    
    
Lc=5.0#m
Nc=5#nombre d'éléments


rho_c=7000.0
c=340.0
Sc=Ss

dxc=Lc/Nc 


#------------------------------------------------------

# W,V=Analyse_modale(Ls, bs, hs, Es, Ns, rho_s, Ss, Lc, Nc, rho_c, c, Sc )
# graph(V, Ns, Ls, Lc, Nc)


