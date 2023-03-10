# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 12:29:39 2023

@author: Bastien Piron
"""

import ballotement
import Poutre_flexion
import cavite_2D_carre

import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.linalg as spl


"""

PROGRAMME PRINCIPAL

Ce programme permet de réaliser une analyse d une poutre modale en 1D
Les variables modifiables sont présentées ci-dessous.

Variables:
    Poutre:
        Lx: Longueur de la poutre
        bs: épaisseur de la poutre
        hs: hauteur de la poutre
        Es: Module d Young
        Nx: nbr de point de discrétion selon la longueur
        Is: Moment quadratique de la poutre
        rho_s: Densité de la poutre
        Ss: Section de la poutre
        dxs: pas de discrétsation 
    
    Cavité & Ballotement:
        Lx: Longueur de la cavité (identique à la poutre)
        Ly: Hauteur de la cavité
        Nx: nbr de point de discrétion selon la longueur (identique à la poutre)
        Ny: nbr de point de discrétion selon la hauteur
        rho_c: Densité du fluide dans la cavité
        rho_b: Densité du fluide dans la ballotement
        dx: pas de discrétsation de la cavité (identique à la poutre)
        dy: pas de discrétsation de la cavité 
    

"""

def Analyse_modale(Nx,Ny, Lx, Ly, dx, dy, Ss, rho_c, rho_b, rho_s, Es, Is, c):
    
    NE=Nx*Ny#Taille matrices cavité
    Ns=Nx*2


    Ms=np.zeros((Ns,Ns),dtype="float64")
    Ks=np.zeros((Ns,Ns),dtype="float64")


    Mc=np.zeros((NE,NE),dtype="float64")
    Kc=np.zeros((NE,NE),dtype="float64")

    Mb=np.zeros((NE,NE),dtype="float64")
    Kb=np.zeros((NE,NE),dtype="float64")


    Kcouplageb=np.zeros((NE, Ns),dtype="float64")
    Mcouplageb=np.zeros((Ns,NE),dtype="float64")

    Kcouplagec=np.zeros((Ns, NE),dtype="float64")
    Mcouplagec=np.zeros((NE,Ns),dtype="float64")

    M=np.zeros((2*NE+Ns-4,2*NE+Ns-4),dtype="float64")
    K=np.zeros((2*NE+Ns-4,2*NE+Ns-4),dtype="float64")

    ###############################################
    #Construction et assemblage des matrices M et K
    ###############################################


    #----------Création matrices de la poutre---------------
    print('Calcul matrices poutre:', end=" ")
    Ms,Ks=Poutre_flexion.matrices_flexion(Nx-1, Lx, Es, Is, rho_s, Ss)

    print('Done')

    #---------Création matrice de la cavité---------------
    print('Calcul matrices ballottement:', end=" ")
    Mb,Kb,Mcouplageb=ballotement.matrices_2D(Lx, Ly, Nx, Ny, dx, dy)

    Mb=Mb/9.81
    print('Done')

    #---------Création matrice de la cavité---------------
    print('Calcul matrices cavité:', end=" ")
    Mc,Kc,Mcouplagec=cavite_2D_carre.matrices_2D(Lx, Ly, Nx, Ny, dx, dy)

    Mc=Mc/c**2
    print('Done')


    #---------Création matrices de couplage cavité ---------------

    Kcouplagec=np.transpose(Mcouplagec)

    Mcouplagec=rho_c*Mcouplagec

    #---------Création matrices de couplage ballotement ---------------

    Kcouplageb=np.transpose(Mcouplageb)

    Mcouplageb=rho_b*Mcouplageb

    #----------Assemblage de la matice de raideur K ----------

    print('Assemblage du systéme:', end=" ")

    #Matrice de raideur du ballotement:
    K[:NE,:NE]=K[:NE,:NE]+Kb[:,:]

    #Matrice de raideur de la structure:
    K[NE:NE+Ns-4,NE:NE+Ns-4]=K[NE:NE+Ns-4,NE:NE+Ns-4]+Ks[2:Ns-2,:2:Ns-2]

    #Matrice de raideur de la cavité:
    K[NE+Ns-4:,NE+Ns-4:]=K[NE+Ns-4:,NE+Ns-4:]+Kc[:,:]  

    #Matrice de raideur de couplage poutre/ballottement: 
    K[NE:NE+Ns-4,:NE]=K[NE:NE+Ns-4,:NE]-Kcouplageb[2:Ns-2,:]

    #Matrice de raideur de couplage poutre/cavité: 
    K[NE:NE+Ns-4,NE+Ns-4:]=K[NE:NE+Ns-4,NE+Ns-4:]-Kcouplagec[2:Ns-2,:]

    #----------Assemblage de la matice de masse M ----------

    #Matrice de masse du ballotement:
    M[:NE,:NE]=M[:NE,:NE]+Mb[:,:]

    #Matrice de masse de la structure:
    M[NE:NE+Ns-4,NE:NE+Ns-4]=M[NE:NE+Ns-4,NE:NE+Ns-4]+Ms[2:Ns-2,:2:Ns-2]

    #Matrice de masse de la cavité:
    M[NE+Ns-4:,NE+Ns-4:]=M[NE+Ns-4:,NE+Ns-4:]+Mc[:,:]  

    #Matrice de masse de couplage poutre/ballottement: 
    M[:NE,NE:NE+Ns-4]=M[:NE,NE:NE+Ns-4]+Mcouplageb[:,2:Ns-2]

    #Matrice de masse de couplage poutre/cavité: 
    M[NE+Ns-4:,NE:NE+Ns-4]=M[NE+Ns-4:,NE:NE+Ns-4]+Mcouplagec[:,2:Ns-2]



    print('Done')        


    ##########################
    #Résolution analyse modale
    #########################

    print('Résolution:', end=" ")

    W,V=spl.eig(K,M)

    print('Done') 
    print(W)
    W_tri=np.zeros((Ns+Nx-4+NE,2),dtype="float64")

    V_reel=np.zeros((Ns+NE-4+NE,Ns+Nx-4+NE),dtype="float64")
     

    #Tri des valeurs:
    compteur=0
     
    for i in range(len(W)):
        if abs(W[i].real)<10e-1:
            W[i]=0.0 #filtrage
        if W[i].real != np.inf and W[i].real > 0.5:
            W_tri[compteur,0]=np.sqrt(abs(W[i].real))/(2.0*np.pi)
            W_tri[compteur,1]=compteur
            V_reel[:,compteur]=V[:,i].real
            compteur=compteur+1
            
     

    idx = np.argsort(W_tri[:compteur-1,0])

    W_tri2=np.zeros((compteur-1,1),dtype="float64")  
    V_reel2=np.zeros((Ns+NE*2,compteur-1),dtype="float64")

    for j in range(compteur-1):
        W_tri2[j] =W_tri[idx[j],0]
        
        V_reel2[:NE,j]= V_reel[:NE,idx[j]]
        V_reel2[NE+2:NE+Ns-2,j]=V_reel[NE:NE+Ns-4,idx[j]]
        V_reel2[NE+Ns:,j]=V_reel[NE+Ns-4:,idx[j]]



    print("Fréquences de résonnance ballotement+poutre+cavité (10 premières) : \n ", W_tri2[:10])
    
    return W_tri2, V_reel2, compteur



def graph(Lx, Ly, Nx, Ny, compteur,V_reel2, Aff_min):
    
    NE=Nx*Ny#Taille matrices cavité
    Ns=Nx*2
    Z=np.zeros((Ny,Nx),dtype="float64")


    x_graph=np.linspace(0,Lx,Nx)
    y_graph=np.linspace(0,Ly,Ny)

    X,Y=np.meshgrid(x_graph,y_graph)


    Aff_max=Aff_min+2


    if (Aff_min>Aff_max):
        
        print('Aff_min>Aff_max \n')
        sys.exit()
        
    if(Aff_max>compteur-1):
        print('Aff_max supérieur à Dims+Dimc-1\n')
        sys.exit()



    nbr_x=10; #pour le calcul des déformés


    x=np.linspace(0, dx,nbr_x);#discrétisation d'un élémént en une quantité de point

    N=Nx-1

    Def_modale=np.zeros(((nbr_x-1)*N+1,compteur-1),dtype="float64");
    X_plt=np.zeros(((nbr_x-1)*N+1),dtype="float64");


    for j in range(compteur-1):
        for k in range(N):
            for i in range(nbr_x):
                X_plt[k*(nbr_x-1)+i]=k*dx+x[i];#liste des abscisses des différents points
                Def_modale[k*(nbr_x-1)+i,j]=np.sign(V_reel2[NE+2*k,j])*Poutre_flexion.N1(x[i],dx)+V_reel2[NE+2*k+1,j]*Poutre_flexion.N2(x[i],dx)+np.sign(V_reel2[NE+2*k+2,j])*Poutre_flexion.N3(x[i],dx)+V_reel2[NE+2*k+3,j]*Poutre_flexion.N4(x[i],dx)
        #         Def_modale[k*(nbr_x-1)+i,j]=c*Poutre_flexion.N1(x[i],dx)+10.0*V_reel[2*k+1,j]*Poutre_flexion.N2(x[i],dx)+c*Poutre_flexion.N3(x[i],dx)+10.0*V_reel[2*k+3,j]*Poutre_flexion.N4(x[i],dx)
                

        # Def_modale=Def_modale-c
        # Def_modale[0,j]=0.0
        # Def_modale[(nbr_x-1)*N,j]=0.0

    fig, axs = plt.subplots(3, 2,sharex='col', figsize=(10,10), gridspec_kw={'height_ratios': [2,1,2]})

    V_reel_max=max(np.max(V_reel2[:NE,Aff_min:Aff_max]),np.max(V_reel2[NE+Ns:,Aff_min:Aff_max]))*1.5
    V_reel_min=min(np.min(V_reel2[:NE,Aff_min:Aff_max]),np.min(V_reel2[NE+Ns:,Aff_min:Aff_max]))*1.5


    for p in range(0,Aff_max-Aff_min):

            
        for k in range(Ny):
            for j in range(Nx):
                Z[k,j]=V_reel2[Nx*k+j,p+Aff_min]#/np.linalg.norm(V_reel2[Ns:NE+Ns,i+p+Aff_min])
        
        c=axs[0,p].contourf(X,Y,Z,100, cmap='jet', vmin=V_reel_min, vmax=V_reel_max, extend='neither')
        axs[0,p].set_ylabel('y(m)')
        axs[0,p].set_xlabel('x(m)')
        axs[0,p].set_title('Pressions (relatives) - Mode {}'.format(1+p+Aff_min)) 
        #fig.colorbar(c, ax=axs[i+1,p])        
        
        
        axs[1,p].plot(X_plt,Def_modale[:,p+Aff_min],color='red')
        # axs[i,p].plot(x_graph,V_reel[:Ns:2,i+p+Aff_min],color='red')
        axs[1,p].set_yticks([0])
        axs[1,p].grid(True)
        axs[1,p].set(title='Déplacements - Mode {}'.format(p+Aff_min+1))  
        
        for k in range(Ny):
            for j in range(Nx):
                Z[k,j]=V_reel2[NE+Ns+Nx*k+j,p+Aff_min]#/np.linalg.norm(V_reel2[NE+Ns:,i+p+Aff_min])
        
        c=axs[2,p].contourf(X,Y,Z,100, cmap='jet', vmin=V_reel_min, vmax=V_reel_max, extend='neither')
        axs[2,p].set_ylabel('y(m)')
        axs[2,p].set_xlabel('x(m)')
        axs[2,p].set_title('Pressions (relatives) - Mode {}'.format(p+Aff_min+1)) 
        #fig.colorbar(c, ax=axs[i+1,p])
            



    # fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.02, hspace=0.02)
    # cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    # cbar = fig.colorbar(c, ax=axs.ravel().tolist(), orientation='horizontal', extend='max')
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    cax = fig.add_axes([0.12, 0.05, 0.78, 0.02])
    cbar = fig.colorbar(c, cax=cax, orientation='horizontal')

    fig.show() 

    #plt.tight_layout()
    
    
    
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

Ny=40#points de discrétisation selon y

dy=float(Ly/(Ny-1));#pas selon y    

rho_c=1.2
rho_b=1000.0
c=340.0
 


b_inf=0
b_sup=40
pas=2

#------------------------------------------------------

W,V,compteur=Analyse_modale(Nx,Ny, Lx, Ly, dx, dy, Ss, rho_c, rho_b, rho_s, Es, Is, c)

# for Aff_min in range(b_inf,b_sup,pas):

#     graph(Lx, Ly, Nx, Ny, compteur,V, Aff_min)