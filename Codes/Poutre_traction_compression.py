# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 12:05:14 2023

@author: Bastien Piron
"""

"""
Created on Mon Oct 10 14:31:15 2022

@author: Jérémy Archier - jeremy.archier@etu.univ-lyon1.fr

Résolution de l'équation de Helmotz par la méthode des éléments finis sur une
cavité rectangulaire de longueur Lx.

Paramètres:
    
    - n : le mode en x
    - Lx : la longeur de la cavité [m]
    - n : le nombre de point de discrétisation
    - p : le champ de pression de la cavité
    - M : la matrice de masse élémentaire
    - K : la matrice de raideur élémentaire

"""

import matplotlib.pyplot as plt
import numpy as np



def Analyse_modale(n,rho, E, S, L):
    
    
    dx=float(L/n)
    #---------------------------Matrices élémentaires------------------------------
    
    Me = np.zeros((2,2))
    Ke = np.zeros((2,2))
    
    Me[0,0] = 2.0
    Me[1,0] = 1.0
    Me[1,1] = 2.0
    Me[0,1] = 1.0
    
    Ke[0,0] = 1.0
    Ke[1,0] = -1.0
    Ke[1,1] = 1.0
    Ke[0,1] = -1.0
    
    
    Me = Me *rho*S*dx/6.0
    Ke = Ke *E*S/dx
    

    #-------------------------------Assemblage-----------------------------------
    
    M = np.zeros((n+1,n+1))
    K = np.zeros((n+1,n+1))
    
    for k in range(n):
        for i in range(2):
            for j in range(2):
                M[k+i,k+j] = M[k+i,k+j]+Me[i,j];
                K[k+i,k+j] = K[k+i,k+j]+Ke[i,j];
    
    #---------------------------------Résolution-----------------------------------
    
    
    W,V=np.linalg.eig(np.dot(np.linalg.inv(M[1:,1:]),K[1:,1:])) #On récupère les valeurs et vecteurs propres
    
    
    W_tri=np.zeros((n,2),dtype="float64");
    W_tri2=np.zeros((n,1),dtype="float64");    
    V_reel=np.zeros((n+1,n),dtype="float64");
        
    #Tri des valeurs:
    
    for i in range(n):
        W_tri[i,0]=np.sqrt(W[i])/(2.0*np.pi)
        W_tri[i,1]=i;
     
    
    idx = np.argsort(W_tri[:,0])
    
    
    for j in range(n):
        W_tri2[j] =W_tri[idx[j],0]
        for i in range(n):
            V_reel[i+1,j]=V[i,idx[j]];    
    
    print('fréquence de résonance poutre en TC:',W_tri2)
    
    return W_tri2,V_reel
    
 
    
def graph(n, L, W,V_reel):
  
  
    fig, axs = plt.subplots(n, 1,sharex='col',figsize=(6,10))
    X=np.linspace(0, L,n+1)
    
    
    for i in range(n):
    
    
        axs[i].plot(X,V_reel[:,i],color='red');
        axs[i].set(ylabel='M°{}'.format(i+1))
        
    
    axs[0].set(title='Déplacements')  
     
    plt.xlabel('L(m)')
    
    
    return None
      
     
     
     
#--------------------------------Paramètres------------------------------------

n = 5
rho = 7800.0
E=205.0e9 # densité air
S = 0.001
L = 1.0


# W,V=Analyse_modale(n,rho, E, S, L)
# graph(n, L, W,V)