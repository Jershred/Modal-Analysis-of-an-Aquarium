# -*- coding: utf-8 -*-
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
import sympy as sp


def Analyse_modale(n, rho, c, L, S):
    
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
    
    
    Me = Me * S*dx/(6.0*c*c)
    Ke = Ke * S/dx
    
    
    #---------------------------------Assemblage-----------------------------------
    
    M = np.zeros((n+1,n+1))
    K = np.zeros((n+1,n+1))
    
    for k in range(n):
        for i in range(2):
            for j in range(2):
                M[k+i,k+j] = M[k+i,k+j]+Me[i,j];
                K[k+i,k+j] = K[k+i,k+j]+Ke[i,j];
    
    #---------------------------------Résolution-----------------------------------
    
    
    W,V=np.linalg.eig(np.dot(np.linalg.inv(M),K)); #On récupère les valeurs et vecteurs propres
    
    
    W_tri=np.zeros((n+1,2),dtype="float64");
    W_tri2=np.zeros((n+1,1),dtype="float64");    
    V_reel=np.zeros((n+1,n+1),dtype="float64");
        
    #Tri des valeurs:
    
    for i in range(n+1):
        if abs(W[i])<10e-3:
            W[i]=0.0 #filtrage
        W_tri[i,0]=np.sqrt(W[i])/(2.0*np.pi)
        W_tri[i,1]=i;
     
    
    idx = np.argsort(W_tri[:,0])
    
    
    for j in range(n+1):
        W_tri2[j] =W_tri[idx[j],0]
        for i in range(n+1):
            V_reel[i,j]=V[i,idx[j]];    
    
    print("Fréquences de résonnance cavité 1D: \n ", W_tri2)
    
    return W_tri2, V_reel


def graph(V_reel,n):
    
    fig, axs = plt.subplots(4, 1,sharex='col', figsize=(6,10))
    X=np.linspace(0, L,n+1)


    for i in range(4):
        
        axs[i].grid()
        axs[i].plot(X,V_reel[:,i+1],color='red');
        axs[i].set(ylabel='M°{}'.format(i+2))


    axs[0].set(title='Pressions')  
     
    plt.xlabel('L(m)')



#--------------------------------Paramètres------------------------------------

n = 5
rho = 1000.0 # densité air
S = 0.001
L = 5.0
c = 340.0

# W,V=Analyse_modale(n, rho, c, L, S)

# graph(V,n)



