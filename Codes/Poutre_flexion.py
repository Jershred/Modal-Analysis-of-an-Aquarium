# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:55:42 2022

@author: Bastien Piron
"""

import numpy as np

import matplotlib.pyplot as plt

import sys

def N1(x,dx):
    
    e=x/dx;
    
    return 1.0-3.0*e*e+2.0*e*e*e

def N2(x,dx):
    
    e=x/dx;
    
    return dx*e*(1.0-e)*(1.0-e)

def N3(x,dx):
    
    e=x/dx;
    
    return e*e*(3.0-2.0*e)

def N4(x,dx):
    
    e=x/dx;
    
    return dx*e*e*(e-1.0)





def matrices_elementaires_flexion(dx, rho, S, E, I):
    
    Me=np.zeros((4,4),dtype="float64");
    Ke=np.zeros((4,4),dtype="float64");
    
    Ke[0,0]= 12.0
    Ke[0,1]= 6.0 * dx
    Ke[0,2]= - 12.0
    Ke[0,3]= 6.0 * dx
    
    Ke[1,0]= 6.0 * dx
    Ke[1,1]= 4.0 * dx * dx
    Ke[1,2]= - 6.0 * dx
    Ke[1,3]= 2.0 * dx * dx
    
    Ke[2,0]= -12.0
    Ke[2,1]= - 6.0 * dx
    Ke[2,2]= 12.0
    Ke[2,3]= - 6.0 * dx
    
    Ke[3,0]= 6.0 * dx
    Ke[3,1]= 2.0 * dx * dx
    Ke[3,2]= - 6.0 * dx
    Ke[3,3]= 4.0 * dx * dx
    
    Me[0,0]= 156.0
    Me[0,1]= 22.0 * dx
    Me[0,2]= 54.0
    Me[0,3]= - 13.0 * dx
    
    Me[1,0]= 22.0 * dx
    Me[1,1]= 4.0 * dx * dx
    Me[1,2]= 13.0 * dx
    Me[1,3]= - 3.0 * dx * dx
    
    Me[2,0]= 54.0
    Me[2,1]= 13.0 * dx
    Me[2,2]= 156.0
    Me[2,3]= - 22.0 * dx
    
    Me[3,0]= - 13.0 * dx
    Me[3,1]= - 3.0 * dx * dx
    Me[3,2]= - 22.0 * dx
    Me[3,3]= 4.0 * dx * dx

    return Me*rho*S*dx/420.0, Ke*E*I/(dx*dx*dx)



def matrices_flexion(N,L,E,I,rho,S):
    
    Dim=4+(N-1)*2;
    dx=float(L/N)
    
    Me,Ke=matrices_elementaires_flexion(dx, rho, S, E, I)
    
    M=np.zeros((Dim,Dim),dtype="float64");
    K=np.zeros((Dim,Dim),dtype="float64");
    
    for k in range(N):
        for i in range(4):
            for j in range(4):
                M[2*k+i,2*k+j]= M[2*k+i,2*k+j]+Me[i,j];
                K[2*k+i,2*k+j]= K[2*k+i,2*k+j]+Ke[i,j];

    return M,K

def Analyse_modale(N, dx, L, rho, S, E, I, b, h):
    

    Dim=4+(N-1)*2;

    Dim2=Dim-4;

    M=np.zeros((Dim,Dim),dtype="float64");
    K=np.zeros((Dim,Dim),dtype="float64");

    Mc=np.zeros((Dim2,Dim2),dtype="float64");
    Kc=np.zeros((Dim2,Dim2),dtype="float64");

    M,K=matrices_flexion(N,L,E,I,rho,S)

    #---------------------Conditions limites-------------------

    """
    Mettre les conditions limites dans la matrice
    Modifier le tracer des vecteurs propres

    """

    Mc=np.zeros((Dim2,Dim2),dtype="float64");
    Kc=np.zeros((Dim2,Dim2),dtype="float64");

    for i in range(Dim2):
        for j in range(Dim2):
            Mc[i,j]= Mc[i,j]+M[2+i,2+j];
            Kc[i,j]= Kc[i,j]+K[2+i,2+j];



    W,V=np.linalg.eig(np.linalg.inv(Mc)@Kc);


    W_tri=np.zeros(((N-1)*2,2),dtype="float64");
    W_tri2=np.zeros(((N-1)*2,1),dtype="float64");    
    V_reel=np.zeros(((N+1)*2,(N-1)*2),dtype="float64");
        
    #Tri des valeurs:

    for i in range((N-1)*2):
        W_tri[i,0]=np.sqrt(W[i])/(2.0*np.pi)
        W_tri[i,1]=i;
     

    idx = np.argsort(W_tri[:,0])


    for j in range((N-1)*2):
        W_tri2[j] =W_tri[idx[j],0]
        for i in range((N+1)*2-4):
            V_reel[i+2,j]=V[i,idx[j]];    
        


    print("Fréquences de résonnance poutre flexion: \n ", W_tri2)


    return W_tri2, V_reel



def graph_mode(W,N, dx, V_reel):
    

    nbr_x=10; #pour le calcul des déformés
    x=np.linspace(0, dx,nbr_x);#discrétisation d'un éléméent en une quantité de point

    Def_modale=np.zeros(((nbr_x-1)*N+1,(N-1)*2),dtype="float64");
    X_plt=np.zeros(((nbr_x-1)*N+1),dtype="float64");

   


    for j in range((N-1)*2):
        for k in range(N):
            for i in range(nbr_x):
                X_plt[k*(nbr_x-1)+i]=k*dx+x[i];#liste des abscisses des différents points
                Def_modale[k*(nbr_x-1)+i,j]=V_reel[2*k,j]*N1(x[i],dx)+V_reel[2*k+1,j]*N2(x[i],dx)+V_reel[2*k+2,j]*N3(x[i],dx)+V_reel[2*k+3,j]*N4(x[i],dx)
                

      




    Aff_min=0
    Aff_max=4


    if (Aff_min>Aff_max):
        
        print('Aff_min>Aff_max \n')
        sys.exit()
        
    if(Aff_max>(N-1)*2):
        print('Aff_max supérieur à (N-1)*2 \n')
        sys.exit()



    fig, axs = plt.subplots(Aff_max-Aff_min, 1,sharex='col',figsize=(6,10))

    for i in range(Aff_max-Aff_min):
        axs[i].plot(X_plt,Def_modale[:,i+Aff_min]);
        axs[i].set_title('Mode {}'.format(i+Aff_min+1))
        axs[i].set(ylabel='V')
        axs[i].grid()

    plt.xlabel('L(m)')
    plt.tight_layout()
    # for ax in fig.get_axes():
    #     ax.label_outer()

    return None

"""

PROGRAMME PRINCIPAL

Ce programme permet de réaliser une analyse d une poutre modale en 1D
Les variables modifiables sont présentées ci-dessous.

Variables:
    L: Longueur de la poutre
    b: épaisseur de la poutre
    h: hauteur de la poutre
    E: Module d Young
    N: nbr de point de discrétion selon la longueur
    I: Moment quadratique de la poutre
    rho: Densité de la poutre
    S: Section de la poutre
    dx: pas de discrétsation ()

    

"""
#------------------------Paramétres---------------------
L=1.0;#m
b=0.01;#m
h=0.01;#m

E=205.0e9;
N=4#nombre d'éléments


#------------------------------------------------------


I=b*h*(b**2+h**2)/12.0

rho=7800.0;

S=b*h;

dx=L/N; #doit respecter l'hypothése des poutres

# W,V=Analyse_modale(N, dx, L, rho, S, E, I, b, h)
# graph_mode(W,N, dx,V)

