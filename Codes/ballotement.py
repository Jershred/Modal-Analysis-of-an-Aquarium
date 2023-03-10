# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 23:43:44 2023

@author: Bastien Piron
"""
import numpy as np
import sympy as sp
import sys
import matplotlib.pyplot as plt 
import Poutre_flexion
import scipy.linalg as spl

#Fonction créeant le maillage:
def tables(Lx,Ly,Nx,Ny,e):
    """ Description : Creation a partir du nombre de point et de la longueur du domaine la table de coord. globale des noeuds et la table de connexion pour un maillage triangulaire

        Donnees :   * Lx - Float : Longueur selon la direction x
                    * Ly - Float : Longueur selon la direction y
                    * Nx - Int   : Nbre point de discretisation selon x
                    * Ny - Int   : Nbre point de discretisation selon y

        Resultats : * Noeud : Table coord. gloable Noeuds
                    * Tbc : Table de connexion
    """
    nx = Nx - 1 # Nbre element sur x
    ny = Ny - 1 # Nbre element sur y


    lx = np.linspace(0,Lx,Nx)
    ly = np.linspace(0,Ly,Ny)
    Noeud = np.zeros((Nx*Ny,2))
    if e=='triangle':
        Tbc = np.zeros((2*nx*ny,3),dtype='int')
    elif e == 'carre':
        Tbc = np.zeros((nx*ny,4),dtype='int')
    

    Ne = 0
    Nn = 0
    i=0
    j=0
    compteur = 0

    while j < Ny-1:     # On se deplace sur les points sur y
        i = 0
        while i< Nx:  # On se deplace sur les points sur x
            if e == 'triangle':
                if 0<i and (Ne+1)%2 == 0:
                    A1=j*Nx+i
                    xA1 = lx[i]
                    yA1 = ly[j]
                    A2=(j+1)*Nx+i
                    xA2 = lx[i]
                    yA2 = ly[j+1]
                    A3=(j+1)*Nx+i-1
                    xA3 = lx[i-1]
                    yA3 = ly[j+1]

                elif 0<=i<=Nx-2:
                    A1=j*Nx+i
                    xA1 = lx[i]
                    yA1 = ly[j]
                    A2=j*Nx+i+1
                    xA2 = lx[i+1]
                    yA2 = ly[j]
                    A3=(j+1)*Nx+i
                    xA3 = lx[i]
                    yA3 = ly[j+1]
                    i=i+1

                elif (i+1)%Nx == 0:
                    break;

            elif e == 'carre':
                if (i+1)%Nx == 0:
                    break;
                else:
                    A1=j*Nx+i
                    xA1 = lx[i]
                    yA1 = ly[j]
                    A2=j*Nx+i+1
                    xA2 = lx[i+1]
                    yA2 = ly[j]
                    A3=(j+1)*Nx+i+1
                    xA3 = lx[i+1]
                    yA3 = ly[j+1]
                    A4=(j+1)*Nx+i
                    xA4 = lx[i]
                    yA4 = ly[j+1]
                    i=i+1

            if e == 'triangle':
                Tbc[Ne,0]=int(A1)
                Tbc[Ne,1]=int(A2)
                Tbc[Ne,2]=int(A3)
            elif e == 'carre':
                Tbc[Ne,0]=int(A1)
                Tbc[Ne,1]=int(A2)
                Tbc[Ne,2]=int(A3)
                Tbc[Ne,3]=int(A4)

            Noeud[A1,0] = xA1
            Noeud[A1,1] = yA1
            Noeud[A2,0] = xA2
            Noeud[A2,1] = yA2
            Noeud[A3,0] = xA3
            Noeud[A3,1] = yA3

            if e == 'carre':
                Noeud[A4,0] = xA4
                Noeud[A4,1] = yA4
            Ne = Ne + 1 # Numero de element
        j=j+1
    return Tbc,Noeud


#Fonction calculant les matrices Kx et Ky:
def MatK (hi, hj):
    
    K=np.zeros((4,4),dtype="float64");
   
    K[0,0]=(hi**2.0+hj**2.0)/(3.0*hi*hj)
    K[0,1]=(hi**2.0-2.0*hj**2.0)/(6.0*hi*hj)
    K[0,2]=-(hi**2.0+hj**2.0)/(6.0*hi*hj)
    K[0,3]=(hj**2.0-2.0*hi**2.0)/(6.0*hi*hj)
    K[1,0]=K[0,1]
    K[1,1]=K[0,0]
    K[1,2]=K[0,3]
    K[1,3]=K[0,2]
    K[2,0]=K[0,2]
    K[2,1]=K[0,3]
    K[2,2]=K[0,0]
    K[2,3]=K[0,1]
    K[3,0]=K[0,3]
    K[3,1]=K[0,2]
    K[3,2]=K[0,1]
    K[3,3]=K[0,0]

    
    return K



def MatM_ana (dx,dy):
    M=np.zeros((2,2),dtype="float64");
    
    M[0,0]=2.0
    M[0,1]=1.0
    M[1,0]=M[0,1]
    M[1,1]=M[0,0]
 
    return M*dx/6.0 



def MatCouplage(x, Ly, dx, a1, a2, Tbc, Noeud, E):
    
    Mcouplage=np.zeros((4,4),dtype="float64");
    
        
    Mcouplage[0,0]=sp.integrate( H1(x,Ly,E,Noeud, Tbc) * Poutre_flexion.N1(x,dx) , (x,a1,a2));
    Mcouplage[0,1]=sp.integrate( H1(x,Ly,E,Noeud, Tbc) * Poutre_flexion.N2(x,dx) , (x,a1,a2));
    Mcouplage[0,2]=sp.integrate( H1(x,Ly,E,Noeud, Tbc) * Poutre_flexion.N3(x,dx) , (x,a1,a2));
    Mcouplage[0,3]=sp.integrate( H1(x,Ly,E,Noeud, Tbc) * Poutre_flexion.N4(x,dx) , (x,a1,a2));
    Mcouplage[1,0]=sp.integrate( H2(x,Ly,E,Noeud, Tbc) * Poutre_flexion.N1(x,dx) , (x,a1,a2));
    Mcouplage[1,1]=sp.integrate( H2(x,Ly,E,Noeud, Tbc) * Poutre_flexion.N2(x,dx) , (x,a1,a2));
    Mcouplage[1,2]=sp.integrate( H2(x,Ly,E,Noeud, Tbc) * Poutre_flexion.N3(x,dx) , (x,a1,a2));
    Mcouplage[1,3]=sp.integrate( H2(x,Ly,E,Noeud, Tbc) * Poutre_flexion.N4(x,dx) , (x,a1,a2));
    Mcouplage[2,0]=sp.integrate( H3(x,Ly,E,Noeud, Tbc) * Poutre_flexion.N1(x,dx) , (x,a1,a2));
    Mcouplage[2,1]=sp.integrate( H3(x,Ly,E,Noeud, Tbc) * Poutre_flexion.N2(x,dx) , (x,a1,a2));
    Mcouplage[2,2]=sp.integrate( H3(x,Ly,E,Noeud, Tbc) * Poutre_flexion.N3(x,dx) , (x,a1,a2));
    Mcouplage[2,3]=sp.integrate( H3(x,Ly,E,Noeud, Tbc) * Poutre_flexion.N4(x,dx) , (x,a1,a2));
    Mcouplage[3,0]=sp.integrate( H4(x,Ly,E,Noeud, Tbc) * Poutre_flexion.N1(x,dx) , (x,a1,a2));
    Mcouplage[3,1]=sp.integrate( H4(x,Ly,E,Noeud, Tbc) * Poutre_flexion.N2(x,dx) , (x,a1,a2));
    Mcouplage[3,2]=sp.integrate( H4(x,Ly,E,Noeud, Tbc) * Poutre_flexion.N3(x,dx) , (x,a1,a2));
    Mcouplage[3,3]=sp.integrate( H4(x,Ly,E,Noeud, Tbc) * Poutre_flexion.N4(x,dx) , (x,a1,a2));
    
    
    return Mcouplage
 

def MatCouplage_ana(dx):
    
    Mcouplage=np.zeros((4,4),dtype="float64");
    
    Mcouplage[1,0]=3.0/20.0
    Mcouplage[1,1]=dx/30.0
    Mcouplage[1,2]=7.0/20.0
    Mcouplage[1,3]=-dx/20.0
    
    Mcouplage[0,0]=7.0/20.0
    Mcouplage[0,1]=dx/20.0     
    Mcouplage[0,2]=3.0/20.0
    Mcouplage[0,3]=-dx/30.0
    
    
    return Mcouplage*dx

   

#Fonction test H1 :   
def H1 (x,y,E, Noeud, Tbc):
    
    return  ((x-Noeud[Tbc[E,2],0])*(y-Noeud[Tbc[E,2],1]))/((Noeud[Tbc[E,0],0]-Noeud[Tbc[E,2],0])*(Noeud[Tbc[E,0],1]-Noeud[Tbc[E,2],1]))

#Fonction test H2  :  
def H2 (x,y,E, Noeud, Tbc):
    
    return ((x-Noeud[Tbc[E,3],0])*(y-Noeud[Tbc[E,3],1]))/((Noeud[Tbc[E,1],0]-Noeud[Tbc[E,3],0])*(Noeud[Tbc[E,1],1]-Noeud[Tbc[E,3],1]))   


#Fonction test H3  :  
def H3 (x,y,E, Noeud, Tbc):
    
    return ((x-Noeud[Tbc[E,0],0])*(y-Noeud[Tbc[E,0],1]))/((Noeud[Tbc[E,2],0]-Noeud[Tbc[E,0],0])*(Noeud[Tbc[E,2],1]-Noeud[Tbc[E,0],1]))    

#Fonction test H4  :
def H4 (x,y,E, Noeud, Tbc):
    
    return  ((x-Noeud[Tbc[E,1],0])*(y-Noeud[Tbc[E,1],1]))/((Noeud[Tbc[E,3],0]-Noeud[Tbc[E,1],0])*(Noeud[Tbc[E,3],1]-Noeud[Tbc[E,1],1]))


def matrices_2D(Lx,Ly,Nx,Ny, dx, dy):
    


    #----------------Creation maillage------------------------------

    NE=(Nx)*(Ny);#nombre de noeuds
    Ns=4+(Nx-2)*2
    ne=4;#taille matrice élémentaire

    x, y = sp.symbols('x ,y');#variables pour le calcul formel

    K=np.zeros((NE,NE),dtype="float64")# matrice 
    M=np.zeros((NE,NE),dtype="float64")# matrice M
    Ke=np.zeros((ne,ne),dtype="float64")#matrice élémentaire 
    Me=np.zeros((ne,ne),dtype="float64")#matrice élémentaire 
    Mecouplage=np.zeros((ne,ne),dtype="float64")
    Mcouplage=np.zeros((NE,Ns),dtype="float64")
    
    Tbc,Noeud=tables (Lx,Ly,Nx,Ny,'carre')
    
    
    #-------------------Assemblage-----------------------------
    
    #------Assemblage de k et M avec les matrices Ke et Me:
  
        
    Ke=MatK(dx, dy)
    Me=MatM_ana(dx, dy)
    
    
    
          
    for i in range(Nx*(Ny-1),Nx*Ny-1):
        
        M[i:i+2,i:i+2]=M[i:i+2,i:i+2]+Me
        
        
    for k in range((Nx-1)*(Ny-1)):
        list=np.array([Tbc[k,0],Tbc[k,1],Tbc[k,2],Tbc[k,3]])
        
        
        
        # Me=MatM(x,y,k,Noeud[Tbc[k,0],1], Noeud[Tbc[k,3],1],Noeud[Tbc[k,0],0], Noeud[Tbc[k,1],0],Noeud, Tbc)
        # print('K:\n',K)
    
        for i in range(ne):
            for j in range(ne):
                K[list[i],list[j]]= K[list[i],list[j]]+Ke[i,j]
        
     
    #-----Assemblage de la matrice de couplage:
        
    E_list=np.zeros(Nx-1,dtype="int");#liste qui contiendra les éléments sur le bord
   
        
        
    for k in range(Nx-1):
        E_list[k]=k #remplissage de la liste avec les éléments sur le bord
    

    # Mecouplage=MatCouplage_ana(dx)
    # Mecouplage=MatCouplage(x, 0.0, dx, 0.0, dx, Tbc, Noeud, 0)
    for k in range(Nx-1):
        
        list=np.array([Tbc[E_list[k],0], Tbc[E_list[k],1], Tbc[E_list[k],2],Tbc[E_list[k],3]])#liste avec les noeuds d'un élément k
     
        Mecouplage=MatCouplage(x, Ly, dx, Noeud[Tbc[k,0],0], Noeud[Tbc[k,1],0], Tbc, Noeud, k)
        #Assemblage de la matrice de couplage:   
        for i in range(ne):
            for j in range(ne):
                Mcouplage[list[i],list[j]]= Mcouplage[list[i],list[j]]+Mecouplage[i,j]




    return M, K, Mcouplage

def graph_cavite2D(Nx, Ny, Lx, Ly, Aff_min, V_reel, NE, compteur):
    
    Z=np.zeros((Ny,Nx),dtype="float64")
    
    x_graph=np.linspace(0,Lx,Nx)
    y_graph=np.linspace(0,Ly,Ny)
    
    X,Y=np.meshgrid(x_graph,y_graph)
    
    
    Aff_max=Aff_min+4
    
    
    if (Aff_min>Aff_max):
        
        print('Aff_min>Aff_max \n')
        sys.exit()
        
    if(Aff_max>compteur-1):
        print('Aff_max supérieur à Dims+Dimc-1\n')
        sys.exit()
    

    
    fig, axs = plt.subplots(2, 2,sharex='col', figsize=(10,10))
    
    V_reel_max=np.max(V_reel)
    V_reel_min=np.min(V_reel)
    
    compteur=0
    for i in range(2):
        for p in range(2):
            for k in range(Ny):
                for j in range(Nx):
                    
                    Z[k,j]=V_reel[Nx*k+j,compteur+Aff_min]
            
            c=axs[i,p].contourf(X,Y,Z,100, cmap='jet', vmin=V_reel_min, vmax=V_reel_max)
            axs[i,p].set_ylabel('y(m)')
            axs[i,p].set_xlabel('x(m)')
            axs[i,p].set_title('Pressions (relatives) - Mode {}'.format(compteur+Aff_min+1)) 
            #fig.colorbar(c, ax=axs[i+1,p])
            compteur=compteur+1
    
    # fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.02, hspace=0.02)
    # cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    # cbar = fig.colorbar(c, ax=axs.ravel().tolist(), orientation='horizontal', extend='max')
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    cax = fig.add_axes([0.12, 0.05, 0.78, 0.02])
    fig.colorbar(c, cax=cax, orientation='horizontal')
    
    fig.show() 
    
    return None



def Analyse_modale(dx,dy,c, rho, Lx, Ly, Nx,Ny):
    
    
   NE=Nx*Ny; 
   M,K, Mc = matrices_2D(Lx,Ly,Nx,Ny, dx, dy)
    
   M=M/9.81
   
   ##########################
   #Résolution analyse modale
   ##########################
   
   #W,V=np.linalg.eig(np.linalg.inv(M)@K);
   W,V=spl.eig(K,M) 
 


   W_tri=np.zeros((Nx,2),dtype="float64");
   W_tri2=np.zeros((Nx,1),dtype="float64");    
   V_reel=np.zeros((NE,Nx),dtype="float64");
   V_reel2=np.zeros((NE,Nx),dtype="float64");    

   #Tri des valeurs:
   compteur=0
    
   for i in range(NE):
       if abs(W[i].real)<10e-3:
           W[i]=0.0 #filtrage
       if W[i].real != np.inf:
           W_tri[compteur,0]=np.sqrt(W[i].real)/(2.0*np.pi)
           W_tri[compteur,1]=compteur
           V_reel[:,compteur]=V[:,i].real
           compteur=compteur+1
           
    

   idx = np.argsort(W_tri[:,0])


   for j in range(Nx):
       W_tri2[j] =W_tri[idx[j],0]
       for i in range(NE):
           V_reel2[i,j]=V_reel[i,idx[j]];

  

   print("Fréquences de résonnance de la ballotement (10 premières) : \n ", W_tri2[:10])

   return W_tri2, V_reel2,compteur
   
   
Lx=1.0;#m
Ly=0.5;#m

Nx=10;#points de discrétisation selon x
Ny=10;#points de discrétisation selon y

NE=Nx*Ny

dx=float(Lx/(Nx-1));#pas selon x
dy=float(Ly/(Ny-1));#pas selon y    

rho_c=1000.0
c=340.0
 
Aff_min=1



# W,V_reel,compteur=Analyse_modale(dx,dy,c, rho_c, Lx, Ly, Nx,Ny)   

# graph_cavite2D(Nx, Ny, Lx, Ly, Aff_min, V_reel, NE, compteur)







   
