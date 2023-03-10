# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 13:06:21 2022

@author: Jérémy Archier - jeremy.archier@etu.univ-lyon1.fr

Résolution des mode de ballottement d'une surface libre dans un cavité fermé 
d'un fluide 2D à l'aide des EF.

Paramètres:
    
    - L : la longeur de la cavité [m]
    - h : la hauteur de la cavité [m]
    - g : Constante gravitationnelle [m^3 kg^−1 s^−2]

"""

import matplotlib.pyplot as plt
import scipy.linalg as spl
import numpy as np

def tables(Lx=10,Ly=20,Nx=3,Ny=5,e='triangle'):
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
    i=0
    j=0

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

def dist(y1,y2,y3,y4):
    #Pour faciliter l'écriture dans l'initialisation des matrices élémentaires
    return (y1-y2)*(y3-y4)

def calculMatricesElement(dx, dz): 
    """
    Calcule les matrices de raideur pour un élément quadrangulaire
    
    Paramètres d'entrée:
        
        dx, dz les pas d'espace selon les directions x et z 
        
    Retourne:
        - Kx : matrice de raideur x élémentaire (voir rapport) de taille (4,4)
        - Kz : matrice de raideur z élémentaire (voir rapport) de taille (4,4)
    
    """
    
    K=np.zeros((4,4),dtype="float64");
   
    K[0,0]=(dx**2.0+dz**2.0)/(3.0*dx*dz)
    K[0,1]=(dx**2.0-2.0*dz**2.0)/(6.0*dx*dz)
    K[0,2]=-(dx**2.0+dz**2.0)/(6.0*dx*dz)
    K[0,3]=(dz**2.0-2.0*dx**2.0)/(6.0*dx*dz)
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

def calculMatriceMasse(dx):
    """
    Calcule la matrice de masse élémentaire Me
    
    Paramètre d'entrée:
        - dx : pas de discrétisation 
    
    Retourne:
        - Me : Matrice de masse élémentaire (voir rapport) de dimension (4,4)
    """
    # M=np.zeros((4,4),dtype="float64");
    
    # M[0,0]=1.0
    # M[0,1]=0.5
    # M[0,2]=0.25
    # M[0,3]=0.5
    # M[1,0]=M[0,1]
    # M[1,1]=M[0,0]
    # M[1,2]=0.5
    # M[1,3]=0.25
    # M[2,0]=M[0,2]
    # M[2,1]=M[1,2]
    # M[2,2]=M[0,0]
    # M[2,3]=0.5
    # M[3,0]=M[0,3]
    # M[3,1]=M[1,3]
    # M[3,2]=M[2,3]
    # M[3,3]=M[0,0]

    # M = dx/(9.81*6.) * M
    
    M = np.zeros((2,2))
    
    M[0,:] = np.array([2,1])
    M[1,:] = np.array([1,2])
    
    M = dx/(9.81*6.) * M
    
    return M

def assembler2D(K, Ne, tbc, coord, dx, dz):
    """
    Assemble les matrices de raideur Kx,Kz élémentaires dans la matrice de raideur globale K.
    
    Paramètres d'entrée:
        - Ne : Nombre d'éléments
        - Npoints : Nombre de points
        - tbc : table de connexions
        - coord : table des coordonnées 
        - K : matrice de raideur globale de taille (Npoints,Npoints)
    """

    for e in range(Ne): #Pour chaque élément
        Ke = calculMatricesElement(dx, dz) #Génération des matrices élémentaires
        for j in range(4):
            for k in range(4):
                K[tbc[e,j],tbc[e,k]] += Ke[j,k] #Assembler
        
def assembler1D(M, Npoints, dx, n_x):
    """
    Assemble la matrice de masse élémentaire Me dans la matrice de masse globale M.
    
    La matrice Me n'agit que sur des éléments sur l'axe z=0 (axe horizontal supérieur). Ces
    points correspondent aux n_x derniers points du vecteur solution 1D, c'est-à-dire les points
    de Npoints - n_x à Npoints-1.
    
    
    Paramètres d'entrée:
        - M: Matrice de masse globale de taille (Npoints, Npoints)
        - Npoints : nombre de points/noeuds de discrétisation TOTAL
        - dx : pas de discrétisation
        - n_x : nombre de points sur l'axe x
    """
    
    Me = calculMatriceMasse(dx) #générer la matrice de masse élémentare
    
    for i in range(Npoints-n_x,Npoints-1): #Sur tous les points de l'axe 
        M[i:i+2,i:i+2] += Me
    
       
def remplirMatrice2D(n_x,n_y,P):
    """
    Transforme le vecteur solution obtenu par éléments finis en matrice 2D 
    pour les tracés.
    
    Paramètres d'entrée:
        - n_x : nombre de points en x
        - n_z : nombre de points en z
        - P   : vecteur solution 1D de longueur Npoints*Npoint
        
    Retourne:
        - Press2D : Matrice de taille (Npoints,Npoints)
    """
    Press2D = np.zeros((n_y,n_x))
    for i in range(n_y):
        Press2D[i,:] = P[i*n_x:i*n_x+n_x]
        
    return Press2D

def find_min(val):
    """
    Trouve la position du minimum d'un vecteur.
    
    Paramètres d'entrée:
        - val : vecteur 1D 
        
    Retourne:
        - index : la position du minimum dans le vecteur
        - mini  : le minimum du vecteur
        
    Exemple: Si val = [1,4,1.5,9], find_min(val) renvoie (2,1.5)
    """
    
    index = 0;
    mini = np.min(val);
    for i in val:
        if abs(mini-i) < 0.00000001:
            return index,mini
        else:
            index += 1
    return index, mini;

def trier_modes(val,vec):
    """
    Trie les modes dans l'ordre croissant car linalg.eig ne rend pas les 
    valeurs propres dans l'ordre
    
    Paramètres d'entrée:
        - val : vecteur des valeurs propres 
        - vec : matrice de vecteurs propres
    
    Retourne:
        - tri_val : les valeurs propres triées dans l'ordre croissant
        - tri_vec : les vecteurs propres associés aux valeurs propres triées
    """
    tri_val = np.copy(val); #Va contenir les valeurs propres triées
    tri_vec = np.copy(vec); #Va contenir les vecteurs propres triés, correspondant aux valeur propres
    i=0;
    while (len(val) != 0):
        index, mini = find_min(val); #On récupère le minimum des valeurs propres et l'index correspondant
        tri_val[i] = mini; 
        tri_vec[:,i] = vec[:,index];
        val = np.delete(val, index);
        vec = np.delete(vec, index, axis=1);
        i += 1

    return tri_val, tri_vec;
    
def tracer_potentiel(potentiel, index_mode, ax, fig,  L, n_x, H, n_z, colormap = 'jet'):
    """
    Trace le potentiel de déplacement calculé avec par éléments finis (vecteur propre) sur une 
    figure de type subplots : fig,ax = plt.subplots()
        
    Paramètres d'entrée : 
        - potentiel : la matrice des vecteurs propres du potentiel pour chaque mode
        - index_mode : l'index du mode à tracer (position dans la matrice des vecteurs propres)
        - ax  : axis du plt.subplots()
        - fig : fig du plt.subplots()
        - L   : dimension x de la cavité
        - n_x : nombre de points en x
        - H   : dimension z de la cavité
        - n_z : nombre de points en z
        - colormap : le jeu de couleurs pour l'affichage des contours. Quelques exemples 
        pour d'autres couleurs : jet, rainbow, ocean, brg... 
        Plus de détails : https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    """
    x,z = np.meshgrid(np.linspace(-0.5*L, 0.5*L, n_x), np.linspace(-H, 0., n_z)) 
    potentiel_a_tracer = remplirMatrice2D(n_x, n_z, potentiel[:,index_mode]) 
    
    
    """
    Quelques exemples de jeu de couleurs pour le contourf:
        - jet 
        - rainbow 
        - 
        
    """
    
    cset = ax.contourf(x, z, potentiel_a_tracer, 100, cmap=colormap)
    fig.colorbar(cset, ax = ax)
    
    ax.set_title('Mode {} EF'.format(index_mode),fontsize=18)

def tracer_deplacement(potentiel, index_mode, ax, fig,  L, n_x, H, n_z, colormap = 'jet'):
    """
    Trace le champ déplacement sur une figure de type subplots : fig,ax = plt.subplots(),
    calculé avec la fonction np.gradient() sur le potentiel de déplacement, car le déplacement
    u peut s'exprimer u = d(potentiel)/dz
        
    Paramètres d'entrée : 
        - potentiel : la matrice des vecteurs propres du potentiel pour chaque mode
        - index_mode : l'index du mode à tracer (position dans la matrice des vecteurs propres)
        - ax  : axis du plt.subplots()
        - fig : fig du plt.subplots()
        - L   : dimension x de la cavité
        - n_x : nombre de points en x
        - H   : dimension z de la cavité
        - n_z : nombre de points en z
        - colormap : le jeu de couleurs pour l'affichage des contours. 
        
        Quelques exemples pour d'autres couleurs : jet, rainbow, ocean, brg... 
        Plus de détails : https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    """
    x,z = np.meshgrid(np.linspace(-0.5*L, 0.5*L, n_x), np.linspace(-H, 0., n_z)) #Domaine 2D pour le tracé

    potentiel_a_tracer = remplirMatrice2D(n_x, n_z, potentiel[:,index_mode]) #Transformer vecteur en matrice
    deplacement_a_tracer = np.gradient(potentiel_a_tracer, edge_order=2)[0] #Dériver le potentiel pour avoir déplacement
    
    cset = ax.contourf(x,z,deplacement_a_tracer, 100, cmap=colormap)
    fig.colorbar(cset, ax = ax)
    ax.set_title('Mode {} EF'.format(index_mode),fontsize=18)

def resoudre_vp(K, M):
    """
    Résout le problème aux valeurs propres
    
    Paramètres d'entrée:
        - K : Matrice de raideur de dimensions (Npoints, Npoints)
        - M : Matrice de masse de dimensions (Npoints, Npoints)
        
    Retourne:
        - omega_carre : valeurs propres triées dans l'ordre croissant
        - potentiel : vecteurs propres associés aux valeurs propres
    """
    
    
    val, vec = spl.eig(K,M)  #Résolution avec le module scipy.linalg
    omega = np.zeros_like(val, dtype = np.float64) #Va stocker les vp triées
    
    #On élimine les valeurs infinies et négatives, et on convertit tout en réel (car spl.eig donne 
    #des valeurs complexes)
    for i in range(len(val)):
        
        if val[i] == np.inf: #Si on trouve des valeurs infinies, on met juste une grosse valeur
            omega[i] = 300000000.
        
        elif val[i] < 10**(-8): #Si c'est proche de 0, c'est que c'est 0...
            omega[i] = 0.
        
        #Sinon, il faut penser à récupérer uniquement la partie réelle (même si la partie 
        #imaginaire est nulle)
        else:
            omega[i] = val[i].real
    
    omega_carre, potentiel = trier_modes(omega, vec)
    
    return omega_carre, potentiel
    

#=================================================================================================================
 #                                        ASSEMBLAGE ET CALCULS
#=================================================================================================================

L = 1.0 #Dimension x
H = 0.5 #Dimension y

n_x = 10 #Nombre de points de discrétisation en x
n_z = 10     #Nombre de points de discrétisation en y

dx = L/(float(n_x-1)) #Pas de discrétisation en x
dz = L/(float(n_z-1)) #Pas de discrétisation en x

[tbc,coord] = tables(Lx = L, Ly = H, Nx = n_x, Ny = n_z,e = 'carre'); #On récupère la table de coordonnées et table de connexion

#Ajuster les coordonnées de -L/2 à L/2 et de -H à 0
coord[:,0] -= L*0.5
coord[:,1] -= H

Ne = len(tbc) #Nombre d'éléments
Npoints = len(coord) #Nombre de points

print(f"Calcul lancé avec {Ne} éléments")

#Assemblage de la matrice avec les éléments 2D et 1D

print("Assemblage des matrices K et M...")
K,M = np.zeros((Npoints,Npoints)),np.zeros((Npoints,Npoints))
assembler2D(K, Ne, tbc, coord, dx, dz) #Assembler matrice de raideur
assembler1D(M, Npoints, dx, n_x) #Assembler matrice de masse
print("Assemblage : fait")

print("Calcul des valeur propres...")
val, vec = resoudre_vp(K, M) #Calculer les valeurs propres 
print("Calcul des valeurs propres : fait")

mode_max = 4
f = np.zeros(mode_max) #pour stocker les 4 premiers modes

index = 0

for i in val[1:mode_max+1]: #On parcourt les 4 premiers modes (on ne prend pas 0)
    f[index] = 1./(2.0*np.pi)*np.sqrt(i)
    print(f'\nMode {index+1} (EF): {f[index]} Hz')
    
    #----------------------- valeurs analytiques pour comparaison --------------------------
    k_ana = np.pi*(index+1)/L; #Nombre d'onde
    omega_ana = np.sqrt(k_ana*9.81*np.tanh(k_ana*H)); #Pulsation
    f_ana = omega_ana/(2.*np.pi)
    print(f'Mode {index+1} (analytique): {f_ana} Hz')
    #---------------------------------------------------------------------------------------
    
    
    err_rel = abs(f[index]-f_ana)/f_ana*100. #Erreur relative 
    print(f'--> erreur relative : {err_rel} %\n')
    index += 1


#=========================================================================================
#                                   TRACE DES SOLUTIONS
#=========================================================================================

#--------------------------- AJUSTER CE QU'ON VEUT TRACER ICI ----------------------------
trace_potentiel = False #Mettre à True pour tracer le potentiel
trace_deplacement = True #Mettreà True pour tracer le déplacement

if trace_potentiel:
    print("Tracé du potentiel")
    
if trace_deplacement:
    print("Tracé du champ de déplacement")

#---------------------------------- TRACER POTENTIEL -----------------------------------
if (trace_potentiel): 
    
    # Plage de modes à tracer 
    index_mode_min = 1 # *
    index_mode_max = 5 # *
    
    fig, axs = plt.subplots(nrows=2,ncols=2,sharey=True,sharex=True, figsize=(9.0,7.0)) # * (sauf le figsize)
    
    for index_mode in range(1,len(val[index_mode_min:index_mode_max+1])):    # *
        tracer_potentiel(vec, index_mode, axs[(index_mode-1)//2,(index_mode-1)%2], # *
                                  fig, L, n_x, H, n_z) # *
        
    axs[1,0].set_xlabel("x (m)",fontsize=18)
    axs[1,1].set_xlabel("x (m)",fontsize=18)
    axs[1,0].set_ylabel("z (m)",fontsize=18)
    axs[0,0].set_ylabel("z (m)",fontsize=18)
#---------------------------------------------------------------------------------------

#---------------------------------- TRACER DEPLACEMENT -----------------------------------

if (trace_deplacement): 
    
    # Plage de modes à tracer
    index_mode_min = 1 # * 
    index_mode_max = 5 # * 
    fig, axs = plt.subplots(nrows=2,ncols=2,sharey=True,sharex=True, figsize=(9.0,7.0)) # * (sauf le figsize)
    
    for index_mode in range(1,len(val[index_mode_min:index_mode_max+1])): # * 
        tracer_deplacement(vec, index_mode, axs[(index_mode-1)//2,(index_mode-1)%2], # * 
                                  fig, L, n_x, H, n_z) # * 
        
    axs[1,0].set_xlabel("x (m)",fontsize=18)
    axs[1,1].set_xlabel("x (m)",fontsize=18)
    axs[1,0].set_ylabel("z (m)",fontsize=18)
    axs[0,0].set_ylabel("z (m)",fontsize=18)
#---------------------------------------------------------------------------------------